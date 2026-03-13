"""Agent Runtime — the main event loop that ties everything together.

This is the beating heart of the living agent. It:
  1. Listens for audio input (Whisper) or system events
  2. Routes through the Thinker (Qwen 3.5 with LoRA)
  3. Parses output into think/tool/speak blocks
  4. Executes tools, manages memory
  5. Sends speech blocks to the Talker (Chatterbox)
  6. Manages state transitions and interruptions
  7. Runs background tasks (compression, memory) on base model

Platform: Designed for Windows (WSL/Docker for Linux-specific tools).
"""

import re
import json
import time
import asyncio
import logging
import uuid
from typing import Optional, Callable, AsyncGenerator

from .state_machine import AgentState, StateMachine
from .tools import ToolExecutor
from .memory import MemoryManager

log = logging.getLogger(__name__)


# ── Output block parsing ─────────────────────────────────────────────

BLOCK_PATTERN = re.compile(
    r"<(think|tool_call|tool_result|speak|interrupted|tool_running)(?:\s+([^>]*))?>(?:(.*?)</\1>)?",
    re.DOTALL,
)


def parse_speak_attrs(attr_str: str) -> dict:
    """Parse speak block attributes like emotion='happy' speed='1.0'."""
    attrs = {}
    if not attr_str:
        return attrs
    for m in re.finditer(r'(\w+)=["\']([^"\']*)["\']', attr_str):
        key, val = m.group(1), m.group(2)
        try:
            attrs[key] = float(val)
        except ValueError:
            attrs[key] = val
    return attrs


def parse_output_blocks(text: str) -> list[dict]:
    """Parse model output into structured blocks.

    Returns list of dicts with 'type' and type-specific fields:
      {'type': 'think', 'content': '...'}
      {'type': 'tool_call', 'name': '...', 'args': {...}}
      {'type': 'speak', 'text': '...', 'emotion': '...', 'speed': ..., 'energy': ...}
      {'type': 'interrupted'}
      {'type': 'tool_running'}
    """
    blocks = []
    for m in BLOCK_PATTERN.finditer(text):
        block_type = m.group(1)
        attrs = m.group(2) or ""
        content = m.group(3) or ""

        if block_type == "think":
            blocks.append({"type": "think", "content": content.strip()})

        elif block_type == "tool_call":
            try:
                call = json.loads(content.strip())
                blocks.append({"type": "tool_call", "name": call["name"], "args": call.get("args", {})})
            except (json.JSONDecodeError, KeyError) as e:
                log.warning(f"Bad tool_call JSON: {e}")

        elif block_type == "speak":
            parsed_attrs = parse_speak_attrs(attrs)
            blocks.append({
                "type": "speak",
                "text": content.strip(),
                "emotion": parsed_attrs.get("emotion", "neutral"),
                "speed": parsed_attrs.get("speed", 1.0),
                "energy": parsed_attrs.get("energy", 0.7),
                "pitch": parsed_attrs.get("pitch", 0.0),
            })

        elif block_type == "interrupted":
            blocks.append({"type": "interrupted"})

        elif block_type == "tool_running":
            blocks.append({"type": "tool_running"})

    return blocks


# ── Main Runtime ─────────────────────────────────────────────────────

class AgentRuntime:
    """The living agent runtime.

    Wires together: state machine, tools, memory, and the model.
    Runs as an async event loop.

    Usage:
        runtime = AgentRuntime(workspace="~/my-project")
        await runtime.start()

        # Feed user input
        await runtime.on_user_speech("hey how's it going?")

        # Or system events
        await runtime.on_system_event("[CI webhook: build failed]")
    """

    def __init__(
        self,
        workspace: str = ".",
        conversation_id: str = None,
        # These are injected — the runtime doesn't own the models
        generate_fn: Callable = None,  # async fn(messages, **kwargs) -> str
        summarize_fn: Callable = None,  # async fn(text) -> str (base model, no LoRA)
        speak_fn: Callable = None,     # async fn(text, emotion, speed, energy) -> audio_bytes
        stream_speak_fn: Callable = None,  # async generator fn(text, emotion, ...) -> yields audio chunks
        transcribe_fn: Callable = None,  # async fn(audio_bytes, sample_rate) -> str (STT)
    ):
        self.workspace = workspace
        self.conversation_id = conversation_id or str(uuid.uuid4())[:8]

        # Core components
        self.state = StateMachine()
        self.tools = ToolExecutor(workspace=workspace)
        self.memory = MemoryManager(workspace=workspace)

        # Model functions (injected)
        self._generate = generate_fn
        self._summarize = summarize_fn
        self._speak = speak_fn
        self._stream_speak = stream_speak_fn
        self._transcribe = transcribe_fn

        # Conversation history
        self.messages: list[dict] = []

        # Background tasks
        self._bg_tasks: list[asyncio.Task] = []
        self._running = False

        # Callbacks
        self._on_speak: list[Callable] = []  # (text, emotion, audio_bytes) -> None
        self._on_audio_chunk: list[Callable] = []  # (audio_chunk) -> None (streaming)
        self._on_state_change: list[Callable] = []

        # Wire up state machine listener
        self.state.on_transition(self._on_transition)

    def on_speak(self, callback: Callable):
        """Register callback for when agent speaks. Gets (text, emotion, audio)."""
        self._on_speak.append(callback)

    def on_audio_chunk(self, callback: Callable):
        """Register callback for streaming audio chunks. Gets (AudioChunk)."""
        self._on_audio_chunk.append(callback)

    async def start(self):
        """Start the runtime event loop."""
        self._running = True
        log.info(f"Agent runtime started (conversation: {self.conversation_id})")

        # Load any existing context
        context = self.memory.context.get_context(self.conversation_id)
        if context:
            for chunk in context:
                if chunk["type"] == "summary":
                    self.messages.append({
                        "role": "system",
                        "content": f"[ROLLING_CONTEXT_SUMMARY]\n{chunk['content']}\n[/ROLLING_CONTEXT_SUMMARY]"
                    })

        # Start background maintenance loop
        self._bg_tasks.append(asyncio.create_task(self._background_loop()))

    async def stop(self):
        """Stop the runtime."""
        self._running = False
        for task in self._bg_tasks:
            task.cancel()
        log.info("Agent runtime stopped")

    # ── Input handlers ───────────────────────────────────────────────

    async def on_user_speech(self, text: str) -> list[dict]:
        """Handle user speech input. Returns list of output blocks."""
        # Interrupt if needed
        if self.state.state in {AgentState.SPEAKING, AgentState.THINKING, AgentState.WORKING}:
            self.state.request_interrupt()
        elif self.state.state == AgentState.TOOL_USING:
            self.state.request_interrupt()  # Will trigger after tool finishes

        self.state.transition(AgentState.LISTENING)

        # Add to conversation
        self.messages.append({"role": "user", "content": text})
        self.memory.context.store_turn(self.conversation_id, f"User: {text}")

        # Process
        self.state.transition(AgentState.THINKING)
        return await self._process()

    async def on_system_event(self, event: str) -> list[dict]:
        """Handle a system event (CI webhook, timer, etc.)."""
        self.messages.append({"role": "system", "content": event})
        self.memory.context.store_turn(self.conversation_id, f"System: {event}")

        if self.state.state == AgentState.IDLE:
            self.state.transition(AgentState.THINKING)
            return await self._process()
        return []

    # ── Core processing loop ─────────────────────────────────────────

    async def _process(self) -> list[dict]:
        """Run the think → tool → speak loop. Returns all output blocks."""
        if not self._generate:
            log.error("No generate_fn provided")
            return []

        all_blocks = []
        max_iterations = 10  # Safety limit

        for _ in range(max_iterations):
            # Check for interruption
            if self.state.check_interrupt():
                all_blocks.append({"type": "interrupted"})
                break

            # Generate next output from model
            raw_output = await self._generate(self.messages)
            blocks = parse_output_blocks(raw_output)

            if not blocks:
                self.state.transition(AgentState.IDLE)
                break

            for block in blocks:
                if block["type"] == "think":
                    all_blocks.append(block)

                elif block["type"] == "tool_call":
                    self.state.transition(AgentState.TOOL_USING)
                    name, args = block["name"], block["args"]

                    # Route to memory or tool executor
                    if name in self.memory.memory_tools:
                        if name == "context_compress" and self._summarize:
                            result = await self.memory.context.compress(
                                args.get("conversation_id", self.conversation_id),
                                self._summarize,
                            )
                            result = json.dumps(result)
                        else:
                            result = await self.memory.execute(name, args)
                    else:
                        result = await self.tools.execute(name, args)

                    # Add tool result to conversation
                    tool_block = {"type": "tool_result", "content": result}
                    all_blocks.append(block)
                    all_blocks.append(tool_block)

                    # Add to messages for context
                    self.messages.append({
                        "role": "assistant",
                        "content": f'<tool_call>{json.dumps({"name": name, "args": args})}</tool_call>\n<tool_result>{result}</tool_result>'
                    })

                    # Check interrupt after tool
                    if self.state.check_interrupt():
                        all_blocks.append({"type": "interrupted"})
                        return all_blocks

                elif block["type"] == "speak":
                    self.state.transition(AgentState.SPEAKING)
                    all_blocks.append(block)

                    # Streaming TTS — yields audio chunks as they're generated
                    if self._stream_speak and self._on_audio_chunk:
                        async for chunk in self._stream_speak(
                            block["text"], block["emotion"],
                            block["speed"], block["energy"],
                        ):
                            # Check for interruption mid-speech
                            if self.state.check_interrupt():
                                all_blocks.append({"type": "interrupted"})
                                return all_blocks

                            for cb in self._on_audio_chunk:
                                try:
                                    await cb(chunk)
                                except Exception as e:
                                    log.error(f"Audio chunk callback error: {e}")

                    # Fallback: non-streaming TTS
                    elif self._speak:
                        audio = await self._speak(
                            block["text"], block["emotion"],
                            block["speed"], block["energy"],
                        )
                        block["audio"] = audio

                    # Notify speak-complete listeners
                    for cb in self._on_speak:
                        try:
                            await cb(block["text"], block["emotion"], block.get("audio"))
                        except Exception as e:
                            log.error(f"Speak callback error: {e}")

                    # Store in context
                    self.memory.context.store_turn(
                        self.conversation_id,
                        f"Agent [{block['emotion']}]: {block['text']}"
                    )

                elif block["type"] == "interrupted":
                    all_blocks.append(block)
                    self.state.transition(AgentState.LISTENING)
                    return all_blocks

            # If last block was speak, we're done for now
            if blocks and blocks[-1]["type"] == "speak":
                self.state.transition(AgentState.IDLE)
                break

        return all_blocks

    # ── Background maintenance ───────────────────────────────────────

    async def _background_loop(self):
        """Background maintenance loop (runs on base model, LoRA off)."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Check every minute

                # Only do background work when idle
                if self.state.state != AgentState.IDLE:
                    continue

                self.state.transition(AgentState.BACKGROUND)

                # Check if context needs compression
                if self.memory.context.needs_compression(self.conversation_id):
                    if self._summarize:
                        log.info("Background: compressing context")
                        result = await self.memory.context.compress(
                            self.conversation_id, self._summarize,
                        )
                        log.info(f"Background: compressed {result}")

                self.state.transition(AgentState.IDLE)

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Background loop error: {e}")
                self.state.force_state(AgentState.IDLE)

    # ── State change handler ─────────────────────────────────────────

    def _on_transition(self, old: AgentState, new: AgentState):
        """Called on every state transition."""
        for cb in self._on_state_change:
            try:
                cb(old, new)
            except Exception as e:
                log.error(f"State change callback error: {e}")
