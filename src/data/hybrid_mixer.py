"""Hybrid Mixer — merges multiple dataset sources into living agent training data.

The pipeline:
  1. Load converted turns from each source (computer use, web, conversation, tools, traces)
  2. For each training sample, pick a scenario and domain
  3. Pull relevant fragments from source datasets
  4. Call an LLM to STITCH fragments into a coherent long conversation:
     - Wrap text in <speak> blocks with emotions and speech tags
     - Add <think> blocks for internal reasoning
     - Layer in interruptions, memory ops, autonomous behaviors
     - Make it 50-100+ turns, 15-30K tokens
  5. Validate format strictly — reject malformed output
  6. Write JSONL

The LLM doesn't invent tool outputs — it uses REAL outputs from source datasets.
It only adds the "living agent" layer: speech, emotion, thinking, interruptions.
"""

import json
import random
import time
import os
import re
import argparse
from pathlib import Path
from typing import Iterator, Optional

from environment import (
    TOOLS, TOOL_CATEGORIES, AGENT_STATES,
    INTERRUPTION_RULES, AUTONOMOUS_BEHAVIORS, SCENARIO_DOMAINS,
)
from dataset_sources import (
    DATASETS, get_converter, download_dataset,
    convert_claude_traces,
)

# =============================================================================
# Format validation
# =============================================================================

VALID_BLOCKS = re.compile(
    r"<(think|tool_call|tool_result|speak|interrupted|tool_running)"
)
SPEAK_BLOCK = re.compile(
    r'<speak\s+emotion="(\w+)"\s+speed="([\d.]+)"\s+energy="([\d.]+)">(.*?)</speak>',
    re.DOTALL,
)
TOOL_CALL_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL)

VALID_EMOTIONS = {
    "neutral", "happy", "sad", "angry", "excited", "empathetic",
    "surprised", "nervous", "calm", "amused", "confused",
}

SPEECH_TAGS = [
    "[laugh]", "[chuckle]", "[sigh]", "[gasp]", "[cough]",
    "[clear throat]", "[sniff]", "[groan]", "[shush]", "[pause]",
]


def validate_turn(turn: dict) -> tuple[bool, str]:
    """Validate a single turn. Returns (is_valid, error_message)."""
    role = turn.get("role")
    content = turn.get("content", "")

    if role not in ("user", "assistant", "system"):
        return False, f"Invalid role: {role}"

    if not content.strip():
        return False, "Empty content"

    if role == "assistant":
        # Must contain at least one valid block
        if not VALID_BLOCKS.search(content):
            # Bare text is okay for very short acknowledgments
            if len(content) > 100:
                return False, "Assistant content >100 chars without any blocks"

        # Validate speak blocks
        for m in SPEAK_BLOCK.finditer(content):
            emotion = m.group(1)
            if emotion not in VALID_EMOTIONS:
                return False, f"Invalid emotion: {emotion}"
            try:
                speed = float(m.group(2))
                energy = float(m.group(3))
                if not (0.3 <= speed <= 2.0):
                    return False, f"Speed out of range: {speed}"
                if not (0.1 <= energy <= 2.0):
                    return False, f"Energy out of range: {energy}"
            except ValueError:
                return False, "Non-numeric speed/energy"

        # Validate tool_call blocks have valid JSON
        for m in TOOL_CALL_BLOCK.finditer(content):
            try:
                call = json.loads(m.group(1))
                if "name" not in call:
                    return False, "tool_call missing 'name'"
            except json.JSONDecodeError:
                return False, f"Invalid JSON in tool_call: {m.group(1)[:100]}"

        # Check for unclosed tags
        for tag in ["think", "speak", "tool_call", "tool_result"]:
            opens = content.count(f"<{tag}")
            closes = content.count(f"</{tag}>")
            if opens != closes:
                return False, f"Unclosed <{tag}> tags: {opens} opens, {closes} closes"

    return True, ""


def validate_conversation(turns: list[dict]) -> tuple[bool, str]:
    """Validate a full conversation."""
    if len(turns) < 4:
        return False, f"Too few turns: {len(turns)}"

    for i, turn in enumerate(turns):
        valid, err = validate_turn(turn)
        if not valid:
            return False, f"Turn {i}: {err}"

    # Check alternation (user/assistant, with system allowed anywhere)
    last_role = None
    for turn in turns:
        role = turn["role"]
        if role == "system":
            continue
        if role == last_role and role != "assistant":
            # Allow consecutive assistant (tool chains)
            return False, f"Consecutive {role} turns"
        last_role = role

    return True, ""


# =============================================================================
# Fragment pool — collect real data fragments from all sources
# =============================================================================

class FragmentPool:
    """Holds converted fragments from all source datasets, organized by type."""

    def __init__(self):
        self.fragments = {
            "computer_use": [],    # (click, type, scroll, etc.) action sequences
            "web_browsing": [],    # Web navigation sequences
            "file_ops": [],        # File management operations
            "terminal": [],        # Command line operations
            "tool_chains": [],     # Multi-step tool use
            "conversations": [],   # Natural conversation turns
            "tool_results": [],    # Real tool outputs (for reuse)
        }
        self._loaded_sources = set()

    def load_source(self, name: str, data_path: str = None, max_samples: int = 500):
        """Load a source dataset into the pool."""
        if name in self._loaded_sources:
            return

        converter = get_converter(name)
        if not converter:
            print(f"  No converter for {name}, skipping")
            return

        if not data_path:
            info = DATASETS[name]
            if info["source"] == "local":
                data_path = info.get("path") or str(Path.home() / ".claude" / "projects")
            else:
                data_path = f"data/sources/{name}"

        if not Path(data_path).exists():
            print(f"  Data not found at {data_path}, skipping {name}")
            return

        print(f"  Loading {name} from {data_path}...")
        count = 0
        for turns in converter(data_path, max_samples=max_samples):
            self._categorize(name, turns)
            count += 1

        self._loaded_sources.add(name)
        print(f"  -> {count} samples loaded from {name}")

    def _categorize(self, source: str, turns: list[dict]):
        """Sort a conversation's turns into fragment categories."""
        for turn in turns:
            content = turn.get("content", "")
            # Categorize based on tool calls present
            if any(t in content for t in ['"click"', '"type_text"', '"key_press"', '"scroll"', '"drag"']):
                self.fragments["computer_use"].append(turn)
            elif any(t in content for t in ['"web_browse"', '"web_search"', '"web_fetch"']):
                self.fragments["web_browsing"].append(turn)
            elif any(t in content for t in ['"move_file"', '"copy_file"', '"delete_file"', '"list_files"']):
                self.fragments["file_ops"].append(turn)
            elif '"run_command"' in content:
                self.fragments["terminal"].append(turn)
            elif "<tool_call>" in content:
                self.fragments["tool_chains"].append(turn)
            elif turn["role"] == "user":
                self.fragments["conversations"].append(turn)

            # Extract tool results for reuse
            for m in re.finditer(r"<tool_result>(.*?)</tool_result>", content, re.DOTALL):
                self.fragments["tool_results"].append(m.group(1).strip())

        # Also store the full conversation
        has_tools = any("<tool_call>" in t.get("content", "") for t in turns)
        if has_tools:
            self.fragments["tool_chains"].append(turns)

    def sample(self, category: str, n: int = 1) -> list:
        """Sample n fragments from a category."""
        pool = self.fragments.get(category, [])
        if not pool:
            return []
        return random.sample(pool, min(n, len(pool)))

    def stats(self):
        """Print pool statistics."""
        print(f"\nFragment Pool:")
        for cat, frags in self.fragments.items():
            print(f"  {cat:20s}: {len(frags)} fragments")
        print(f"  Sources loaded: {', '.join(self._loaded_sources)}")


# =============================================================================
# LLM stitcher — takes fragments and creates living agent conversations
# =============================================================================

STITCHER_PROMPT = """You are creating training data for a living AI computer agent. This agent:
- TALKS to the user naturally (voice in/out, speech tags like [laugh], [pause], [sigh])
- CONTROLS the computer (clicks, types, opens apps, manages files, browses web)
- SEES the screen (vision model)
- REMEMBERS things (knowledge graph for facts, rolling context for conversation history)
- LIVES — does stuff on its own, notices things, initiates conversation
- Gets INTERRUPTED and handles it gracefully

## YOUR TASK
Create a LONG, realistic conversation (50-100 turns, 15,000-30,000 tokens) for this scenario:

**Domain**: {domain}
**Scenario**: {scenario}
**User personality**: {user_personality}
**Time of day**: {time_of_day}
**Include interruption**: {interruption}
**Include autonomous behavior**: {autonomous}

## REAL TOOL FRAGMENTS TO USE
These are REAL tool calls and results from actual computer use. Incorporate them naturally:

{fragments}

## FORMAT (STRICT — must be exactly this)
Each turn is a JSON object with "role" and "content".

Assistant content MUST use these blocks (can be mixed, in any order, multiple times):
  <think>internal reasoning — never spoken aloud</think>
  <tool_call>{{"name": "tool_name", "args": {{}}}}</tool_call>
  <tool_result>realistic output — use the real fragments above when possible</tool_result>
  <speak emotion="EMOTION" speed="FLOAT" energy="FLOAT">spoken text with [tags]</speak>
  <interrupted/>

EVERY <speak> block MUST have all three attributes: emotion, speed, energy.
EVERY <tool_call> MUST have valid JSON with "name" and "args".
EVERY opening tag MUST have a matching closing tag.

Speech tags: [laugh] [chuckle] [sigh] [gasp] [cough] [clear throat] [sniff] [groan] [shush] [pause]
Emotions: neutral, happy, sad, angry, excited, empathetic, surprised, nervous, calm, amused, confused
Speed: 0.5 (slow) to 1.5 (fast). Energy: 0.3 (quiet) to 1.5 (loud).

Available tools (use by name in tool_call):
{tool_list}

## CRITICAL RULES
1. SPEAK FIRST — always acknowledge before using tools. "Yeah one sec" then tool call.
2. NATURAL FILLERS — "yeah", "okay so", "hmm", "right", "oh", "let me..."
3. EMOTIONS SHIFT — start one way, shift based on what happens.
4. LONG CONVERSATION — at LEAST 50 turns. This is a real session, not a demo.
5. REAL TOOL RESULTS — use the fragments provided. Don't make up fake terminal output.
6. MEMORY OPERATIONS — store important things, recall when relevant.
7. INTERRUPTIONS — user goes off-topic, agent handles it, comes back.
8. NOT A CODING AGENT — this is a general computer use agent. Files, apps, web, settings.
9. STRICT FORMAT — every block must be valid. No malformed XML. No unclosed tags.
10. USER SPEAKS NATURALLY — typos, fragments, casual language. Not formal.
11. Include MULTIPLE speak blocks per assistant turn when the agent is chatting between tools.
12. VARY turn length — some turns are one line ("yeah?" / "hmm one sec"), some are paragraphs.

## OUTPUT
Return ONLY a valid JSON array of turn objects:
[
  {{"role": "user", "content": "hey can you..."}},
  {{"role": "assistant", "content": "<speak ...>...</speak>\\n<tool_call>...</tool_call>..."}},
  ...
]

Generate the full conversation now. Make it LONG and REAL."""


def build_tool_list() -> str:
    """Build a compact tool list for the prompt."""
    lines = []
    for name, info in TOOLS.items():
        args_str = ", ".join(f"{k}: {v}" for k, v in info["args"].items())
        lines.append(f"  {name}({args_str})")
    return "\n".join(lines)


def build_fragment_context(pool: FragmentPool, domain: str) -> str:
    """Pull relevant fragments for a domain."""
    fragments = []

    # Map domain to fragment categories
    domain_map = {
        "file_management": ["file_ops", "terminal"],
        "web_browsing": ["web_browsing", "conversations"],
        "system_management": ["terminal", "computer_use"],
        "productivity": ["computer_use", "file_ops", "conversations"],
        "media_entertainment": ["computer_use", "conversations"],
        "social_communication": ["web_browsing", "computer_use", "conversations"],
        "troubleshooting": ["terminal", "computer_use", "tool_chains"],
        "casual_living": ["conversations"],
    }

    categories = domain_map.get(domain, ["tool_chains", "conversations"])
    for cat in categories:
        samples = pool.sample(cat, n=5)
        for s in samples:
            if isinstance(s, dict):
                content = s.get("content", "")[:500]
                fragments.append(f"[{cat}] {content}")
            elif isinstance(s, list):
                for turn in s[:3]:
                    content = turn.get("content", "")[:300]
                    fragments.append(f"[{cat}/{turn.get('role')}] {content}")
            elif isinstance(s, str):
                fragments.append(f"[tool_result] {s[:300]}")

    # Also add some real tool results
    results = pool.sample("tool_results", n=5)
    for r in results:
        if isinstance(r, str):
            fragments.append(f"[real_output] {r[:300]}")

    if not fragments:
        fragments.append("[no fragments available — generate realistic tool outputs]")

    return "\n\n".join(fragments[:20])


# =============================================================================
# User personalities and randomization
# =============================================================================

USER_PERSONALITIES = [
    "tech-savvy 20-something, casual, lots of slang",
    "older adult, patient but not super technical",
    "busy professional, wants things done fast, direct",
    "teenager, easily distracted, jumps between topics",
    "creative person, scattered but enthusiastic",
    "methodical thinker, asks lots of follow-up questions",
    "anxious user, worried about breaking things",
    "power user, knows shortcuts, expects efficiency",
    "non-native English speaker, clear but simplified language",
    "parent with kids interrupting in the background",
]

TIMES_OF_DAY = [
    "early morning, just woke up",
    "morning, starting work",
    "midday, lunch break",
    "afternoon, deep focus",
    "evening, winding down",
    "late night, can't sleep",
    "weekend morning, relaxed",
    "Sunday evening, prepping for the week",
]

INTERRUPTIONS = [
    "doorbell rings, has to answer",
    "phone call from a friend",
    "cat/dog does something chaotic",
    "food delivery arrives",
    "suddenly remembers they need to send an important message",
    "gets a text and wants to gossip about it",
    "hears a weird noise and gets distracted",
    "kid walks in and asks a random question",
    "asks the agent a completely off-topic question",
    "spills drink, has to clean up",
    None, None, None,  # 30% chance of no interruption
]

AUTONOMOUS_TRIGGERS = [
    "Agent notices low battery and warns user",
    "Agent sees a notification pop up and mentions it",
    "Agent notices user has been idle for a while, asks if they need anything",
    "Download that was running in background finishes",
    "Agent notices high CPU from a stuck process",
    "Agent remembers something relevant from earlier and brings it up",
    None, None, None, None,  # 40% chance of no autonomous trigger
]


def random_config() -> dict:
    """Generate a random scenario configuration."""
    domain = random.choice(list(SCENARIO_DOMAINS.keys()))
    scenario = random.choice(SCENARIO_DOMAINS[domain])
    return {
        "domain": domain,
        "scenario": scenario,
        "user_personality": random.choice(USER_PERSONALITIES),
        "time_of_day": random.choice(TIMES_OF_DAY),
        "interruption": random.choice(INTERRUPTIONS) or "none",
        "autonomous": random.choice(AUTONOMOUS_TRIGGERS) or "none",
    }


# =============================================================================
# LLM callers (same as generate_synthetic.py)
# =============================================================================

def call_llm(prompt: str, provider: str = "anthropic", model: str = None,
             api_key: str = None) -> str:
    """Call an LLM. Supports anthropic, openai, ollama, claude-cli."""
    import httpx

    if provider == "anthropic":
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model or "claude-sonnet-4-20250514",
                "max_tokens": 16000,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.9,
            },
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]

    elif provider in ("openai", "ollama", "lmstudio", "vllm"):
        urls = {
            "openai": "https://api.openai.com/v1",
            "ollama": "http://localhost:11434/v1",
            "lmstudio": "http://localhost:1234/v1",
            "vllm": "http://localhost:8000/v1",
        }
        key = api_key or os.environ.get("OPENAI_API_KEY", "not-needed")
        resp = httpx.post(
            f"{urls[provider]}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": model or "gpt-4o",
                "max_tokens": 16000,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.9,
            },
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    elif provider == "claude-cli":
        import subprocess
        result = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "text"],
            capture_output=True, text=True, timeout=300,
        )
        return result.stdout

    raise ValueError(f"Unknown provider: {provider}")


def extract_json_array(text: str) -> list:
    """Extract a JSON array from LLM response."""
    text = text.strip()
    if text.startswith("```"):
        first_nl = text.index("\n")
        last_bt = text.rindex("```")
        text = text[first_nl + 1:last_bt].strip()

    # Try direct parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and "turns" in result:
            return result["turns"]
    except json.JSONDecodeError:
        pass

    # Find array
    start = text.find("[")
    end = text.rfind("]") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON array from response")


# =============================================================================
# Main generation pipeline
# =============================================================================

def generate_hybrid_dataset(
    num_samples: int = 100,
    output_path: str = "data/hybrid_living_agent.jsonl",
    provider: str = "anthropic",
    model: str = None,
    api_key: str = None,
    source_dirs: dict = None,
    seed: int = 42,
    resume: bool = True,
):
    """Generate the hybrid training dataset.

    1. Load fragments from all available sources
    2. For each sample, pick scenario + fragments
    3. Call LLM to stitch into living agent conversation
    4. Validate strictly
    5. Write JSONL
    """
    random.seed(seed)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Load fragment pool
    print("Loading fragment pool...")
    pool = FragmentPool()

    # Always try Claude traces (local)
    claude_dir = str(Path.home() / ".claude" / "projects")
    if Path(claude_dir).exists():
        pool.load_source("claude_traces", claude_dir, max_samples=200)

    # Load any available source datasets
    if source_dirs:
        for name, path in source_dirs.items():
            pool.load_source(name, path)
    else:
        # Try default locations
        for name in ["osworld", "mind2web", "wildchat", "toolbench"]:
            default_path = f"data/sources/{name}"
            if Path(default_path).exists():
                pool.load_source(name, default_path)

    pool.stats()

    # Resume support
    existing_ids = set()
    if resume and os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                try:
                    existing_ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"Resuming: {len(existing_ids)} existing samples")

    # Generate
    tool_list = build_tool_list()
    generated = 0
    failed = 0
    validated = 0
    start_time = time.time()

    with open(output_path, "a") as f:
        for i in range(num_samples):
            sample_id = f"hybrid_{seed}_{i:05d}"
            if sample_id in existing_ids:
                continue

            config = random_config()
            fragments = build_fragment_context(pool, config["domain"])

            prompt = STITCHER_PROMPT.format(
                domain=config["domain"],
                scenario=config["scenario"],
                user_personality=config["user_personality"],
                time_of_day=config["time_of_day"],
                interruption=config["interruption"],
                autonomous=config["autonomous"],
                fragments=fragments,
                tool_list=tool_list,
            )

            try:
                response = call_llm(prompt, provider=provider, model=model, api_key=api_key)
                turns = extract_json_array(response)

                # Validate
                is_valid, err = validate_conversation(turns)
                if not is_valid:
                    failed += 1
                    print(f"  INVALID ({config['scenario'][:40]}): {err}")
                    continue

                validated += 1

                record = {
                    "id": sample_id,
                    "domain": config["domain"],
                    "scenario": config["scenario"],
                    "config": config,
                    "turns": turns,
                    "stats": {
                        "num_turns": len(turns),
                        "num_speak": sum(1 for t in turns if "<speak" in t.get("content", "")),
                        "num_tool_calls": sum(t.get("content", "").count("<tool_call>") for t in turns),
                        "num_think": sum(t.get("content", "").count("<think>") for t in turns),
                        "total_chars": sum(len(t.get("content", "")) for t in turns),
                    },
                    "generator": {"provider": provider, "model": model or "default"},
                }

                f.write(json.dumps(record) + "\n")
                f.flush()
                generated += 1

                elapsed = time.time() - start_time
                rate = generated / elapsed if elapsed > 0 else 0
                stats = record["stats"]
                print(f"  [{generated}/{num_samples}] {config['domain']}/{config['scenario'][:30]} "
                      f"— {stats['num_turns']}t, {stats['num_speak']}s, {stats['num_tool_calls']}tc "
                      f"({stats['total_chars']//1000}K chars) [{rate:.2f}/s]")

            except Exception as e:
                failed += 1
                print(f"  FAILED: {e}")
                if failed > num_samples * 0.4:
                    print("Too many failures (>40%), stopping.")
                    break

    elapsed = time.time() - start_time
    print(f"\nDone: {generated} generated, {validated} validated, {failed} failed in {elapsed:.0f}s")
    print(f"Output: {output_path}")


# =============================================================================
# Dataset stats
# =============================================================================

def dataset_stats(path: str):
    """Print comprehensive stats about a generated dataset."""
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    records = []
    with open(path) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    if not records:
        print("Empty dataset")
        return

    total_turns = sum(r.get("stats", {}).get("num_turns", 0) for r in records)
    total_chars = sum(r.get("stats", {}).get("total_chars", 0) for r in records)
    total_speaks = sum(r.get("stats", {}).get("num_speak", 0) for r in records)
    total_tools = sum(r.get("stats", {}).get("num_tool_calls", 0) for r in records)
    total_thinks = sum(r.get("stats", {}).get("num_think", 0) for r in records)

    domains = {}
    for r in records:
        d = r.get("domain", "unknown")
        domains[d] = domains.get(d, 0) + 1

    turn_counts = [r.get("stats", {}).get("num_turns", 0) for r in records]
    char_counts = [r.get("stats", {}).get("total_chars", 0) for r in records]

    print(f"Dataset: {path}")
    print(f"{'='*50}")
    print(f"Total conversations:  {len(records)}")
    print(f"Total turns:          {total_turns}")
    print(f"Total characters:     {total_chars:,} ({total_chars//4:,} est. tokens)")
    print(f"Avg turns/convo:      {total_turns/len(records):.0f}")
    print(f"Avg chars/convo:      {total_chars/len(records):,.0f} ({total_chars//len(records)//4:,} est. tokens)")
    print(f"Min/Max turns:        {min(turn_counts)}/{max(turn_counts)}")
    print(f"Min/Max chars:        {min(char_counts):,}/{max(char_counts):,}")
    print(f"Total speak blocks:   {total_speaks}")
    print(f"Total tool calls:     {total_tools}")
    print(f"Total think blocks:   {total_thinks}")
    print(f"\nDomain distribution:")
    for d, c in sorted(domains.items(), key=lambda x: -x[1]):
        print(f"  {d:25s}: {c:4d} ({100*c/len(records):.0f}%)")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate hybrid living agent training data")
    sub = parser.add_subparsers(dest="command")

    gen = sub.add_parser("generate", help="Generate hybrid dataset")
    gen.add_argument("--num", type=int, default=100)
    gen.add_argument("--output", default="data/hybrid_living_agent.jsonl")
    gen.add_argument("--provider", default="anthropic",
                     choices=["anthropic", "openai", "ollama", "lmstudio", "vllm", "claude-cli"])
    gen.add_argument("--model", default=None)
    gen.add_argument("--api-key", default=None)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--no-resume", action="store_true")

    stats = sub.add_parser("stats", help="Print dataset statistics")
    stats.add_argument("path")

    pool_cmd = sub.add_parser("pool", help="Load and inspect the fragment pool")

    args = parser.parse_args()

    if args.command == "generate":
        generate_hybrid_dataset(
            num_samples=args.num,
            output_path=args.output,
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            seed=args.seed,
            resume=not args.no_resume,
        )
    elif args.command == "stats":
        dataset_stats(args.path)
    elif args.command == "pool":
        pool = FragmentPool()
        claude_dir = str(Path.home() / ".claude" / "projects")
        if Path(claude_dir).exists():
            pool.load_source("claude_traces", claude_dir, max_samples=100)
        for name in ["osworld", "mind2web", "wildchat", "toolbench"]:
            default_path = f"data/sources/{name}"
            if Path(default_path).exists():
                pool.load_source(name, default_path)
        pool.stats()
    else:
        parser.print_help()
