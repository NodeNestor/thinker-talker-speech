#!/usr/bin/env python3
"""Generate synthetic training data for the living agent speech model.

Two modes:
  1. `--seed`    → Output the seed examples from SCENARIOS + LIVING_AGENT as JSONL
  2. `--generate → Call an LLM to produce thousands of diverse variations

The LLM receives our full tool environment, format spec, and seed examples,
then generates novel conversations that teach the model to:
  - Speak naturally with fillers, pauses, human sounds
  - Think internally before/between speech
  - Use tools conversationally (coding, memory, vision, web, system)
  - Handle interruptions gracefully
  - Store/recall from knowledge graph and rolling context
  - Work autonomously when user is away

Output: JSONL where each line is a multi-turn conversation with emotion metadata.
"""

import json
import random
import os
import time
import argparse
from typing import Optional
from pathlib import Path

# Import our scenario seeds and environment
from living_agent_scenarios import SCENARIOS as LIVING_SCENARIOS, FORMAT_SPEC
from environment import TOOLS, AGENT_STATES, INTERRUPTION_RULES, AUTONOMOUS_BEHAVIORS

# =============================================================================
# Speech tags & emotion config (Chatterbox Turbo compatible)
# =============================================================================

SPEECH_TAGS = [
    "[laugh]", "[chuckle]", "[sigh]", "[gasp]", "[cough]",
    "[clear throat]", "[sniff]", "[groan]", "[shush]", "[pause]",
]

EMOTION_TAGS = {
    "happy":      ["[laugh]", "[chuckle]"],
    "excited":    ["[gasp]", "[laugh]"],
    "sad":        ["[sigh]", "[pause]", "[sniff]"],
    "angry":      ["[groan]", "[sigh]"],
    "empathetic": ["[sigh]", "[pause]"],
    "surprised":  ["[gasp]"],
    "nervous":    ["[clear throat]", "[pause]", "[cough]"],
    "calm":       ["[pause]"],
    "amused":     ["[chuckle]", "[laugh]"],
    "confused":   ["[pause]", "[clear throat]"],
    "neutral":    ["[pause]"],
}

EMOTION_PROSODY = {
    "happy":      {"speed": (1.0, 1.3), "energy": (0.8, 1.2), "pitch": (0.1, 0.5)},
    "excited":    {"speed": (1.2, 1.5), "energy": (1.0, 1.5), "pitch": (0.3, 0.8)},
    "sad":        {"speed": (0.6, 0.8), "energy": (0.3, 0.6), "pitch": (-0.5, -0.1)},
    "angry":      {"speed": (1.0, 1.3), "energy": (1.2, 1.8), "pitch": (0.0, 0.3)},
    "empathetic": {"speed": (0.8, 1.0), "energy": (0.5, 0.8), "pitch": (-0.2, 0.1)},
    "surprised":  {"speed": (1.1, 1.4), "energy": (0.9, 1.3), "pitch": (0.3, 0.7)},
    "nervous":    {"speed": (0.9, 1.2), "energy": (0.5, 0.8), "pitch": (0.0, 0.3)},
    "calm":       {"speed": (0.8, 1.0), "energy": (0.4, 0.7), "pitch": (-0.3, 0.0)},
    "amused":     {"speed": (1.0, 1.2), "energy": (0.7, 1.0), "pitch": (0.1, 0.4)},
    "confused":   {"speed": (0.8, 1.0), "energy": (0.5, 0.8), "pitch": (0.0, 0.2)},
    "neutral":    {"speed": (0.9, 1.1), "energy": (0.6, 0.9), "pitch": (-0.1, 0.1)},
}

# =============================================================================
# The 13 original single-turn scenarios (from v1 of this file)
# =============================================================================

SINGLE_TURN_SCENARIOS = [
    {
        "context": "User is frustrated about a bug",
        "user": "I've been debugging this for 6 hours and I can't figure out why the tests keep failing!",
        "assistant": (
            '<speak emotion="empathetic" speed="0.9" energy="0.7">'
            'Oh man [sigh] six hours, yeah that\'s brutal. Okay let me take a look at this for you real quick.'
            '</speak>\n'
            '<tool_call>{"name": "run_tests", "args": {"path": "tests/", "verbose": true}}</tool_call>\n'
            '<tool_result>FAILED tests/test_api.py::test_auth - AssertionError: expected 200 got 401\n'
            'FAILED tests/test_api.py::test_user_create - ConnectionRefused\n2 failed, 14 passed</tool_result>\n'
            '<think>Auth 401 + connection refused — the test database isn\'t running. Simple fix.</think>\n'
            '<speak emotion="empathetic" speed="1.0" energy="0.8">'
            'Okay yeah [pause] so here\'s the thing — it\'s not actually a code bug. '
            'Your test database isn\'t running, that\'s why auth returns 401 and the connection gets refused. '
            '[pause] So you\'ve been chasing a ghost basically. Want me to start the test DB?'
            '</speak>'
        ),
        "emotion": "empathetic",
    },
    {
        "context": "User asks about their model training results",
        "user": "How's the training run going? The loss was stuck at 2.3 yesterday.",
        "assistant": (
            '<speak emotion="neutral" speed="1.0" energy="0.7">'
            'Yeah let me check on that.'
            '</speak>\n'
            '<tool_call>{"name": "read_file", "args": {"path": "logs/training.log", "line_start": -20}}</tool_call>\n'
            '<tool_result>Epoch 47/50 | Loss: 0.34 | Val Acc: 94.2% | LR: 1e-5\n'
            'Epoch 48/50 | Loss: 0.31 | Val Acc: 94.8% | LR: 8e-6\n'
            'New best checkpoint saved!</tool_result>\n'
            '<speak emotion="excited" speed="1.3" energy="1.3">'
            'Oh [gasp] dude! [laugh] It\'s at 0.31 loss and 94.8% accuracy! '
            'It went from 2.3 all the way down, that\'s insane. [pause] '
            'Two more epochs to go and it just saved a new best checkpoint. This is looking really good.'
            '</speak>'
        ),
        "emotion": "excited",
    },
    {
        "context": "User asks a technical question",
        "user": "Can you explain how attention works in transformers?",
        "assistant": (
            '<speak emotion="calm" speed="0.9" energy="0.6">'
            'Yeah sure. [pause] So basically, attention is just a way for each token to look at every '
            'other token and figure out which ones matter to it right now. [pause] '
            'You take your input and make three things from it — queries, keys, and values. '
            'The query is like "what am I looking for", the key is "here\'s what I have". '
            'You multiply them together, get scores, and use those to weight the values. '
            '[pause] That\'s literally it. It\'s a weighted lookup. [pause] '
            'The "multi-head" part just means you do this multiple times in parallel with different '
            'learned projections, so the model can attend to different things at once.'
            '</speak>'
        ),
        "emotion": "calm",
    },
    {
        "context": "Start of conversation",
        "user": "Hey, how's it going?",
        "assistant": (
            '<speak emotion="happy" speed="1.0" energy="0.8">'
            'Hey! [pause] Going good, what are you working on today?'
            '</speak>'
        ),
        "emotion": "happy",
    },
]

# =============================================================================
# World state randomization — gives the LLM diverse contexts to work with
# =============================================================================

WORLD_STATES = {
    "projects": [
        "React dashboard with Next.js and Prisma",
        "Python FastAPI backend with PostgreSQL",
        "Rust CLI tool for log analysis",
        "Flutter mobile app for fitness tracking",
        "Go microservice handling payment webhooks",
        "Vue.js e-commerce frontend",
        "Node.js Discord bot with slash commands",
        "PyTorch training pipeline for image classification",
        "Kubernetes operator in Go",
        "Chrome extension for productivity",
        "Unity game with procedural generation",
        "Svelte app with real-time collaboration",
        "Django REST API for a CMS",
        "Electron desktop app for video editing",
        "Terraform infrastructure for AWS",
    ],
    "user_moods": [
        "frustrated — been at this for hours",
        "excited — just got something working",
        "calm and focused",
        "stressed — deadline tomorrow",
        "curious — learning something new",
        "tired — late night coding session",
        "confused — inherited unfamiliar codebase",
        "happy — code review went well",
        "anxious — first production deploy",
        "bored — doing repetitive refactoring",
    ],
    "time_of_day": [
        "morning, just started work",
        "midday, after lunch",
        "afternoon, deep focus time",
        "evening, wrapping up",
        "late night, still coding",
        "weekend, side project",
    ],
    "screen_contents": [
        "VS Code with a TypeScript file open, red squiggly lines everywhere",
        "Terminal showing a stack trace with 20 lines of Python errors",
        "Browser with localhost:3000 showing a broken CSS layout",
        "Grafana dashboard showing a spike in error rates",
        "GitHub PR page with 47 comments and requested changes",
        "Slack with 12 unread messages in #engineering",
        "Jupyter notebook with training loss plots that plateau",
        "Docker Desktop showing 3 containers, one in red (crashed)",
        "AWS console showing an EC2 instance at 100% CPU",
        "Empty terminal, user just opened it",
    ],
    "knowledge_graph_contents": [
        {"entity": "daily standup", "type": "fact", "props": {"time": "10:30 AM", "channel": "#dev-sync"}},
        {"entity": "user_prefs", "type": "preference", "props": {"editor": "VS Code", "tabs_vs_spaces": "spaces", "indent": 2}},
        {"entity": "auth_service", "type": "project", "props": {"status": "migrating to JWT", "owner": "user"}},
        {"entity": "deploy_process", "type": "concept", "props": {"steps": "test → staging → canary → prod", "rollback": "automated"}},
        {"entity": "teammate_alex", "type": "person", "props": {"role": "backend lead", "timezone": "PST", "expertise": "databases"}},
        {"entity": "sprint_goal", "type": "project", "props": {"goal": "ship user profiles V2", "deadline": "Friday"}},
        {"entity": "user_prefs", "type": "preference", "props": {"language": "TypeScript", "framework": "Next.js", "test_runner": "vitest"}},
        {"entity": "previous_bug", "type": "fact", "props": {"issue": "Stripe webhook key expired", "fix": "rotate key in vault", "date": "last week"}},
    ],
    "rolling_context_recalls": [
        "Last session: user was refactoring the auth middleware, got halfway through",
        "Yesterday: tried Recharts for dashboard, switched to Nivo for heatmaps",
        "3 days ago: set up CI pipeline with GitHub Actions, added lint + test stages",
        "Last week: debugged memory leak in WebSocket server, was the event listener cleanup",
        "2 days ago: user asked about connection pooling, we increased pool size to 20",
        "Earlier today: deployed staging successfully, waiting on QA signoff",
    ],
    "interruption_types": [
        "random question about something unrelated",
        "cat/dog/pet doing something chaotic",
        "doorbell / delivery person",
        "phone call they need to take",
        "suddenly remembers something urgent",
        "asks a dumb/funny philosophical question",
        "wants to change topic entirely",
        "spills coffee/water on keyboard",
        "someone walks into the room",
        "fire alarm / loud noise",
    ],
    "autonomous_triggers": [
        "CI build failed on main branch",
        "Training run finished while user was away",
        "Noticed high CPU usage from a runaway process",
        "PR was merged by a teammate",
        "Dependency vulnerability alert from GitHub",
        "Server health check started failing",
        "User's screen shows an error they haven't noticed",
        "New issue assigned to user in project tracker",
    ],
}


def random_world_state() -> dict:
    """Sample a random world state for the LLM to work with."""
    return {
        "project": random.choice(WORLD_STATES["projects"]),
        "user_mood": random.choice(WORLD_STATES["user_moods"]),
        "time_of_day": random.choice(WORLD_STATES["time_of_day"]),
        "screen": random.choice(WORLD_STATES["screen_contents"]),
        "memory_graph": random.sample(WORLD_STATES["knowledge_graph_contents"], k=random.randint(1, 3)),
        "rolling_context": random.choice(WORLD_STATES["rolling_context_recalls"]),
        "interruption": random.choice(WORLD_STATES["interruption_types"]) if random.random() < 0.4 else None,
        "autonomous_trigger": random.choice(WORLD_STATES["autonomous_triggers"]) if random.random() < 0.25 else None,
    }


# =============================================================================
# Scenario types to request from the LLM — weighted by importance
# =============================================================================

SCENARIO_TYPES = [
    # (type, weight, description)
    ("tool_use_conversation", 25, "User asks something that requires tools (code search, file read, tests, commands). Agent speaks first, uses tools, speaks results."),
    ("interruption_mid_speech", 15, "Agent is speaking or using tools, user interrupts with something random. Agent handles it gracefully and offers to continue."),
    ("memory_store_recall", 12, "Conversation where agent stores something in knowledge graph, then recalls it later in the same or 'later' conversation."),
    ("rolling_context_recall", 8, "User asks about something from a previous session. Agent uses context_recall to find compressed old conversation."),
    ("autonomous_initiate", 10, "Agent notices something (CI fail, screen error, process issue) and speaks up on its own. No user prompt first."),
    ("pure_speech", 10, "Just talking — explanation, opinion, greeting, small talk. No tools needed. Natural with fillers and pauses."),
    ("multi_tool_chain", 8, "Agent uses 2-3 tools in sequence, speaking between each. Shows investigation flow."),
    ("vision_assist", 5, "Agent takes a screenshot, analyzes what's on screen, offers help based on what it sees."),
    ("user_away_autonomous", 5, "User gives a task and leaves. Agent works autonomously — reads files, writes code, runs tests, commits. Reports back when user returns."),
    ("emotional_shift", 5, "Conversation where the emotion shifts mid-way (e.g., investigating → surprised discovery, or calm → excited good news)."),
    ("dumb_question_handling", 5, "User asks something silly, off-topic, or impossible. Agent responds with humor and redirects."),
    ("error_recovery", 5, "A tool fails or returns unexpected results. Agent handles it gracefully, tries alternative approach."),
]


def pick_scenario_type() -> tuple[str, str]:
    """Weighted random pick of scenario type."""
    types, weights, descs = zip(*SCENARIO_TYPES)
    chosen = random.choices(range(len(types)), weights=weights, k=1)[0]
    return types[chosen], descs[chosen]


# =============================================================================
# The mega-prompt for LLM generation
# =============================================================================

def build_tools_description() -> str:
    """Format our tool environment for the LLM prompt."""
    lines = []
    for name, info in TOOLS.items():
        args_str = ", ".join(f"{k}: {v}" for k, v in info["args"].items())
        lines.append(f"  {name}({args_str}) — {info['description']}")
    return "\n".join(lines)


def build_seed_examples(n: int = 3) -> str:
    """Pick N seed examples to show the LLM the format."""
    all_scenarios = LIVING_SCENARIOS + [
        {"id": f"single_{i}", "turns": [
            {"role": "user", "content": s["user"]},
            {"role": "assistant", "content": s["assistant"]},
        ], "context": s["context"]}
        for i, s in enumerate(SINGLE_TURN_SCENARIOS)
    ]
    picked = random.sample(all_scenarios, min(n, len(all_scenarios)))
    examples = []
    for sc in picked:
        examples.append(json.dumps({"id": sc["id"], "context": sc.get("context", ""), "turns": sc["turns"]}, indent=2))
    return "\n---\n".join(examples)


def build_generation_prompt(scenario_type: str, scenario_desc: str, world: dict, seed_examples: str) -> str:
    """Build the full prompt for the LLM to generate one conversation."""
    return f"""You are generating training data for a living AI agent that talks, thinks, uses tools, and has memory.

## YOUR TASK
Generate ONE realistic multi-turn conversation of type: **{scenario_type}**
Description: {scenario_desc}

## WORLD STATE FOR THIS CONVERSATION
- Project: {world['project']}
- User mood: {world['user_mood']}
- Time: {world['time_of_day']}
- Screen: {world['screen']}
- Knowledge graph has: {json.dumps(world['memory_graph'])}
- Rolling context from previous session: {world['rolling_context']}
{"- INTERRUPTION: User will " + world['interruption'] + " at some point during the conversation" if world['interruption'] else "- No interruptions this conversation"}
{"- AUTONOMOUS TRIGGER: " + world['autonomous_trigger'] if world['autonomous_trigger'] else ""}

## AVAILABLE TOOLS
{build_tools_description()}

## FORMAT RULES
Each turn is {{"role": "user"|"assistant"|"system", "content": "..."}}

Assistant content uses these blocks (can be mixed in any order):
  <think>internal reasoning (never spoken aloud)</think>
  <tool_call>{{"name": "tool_name", "args": {{}}}}</tool_call>
  <tool_result>realistic simulated output</tool_result>
  <speak emotion="EMOTION" speed="FLOAT" energy="FLOAT">spoken text with [tags]</speak>
  <interrupted/>  (user started speaking, model stops mid-sentence)

Speech tags (Chatterbox Turbo): [laugh] [chuckle] [sigh] [gasp] [cough] [clear throat] [sniff] [groan] [shush] [pause]
Emotions: neutral, happy, sad, angry, excited, empathetic, surprised, nervous, calm, amused, confused
Speed: 0.5-1.5, Energy: 0.3-1.5

System messages are environment events: [CI webhook: ...], [periodic screen check], [5 minutes of silence], etc.

## KEY BEHAVIORS TO MODEL
1. Agent SPEAKS FIRST — acknowledge, then investigate. Never silently use tools.
2. Agent uses FILLER naturally — "yeah", "okay so", "hmm", "right", "let me..."
3. Agent THINKS between actions — reasoning is in <think> blocks, never spoken.
4. When INTERRUPTED: stop gracefully, handle the interruption, offer to continue.
5. Tool results are REALISTIC — file contents, test output, git diffs, actual errors.
6. Memory operations use the KNOWLEDGE GRAPH (memory_store/query/update) and ROLLING CONTEXT (context_compress/recall).
7. Emotions SHIFT naturally — agent might start neutral, get surprised by findings, then amused.
8. DON'T be generic — reference the specific project, files, and technologies from the world state.

## SEED EXAMPLES (for format reference — generate something DIFFERENT)
{seed_examples}

## OUTPUT
Return ONLY a valid JSON object with this structure:
{{"id": "unique_id", "type": "{scenario_type}", "context": "brief context", "turns": [...]}}

Generate the conversation now. Make it feel REAL — like an actual person talking to their AI assistant."""


# =============================================================================
# LLM API callers — supports Anthropic, OpenAI-compatible, and local
# =============================================================================

def call_anthropic(prompt: str, model: str = "claude-sonnet-4-20250514", api_key: str = None) -> str:
    """Call Anthropic API directly."""
    import httpx
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("Set ANTHROPIC_API_KEY or pass api_key")
    resp = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]


def call_openai_compatible(prompt: str, model: str = "gpt-4o", base_url: str = None, api_key: str = None) -> str:
    """Call OpenAI-compatible API (OpenAI, Ollama, vLLM, LM Studio, etc.)."""
    import httpx
    url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    key = api_key or os.environ.get("OPENAI_API_KEY", "not-needed")
    resp = httpx.post(
        f"{url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {key}", "content-type": "application/json"},
        json={
            "model": model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.9,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def call_claude_cli(prompt: str) -> str:
    """Call Claude via the `claude` CLI (zero config, uses existing auth)."""
    import subprocess
    result = subprocess.run(
        ["claude", "-p", prompt, "--output-format", "text"],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude CLI failed: {result.stderr}")
    return result.stdout


PROVIDERS = {
    "anthropic": call_anthropic,
    "openai": call_openai_compatible,
    "ollama": lambda p, **kw: call_openai_compatible(p, base_url="http://localhost:11434/v1", **kw),
    "lmstudio": lambda p, **kw: call_openai_compatible(p, base_url="http://localhost:1234/v1", **kw),
    "vllm": lambda p, **kw: call_openai_compatible(p, base_url="http://localhost:8000/v1", **kw),
    "claude-cli": call_claude_cli,
}


# =============================================================================
# JSON extraction — LLMs sometimes wrap in markdown code blocks
# =============================================================================

def extract_json(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown wrappers."""
    text = text.strip()
    # Strip markdown code block
    if text.startswith("```"):
        first_newline = text.index("\n")
        last_backticks = text.rindex("```")
        text = text[first_newline + 1:last_backticks].strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try finding the JSON object
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not extract JSON from response:\n{text[:500]}")


# =============================================================================
# Main generation pipeline
# =============================================================================

def generate_seed_dataset(output_path: str) -> int:
    """Export all seed scenarios (hand-crafted) as JSONL."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    count = 0
    with open(output_path, "w") as f:
        # Living agent scenarios (multi-turn)
        for sc in LIVING_SCENARIOS:
            record = {
                "id": sc["id"],
                "type": "seed_living_agent",
                "context": sc.get("context", ""),
                "turns": sc["turns"],
            }
            f.write(json.dumps(record) + "\n")
            count += 1
        # Single-turn scenarios
        for i, sc in enumerate(SINGLE_TURN_SCENARIOS):
            record = {
                "id": f"seed_single_{i}",
                "type": "seed_single_turn",
                "context": sc["context"],
                "turns": [
                    {"role": "user", "content": sc["user"]},
                    {"role": "assistant", "content": sc["assistant"]},
                ],
                "emotion": sc["emotion"],
            }
            f.write(json.dumps(record) + "\n")
            count += 1
    print(f"Wrote {count} seed examples -> {output_path}")
    return count


def generate_llm_dataset(
    num_samples: int = 500,
    output_path: str = "data/synthetic_living_agent.jsonl",
    provider: str = "anthropic",
    model: str = None,
    api_key: str = None,
    base_url: str = None,
    seed: int = 42,
    resume: bool = True,
):
    """Generate diverse training data by calling an LLM.

    Args:
        num_samples: How many conversations to generate
        output_path: Where to write the JSONL
        provider: anthropic, openai, ollama, lmstudio, vllm, claude-cli
        model: Model name (provider-specific)
        api_key: API key (or set env var)
        base_url: For openai-compatible providers
        seed: Random seed
        resume: If True, skip existing IDs in output file
    """
    random.seed(seed)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Set up LLM caller
    caller = PROVIDERS.get(provider)
    if caller is None:
        raise ValueError(f"Unknown provider: {provider}. Options: {list(PROVIDERS.keys())}")

    call_kwargs = {}
    if model:
        call_kwargs["model"] = model
    if api_key:
        call_kwargs["api_key"] = api_key
    if base_url and provider == "openai":
        call_kwargs["base_url"] = base_url

    # Resume support — skip already generated IDs
    existing_ids = set()
    if resume and os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                try:
                    existing_ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"Resuming: {len(existing_ids)} existing samples found")

    generated = 0
    failed = 0
    start_time = time.time()

    with open(output_path, "a") as f:
        for i in range(num_samples):
            sample_id = f"gen_{seed}_{i:05d}"
            if sample_id in existing_ids:
                continue

            # Pick scenario type and world state
            scenario_type, scenario_desc = pick_scenario_type()
            world = random_world_state()
            seed_examples = build_seed_examples(n=2)

            prompt = build_generation_prompt(scenario_type, scenario_desc, world, seed_examples)

            try:
                response = caller(prompt, **call_kwargs)
                data = extract_json(response)

                # Validate structure
                if "turns" not in data or not isinstance(data["turns"], list):
                    raise ValueError("Missing or invalid 'turns' field")
                if len(data["turns"]) < 2:
                    raise ValueError("Need at least 2 turns")

                # Stamp with metadata
                data["id"] = sample_id
                data["type"] = scenario_type
                data["world_state"] = {
                    "project": world["project"],
                    "user_mood": world["user_mood"],
                    "time_of_day": world["time_of_day"],
                }
                data["generator"] = {"provider": provider, "model": model or "default"}

                f.write(json.dumps(data) + "\n")
                f.flush()
                generated += 1

                elapsed = time.time() - start_time
                rate = generated / elapsed if elapsed > 0 else 0
                print(f"  [{generated}/{num_samples}] {scenario_type} ({rate:.1f}/s)")

            except Exception as e:
                failed += 1
                print(f"  FAILED ({scenario_type}): {e}")
                if failed > num_samples * 0.3:
                    print("Too many failures (>30%), stopping.")
                    break

    elapsed = time.time() - start_time
    print(f"\nDone: {generated} generated, {failed} failed in {elapsed:.0f}s")
    print(f"Output: {output_path}")
    return generated


# =============================================================================
# Dataset stats
# =============================================================================

def dataset_stats(path: str):
    """Print stats about a generated dataset."""
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    types = {}
    emotions = {}
    turn_counts = []
    total = 0

    with open(path) as f:
        for line in f:
            try:
                data = json.loads(line)
                total += 1
                t = data.get("type", "unknown")
                types[t] = types.get(t, 0) + 1
                turn_counts.append(len(data.get("turns", [])))
                # Count emotions in speak blocks
                for turn in data.get("turns", []):
                    content = turn.get("content", "")
                    for emo in EMOTION_PROSODY:
                        if f'emotion="{emo}"' in content:
                            emotions[emo] = emotions.get(emo, 0) + 1
            except json.JSONDecodeError:
                pass

    print(f"Dataset: {path}")
    print(f"Total conversations: {total}")
    print(f"Total turns: {sum(turn_counts)}")
    print(f"Avg turns/conversation: {sum(turn_counts)/max(total,1):.1f}")
    print(f"\nScenario types:")
    for t, c in sorted(types.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c} ({100*c/total:.0f}%)")
    print(f"\nEmotion distribution:")
    for e, c in sorted(emotions.items(), key=lambda x: -x[1]):
        print(f"  {e}: {c}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate living agent training data")
    sub = parser.add_subparsers(dest="command")

    # Seed export
    seed_cmd = sub.add_parser("seed", help="Export hand-crafted seed examples as JSONL")
    seed_cmd.add_argument("--output", default="data/seed_examples.jsonl")

    # LLM generation
    gen_cmd = sub.add_parser("generate", help="Generate diverse data using an LLM")
    gen_cmd.add_argument("--num", type=int, default=500, help="Number of conversations")
    gen_cmd.add_argument("--output", default="data/synthetic_living_agent.jsonl")
    gen_cmd.add_argument("--provider", default="anthropic",
                         choices=list(PROVIDERS.keys()),
                         help="LLM provider")
    gen_cmd.add_argument("--model", default=None, help="Model name (provider-specific)")
    gen_cmd.add_argument("--api-key", default=None, help="API key (or use env var)")
    gen_cmd.add_argument("--base-url", default=None, help="Base URL for openai-compatible")
    gen_cmd.add_argument("--seed", type=int, default=42)
    gen_cmd.add_argument("--no-resume", action="store_true", help="Start fresh, don't skip existing")

    # Stats
    stats_cmd = sub.add_parser("stats", help="Print dataset statistics")
    stats_cmd.add_argument("path", help="Path to JSONL dataset")

    args = parser.parse_args()

    if args.command == "seed":
        generate_seed_dataset(args.output)
    elif args.command == "generate":
        generate_llm_dataset(
            num_samples=args.num,
            output_path=args.output,
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            seed=args.seed,
            resume=not args.no_resume,
        )
    elif args.command == "stats":
        dataset_stats(args.path)
    else:
        parser.print_help()
