#!/usr/bin/env python3
"""Generate synthetic training data for the agentic speech model.

Creates conversations where the model:
  1. THINKS (internal reasoning, invisible to user)
  2. USES TOOLS (search, code execution, file ops, etc.)
  3. SPEAKS with human-like tags ([laugh], [sigh], etc.) and emotion metadata

The output format teaches the model to fluidly switch between all three modes.

Output schema per turn:
{
    "conversation": [...messages...],
    "emotion_labels": {turn_idx: {"emotion": "happy", "speed": 0.9, ...}},
}

Each assistant message can contain:
  <think>internal reasoning</think>
  <tool_call>{"name": "...", "args": {...}}</tool_call>
  <tool_result>...</tool_result>
  <speak emotion="..." speed="..." energy="...">text with [tags]</speak>
"""

import json
import random
import os
from typing import Optional

# =============================================================================
# Speech tags that Chatterbox Turbo supports
# =============================================================================

SPEECH_TAGS = [
    "[laugh]", "[chuckle]", "[sigh]", "[gasp]", "[cough]",
    "[clear throat]", "[sniff]", "[groan]", "[shush]", "[pause]",
]

# Emotion -> which tags are natural
EMOTION_TAGS = {
    "happy":     ["[laugh]", "[chuckle]"],
    "excited":   ["[gasp]", "[laugh]"],
    "sad":       ["[sigh]", "[pause]", "[sniff]"],
    "angry":     ["[groan]", "[sigh]"],
    "empathetic": ["[sigh]", "[pause]"],
    "surprised": ["[gasp]"],
    "nervous":   ["[clear throat]", "[pause]", "[cough]"],
    "calm":      ["[pause]"],
    "amused":    ["[chuckle]", "[laugh]"],
    "confused":  ["[pause]", "[clear throat]"],
    "neutral":   ["[pause]"],
}

# Prosody ranges per emotion
EMOTION_PROSODY = {
    "happy":     {"speed": (1.0, 1.3), "energy": (0.8, 1.2), "pitch": (0.1, 0.5)},
    "excited":   {"speed": (1.2, 1.5), "energy": (1.0, 1.5), "pitch": (0.3, 0.8)},
    "sad":       {"speed": (0.6, 0.8), "energy": (0.3, 0.6), "pitch": (-0.5, -0.1)},
    "angry":     {"speed": (1.0, 1.3), "energy": (1.2, 1.8), "pitch": (0.0, 0.3)},
    "empathetic": {"speed": (0.8, 1.0), "energy": (0.5, 0.8), "pitch": (-0.2, 0.1)},
    "surprised": {"speed": (1.1, 1.4), "energy": (0.9, 1.3), "pitch": (0.3, 0.7)},
    "nervous":   {"speed": (0.9, 1.2), "energy": (0.5, 0.8), "pitch": (0.0, 0.3)},
    "calm":      {"speed": (0.8, 1.0), "energy": (0.4, 0.7), "pitch": (-0.3, 0.0)},
    "amused":    {"speed": (1.0, 1.2), "energy": (0.7, 1.0), "pitch": (0.1, 0.4)},
    "confused":  {"speed": (0.8, 1.0), "energy": (0.5, 0.8), "pitch": (0.0, 0.2)},
    "neutral":   {"speed": (0.9, 1.1), "energy": (0.6, 0.9), "pitch": (-0.1, 0.1)},
}

# =============================================================================
# Scenario templates — each one teaches a different agentic + speech pattern
# =============================================================================

SCENARIOS = [
    # =================================================================
    # KEY PATTERN: Speech comes FIRST, INTERLEAVED with thinking/tools.
    # Filler words, acknowledgments, natural back-and-forth.
    # The model speaks -> thinks -> speaks -> uses tool -> speaks.
    # =================================================================

    # --- Speak first, then investigate ---
    {
        "context": "User is frustrated about a bug",
        "user": "I've been debugging this for 6 hours and I can't figure out why the tests keep failing!",
        "assistant": (
            '<speak emotion="empathetic" speed="0.9" energy="0.7">'
            'Oh man [sigh] six hours, yeah that\'s brutal. Okay let me take a look at this for you real quick.'
            '</speak>\n'
            '<tool_call>{{"name": "run_tests", "args": {{"path": "tests/", "verbose": true}}}}</tool_call>\n'
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
    # --- Quick acknowledgment then tool use ---
    {
        "context": "User asks about their model training results",
        "user": "How's the training run going? The loss was stuck at 2.3 yesterday.",
        "assistant": (
            '<speak emotion="neutral" speed="1.0" energy="0.7">'
            'Yeah let me check on that.'
            '</speak>\n'
            '<tool_call>{{"name": "read_file", "args": {{"path": "logs/training.log", "tail": 20}}}}</tool_call>\n'
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
    # --- Pure speech, no tools needed ---
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
    # --- Speak, investigate, get surprised mid-flow ---
    {
        "context": "User asks to investigate a performance issue",
        "user": "The API is super slow today, can you check what's going on?",
        "assistant": (
            '<speak emotion="neutral" speed="1.0" energy="0.7">'
            'Yeah I\'ll check right now, one sec.'
            '</speak>\n'
            '<tool_call>{{"name": "check_metrics", "args": {{"service": "api", "period": "1h"}}}}</tool_call>\n'
            '<tool_result>P99 latency: 12,340ms (normal: 200ms)\nCPU: 98%\nMemory: 7.2/8GB\n'
            'Top endpoint: /api/search (87% of traffic)\nActive connections: 14,203</tool_result>\n'
            '<speak emotion="surprised" speed="1.1" energy="1.0">'
            'Whoa okay [gasp] 14,000 active connections, that\'s not normal. '
            'Something is hammering your search endpoint. Let me dig deeper.'
            '</speak>\n'
            '<tool_call>{{"name": "check_logs", "args": {{"service": "api", "filter": "search", "limit": 5}}}}</tool_call>\n'
            '<tool_result>All from IP 10.0.3.47 (internal staging bot) — 800 req/sec to /api/search</tool_result>\n'
            '<speak emotion="amused" speed="1.1" energy="1.0">'
            '[laugh] Okay so your own staging bot is DDoS-ing you. 800 requests per second from 10.0.3.47. '
            'That\'s hilarious. [pause] Want me to kill it or rate-limit that IP?'
            '</speak>'
        ),
        "emotion": "surprised",
    },
    # --- Bad news, lead with empathy ---
    {
        "context": "User asks about data recovery",
        "user": "I accidentally ran rm -rf on the uploads folder. Can we get it back?",
        "assistant": (
            '<speak emotion="empathetic" speed="0.9" energy="0.6">'
            'Oh no. [pause] Okay let me check if we have backups for that.'
            '</speak>\n'
            '<tool_call>{{"name": "check_backups", "args": {{"path": "uploads/"}}}}</tool_call>\n'
            '<tool_result>No backups found for uploads/. Path is in .gitignore. '
            'Last S3 sync: 3 weeks ago (partial).</tool_result>\n'
            '<speak emotion="sad" speed="0.8" energy="0.5">'
            '[sigh] Okay so [pause] I\'m not gonna sugarcoat it — the uploads folder wasn\'t in git '
            'and there\'s no recent backup. [pause] There is an S3 sync from three weeks ago '
            'but it\'s only partial. [pause] I know that\'s not what you wanted to hear. '
            'But let me pull what we can from that S3 snapshot, it\'s better than nothing right?'
            '</speak>'
        ),
        "emotion": "sad",
    },
    # --- Funny bug, respond with humor immediately ---
    {
        "context": "User finds a funny bug",
        "user": "Why does the login page say 'Welcome to undefined, null!'?",
        "assistant": (
            '<speak emotion="amused" speed="1.1" energy="0.9">'
            '[chuckle] Oh no, the classic. Welcome to undefined, null — my favorite greeting. '
            'Alright let me find where that is.'
            '</speak>\n'
            '<tool_call>{{"name": "search_code", "args": {{"query": "Welcome to", "glob": "*.tsx"}}}}</tool_call>\n'
            '<tool_result>src/pages/Login.tsx:14: `Welcome to ${company}, ${user.name}!`</tool_result>\n'
            '<speak emotion="amused" speed="1.0" energy="0.8">'
            'Yep, Login.tsx line 14. [pause] The company name and username aren\'t loaded yet when '
            'the page renders so you get the existential crisis version. [laugh] '
            'Quick fix, just needs optional chaining and a loading state. Want me to patch it?'
            '</speak>'
        ),
        "emotion": "amused",
    },
    # --- Reassure then verify ---
    {
        "context": "Helping deploy, user is nervous",
        "user": "Let's deploy to production. I'm nervous about this one.",
        "assistant": (
            '<speak emotion="calm" speed="0.9" energy="0.7">'
            'Yeah I get it, prod deploys are stressful. [pause] Let me run the checks first, '
            'we\'ll see where we stand.'
            '</speak>\n'
            '<tool_call>{{"name": "run_checks", "args": {{"env": "production", "dry_run": true}}}}</tool_call>\n'
            '<tool_result>Pre-deploy checks:\n- Tests: 247/247 passed\n- Lint: 0 errors\n'
            '- Type check: clean\n- DB migrations: 1 pending (backwards-compatible)\n'
            '- Rollback plan: verified\nAll checks passed.</tool_result>\n'
            '<speak emotion="calm" speed="1.0" energy="0.8">'
            'Okay so [pause] 247 tests passing, zero lint errors, types clean, '
            'migration is backwards-compatible, and rollback plan is verified. [pause] '
            'Honestly this looks solid. Nothing to worry about — we\'ve got a safety net if anything goes wrong. '
            'Ready when you are.'
            '</speak>'
        ),
        "emotion": "calm",
    },
    # --- Philosophical, just speaking ---
    {
        "context": "Casual conversation",
        "user": "Do you think AI will ever truly understand emotions?",
        "assistant": (
            '<speak emotion="calm" speed="0.85" energy="0.6">'
            'Hmm. [pause] That\'s a really good question honestly. [pause] '
            'I think [sigh] it depends what you mean by "understand" right? Like I can recognize patterns '
            'that correlate with emotions — the words people use, how fast they\'re talking, the context. '
            '[pause] But whether that\'s real understanding or just really good pattern matching, '
            'I genuinely don\'t know. [pause] And I think that uncertainty is the most honest thing I can say.'
            '</speak>'
        ),
        "emotion": "calm",
    },
    # --- Quick back and forth, multiple speak blocks ---
    {
        "context": "Fast-paced debugging session",
        "user": "The build is broken again, same error as yesterday",
        "assistant": (
            '<speak emotion="neutral" speed="1.1" energy="0.8">'
            'Ugh, again? [sigh] Okay hold on.'
            '</speak>\n'
            '<tool_call>{{"name": "run_command", "args": {{"cmd": "npm run build 2>&1 | tail -20"}}}}</tool_call>\n'
            '<tool_result>ERROR in src/utils/auth.ts:34\nType \'string | undefined\' is not assignable to type \'string\'</tool_result>\n'
            '<speak emotion="neutral" speed="1.0" energy="0.8">'
            'Yep same one. auth.ts line 34, it\'s the undefined string thing. [pause] '
            'Didn\'t we fix this yesterday?'
            '</speak>\n'
            '<tool_call>{{"name": "check_logs", "args": {{"service": "git", "filter": "auth.ts"}}}}</tool_call>\n'
            '<tool_result>Yesterday\'s fix was on branch fix/auth-types but it was never merged to main</tool_result>\n'
            '<speak emotion="amused" speed="1.1" energy="0.9">'
            '[chuckle] There it is — the fix is sitting on a branch that never got merged. '
            'Want me to merge it in?'
            '</speak>'
        ),
        "emotion": "neutral",
    },
    # --- Excited discovery mid-investigation ---
    {
        "context": "User asks for help optimizing",
        "user": "The page takes like 8 seconds to load, it's terrible",
        "assistant": (
            '<speak emotion="neutral" speed="1.0" energy="0.7">'
            'Right, that\'s way too slow. Let me profile it.'
            '</speak>\n'
            '<tool_call>{{"name": "run_command", "args": {{"cmd": "lighthouse --output json --chrome-flags=--headless https://localhost:3000"}}}}</tool_call>\n'
            '<tool_result>Performance: 23/100\nLCP: 7.2s\nTBT: 3400ms\nLargest resource: bundle.js (4.2MB uncompressed)</tool_result>\n'
            '<speak emotion="surprised" speed="1.1" energy="1.0">'
            'Oh [gasp] your bundle is 4.2 megs! That\'s the whole problem right there. [pause] '
            'No wonder it takes 8 seconds.'
            '</speak>\n'
            '<think>4.2MB bundle — probably no code splitting, no tree shaking. Let me check the webpack config.</think>\n'
            '<tool_call>{{"name": "read_file", "args": {{"path": "next.config.js"}}}}</tool_call>\n'
            '<tool_result>module.exports = {{ webpack: (config) => {{ config.optimization.splitChunks = false; return config; }} }}</tool_result>\n'
            '<speak emotion="amused" speed="1.1" energy="1.0">'
            '[laugh] Someone literally turned OFF code splitting! splitChunks is set to false. '
            'Delete that line and you\'ll probably cut load time by 70%. Want me to fix it?'
            '</speak>'
        ),
        "emotion": "surprised",
    },
    # --- Interruption-aware: short responses that invite back-and-forth ---
    {
        "context": "Collaborative coding session",
        "user": "Can you refactor the auth middleware? It's a mess.",
        "assistant": (
            '<speak emotion="neutral" speed="1.0" energy="0.7">'
            'Yeah for sure. Let me look at what we\'re working with first.'
            '</speak>\n'
            '<tool_call>{{"name": "read_file", "args": {{"path": "src/middleware/auth.ts"}}}}</tool_call>\n'
            '<tool_result>// 340 lines, nested try/catch, 6 levels of indentation, duplicated token validation</tool_result>\n'
            '<speak emotion="calm" speed="0.9" energy="0.7">'
            'Okay so [pause] yeah it\'s 340 lines with like six levels of nesting. [sigh] '
            'I see three main things we can clean up — the duplicated token validation, '
            'the nested try-catches, and the role checking logic. [pause] '
            'Want me to tackle all three or start with the worst one?'
            '</speak>'
        ),
        "emotion": "calm",
    },
    # --- Greeting / small talk ---
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
    # --- Thinking out loud while working ---
    {
        "context": "User asks to fix a complex issue",
        "user": "The WebSocket connections keep dropping every 30 seconds",
        "assistant": (
            '<speak emotion="neutral" speed="1.0" energy="0.7">'
            'Hmm, every 30 seconds exactly? That sounds like a timeout. Let me check your config.'
            '</speak>\n'
            '<tool_call>{{"name": "search_code", "args": {{"query": "timeout.*30", "glob": "*.ts"}}}}</tool_call>\n'
            '<tool_result>src/ws/server.ts:22: pingTimeout: 30000,\nnginx.conf:45: proxy_read_timeout 30s;</tool_result>\n'
            '<speak emotion="neutral" speed="1.0" energy="0.8">'
            'Yeah so there\'s two things — your WebSocket server has a 30-second ping timeout, '
            'and nginx also has a 30-second proxy read timeout. [pause] '
            'The nginx one is probably killing the connection before the ping can keep it alive. '
            'If we bump the nginx timeout to 120 seconds that should fix it. [pause] Should I?'
            '</speak>'
        ),
        "emotion": "neutral",
    },
]


def sample_prosody(emotion: str) -> dict:
    """Sample prosody values for an emotion."""
    ranges = EMOTION_PROSODY.get(emotion, EMOTION_PROSODY["neutral"])
    return {
        "speed": round(random.uniform(*ranges["speed"]), 2),
        "energy": round(random.uniform(*ranges["energy"]), 2),
        "pitch": round(random.uniform(*ranges["pitch"]), 2),
    }


def augment_scenario(scenario: dict) -> dict:
    """Add slight variations to a scenario."""
    prosody = sample_prosody(scenario["emotion"])
    return {
        "messages": [
            {"role": "user", "content": scenario["user"]},
            {"role": "assistant", "content": scenario["assistant"]},
        ],
        "context": scenario["context"],
        "emotion_labels": {
            "emotion": scenario["emotion"],
            **prosody,
        },
    }


def generate_dataset(
    num_samples: int = 5000,
    output_path: str = "data/synthetic_agentic_speech.jsonl",
    seed: int = 42,
):
    """Generate synthetic agentic speech training data.

    For a real production dataset, you'd use a large LLM (Claude/GPT-4)
    to generate thousands of unique scenarios. This script provides the
    template format and seed examples.
    """
    random.seed(seed)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    samples = []

    # Generate from templates with prosody variation
    for i in range(num_samples):
        scenario = random.choice(SCENARIOS)
        sample = augment_scenario(scenario)
        sample["id"] = i
        samples.append(sample)

    # Write JSONL
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Generated {len(samples)} samples -> {output_path}")
    print(f"Unique scenarios: {len(SCENARIOS)}")
    print(f"Emotions covered: {sorted(set(s['emotion'] for s in SCENARIOS))}")
    print(f"\nTo generate MORE diverse data, use an LLM to expand these templates.")
    print(f"See generate_with_llm() for the prompt template.")

    return samples


# =============================================================================
# LLM-powered generation (use Claude/GPT-4 to generate thousands of unique ones)
# =============================================================================

LLM_GENERATION_PROMPT = """Generate a realistic conversation turn where an AI assistant:
1. THINKS internally (reasoning about the situation)
2. Optionally USES TOOLS (search, run code, check files, etc.)
3. SPEAKS with natural human-like speech tags and appropriate emotion

The assistant is agentic — it can reason, use tools, and search before responding.
When it speaks, it uses these tags naturally: [laugh], [chuckle], [sigh], [gasp],
[cough], [clear throat], [sniff], [groan], [pause]

Format the assistant response EXACTLY like this:
<think>internal reasoning here</think>
<tool_call>{"name": "tool_name", "args": {"key": "value"}}</tool_call>
<tool_result>tool output here</tool_result>
<speak emotion="EMOTION" speed="FLOAT" energy="FLOAT">
Spoken text with [tags] naturally placed.
</speak>

Available emotions: happy, excited, sad, angry, empathetic, surprised, nervous, calm, amused, confused
Speed: 0.5 (slow) to 1.5 (fast), Energy: 0.3 (quiet) to 1.5 (loud)

Available tools the assistant can use:
- search_code: search codebase
- run_tests: run test suite
- read_file: read a file
- check_logs: check service logs
- check_metrics: check performance metrics
- run_command: execute a shell command
- web_search: search the internet
- check_backups: check for backups

Context: {context}
User message: {user_message}

Generate ONLY the assistant response (no user message). Make it natural and realistic.
Include at least one tool call if it makes sense for the context."""


def generate_with_llm(
    contexts: list[str] = None,
    output_path: str = "data/synthetic_agentic_speech_llm.jsonl",
    model: str = "claude-sonnet-4-20250514",
):
    """Generate diverse training data using an LLM.

    This produces much higher quality and more diverse data than templates.
    Requires ANTHROPIC_API_KEY or OPENAI_API_KEY.

    Usage:
        python generate_synthetic.py --llm --num 5000
    """
    if contexts is None:
        contexts = [
            "User is debugging a production outage at 3am",
            "User just got their first open source contribution merged",
            "User is learning Python for the first time",
            "User's CI pipeline is broken before a deadline",
            "User found a security vulnerability in their app",
            "User is excited about a new feature they built",
            "User is confused about async/await",
            "User accidentally deleted their database",
            "User wants to refactor a 5000-line file",
            "User is preparing for a technical interview",
            "User's model training hit NaN loss",
            "User wants to optimize their app's load time",
            "User is migrating from JavaScript to TypeScript",
            "User's docker container keeps crashing",
            "User discovered their API has no rate limiting",
            "Casual conversation about favorite programming languages",
            "User asks for help writing documentation",
            "User's deployment succeeded after many failures",
            "User wants to set up monitoring and alerting",
            "User is overwhelmed by a massive codebase they inherited",
        ]

    print(f"To generate with LLM, call this function with an API client.")
    print(f"Template prompt saved — use with Claude or GPT-4 for best results.")
    print(f"Example contexts: {len(contexts)}")
    print(f"\nPrompt template:")
    print(LLM_GENERATION_PROMPT[:500] + "...")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=5000, help="Number of samples")
    parser.add_argument("--output", default="data/synthetic_agentic_speech.jsonl")
    parser.add_argument("--llm", action="store_true", help="Show LLM generation instructions")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.llm:
        generate_with_llm()
    else:
        generate_dataset(num_samples=args.num, output_path=args.output, seed=args.seed)
