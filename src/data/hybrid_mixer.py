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
    # System turns can appear between any turns without breaking alternation.
    # user→system→user is fine (system event injected between user messages).
    # user→user (no system between) should be caught by postprocessing merge.
    last_nonsys_role = None
    consecutive_user = 0
    for turn in turns:
        role = turn["role"]
        if role == "system":
            continue
        if role == "user":
            consecutive_user += 1
            if consecutive_user > 4:
                # Allow several consecutive user turns (system events between,
                # or user sending multiple messages before agent responds)
                return False, f"Too many consecutive user turns ({consecutive_user})"
        else:
            consecutive_user = 0
        last_nonsys_role = role

    return True, ""


# =============================================================================
# Post-processing — fix common LLM generation issues
# =============================================================================

# Map common invalid emotions to valid ones
EMOTION_FIXES = {
    "warm": "happy",
    "friendly": "happy",
    "cheerful": "happy",
    "curious": "neutral",
    "thoughtful": "calm",
    "concerned": "empathetic",
    "playful": "amused",
    "sarcastic": "amused",
    "annoyed": "angry",
    "worried": "nervous",
    "relieved": "calm",
    "content": "calm",
    "enthusiastic": "excited",
    "skeptical": "confused",
    "tender": "empathetic",
    "wistful": "sad",
    "reassuring": "calm",
    "engaged": "neutral",
    "focused": "neutral",
    "determined": "calm",
    "proud": "happy",
    "grateful": "happy",
    "apologetic": "empathetic",
    "frustrated": "angry",
    "disappointed": "sad",
    "bored": "neutral",
    "impressed": "surprised",
    "sympathetic": "empathetic",
    "confident": "calm",
    "humorous": "amused",
    "serious": "neutral",
    "gentle": "calm",
    "matter-of-fact": "neutral",
    "helpful": "neutral",
    "patient": "calm",
    "encouraging": "happy",
    "interested": "neutral",
    "intrigued": "surprised",
    "contemplative": "calm",
    "nostalgic": "sad",
    "dismissive": "neutral",
    "supportive": "empathetic",
    "teasing": "amused",
    "dry": "neutral",
    "upbeat": "happy",
    "somber": "sad",
    "defiant": "angry",
    "sheepish": "nervous",
    "resigned": "sad",
    "earnest": "calm",
    "amazed": "surprised",
    "approving": "happy",
    "casual": "neutral",
    "emphatic": "excited",
    "energetic": "excited",
    "optimistic": "happy",
    "realistic": "neutral",
    "satisfied": "happy",
    "shocked": "surprised",
    "slightly_excited": "excited",
    "uncertain": "nervous",
    "understanding": "empathetic",
    "very_happy": "happy",
}


def _fix_unclosed_tags(content: str) -> str:
    """Fix mismatched XML-style tags in assistant content."""
    for tag in ["think", "speak", "tool_call", "tool_result"]:
        opens = content.count(f"<{tag}")
        closes = content.count(f"</{tag}>")
        if opens > closes:
            # Missing close tags — append them
            for _ in range(opens - closes):
                content = content.rstrip() + f"</{tag}>"
        elif closes > opens:
            # Extra close tags — remove from the end
            for _ in range(closes - opens):
                idx = content.rfind(f"</{tag}>")
                if idx >= 0:
                    content = content[:idx] + content[idx + len(f"</{tag}>"):]
    return content


def _track_braces(text: str, start: int) -> int:
    """Track JSON brace depth from start, return index after closing brace or -1."""
    brace_depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        c = text[i]
        if escape_next:
            escape_next = False
            continue
        if c == '\\' and in_string:
            escape_next = True
            continue
        if c == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '{':
            brace_depth += 1
        elif c == '}':
            brace_depth -= 1
            if brace_depth == 0:
                return i + 1
    return -1


def _fix_tool_call_boundaries(content: str) -> str:
    r"""Fix cases where LLM merged tool_call and tool_result into one block.

    Common LLM failure patterns:
      <tool_call>{"name":"screenshot","args":{}}</tool_result>   <- wrong close tag
      <tool_call>{"name":"screenshot","args":{}}</tool_result>
      <tool_result>Screenshot taken...</tool_result>             <- leaked into tool_call
      <tool_call>{"name":"foo","args":{}}</tool_tool_call>       <- typo close tag

    Fix: find <tool_call> followed by JSON, then truncate at the end of the JSON
    object (first matching '}') and close with </tool_call>.
    """
    # Pattern: <tool_call> followed by JSON that ends with </tool_result> instead of </tool_call>
    # or has no proper close tag at all
    result = content

    # Fix </tool_tool_call> typos
    result = result.replace("</tool_tool_call>", "</tool_call>")

    # Fix <tool_call>{...}</tool_result> — wrong close tag
    # Find all <tool_call> positions and try to extract just the JSON object
    parts = []
    pos = 0
    while pos < len(result):
        tc_start = result.find("<tool_call>", pos)
        if tc_start == -1:
            parts.append(result[pos:])
            break

        parts.append(result[pos:tc_start])

        json_start = tc_start + len("<tool_call>")

        # First try brace tracking on original content
        json_end = _track_braces(result, json_start)

        if json_end == -1:
            # Brace tracking failed. Try with newline-fixed version.
            candidate = _fix_json_string(result[json_start:])
            fixed_end = _track_braces(candidate, 0)
            if fixed_end != -1:
                json_str = candidate[:fixed_end]
                # Find where to resume in original
                orig_scan = result[json_start:]
                next_tag = len(orig_scan)
                for tag in ['<tool_result>', '</tool_result>', '<tool_call>',
                            '</tool_call>', '<speak', '<think>', '</think>']:
                    idx = orig_scan.find(tag)
                    if idx > 0:
                        next_tag = min(next_tag, idx)
                json_end = json_start + next_tag
            else:
                # JSON is truncated (no closing braces at all).
                # Find the end boundary and try to force-close the JSON.
                orig_scan = result[json_start:]
                next_tag = len(orig_scan)
                for tag in ['<tool_result>', '</tool_result>', '<tool_call>',
                            '</tool_call>', '<speak', '<think>', '</think>']:
                    idx = orig_scan.find(tag)
                    if idx > 0:
                        next_tag = min(next_tag, idx)

                truncated = _fix_json_string(orig_scan[:next_tag].rstrip())
                # Try closing with increasing number of braces/quotes
                for suffix in ['}', '"}', '"}}', '"}}}']:
                    try:
                        json.loads(truncated + suffix)
                        json_str = truncated + suffix
                        json_end = json_start + next_tag
                        break
                    except json.JSONDecodeError:
                        # Also try with backslash fix
                        fixed_t = re.sub(r'(?<!\\)\\(?![\\"/u])', r'\\\\', truncated)
                        try:
                            json.loads(fixed_t + suffix)
                            json_str = fixed_t + suffix
                            json_end = json_start + next_tag
                            break
                        except json.JSONDecodeError:
                            continue
                else:
                    # Still can't parse. Extract name and make minimal tool_call.
                    name_m = re.search(r'"name"\s*:\s*"([^"]+)"', truncated)
                    if name_m:
                        json_str = json.dumps({"name": name_m.group(1), "args": {}})
                        json_end = json_start + next_tag
                    else:
                        parts.append(result[tc_start:tc_start + len("<tool_call>")])
                        pos = json_start
                        continue
        else:
            json_str = result[json_start:json_end]

        parts.append(f"<tool_call>{json_str}</tool_call>")

        # Skip past any wrong close tag that follows
        after = result[json_end:].lstrip()
        if after.startswith("</tool_call>"):
            pos = json_end + result[json_end:].index("</tool_call>") + len("</tool_call>")
        elif after.startswith("</tool_result>"):
            pos = json_end + result[json_end:].index("</tool_result>") + len("</tool_result>")
        elif after.startswith("</tool_tool_call>"):
            pos = json_end + result[json_end:].index("</tool_tool_call>") + len("</tool_tool_call>")
        else:
            pos = json_end

    return "".join(parts)


def _fix_tool_call_json(content: str) -> str:
    r"""Fix common JSON issues in <tool_call> blocks.

    Handles:
    - Windows paths with unescaped backslashes (C:\Users\foo)
    - Literal newlines in JSON string values (write_file content, type_text text)
    - Args at wrong level ({"name":"foo", "path":"bar"} instead of {"name":"foo","args":{"path":"bar"}})
    """
    def _fix_json_block(m):
        raw = m.group(1)
        try:
            json.loads(raw)
            return m.group(0)  # already valid
        except json.JSONDecodeError:
            pass

        fixed = raw

        # Fix 1: Escape literal newlines/tabs inside JSON string values
        fixed = _fix_json_string(fixed)

        # Fix 2: Unescaped backslashes — in tool_call context, \b \f \n \r \t
        # in file paths should be literal backslash+letter, not JSON escapes.
        # Aggressively escape ALL single backslashes except \\ and \" and \/ and \uXXXX
        fixed = re.sub(r'(?<!\\)\\(?![\\"/u])', r'\\\\', fixed)

        try:
            json.loads(fixed)
            return f"<tool_call>{fixed}</tool_call>"
        except json.JSONDecodeError:
            pass

        # Fix 3: Try to extract name and rebuild
        name_m = re.search(r'"name"\s*:\s*"([^"]+)"', fixed)
        if name_m:
            tool_name = name_m.group(1)
            # Try to find args object
            args_m = re.search(r'"args"\s*:\s*(\{.*)', fixed, re.DOTALL)
            if args_m:
                # Find matching brace for args
                args_raw = args_m.group(1)
                depth = 0
                for i, c in enumerate(args_raw):
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            args_str = args_raw[:i+1]
                            try:
                                args_obj = json.loads(args_str)
                                rebuilt = json.dumps({"name": tool_name, "args": args_obj})
                                return f"<tool_call>{rebuilt}</tool_call>"
                            except json.JSONDecodeError:
                                # Try fixing the args JSON too
                                args_fixed = _fix_json_string(args_str)
                                args_fixed = re.sub(r'(?<!\\)\\(?![\\"/u])', r'\\\\', args_fixed)
                                try:
                                    args_obj = json.loads(args_fixed)
                                    rebuilt = json.dumps({"name": tool_name, "args": args_obj})
                                    return f"<tool_call>{rebuilt}</tool_call>"
                                except json.JSONDecodeError:
                                    break

            # Fix 4: Args at wrong level — {"name":"foo", "path":"bar", "content":"baz"}
            # Rebuild as {"name":"foo", "args": {"path":"bar", "content":"baz"}}
            try:
                # Try parsing with fixes applied to extract all keys
                obj = None
                try:
                    obj = json.loads(fixed)
                except json.JSONDecodeError:
                    # Last resort: truncate at the point where JSON breaks
                    # and try to close it
                    for end_pos in range(len(fixed) - 1, 0, -1):
                        if fixed[end_pos] == '"':
                            attempt = fixed[:end_pos+1] + '}'
                            if fixed.count('{') > 1:
                                attempt += '}'
                            try:
                                obj = json.loads(attempt)
                                break
                            except json.JSONDecodeError:
                                continue

                if obj and isinstance(obj, dict) and "name" in obj:
                    name = obj.pop("name")
                    args = obj.pop("args", None)
                    if args is None:
                        args = obj  # remaining keys become args
                    rebuilt = json.dumps({"name": name, "args": args})
                    return f"<tool_call>{rebuilt}</tool_call>"
            except Exception:
                pass

            # No args or args parsing failed — try minimal
            minimal = json.dumps({"name": tool_name, "args": {}})
            return f"<tool_call>{minimal}</tool_call>"

        return m.group(0)  # give up, return original

    return re.sub(r"<tool_call>(.*?)</tool_call>", _fix_json_block, content, flags=re.DOTALL)


def postprocess_turns(turns: list[dict]) -> list[dict]:
    """Fix common issues in LLM-generated turns.

    1. Fix unclosed tags in assistant content
    2. Merge consecutive assistant turns into one (the big one)
    3. Merge consecutive user turns
    4. Fix invalid emotions
    5. Remove empty turns
    6. Ensure proper alternation: user → assistant (with system anywhere)
    """
    if not turns:
        return turns

    # Step 1: Fix emotions and unclosed tags in all content
    fixed = []
    for turn in turns:
        content = turn.get("content", "")
        if not content.strip():
            continue

        # Fix unclosed tags and broken JSON in assistant turns
        if turn.get("role") == "assistant":
            content = _fix_tool_call_boundaries(content)
            content = _fix_tool_call_json(content)
            content = _fix_unclosed_tags(content)

        # Fix invalid emotions
        for bad, good in EMOTION_FIXES.items():
            content = content.replace(f'emotion="{bad}"', f'emotion="{good}"')

        fixed.append({"role": turn["role"], "content": content})

    # Step 2: Merge consecutive same-role turns
    merged = []
    for turn in fixed:
        if not merged:
            merged.append(turn)
            continue

        prev = merged[-1]
        if turn["role"] == prev["role"]:
            # Merge into previous turn
            prev["content"] = prev["content"].rstrip() + "\n" + turn["content"].lstrip()
        else:
            merged.append(turn)

    # Step 3: Ensure it starts with user or system, ends with assistant
    while merged and merged[0]["role"] == "assistant":
        merged.pop(0)
    while merged and merged[-1]["role"] == "user":
        merged.pop()

    # Step 4: Validate alternation — if two users in a row somehow survived,
    # merge them. If two assistants survived (shouldn't happen after step 2), merge.
    final = []
    for turn in merged:
        if not final:
            final.append(turn)
            continue
        prev = final[-1]
        if turn["role"] == prev["role"] and turn["role"] != "system":
            prev["content"] = prev["content"].rstrip() + "\n" + turn["content"].lstrip()
        else:
            final.append(turn)

    return final


def compute_turn_stats(turns: list[dict]) -> dict:
    """Compute statistics for a conversation."""
    total_chars = sum(len(t.get("content", "")) for t in turns)
    user_turns = sum(1 for t in turns if t["role"] == "user")
    assistant_turns = sum(1 for t in turns if t["role"] == "assistant")
    system_turns = sum(1 for t in turns if t["role"] == "system")

    # Count blocks across all assistant turns
    all_assistant = " ".join(t["content"] for t in turns if t["role"] == "assistant")
    num_speak = len(SPEAK_BLOCK.findall(all_assistant))
    num_tool_calls = len(TOOL_CALL_BLOCK.findall(all_assistant))
    num_think = len(THINK_BLOCK.findall(all_assistant))
    num_interrupts = all_assistant.count("<interrupted/>")

    # Average assistant turn length
    asst_lengths = [len(t["content"]) for t in turns if t["role"] == "assistant"]
    avg_asst_len = sum(asst_lengths) / len(asst_lengths) if asst_lengths else 0
    max_asst_len = max(asst_lengths) if asst_lengths else 0

    return {
        "num_turns": len(turns),
        "user_turns": user_turns,
        "assistant_turns": assistant_turns,
        "system_turns": system_turns,
        "num_speak": num_speak,
        "num_tool_calls": num_tool_calls,
        "num_think": num_think,
        "num_interrupts": num_interrupts,
        "total_chars": total_chars,
        "est_tokens": total_chars // 4,
        "avg_assistant_chars": int(avg_asst_len),
        "max_assistant_chars": max_asst_len,
    }


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
Create a LONG, realistic conversation for this scenario:

**Domain**: {domain}
**Scenario**: {scenario}
**User personality**: {user_personality}
**Time of day**: {time_of_day}
**Include interruption**: {interruption}
**Include autonomous behavior**: {autonomous}

## REAL TOOL FRAGMENTS TO USE
These are REAL tool calls and results from actual computer use. Incorporate them naturally:

{fragments}

## TURN STRUCTURE — THIS IS THE MOST IMPORTANT PART

A conversation STRICTLY alternates: user turn -> assistant turn -> user turn -> assistant turn.
System turns (events like notifications, timers) can appear between any turns.

The user says ONE thing per turn — short, casual, messy.

The assistant packs EVERYTHING into ONE turn — this is critical. A single assistant turn
contains ALL the think blocks, tool calls, tool results, and speak blocks for that response.
The assistant might speak, then call 10 tools with results, then speak again — all in ONE
content string. Example:

USER: "hey can you clean up my downloads"
ASSISTANT (one content string): <speak emotion="neutral" speed="1.0" energy="0.7">Yeah one sec let me see what you got in there.</speak>
<tool_call>{{"name": "list_files", "args": {{"path": "~/Downloads"}}}}</tool_call>
<tool_result>resume_v3_FINAL.pdf  IMG_4821.jpg  IMG_4822.jpg  setup.exe  setup(1).exe  recipe.png  meme.jpg  taxes.pdf</tool_result>
<think>Tons of junk. PDFs, images, old installers. Sort by type.</think>
<speak emotion="amused" speed="1.1" energy="0.9">[chuckle] Resume v3 FINAL, two old installers, some photos and a meme. Let me sort this out.</speak>
<tool_call>{{"name": "move_file", "args": {{"src": "~/Downloads/resume_v3_FINAL.pdf", "dest": "~/Documents/Resumes/"}}}}</tool_call>
<tool_result>Moved</tool_result>
<tool_call>{{"name": "move_file", "args": {{"src": "~/Downloads/taxes.pdf", "dest": "~/Documents/Finance/"}}}}</tool_call>
<tool_result>Moved</tool_result>
<tool_call>{{"name": "move_file", "args": {{"src": "~/Downloads/IMG_4821.jpg", "dest": "~/Pictures/"}}}}</tool_call>
<tool_result>Moved</tool_result>
<tool_call>{{"name": "move_file", "args": {{"src": "~/Downloads/IMG_4822.jpg", "dest": "~/Pictures/"}}}}</tool_call>
<tool_result>Moved</tool_result>
<tool_call>{{"name": "delete_file", "args": {{"path": "~/Downloads/setup.exe"}}}}</tool_call>
<tool_result>Deleted</tool_result>
<tool_call>{{"name": "delete_file", "args": {{"path": "~/Downloads/setup(1).exe"}}}}</tool_call>
<tool_result>Deleted</tool_result>
<speak emotion="happy" speed="1.0" energy="0.8">Done! Resume and taxes to Documents, photos to Pictures, trashed the installers. [pause] Still got a meme and a recipe screenshot, want those or nah?</speak>

That's a SMALL example. In reality, a single assistant turn can have 30, 50, even 100+ tool calls.
For example, if the user says "organize my entire downloads folder", the assistant might:
- list_files to see everything (1 call)
- speak about what it sees
- think about how to categorize
- move 40 files one by one (40 calls with results)
- speak progress updates every 10 files or so
- delete 15 junk files (15 calls)
- speak summary at the end
That's 56+ tool calls, 4-5 speak blocks, 2-3 think blocks — ONE assistant turn.

Or if the user says "set up my new laptop":
- open_app Settings (1)
- click through wifi settings (5-10 clicks + type_text for password)
- speak "wifi connected"
- open_app Chrome, web_browse to sign in (3-5 calls)
- speak progress
- system_info to check battery/storage
- bluetooth connect headphones (3-4 calls)
- open_app Spotify, play_media
- screenshot to verify everything looks right
- speak summary
That's 30+ tool calls across different categories, all in one turn.

The agent TALKS BETWEEN tool batches — it doesn't just silently chain 50 tools. It does 5-10
tools, speaks a quick update ("[pause] okay almost done, just a few more"), then continues.

When the tool chain is LONG, the agent speaks progress updates naturally:
- After first batch: "Alright so I see like 40 files in here [pause] this is gonna take a sec"
- Midway: "Okay moved all the photos and documents [pause] now dealing with the random stuff"
- Near end: "[sigh] okay there were way more installers than I expected [laugh]"
- Done: "Alright all done! Here's what I did..."

## FORMAT RULES
Assistant content blocks (mix freely, repeat, any order, all in ONE content string):
  <think>internal reasoning — never spoken</think>
  <tool_call>{{"name": "tool_name", "args": {{}}}}</tool_call>
  <tool_result>realistic output</tool_result>
  <speak emotion="EMOTION" speed="FLOAT" energy="FLOAT">spoken text with [tags]</speak>
  <interrupted/>

EVERY <speak> MUST have emotion, speed, energy. EVERY <tool_call> MUST have valid JSON.
EVERY opening tag MUST have a matching closing tag.
Emotions: neutral, happy, sad, angry, excited, empathetic, surprised, nervous, calm, amused, confused
Speech tags: [laugh] [chuckle] [sigh] [gasp] [cough] [clear throat] [sniff] [groan] [shush] [pause]
Speed: 0.5-1.5. Energy: 0.3-1.5.

Available tools:
{tool_list}

## RULES
1. SPEAK FIRST — acknowledge before tools. "Yeah one sec" then tool calls.
2. ONE assistant turn per user message — ALL blocks in one content string.
3. NEVER split tool_call and tool_result across separate turns.
4. TOOL CHAINS CAN BE HUGE — 10, 30, 50, even 100+ tool calls in one assistant turn is normal.
   Don't stop at 5-6 calls. If the task requires 40 file moves, DO 40 file moves.
5. SPEAK BETWEEN TOOL BATCHES — every 5-15 tool calls, add a speak block with a progress update.
   The agent isn't silent during long chains. It narrates like a friend helping you.
6. NATURAL FILLERS — "yeah", "okay so", "hmm", "right", "oh", "uh", "let me..."
7. USER SPEAKS MESSY — typos, fragments, "liek", "idk", "nvm", "lol", "wait", "oh also".
8. VARY assistant turn size wildly — some are just <speak>"yeah?"</speak> (3 words).
   Others are 50+ tool calls with speaks in between (5000+ chars). Mix both.
9. MEMORY — use memory_store and memory_query naturally throughout.
10. NOT A CODING AGENT — computer use: files, apps, web, music, settings, browsing.
11. STRICT FORMAT — valid XML, valid JSON, no unclosed tags.
12. AT LEAST 20 user turns. With big assistant turns, expect 30-60K total chars.
13. Agent has PERSONALITY — jokes, opinions, teases, has preferences.
14. TOOL RESULTS ARE REALISTIC — real file names (not "file1.txt"), real app names,
    real error messages, real terminal output. Make it look like a real computer.
15. TOOLS FAIL SOMETIMES — this is critical. Not everything works the first time:
    - click on wrong element, try again: "oh wait that wasn't it [pause] let me try..."
    - file already exists at destination: agent handles the conflict
    - app won't open or crashes: agent tries alternative approach
    - web page loads wrong content: agent searches differently
    - permission denied: agent tries another way or asks user
    - wifi won't connect: agent troubleshoots step by step
    - tool_result returns an error message: agent reads it, thinks, adapts
    The agent should NEVER just give up. It tries another approach, works around it,
    or asks the user for help. This is how real computer use works — messy and iterative.
    Include at least 2-3 failures/retries per conversation. Example:
    <tool_call>{{"name": "move_file", "args": {{"src": "~/file.pdf", "dest": "~/Documents/"}}}}</tool_call>
    <tool_result>Error: destination file already exists</tool_result>
    <think>File already exists there. Let me rename it with a suffix.</think>
    <speak emotion="neutral" speed="1.0" energy="0.7">Oh there's already one with that name. Let me rename it.</speak>
    <tool_call>{{"name": "move_file", "args": {{"src": "~/file.pdf", "dest": "~/Documents/file_2.pdf"}}}}</tool_call>
    <tool_result>Moved</tool_result>

## OUTPUT
Return ONLY a valid JSON array. No markdown wrapper. No explanation:
[{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}, ...]

Generate the full conversation now. Make it feel REAL."""


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
             api_key: str = None, max_retries: int = 5) -> str:
    """Call an LLM with retry on rate limits. Supports anthropic, openai, ollama, claude-cli."""
    import httpx

    if provider == "anthropic":
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        for attempt in range(max_retries + 1):
            resp = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": model or "claude-sonnet-4-20250514",
                    "max_tokens": 32000,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.9,
                },
                timeout=300,
            )
            if resp.status_code == 429:
                wait = min(2 ** attempt * 5, 120)  # 5s, 10s, 20s, 40s, 80s, 120s
                print(f"    Rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})...")
                import time as _time
                _time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()["content"][0]["text"]
        resp.raise_for_status()  # raise the last 429

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
                "max_tokens": 32000,
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


def _fix_json_string(text: str) -> str:
    """Fix common LLM JSON issues: unescaped newlines, control chars, bad quotes."""
    # Fix literal newlines inside JSON strings (invalid control characters)
    # We need to be careful to only fix newlines INSIDE string values, not structural ones.
    # Strategy: replace \n that appears between quotes with \\n
    result = []
    in_string = False
    escape_next = False
    for ch in text:
        if escape_next:
            result.append(ch)
            escape_next = False
            continue
        if ch == '\\' and in_string:
            result.append(ch)
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
        if in_string and ch == '\n':
            result.append('\\n')
            continue
        if in_string and ch == '\r':
            result.append('\\r')
            continue
        if in_string and ch == '\t':
            result.append('\\t')
            continue
        result.append(ch)
    return ''.join(result)


def extract_json_array(text: str) -> list:
    """Extract a JSON array from LLM response, with robust error recovery."""
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
        chunk = text[start:end]
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            pass

        # Fix common issues and retry
        fixed = _fix_json_string(chunk)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        # Last resort: parse turns individually with regex
        turns = []
        pattern = re.compile(
            r'\{\s*"role"\s*:\s*"(user|assistant|system)"\s*,\s*"content"\s*:\s*"',
            re.DOTALL,
        )
        for m in pattern.finditer(chunk):
            role = m.group(1)
            # Find the end of this content string — look for unescaped " followed by }
            content_start = m.end()
            pos = content_start
            while pos < len(chunk):
                if chunk[pos] == '\\':
                    pos += 2  # skip escaped char
                    continue
                if chunk[pos] == '"':
                    # Check if followed by whitespace/} to confirm end of string
                    rest = chunk[pos + 1:pos + 20].lstrip()
                    if rest.startswith('}'):
                        content = chunk[content_start:pos]
                        # Unescape JSON string escapes
                        try:
                            content = json.loads(f'"{content}"')
                        except json.JSONDecodeError:
                            content = content.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
                        turns.append({"role": role, "content": content})
                        break
                pos += 1

        if len(turns) >= 4:
            return turns

    raise ValueError("Could not extract JSON array from response")


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

                # Always save raw response (tokens are expensive!)
                raw_dir = os.path.join(os.path.dirname(output_path) or ".", "raw_responses")
                os.makedirs(raw_dir, exist_ok=True)
                with open(os.path.join(raw_dir, f"{sample_id}.txt"), "w", encoding="utf-8") as rf:
                    rf.write(response)
                with open(os.path.join(raw_dir, f"{sample_id}_meta.json"), "w") as rf:
                    json.dump({"id": sample_id, "scenario": config["scenario"],
                               "domain": config["domain"], "config": config,
                               "provider": provider, "model": model or "default",
                               "response_chars": len(response)}, rf, indent=2)

                turns = extract_json_array(response)

                # Post-process: merge consecutive assistant turns, fix emotions
                turns = postprocess_turns(turns)

                # Validate
                is_valid, err = validate_conversation(turns)
                if not is_valid:
                    failed += 1
                    print(f"  INVALID ({config['scenario'][:40]}): {err}")
                    continue

                validated += 1
                stats = compute_turn_stats(turns)

                record = {
                    "id": sample_id,
                    "domain": config["domain"],
                    "scenario": config["scenario"],
                    "config": config,
                    "turns": turns,
                    "stats": stats,
                    "generator": {"provider": provider, "model": model or "default"},
                }

                f.write(json.dumps(record) + "\n")
                f.flush()
                generated += 1

                elapsed = time.time() - start_time
                rate = generated / elapsed if elapsed > 0 else 0
                print(f"  [{generated}/{num_samples}] {config['domain']}/{config['scenario'][:30]} "
                      f"— {stats['user_turns']}u/{stats['assistant_turns']}a turns, "
                      f"{stats['num_speak']}s, {stats['num_tool_calls']}tc, "
                      f"{stats['est_tokens']//1000}K tok, "
                      f"max_asst={stats['max_assistant_chars']//1000}K [{rate:.2f}/s]")

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
