"""Dataset Sources — download, parse, and convert external datasets into our format.

The hybrid pipeline:
  1. Download/load source datasets (computer use, web, conversations, tools, emotions)
  2. Convert each into our unified turn format (<think>/<tool_call>/<speak> blocks)
  3. Merge and interleave — stitch fragments into long, realistic living agent sessions
  4. Layer on speech tags, emotions, interruptions, memory ops
  5. Validate format strictly

Source datasets and what we extract from each:
  - Claude Code traces   → real tool calls, terminal output, file operations
  - OS-World / GUI data  → computer use actions (click, type, scroll, screen descriptions)
  - Mind2Web / WebArena  → web browsing actions (navigate, fill forms, click links)
  - UltraChat / WildChat → natural conversation patterns, casual talk
  - ToolBench            → tool use chains, multi-step reasoning
  - GoEmotions / MELD    → emotion labels (for probe training, separate)
  - Expresso             → expressive speech patterns (for Talker training, separate)
"""

import json
import os
import random
import re
from pathlib import Path
from typing import Optional, Iterator

# =============================================================================
# Dataset registry — what to download and how to parse it
# =============================================================================

DATASETS = {
    # ── Computer Use ─────────────────────────────────────────────────
    "osworld": {
        "source": "huggingface",
        "repo": "xlangai/os-world",
        "description": "OS-level computer use trajectories — click, type, scroll on real desktop",
        "format": "json",
        "license": "apache-2.0",
        "converter": "convert_osworld",
    },
    "gui_odyssey": {
        "source": "huggingface",
        "repo": "showlab/GUI-Odyssey",
        "description": "Cross-app GUI navigation trajectories on Android/desktop",
        "format": "json",
        "license": "cc-by-4.0",
        "converter": "convert_gui_odyssey",
    },
    "screenspot": {
        "source": "huggingface",
        "repo": "KevinQHLin/ScreenSpot",
        "description": "GUI element grounding — find and click specific elements",
        "format": "json",
        "license": "apache-2.0",
        "converter": "convert_screenspot",
    },

    # ── Web Browsing ─────────────────────────────────────────────────
    "mind2web": {
        "source": "huggingface",
        "repo": "osunlp/Mind2Web",
        "description": "Real web browsing tasks — click, type, select on live websites",
        "format": "json",
        "license": "cc-by-4.0",
        "converter": "convert_mind2web",
    },

    # ── Natural Conversation ─────────────────────────────────────────
    "wildchat": {
        "source": "huggingface",
        "repo": "allenai/WildChat-1M",
        "description": "1M real user-AI conversations — natural language patterns",
        "format": "parquet",
        "license": "odc-by",
        "converter": "convert_wildchat",
    },

    # ── Tool Use ─────────────────────────────────────────────────────
    "toolbench": {
        "source": "huggingface",
        "repo": "ToolBench/ToolBench",
        "description": "Multi-step tool use conversations with real API calls",
        "format": "json",
        "license": "apache-2.0",
        "converter": "convert_toolbench",
    },

    # ── Claude Code Traces (local) ───────────────────────────────────
    "claude_traces": {
        "source": "local",
        "path": None,  # Set at runtime — ~/.claude/projects/ or custom
        "description": "Your real Claude Code conversation logs with tool calls",
        "format": "jsonl",
        "converter": "convert_claude_traces",
    },
}


# =============================================================================
# Unified turn format — everything gets converted to this
# =============================================================================

def make_turn(role: str, content: str, metadata: dict = None) -> dict:
    """Create a turn in our unified format.

    role: "user", "assistant", "system"
    content: raw text with <think>, <tool_call>, <speak>, etc. blocks
    metadata: optional — emotion labels, prosody, source dataset, etc.
    """
    turn = {"role": role, "content": content}
    if metadata:
        turn["metadata"] = metadata
    return turn


# =============================================================================
# Converters — one per source dataset
# =============================================================================

def convert_claude_traces(traces_dir: str, max_samples: int = None) -> Iterator[list[dict]]:
    """Convert Claude Code JSONL traces into our turn format.

    Extracts: tool call patterns, terminal output, file contents, thinking.
    Wraps assistant text in <speak> blocks (as if the agent were speaking it).
    """
    from pathlib import Path

    traces = sorted(Path(traces_dir).rglob("*.jsonl"))
    if max_samples:
        traces = traces[:max_samples]

    for trace_path in traces:
        if "subagents" in str(trace_path):
            continue
        try:
            turns = _parse_claude_trace(trace_path)
            if turns and len(turns) >= 4:
                yield turns
        except Exception:
            continue


def _parse_claude_trace(path: Path) -> list[dict]:
    """Parse a single Claude Code JSONL trace into turns."""
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Merge streamed assistant chunks (same message ID)
    merged_events = _merge_assistant_chunks(events)

    turns = []
    for evt in merged_events:
        evt_type = evt.get("type")
        msg = evt.get("message", {})
        content = msg.get("content", "")

        if evt_type == "user":
            text = _extract_text(content)
            if text.strip():
                turns.append(make_turn("user", text.strip()))

        elif evt_type == "assistant":
            blocks = _extract_assistant_blocks(content)
            if blocks.strip():
                turns.append(make_turn("assistant", blocks.strip()))

    return turns


def _merge_assistant_chunks(events: list) -> list:
    """Merge streamed Claude Code assistant chunks by message ID."""
    merged = []
    buffer = {}

    for evt in events:
        if evt.get("type") != "assistant":
            if evt.get("type") == "user" and buffer:
                for key in list(buffer.keys()):
                    merged.append(buffer.pop(key))
            merged.append(evt)
            continue

        msg = evt.get("message", {})
        key = msg.get("id", "") or evt.get("requestId", "")
        if not key:
            merged.append(evt)
            continue

        if key not in buffer:
            buffer[key] = json.loads(json.dumps(evt))
        else:
            existing = buffer[key]["message"].get("content", [])
            new = msg.get("content", [])
            if isinstance(existing, list) and isinstance(new, list):
                existing.extend(new)
                buffer[key]["message"]["content"] = existing

    for key in buffer:
        merged.append(buffer[key])

    return merged


def _extract_text(content) -> str:
    """Extract plain text from Claude Code content blocks."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif block.get("type") == "tool_result":
                result = block.get("content", "")
                if isinstance(result, list):
                    result = " ".join(
                        b.get("text", "") for b in result if isinstance(b, dict)
                    )
                parts.append(f"<tool_result>{result}</tool_result>")
    return "\n".join(parts)


def _extract_assistant_blocks(content) -> str:
    """Convert Claude Code assistant content into our block format."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    parts = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type", "")

        if btype == "thinking":
            thinking = block.get("thinking", "")
            if thinking.strip():
                parts.append(f"<think>{thinking.strip()}</think>")

        elif btype == "text":
            text = block.get("text", "").strip()
            if text:
                # This is assistant speech — will be wrapped in <speak> during mixing
                parts.append(text)

        elif btype == "tool_use":
            name = block.get("name", "unknown")
            args = block.get("input", {})
            call = json.dumps({"name": name, "args": args})
            parts.append(f"<tool_call>{call}</tool_call>")

    return "\n".join(parts)


def convert_osworld(data_dir: str, max_samples: int = None) -> Iterator[list[dict]]:
    """Convert OS-World trajectories into our turn format.

    OS-World has: task description, sequence of (screenshot, action) pairs.
    We convert actions to our computer use tool calls.
    """
    data_path = Path(data_dir)
    files = sorted(data_path.glob("**/*.json"))
    if max_samples:
        files = files[:max_samples]

    for fpath in files:
        try:
            data = json.loads(fpath.read_text())
            turns = _convert_osworld_task(data)
            if turns:
                yield turns
        except Exception:
            continue


def _convert_osworld_task(data: dict) -> list[dict]:
    """Convert a single OS-World task to turns."""
    task = data.get("instruction", data.get("task", ""))
    if not task:
        return []

    turns = [make_turn("user", task)]

    actions = data.get("actions", data.get("trajectory", []))
    assistant_blocks = []

    for action in actions:
        action_type = action.get("action_type", action.get("type", ""))
        # Map OS-World actions to our tool calls
        tool_call = _map_osworld_action(action_type, action)
        if tool_call:
            assistant_blocks.append(f'<tool_call>{json.dumps(tool_call)}</tool_call>')
            # Add a simulated result based on the action
            result = action.get("result", action.get("observation", "done"))
            if result:
                assistant_blocks.append(f'<tool_result>{result}</tool_result>')

    if assistant_blocks:
        turns.append(make_turn("assistant", "\n".join(assistant_blocks)))

    return turns


def _map_osworld_action(action_type: str, action: dict) -> Optional[dict]:
    """Map an OS-World action to our tool call format."""
    action_type = action_type.lower()

    if action_type in ("click", "left_click", "single_click"):
        return {"name": "click", "args": {
            "x": action.get("coordinate", [0, 0])[0] if isinstance(action.get("coordinate"), list) else action.get("x", 0),
            "y": action.get("coordinate", [0, 0])[1] if isinstance(action.get("coordinate"), list) else action.get("y", 0),
        }}
    elif action_type in ("double_click",):
        return {"name": "click", "args": {
            "x": action.get("coordinate", [0, 0])[0] if isinstance(action.get("coordinate"), list) else 0,
            "y": action.get("coordinate", [0, 0])[1] if isinstance(action.get("coordinate"), list) else 0,
            "button": "double",
        }}
    elif action_type in ("right_click",):
        return {"name": "click", "args": {
            "x": action.get("coordinate", [0, 0])[0] if isinstance(action.get("coordinate"), list) else 0,
            "y": action.get("coordinate", [0, 0])[1] if isinstance(action.get("coordinate"), list) else 0,
            "button": "right",
        }}
    elif action_type in ("type", "input_text", "type_text"):
        return {"name": "type_text", "args": {
            "text": action.get("text", action.get("value", "")),
        }}
    elif action_type in ("key", "hotkey", "key_press", "press"):
        return {"name": "key_press", "args": {
            "keys": action.get("key", action.get("value", "")),
        }}
    elif action_type in ("scroll", "scroll_down", "scroll_up"):
        direction = "down" if "down" in action_type else "up"
        return {"name": "scroll", "args": {
            "direction": action.get("direction", direction),
            "amount": action.get("amount", 3),
        }}
    elif action_type in ("drag", "drag_to"):
        return {"name": "drag", "args": {
            "from_x": action.get("start_coordinate", [0, 0])[0],
            "from_y": action.get("start_coordinate", [0, 0])[1],
            "to_x": action.get("end_coordinate", [0, 0])[0],
            "to_y": action.get("end_coordinate", [0, 0])[1],
        }}

    return None


def convert_mind2web(data_dir: str, max_samples: int = None) -> Iterator[list[dict]]:
    """Convert Mind2Web web browsing tasks into our turn format.

    Mind2Web has: task, website, sequence of (element, action) pairs.
    We convert to web_browse + click + type_text sequences.
    """
    data_path = Path(data_dir)
    files = sorted(data_path.glob("**/*.json"))
    if max_samples:
        files = files[:max_samples]

    for fpath in files:
        try:
            data = json.loads(fpath.read_text())
            if isinstance(data, list):
                for item in data[:max_samples or len(data)]:
                    turns = _convert_mind2web_task(item)
                    if turns:
                        yield turns
            else:
                turns = _convert_mind2web_task(data)
                if turns:
                    yield turns
        except Exception:
            continue


def _convert_mind2web_task(data: dict) -> list[dict]:
    """Convert a single Mind2Web task."""
    task = data.get("confirmed_task", data.get("task", ""))
    website = data.get("website", "")
    if not task:
        return []

    turns = [make_turn("user", task)]
    blocks = []

    if website:
        blocks.append(f'<tool_call>{json.dumps({"name": "web_browse", "args": {"url": website}})}</tool_call>')
        blocks.append(f'<tool_result>Opened {website}</tool_result>')

    for action in data.get("action_reprs", data.get("actions", [])):
        if isinstance(action, str):
            # Action representation string like "Click [button] Submit"
            if action.lower().startswith("click"):
                element = action.split("]", 1)[-1].strip() if "]" in action else action[6:]
                blocks.append(f'<tool_call>{json.dumps({"name": "click", "args": {"element": element}})}</tool_call>')
                blocks.append(f'<tool_result>Clicked: {element}</tool_result>')
            elif action.lower().startswith("type"):
                parts = action.split("]", 1)
                text = parts[-1].strip() if len(parts) > 1 else ""
                blocks.append(f'<tool_call>{json.dumps({"name": "type_text", "args": {"text": text}})}</tool_call>')
                blocks.append(f'<tool_result>Typed: {text}</tool_result>')
            elif action.lower().startswith("select"):
                blocks.append(f'<tool_call>{json.dumps({"name": "click", "args": {"element": action}})}</tool_call>')
                blocks.append(f'<tool_result>Selected: {action}</tool_result>')

    if blocks:
        turns.append(make_turn("assistant", "\n".join(blocks)))

    return turns


def convert_wildchat(data_dir: str, max_samples: int = 1000) -> Iterator[list[dict]]:
    """Convert WildChat conversations into our turn format.

    WildChat has natural human-AI conversations. We extract the conversation
    patterns and will later layer speech tags and emotions on top.
    """
    data_path = Path(data_dir)
    # WildChat is typically parquet — try both
    try:
        import pandas as pd
        files = sorted(data_path.glob("**/*.parquet"))
        count = 0
        for fpath in files:
            df = pd.read_parquet(fpath)
            for _, row in df.iterrows():
                if count >= max_samples:
                    return
                conv = row.get("conversation", [])
                if isinstance(conv, str):
                    conv = json.loads(conv)
                turns = []
                for msg in conv:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role in ("user", "human"):
                        turns.append(make_turn("user", content))
                    elif role in ("assistant", "gpt"):
                        turns.append(make_turn("assistant", content))
                if len(turns) >= 4:
                    yield turns
                    count += 1
    except ImportError:
        # Fallback to JSONL
        for fpath in sorted(data_path.glob("**/*.jsonl")):
            count = 0
            with open(fpath) as f:
                for line in f:
                    if count >= max_samples:
                        return
                    try:
                        data = json.loads(line)
                        conv = data.get("conversation", data.get("messages", []))
                        turns = []
                        for msg in conv:
                            role = msg.get("role", "")
                            content = msg.get("content", "")
                            if role in ("user", "human"):
                                turns.append(make_turn("user", content))
                            elif role in ("assistant", "gpt"):
                                turns.append(make_turn("assistant", content))
                        if len(turns) >= 4:
                            yield turns
                            count += 1
                    except json.JSONDecodeError:
                        continue


def convert_toolbench(data_dir: str, max_samples: int = None) -> Iterator[list[dict]]:
    """Convert ToolBench tool use conversations.

    ToolBench has multi-step tool use with real API calls.
    We map their tool format to ours.
    """
    data_path = Path(data_dir)
    files = sorted(data_path.glob("**/*.json"))
    if max_samples:
        files = files[:max_samples]

    for fpath in files:
        try:
            data = json.loads(fpath.read_text())
            items = data if isinstance(data, list) else [data]
            for item in items:
                turns = _convert_toolbench_item(item)
                if turns and len(turns) >= 2:
                    yield turns
        except Exception:
            continue


def _convert_toolbench_item(data: dict) -> list[dict]:
    """Convert a single ToolBench conversation."""
    turns = []
    for step in data.get("answer_generation", data.get("steps", [])):
        role = step.get("role", "")
        if role == "user":
            turns.append(make_turn("user", step.get("message", "")))
        elif role == "assistant":
            content = step.get("message", "")
            # Convert their tool calls to our format
            if "Action:" in content and "Action Input:" in content:
                think_match = re.search(r"Thought:(.*?)Action:", content, re.DOTALL)
                action_match = re.search(r"Action:\s*(.+)", content)
                input_match = re.search(r"Action Input:\s*(.+)", content)
                blocks = []
                if think_match:
                    blocks.append(f"<think>{think_match.group(1).strip()}</think>")
                if action_match and input_match:
                    try:
                        args = json.loads(input_match.group(1))
                    except json.JSONDecodeError:
                        args = {"input": input_match.group(1)}
                    call = {"name": action_match.group(1).strip(), "args": args}
                    blocks.append(f"<tool_call>{json.dumps(call)}</tool_call>")
                if blocks:
                    turns.append(make_turn("assistant", "\n".join(blocks)))
            else:
                turns.append(make_turn("assistant", content))
        elif role == "tool":
            # Tool result
            if turns and turns[-1]["role"] == "assistant":
                turns[-1]["content"] += f"\n<tool_result>{step.get('message', '')}</tool_result>"
    return turns


# =============================================================================
# Download helpers
# =============================================================================

def download_dataset(name: str, output_dir: str = "data/sources") -> str:
    """Download a dataset to local storage."""
    info = DATASETS.get(name)
    if not info:
        raise ValueError(f"Unknown dataset: {name}. Options: {list(DATASETS.keys())}")

    if info["source"] == "local":
        return info.get("path") or str(Path.home() / ".claude" / "projects")

    dest = Path(output_dir) / name
    dest.mkdir(parents=True, exist_ok=True)

    if info["source"] == "huggingface":
        print(f"Downloading {name} from HuggingFace: {info['repo']}")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=info["repo"],
                repo_type="dataset",
                local_dir=str(dest),
            )
            print(f"  -> {dest}")
            return str(dest)
        except ImportError:
            print("  pip install huggingface_hub to download automatically")
            print(f"  Manual: huggingface-cli download {info['repo']} --repo-type dataset --local-dir {dest}")
            return str(dest)

    return str(dest)


def get_converter(name: str):
    """Get the converter function for a dataset."""
    info = DATASETS.get(name)
    if not info:
        return None
    converter_name = info["converter"]
    return globals().get(converter_name)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Manage dataset sources")
    sub = parser.add_subparsers(dest="command")

    dl = sub.add_parser("download", help="Download a dataset")
    dl.add_argument("name", choices=list(DATASETS.keys()))
    dl.add_argument("--output", default="data/sources")

    ls = sub.add_parser("list", help="List available datasets")

    test = sub.add_parser("test", help="Test a converter with a few samples")
    test.add_argument("name", choices=list(DATASETS.keys()))
    test.add_argument("--path", help="Data path override")
    test.add_argument("--num", type=int, default=3)

    args = parser.parse_args()

    if args.command == "list":
        for name, info in DATASETS.items():
            src = info["source"]
            repo = info.get("repo", info.get("path", "local"))
            print(f"  {name:20s} [{src:12s}] {info['description']}")

    elif args.command == "download":
        download_dataset(args.name, args.output)

    elif args.command == "test":
        converter = get_converter(args.name)
        if not converter:
            print(f"No converter for {args.name}")
        else:
            data_path = args.path
            if not data_path:
                info = DATASETS[args.name]
                if info["source"] == "local":
                    data_path = info.get("path") or str(Path.home() / ".claude" / "projects")
                else:
                    data_path = f"data/sources/{args.name}"

            print(f"Testing {args.name} converter with path: {data_path}")
            count = 0
            for turns in converter(data_path, max_samples=args.num):
                count += 1
                print(f"\n{'='*60}")
                print(f"Sample {count}: {len(turns)} turns")
                for t in turns[:4]:
                    role = t["role"]
                    content = t["content"][:200]
                    print(f"  [{role}] {content}...")
            print(f"\nTotal samples: {count}")
