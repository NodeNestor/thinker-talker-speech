"""Tool Executor — runs tools and returns results.

These are REAL tools the agent can use at inference time.
During training data generation, these same tools are simulated by the LLM.
At runtime, they execute for real.
"""

import os
import subprocess
import json
import glob as glob_module
import re
import logging
import asyncio
from typing import Any, Optional
from pathlib import Path

log = logging.getLogger(__name__)


class ToolExecutor:
    """Executes agent tools against the real system."""

    def __init__(self, workspace: str = ".", timeout: int = 30):
        self.workspace = Path(workspace).resolve()
        self.timeout = timeout
        self._tools = {
            "read_file": self._read_file,
            "write_file": self._write_file,
            "edit_file": self._edit_file,
            "search_code": self._search_code,
            "run_command": self._run_command,
            "run_tests": self._run_tests,
            "git": self._git,
            "screenshot": self._screenshot,
            "read_image": self._read_image,
            "web_search": self._web_search,
            "web_fetch": self._web_fetch,
            "check_processes": self._check_processes,
            "notify": self._notify,
            "set_timer": self._set_timer,
            # Memory tools are handled by MemoryManager, not here
        }

    @property
    def available_tools(self) -> list[str]:
        return list(self._tools.keys())

    async def execute(self, name: str, args: dict) -> str:
        """Execute a tool by name. Returns result as string."""
        fn = self._tools.get(name)
        if fn is None:
            return f"Error: Unknown tool '{name}'"
        try:
            result = await asyncio.to_thread(fn, **args)
            return str(result)
        except Exception as e:
            log.error(f"Tool {name} failed: {e}")
            return f"Error: {e}"

    # ── File operations ──────────────────────────────────────────────

    def _read_file(self, path: str, line_start: int = None, line_end: int = None) -> str:
        fpath = self._resolve(path)
        if not fpath.exists():
            return f"Error: File not found: {path}"
        text = fpath.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        if line_start is not None or line_end is not None:
            start = (line_start or 1) - 1
            end = line_end or len(lines)
            lines = lines[start:end]
        # Cap output
        if len(lines) > 200:
            return "\n".join(lines[:200]) + f"\n... ({len(lines) - 200} more lines)"
        return "\n".join(lines)

    def _write_file(self, path: str, content: str) -> str:
        fpath = self._resolve(path)
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_text(content, encoding="utf-8")
        return f"Written: {path} ({len(content)} chars)"

    def _edit_file(self, path: str, old: str, new: str) -> str:
        fpath = self._resolve(path)
        if not fpath.exists():
            return f"Error: File not found: {path}"
        text = fpath.read_text(encoding="utf-8")
        if old not in text:
            return f"Error: old string not found in {path}"
        count = text.count(old)
        text = text.replace(old, new, 1)
        fpath.write_text(text, encoding="utf-8")
        return f"Edited: {path} (replaced 1 of {count} occurrences)"

    def _search_code(self, query: str, glob: str = None, pattern: str = None, max_results: int = 20) -> str:
        pattern = glob or pattern or "**/*"
        results = []
        for fpath in self.workspace.glob(pattern):
            if not fpath.is_file():
                continue
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")
                for i, line in enumerate(text.splitlines(), 1):
                    if query.lower() in line.lower():
                        rel = fpath.relative_to(self.workspace)
                        results.append(f"{rel}:{i}: {line.strip()}")
                        if len(results) >= max_results:
                            return "\n".join(results)
            except Exception:
                continue
        return "\n".join(results) if results else f"No results for '{query}'"

    # ── Shell / Commands ─────────────────────────────────────────────

    def _run_command(self, cmd: str, cwd: str = None, timeout: int = None) -> str:
        work_dir = self._resolve(cwd) if cwd else self.workspace
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                cwd=str(work_dir), timeout=timeout or self.timeout,
            )
            output = result.stdout
            if result.stderr:
                output += "\n" + result.stderr
            if result.returncode != 0:
                output += f"\n(exit code: {result.returncode})"
            return output.strip()[:5000]  # Cap output
        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {timeout or self.timeout}s"

    def _run_tests(self, path: str = None, filter: str = None, verbose: bool = False) -> str:
        cmd = "python -m pytest"
        if path:
            cmd += f" {path}"
        if filter:
            cmd += f" -k {filter}"
        if verbose:
            cmd += " -v"
        cmd += " --tb=short 2>&1"
        return self._run_command(cmd)

    def _git(self, command: str) -> str:
        return self._run_command(f"git {command}")

    # ── Vision ───────────────────────────────────────────────────────

    def _screenshot(self, window: str = None) -> str:
        """Take a screenshot. Returns path to saved image."""
        try:
            import mss
            with mss.mss() as sct:
                path = self.workspace / ".agent" / "screenshots"
                path.mkdir(parents=True, exist_ok=True)
                import time
                fname = path / f"screen_{int(time.time())}.png"
                sct.shot(output=str(fname))
                return f"[screenshot saved: {fname}]"
        except ImportError:
            return "[screenshot: mss not installed, pip install mss]"
        except Exception as e:
            return f"[screenshot failed: {e}]"

    def _read_image(self, path: str, question: str = None) -> str:
        """Analyze an image. At runtime this calls the vision model."""
        fpath = self._resolve(path)
        if not fpath.exists():
            return f"Error: Image not found: {path}"
        # Placeholder — at runtime, the vision model (Qwen 3.5) processes this
        return f"[image: {path}, {fpath.stat().st_size} bytes — vision analysis pending]"

    # ── Web ──────────────────────────────────────────────────────────

    def _web_search(self, query: str, max_results: int = 5) -> str:
        """Search the web. Requires httpx."""
        try:
            import httpx
            # Use DuckDuckGo instant answer API (no key needed)
            resp = httpx.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": 1},
                timeout=10,
            )
            data = resp.json()
            results = []
            if data.get("AbstractText"):
                results.append(data["AbstractText"])
            for r in data.get("RelatedTopics", [])[:max_results]:
                if isinstance(r, dict) and "Text" in r:
                    results.append(r["Text"])
            return "\n".join(results) if results else f"No results for '{query}'"
        except Exception as e:
            return f"Web search error: {e}"

    def _web_fetch(self, url: str) -> str:
        """Fetch a webpage."""
        try:
            import httpx
            resp = httpx.get(url, timeout=15, follow_redirects=True)
            # Return first 5000 chars of text content
            return resp.text[:5000]
        except Exception as e:
            return f"Fetch error: {e}"

    # ── System ───────────────────────────────────────────────────────

    def _check_processes(self, filter: str = None) -> str:
        """List running processes."""
        import platform
        if platform.system() == "Windows":
            cmd = "tasklist /FO CSV"
        else:
            cmd = "ps aux"
        if filter:
            cmd += f" | grep -i {filter}" if platform.system() != "Windows" else ""
        return self._run_command(cmd)[:3000]

    def _notify(self, title: str, body: str) -> str:
        """Send a desktop notification."""
        import platform
        if platform.system() == "Windows":
            # PowerShell toast
            ps = f'[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null; $null'
            self._run_command(f'powershell -c "Write-Host \'{title}: {body}\'"')
        elif platform.system() == "Darwin":
            self._run_command(f'osascript -e \'display notification "{body}" with title "{title}"\'')
        else:
            self._run_command(f'notify-send "{title}" "{body}"')
        return f"Notification sent: {title}"

    def _set_timer(self, duration: str, message: str) -> str:
        """Set a timer. Returns immediately, fires later."""
        # Parse duration like "5m", "30s", "1h"
        import re
        m = re.match(r"(\d+)\s*(s|m|h)", duration)
        if not m:
            return f"Error: Can't parse duration '{duration}'. Use format like '5m', '30s', '1h'"
        val, unit = int(m.group(1)), m.group(2)
        seconds = val * {"s": 1, "m": 60, "h": 3600}[unit]
        # Fire-and-forget timer
        async def _timer():
            await asyncio.sleep(seconds)
            log.info(f"Timer fired: {message}")
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(_timer())
        except RuntimeError:
            pass  # No event loop yet
        return f"Timer set: {duration} — '{message}'"

    # ── Helpers ──────────────────────────────────────────────────────

    def _resolve(self, path: str) -> Path:
        """Resolve a path relative to workspace."""
        p = Path(path)
        if p.is_absolute():
            return p
        return self.workspace / p
