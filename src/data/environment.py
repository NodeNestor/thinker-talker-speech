"""Agent Environment — defines the tools, memory systems, and world the agent lives in.

This agent is a continuously-running local AI that:
  - Talks naturally (voice in/out via thinker-talker)
  - Sees your screen (Qwen 3.5 vision)
  - Remembers everything (knowledge graph + rolling compression)
  - Uses tools autonomously (code, terminal, search, files)
  - Lives — works on stuff between conversations, initiates when it notices things
  - Gets interrupted, handles it gracefully

The base model (LoRA off) runs background tasks like memory compression
and context summarization, while the main model (LoRA on) handles
conversation and tool use.
"""

# =============================================================================
# TOOLS — what the agent can do
# =============================================================================

TOOLS = {
    # --- Coding / Development ---
    "read_file": {
        "description": "Read a file from the filesystem",
        "args": {"path": "str", "line_start": "int?", "line_end": "int?"},
    },
    "write_file": {
        "description": "Write content to a file",
        "args": {"path": "str", "content": "str"},
    },
    "edit_file": {
        "description": "Edit a specific section of a file",
        "args": {"path": "str", "old": "str", "new": "str"},
    },
    "search_code": {
        "description": "Search codebase for a pattern",
        "args": {"query": "str", "glob": "str?", "max_results": "int?"},
    },
    "run_command": {
        "description": "Execute a shell command",
        "args": {"cmd": "str", "cwd": "str?", "timeout": "int?"},
    },
    "run_tests": {
        "description": "Run test suite",
        "args": {"path": "str?", "filter": "str?", "verbose": "bool?"},
    },
    "git": {
        "description": "Run git operations",
        "args": {"command": "str"},  # e.g., "status", "diff", "log --oneline -5"
    },

    # --- Memory (Knowledge Graph) ---
    "memory_store": {
        "description": "Store an entity/fact in the knowledge graph",
        "args": {
            "entity": "str",
            "type": "str",  # person, concept, project, preference, fact
            "properties": "dict",
            "relations": "list[dict]?",  # [{"type": "uses", "target": "Python"}]
        },
    },
    "memory_query": {
        "description": "Query the knowledge graph for entities and relations",
        "args": {"query": "str", "type": "str?", "limit": "int?"},
    },
    "memory_update": {
        "description": "Update an existing entity in the knowledge graph",
        "args": {"entity": "str", "properties": "dict"},
    },

    # --- Memory (Rolling Context) ---
    "context_compress": {
        "description": "Compress old conversation context (runs on base model in background)",
        "args": {"conversation_id": "str"},
    },
    "context_recall": {
        "description": "Recall compressed context from a previous conversation",
        "args": {"query": "str", "time_range": "str?"},
    },

    # --- Vision ---
    "screenshot": {
        "description": "Take a screenshot of the current screen or a window",
        "args": {"window": "str?"},  # None = full screen
    },
    "read_image": {
        "description": "Analyze an image file",
        "args": {"path": "str", "question": "str?"},
    },

    # --- Web ---
    "web_search": {
        "description": "Search the internet",
        "args": {"query": "str", "max_results": "int?"},
    },
    "web_fetch": {
        "description": "Fetch and read a webpage",
        "args": {"url": "str"},
    },

    # --- System ---
    "check_processes": {
        "description": "List running processes, CPU/memory usage",
        "args": {"filter": "str?"},
    },
    "notify": {
        "description": "Send a desktop notification",
        "args": {"title": "str", "body": "str"},
    },
    "set_timer": {
        "description": "Set a reminder/timer",
        "args": {"duration": "str", "message": "str"},
    },
}

# =============================================================================
# AGENT STATES — what mode the agent is in
# =============================================================================

AGENT_STATES = [
    "idle",           # Chilling, monitoring, might notice something
    "listening",      # User is talking, processing speech input
    "thinking",       # Internal reasoning, can be interrupted
    "speaking",       # Generating speech output, stops if user speaks
    "tool_using",     # Running a tool, finishes tool then checks for interruption
    "working",        # Autonomously doing a task (coding, researching)
    "background",     # Base model running compression/memory in background
]

# =============================================================================
# INTERRUPTION RULES
# =============================================================================

INTERRUPTION_RULES = """
When the user starts speaking:
  - If agent is IDLE: immediately switch to LISTENING
  - If agent is SPEAKING: stop mid-sentence, switch to LISTENING
  - If agent is THINKING: stop thinking, switch to LISTENING
  - If agent is TOOL_USING: let the tool finish, THEN switch to LISTENING
  - If agent is WORKING: pause work, acknowledge user, switch to LISTENING
  - If agent is BACKGROUND: background continues, main model switches to LISTENING

After being interrupted, acknowledge naturally:
  - "Oh, yeah?" / "Hmm?" / "What's up?" / "Yeah go ahead"
  - Don't repeat what you were saying unless asked
  - If interrupted mid-tool, mention "just finishing this up" then listen
"""

# =============================================================================
# AUTONOMOUS BEHAVIORS — things the agent does on its own
# =============================================================================

AUTONOMOUS_BEHAVIORS = [
    "notice_build_failure",      # Sees CI fail, mentions it
    "notice_high_cpu",           # Spots resource usage spike
    "finish_background_task",    # Completes something it was working on
    "remember_relevant_fact",    # Recalls something from memory that's relevant
    "suggest_improvement",       # Notices a pattern and suggests optimization
    "check_on_long_process",     # Training run, deployment, etc.
    "organize_memory",           # Periodically compress/organize knowledge graph
    "read_screen_context",       # Glances at screen, understands what user is doing
]
