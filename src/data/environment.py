"""Agent Environment — defines the tools, memory systems, and world the agent lives in.

This agent is a continuously-running local AI that:
  - Talks naturally (voice in/out via thinker-talker)
  - Sees your screen (Qwen 3.5 vision)
  - Controls your computer (mouse, keyboard, apps, windows)
  - Remembers everything (knowledge graph + rolling compression)
  - Uses tools autonomously (computer, terminal, search, files, web)
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

    # --- Computer Use (mouse, keyboard, windows, apps) ---
    "click": {
        "description": "Click at screen coordinates or on a UI element",
        "args": {"x": "int?", "y": "int?", "element": "str?", "button": "str?"},
        # element: natural language like "the save button", "the search bar"
        # button: "left" (default), "right", "double", "middle"
    },
    "type_text": {
        "description": "Type text at the current cursor position",
        "args": {"text": "str", "press_enter": "bool?"},
    },
    "key_press": {
        "description": "Press a keyboard shortcut or key combination",
        "args": {"keys": "str"},
        # e.g., "ctrl+s", "alt+tab", "enter", "ctrl+shift+t", "win+e"
    },
    "scroll": {
        "description": "Scroll the screen or a specific area",
        "args": {"direction": "str", "amount": "int?", "x": "int?", "y": "int?"},
        # direction: "up", "down", "left", "right"
    },
    "drag": {
        "description": "Drag from one point to another (move files, resize windows, etc.)",
        "args": {"from_x": "int", "from_y": "int", "to_x": "int", "to_y": "int"},
    },
    "move_mouse": {
        "description": "Move the mouse cursor to a position",
        "args": {"x": "int", "y": "int"},
    },
    "open_app": {
        "description": "Open an application by name",
        "args": {"name": "str"},
        # e.g., "Chrome", "File Explorer", "Spotify", "Discord", "Settings"
    },
    "close_app": {
        "description": "Close an application or window",
        "args": {"name": "str?", "force": "bool?"},
    },
    "switch_window": {
        "description": "Switch to a specific window or application",
        "args": {"name": "str"},
    },
    "select_text": {
        "description": "Select text on screen (for copy/paste, etc.)",
        "args": {"method": "str", "value": "str?"},
        # method: "all" (ctrl+a), "word", "line", "range" (with coordinates)
    },
    "clipboard": {
        "description": "Copy, paste, or read clipboard contents",
        "args": {"action": "str", "text": "str?"},
        # action: "copy", "paste", "cut", "read"
    },

    # --- File Management (no coding, just organizing) ---
    "read_file": {
        "description": "Read a file from the filesystem",
        "args": {"path": "str"},
    },
    "write_file": {
        "description": "Write content to a file (notes, lists, etc.)",
        "args": {"path": "str", "content": "str"},
    },
    "move_file": {
        "description": "Move or rename a file or folder",
        "args": {"src": "str", "dest": "str"},
    },
    "copy_file": {
        "description": "Copy a file or folder",
        "args": {"src": "str", "dest": "str"},
    },
    "delete_file": {
        "description": "Delete a file or folder",
        "args": {"path": "str", "confirm": "bool?"},
    },
    "list_files": {
        "description": "List files and folders in a directory",
        "args": {"path": "str?", "pattern": "str?"},
    },
    "search_files": {
        "description": "Search for files by name or content",
        "args": {"query": "str", "path": "str?", "type": "str?"},
        # type: "name", "content", "both"
    },

    # --- Shell / Terminal ---
    "run_command": {
        "description": "Execute a shell command",
        "args": {"cmd": "str", "cwd": "str?", "timeout": "int?"},
    },

    # --- Memory (Knowledge Graph) ---
    "memory_store": {
        "description": "Store an entity/fact in the knowledge graph",
        "args": {
            "entity": "str",
            "type": "str",  # person, concept, project, preference, fact, place, event
            "properties": "dict",
            "relations": "list[dict]?",  # [{"type": "likes", "target": "cats"}]
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
        "description": "Take a screenshot of the current screen or a specific window",
        "args": {"window": "str?", "region": "dict?"},
        # region: {"x": int, "y": int, "width": int, "height": int}
    },
    "read_screen": {
        "description": "Read and understand what's currently visible on screen (OCR + vision)",
        "args": {"question": "str?"},
        # Returns structured description of screen contents
    },
    "read_image": {
        "description": "Analyze an image file (photo, diagram, chart, etc.)",
        "args": {"path": "str", "question": "str?"},
    },
    "find_on_screen": {
        "description": "Find a UI element, text, or icon on screen and return its coordinates",
        "args": {"target": "str"},
        # e.g., "the close button", "the wifi icon", "text saying 'Submit'"
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
    "web_browse": {
        "description": "Open a URL in the browser (visible to user)",
        "args": {"url": "str"},
    },

    # --- System ---
    "check_processes": {
        "description": "List running processes, CPU/memory usage",
        "args": {"filter": "str?"},
    },
    "system_info": {
        "description": "Get system information (battery, storage, network, display, etc.)",
        "args": {"what": "str"},
        # what: "battery", "storage", "network", "displays", "audio", "bluetooth"
    },
    "volume": {
        "description": "Control system audio volume",
        "args": {"level": "int?", "action": "str?"},
        # action: "mute", "unmute", "up", "down"  |  level: 0-100
    },
    "brightness": {
        "description": "Control screen brightness",
        "args": {"level": "int"},  # 0-100
    },
    "wifi": {
        "description": "Manage wifi connection",
        "args": {"action": "str", "network": "str?", "password": "str?"},
        # action: "status", "connect", "disconnect", "list"
    },
    "bluetooth": {
        "description": "Manage bluetooth devices",
        "args": {"action": "str", "device": "str?"},
        # action: "status", "connect", "disconnect", "list"
    },
    "notify": {
        "description": "Send a desktop notification",
        "args": {"title": "str", "body": "str"},
    },
    "set_timer": {
        "description": "Set a reminder/timer",
        "args": {"duration": "str", "message": "str"},
    },
    "set_alarm": {
        "description": "Set an alarm for a specific time",
        "args": {"time": "str", "message": "str", "repeat": "str?"},
    },

    # --- Media ---
    "play_media": {
        "description": "Control media playback (music, video)",
        "args": {"action": "str", "query": "str?"},
        # action: "play", "pause", "next", "previous", "stop"
        # query: "lofi beats", "that song from earlier" — for play action
    },
    "record_audio": {
        "description": "Record audio from microphone",
        "args": {"duration": "str?", "output": "str?"},
    },
    "take_photo": {
        "description": "Take a photo with the webcam",
        "args": {"output": "str?"},
    },
}

# =============================================================================
# TOOL CATEGORIES — for scenario generation and context
# =============================================================================

TOOL_CATEGORIES = {
    "computer_use": ["click", "type_text", "key_press", "scroll", "drag", "move_mouse",
                     "open_app", "close_app", "switch_window", "select_text", "clipboard"],
    "file_management": ["read_file", "write_file", "move_file", "copy_file",
                        "delete_file", "list_files", "search_files"],
    "terminal": ["run_command"],
    "memory": ["memory_store", "memory_query", "memory_update",
               "context_compress", "context_recall"],
    "vision": ["screenshot", "read_screen", "read_image", "find_on_screen"],
    "web": ["web_search", "web_fetch", "web_browse"],
    "system": ["check_processes", "system_info", "volume", "brightness",
               "wifi", "bluetooth", "notify", "set_timer", "set_alarm"],
    "media": ["play_media", "record_audio", "take_photo"],
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
    "working",        # Autonomously doing a task
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
    "notice_screen_change",       # Sees something change on screen, reacts
    "notice_notification",        # System notification pops up, mentions it
    "notice_high_cpu",            # Spots resource usage spike
    "notice_low_battery",         # Battery getting low, warns user
    "notice_download_complete",   # Download finished, lets user know
    "finish_background_task",     # Completes something it was working on
    "remember_relevant_fact",     # Recalls something from memory that's relevant
    "check_on_long_process",     # Download, render, upload, etc.
    "organize_memory",           # Periodically compress/organize knowledge graph
    "read_screen_context",       # Glances at screen, understands what user is doing
    "suggest_help",              # Notices user struggling with something
    "tidy_up",                   # Organize desktop, close unused apps, clear temp files
]

# =============================================================================
# COMPUTER USE SCENARIOS — what kind of tasks the agent handles
# =============================================================================

SCENARIO_DOMAINS = {
    "file_management": [
        "Organize messy desktop",
        "Find a file the user can't locate",
        "Sort downloads folder",
        "Move photos to the right albums",
        "Clean up duplicate files",
        "Rename batch of files",
        "Create folder structure for a project",
        "Back up important files",
        "Extract a zip and organize contents",
    ],
    "web_browsing": [
        "Research a topic and summarize findings",
        "Find a recipe and save it",
        "Compare products/prices",
        "Book something (restaurant, tickets, travel)",
        "Fill out an online form",
        "Check weather forecast",
        "Look up directions",
        "Find a YouTube video",
        "Read and summarize an article",
        "Check social media for something specific",
    ],
    "system_management": [
        "Computer is running slow, diagnose why",
        "Connect to bluetooth speaker/headphones",
        "Fix wifi connection issues",
        "Adjust display/brightness settings",
        "Free up disk space",
        "Update apps or system",
        "Set up a printer",
        "Change default apps",
        "Manage startup programs",
        "Check battery health",
    ],
    "productivity": [
        "Set timers and reminders",
        "Take notes during a meeting",
        "Write and send an email",
        "Create a to-do list",
        "Manage calendar events",
        "Convert a file format",
        "Merge PDFs",
        "Resize/crop images",
        "Create a simple spreadsheet",
        "Write a letter or document",
    ],
    "media_entertainment": [
        "Play music / control playback",
        "Find and play a movie/show",
        "Adjust volume during a call",
        "Record a voice memo",
        "Take a photo or screenshot",
        "Edit a photo (crop, filter)",
        "Create a playlist",
        "Stream to TV/external display",
    ],
    "social_communication": [
        "Read and reply to messages",
        "Video call setup",
        "Share a file with someone",
        "Post something on social media",
        "Check notifications",
        "Find a contact and call them",
    ],
    "troubleshooting": [
        "App won't open / keeps crashing",
        "No sound from speakers",
        "Screen flickering",
        "Can't connect to internet",
        "Printer not working",
        "Running out of storage",
        "Computer froze, what to do",
        "Accidentally deleted something important",
        "Pop-up ads keep appearing",
        "External drive not showing up",
    ],
    "casual_living": [
        "Just chatting about nothing",
        "User asks random questions",
        "User asks agent's opinion on something",
        "User vents about their day",
        "User wants to learn something",
        "User asks for a recommendation",
        "Agent notices something interesting on screen",
        "User and agent have a running joke",
        "User comes home and checks in",
        "User says goodnight",
    ],
}
