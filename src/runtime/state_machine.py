"""Agent State Machine — manages transitions between agent states.

States:
  idle       → Model is on standby (base model, LoRA off). Runs background tasks.
  listening  → User is speaking. Whisper is transcribing.
  thinking   → Internal reasoning. Can be interrupted.
  speaking   → Generating speech via Talker. Stops if user speaks.
  tool_using → Running a tool. Finishes tool, then checks for interruption.
  working    → Autonomously doing a multi-step task (coding, researching).
  background → Base model running compression/memory in background.
"""

import enum
import time
import asyncio
import logging
from typing import Callable, Optional

log = logging.getLogger(__name__)


class AgentState(enum.Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    TOOL_USING = "tool_using"
    WORKING = "working"
    BACKGROUND = "background"


# Valid transitions: from_state -> set of allowed to_states
TRANSITIONS = {
    AgentState.IDLE: {
        AgentState.LISTENING,   # User starts speaking
        AgentState.THINKING,    # Agent decides to do something
        AgentState.BACKGROUND,  # Start background maintenance
        AgentState.WORKING,     # Autonomous task
    },
    AgentState.LISTENING: {
        AgentState.THINKING,    # User finished speaking, agent processes
        AgentState.IDLE,        # False alarm / silence
    },
    AgentState.THINKING: {
        AgentState.SPEAKING,    # Ready to respond
        AgentState.TOOL_USING,  # Need to use a tool first
        AgentState.LISTENING,   # Interrupted by user
        AgentState.IDLE,        # Nothing to say
    },
    AgentState.SPEAKING: {
        AgentState.LISTENING,   # User interrupts
        AgentState.THINKING,    # More to process
        AgentState.TOOL_USING,  # Need to use a tool mid-response
        AgentState.IDLE,        # Done speaking
    },
    AgentState.TOOL_USING: {
        AgentState.THINKING,    # Tool done, process results
        AgentState.SPEAKING,    # Tool done, report to user
        AgentState.LISTENING,   # User interrupted (after tool finishes)
        AgentState.TOOL_USING,  # Chain another tool
    },
    AgentState.WORKING: {
        AgentState.LISTENING,   # User interrupts
        AgentState.SPEAKING,    # Want to report progress
        AgentState.TOOL_USING,  # Using a tool as part of work
        AgentState.THINKING,    # Reasoning about next step
        AgentState.IDLE,        # Task complete
    },
    AgentState.BACKGROUND: {
        AgentState.LISTENING,   # User starts speaking (background continues separately)
        AgentState.IDLE,        # Background task done
    },
}


class StateMachine:
    """Manages agent state transitions with callbacks and interruption support."""

    def __init__(self):
        self._state = AgentState.IDLE
        self._listeners: list[Callable[[AgentState, AgentState], None]] = []
        self._interrupt_requested = False
        self._state_entered_at: float = time.monotonic()
        self._history: list[tuple[float, AgentState]] = [(time.monotonic(), AgentState.IDLE)]

    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def time_in_state(self) -> float:
        """Seconds spent in current state."""
        return time.monotonic() - self._state_entered_at

    @property
    def is_interruptible(self) -> bool:
        """Can the user interrupt the current state?"""
        return self._state in {
            AgentState.SPEAKING,
            AgentState.THINKING,
            AgentState.WORKING,
            AgentState.IDLE,
            AgentState.BACKGROUND,
        }

    def transition(self, new_state: AgentState) -> bool:
        """Attempt a state transition. Returns True if successful."""
        if new_state == self._state:
            return True

        allowed = TRANSITIONS.get(self._state, set())
        if new_state not in allowed:
            log.warning(f"Invalid transition: {self._state.value} -> {new_state.value}")
            return False

        old = self._state
        self._state = new_state
        self._state_entered_at = time.monotonic()
        self._history.append((time.monotonic(), new_state))

        # Keep history bounded
        if len(self._history) > 1000:
            self._history = self._history[-500:]

        log.debug(f"State: {old.value} -> {new_state.value}")

        # Notify listeners
        for cb in self._listeners:
            try:
                cb(old, new_state)
            except Exception as e:
                log.error(f"State listener error: {e}")

        return True

    def request_interrupt(self):
        """Signal that the user wants to interrupt."""
        self._interrupt_requested = True
        if self.is_interruptible and self._state != AgentState.TOOL_USING:
            # Immediately transition to listening (except tool_using waits)
            self.transition(AgentState.LISTENING)

    def check_interrupt(self) -> bool:
        """Check and clear interrupt flag. Called after tool completion."""
        if self._interrupt_requested:
            self._interrupt_requested = False
            self.transition(AgentState.LISTENING)
            return True
        return False

    def on_transition(self, callback: Callable[[AgentState, AgentState], None]):
        """Register a state transition listener."""
        self._listeners.append(callback)

    def force_state(self, state: AgentState):
        """Force a state change (bypasses transition rules). Use sparingly."""
        old = self._state
        self._state = state
        self._state_entered_at = time.monotonic()
        self._history.append((time.monotonic(), state))
        log.warning(f"Forced state: {old.value} -> {state.value}")
