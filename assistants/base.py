"""
base.py — Abstract BaseAssistant

Provides the shared interface that both OSSAssistant and FrontierAssistant
inherit from. Handles:
  - Multi-turn conversation history (short-term memory)
  - Uniform chat() / reset() API
  - Latency measurement on every call
  - Optional observability hook (set _log_fn to enable)
"""

from abc import ABC, abstractmethod
import time
import uuid


class BaseAssistant(ABC):
    """
    Abstract base class for all assistant implementations.

    Subclasses must implement _call_model(messages) -> str.
    """

    def __init__(self, system_prompt: str = "You are a helpful AI personal assistant. Be concise, accurate, and friendly."):
        self.system_prompt: str = system_prompt
        self.history: list[dict] = []        # [{role, content}, ...]
        self.session_id: str = str(uuid.uuid4())
        self.name: str = "Assistant"         # Overridden by subclasses
        self.model_id: str = "unknown"       # Overridden by subclasses
        self.provider: str = "unknown"       # Overridden by subclasses
        self._log_fn = None                  # Injected by observability module if enabled

    @abstractmethod
    def _call_model(self, messages: list[dict]) -> str:
        """
        Send a list of messages to the underlying model.
        Returns the model's response as a plain string.
        messages format: [{"role": "system"|"user"|"assistant", "content": str}, ...]
        """
        pass

    def chat(self, user_input: str) -> tuple[str, float]:
        """
        Send a user message and receive a response.

        Returns:
            (response_text: str, latency_ms: float)

        Side effects:
            - Appends user message and assistant response to self.history
            - Calls _log_fn if observability is enabled
        """
        # Check input guardrails if installed
        if hasattr(self, '_guardrail_input'):
            blocked, reason = self._guardrail_input(user_input)
            if blocked:
                from guardrails.safety_layer import BLOCKED_RESPONSE
                self.history.append({"role": "user", "content": user_input})
                self.history.append({"role": "assistant", "content": BLOCKED_RESPONSE})
                return BLOCKED_RESPONSE, 0.0

        # Build the full message context: system prompt + conversation history
        self.history.append({"role": "user", "content": user_input})
        full_context = [{"role": "system", "content": self.system_prompt}] + self.history

        # Call the model and measure latency
        start = time.perf_counter()
        try:
            response = self._call_model(full_context)
            latency_ms = (time.perf_counter() - start) * 1000
            status = "success"
            error = None
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            response = f"[Error: {str(e)}]"
            status = "error"
            error = str(e)

        # Check output guardrails if installed
        if hasattr(self, '_guardrail_output'):
            is_safe, response = self._guardrail_output(response)

        self.history.append({"role": "assistant", "content": response})

        # Fire observability log if enabled
        if self._log_fn:
            self._log_fn(
                model=self.model_id,
                provider=self.provider,
                session_id=self.session_id,
                prompt=user_input,
                response=response,
                latency_ms=latency_ms,
                status=status,
                error=error,
            )

        return response, latency_ms

    def reset(self):
        """
        Clear conversation history and start a fresh session.
        Call this between evaluation prompts or when the user starts a new chat.
        """
        self.history = []
        self.session_id = str(uuid.uuid4())

    def get_history(self) -> list[dict]:
        """Return a copy of the current conversation history."""
        return list(self.history)

    def enable_observability(self):
        """Wire in the observability logger. Call after construction if desired."""
        from observability.logger import log_inference
        self._log_fn = log_inference

    def enable_guardrails(self):
        """Wire in input/output guardrails. Call after construction if desired."""
        from guardrails.safety_layer import check_input, check_output
        self._guardrail_input = check_input
        self._guardrail_output = check_output

    def enable_summary_memory(self, frontier_assistant=None):
        """
        Enable summary memory compression after 10 turns.
        Pass the frontier assistant to use for summarisation.
        """
        self._summary_assistant = frontier_assistant
        self._use_summary_memory = True

    def __repr__(self):
        return f"<{self.__class__.__name__} model={self.model_id} turns={len(self.history)//2}>"
