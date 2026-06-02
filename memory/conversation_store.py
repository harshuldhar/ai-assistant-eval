"""
conversation_store.py — Summary Memory (Bonus: Memory)

Prevents context window overflow for long conversations by compressing
old turns into a summary after a configurable threshold.

How it works:
  1. While history length < threshold: full history is used (standard behavior)
  2. When history reaches threshold:
     - Summarise the OLDEST turns (all but the last `keep_recent` turns)
     - Replace them with a single system summary message
     - Keep the most recent turns verbatim for continuity

This means the model always has:
  - A summary of everything that happened earlier
  - The last N turns in full detail

Integration:
    Call assistant.enable_summary_memory(frontier_assistant) on any assistant.
    The compression is triggered automatically in chat() when history is long.
    (The BaseAssistant checks self._use_summary_memory in chat())

Note: We use the Frontier assistant to generate summaries because it produces
higher quality summaries than the smaller OSS model.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compress_history(
    history: list[dict],
    summarizer,
    threshold: int = 12,
    keep_recent: int = 4,
) -> list[dict]:
    """
    Compress conversation history when it exceeds the threshold.

    Args:
        history:      Current message history (list of {role, content} dicts)
        summarizer:   An assistant instance whose _call_model() generates the summary
        threshold:    Total messages (user + assistant) before compression triggers
        keep_recent:  How many recent messages to keep verbatim after compression

    Returns:
        Potentially compressed history list.
        If history is short enough, returns it unchanged.
    """
    if len(history) < threshold:
        return history

    # Split into old turns (to summarise) and recent turns (to keep)
    old_turns = history[:-keep_recent]
    recent_turns = history[-keep_recent:]

    if not old_turns:
        return history

    # Build a readable transcript of old turns for the summarizer
    transcript = "\n".join(
        f"{msg['role'].upper()}: {msg['content']}"
        for msg in old_turns
    )

    summary_prompt = (
        "Summarise the following conversation in 3-4 concise sentences. "
        "Focus on key facts, user preferences, and important context that would "
        "help continue the conversation naturally:\n\n" + transcript
    )

    try:
        # Call the summarizer model directly (bypasses history/guardrails)
        summary_messages = [
            {"role": "system", "content": "You are a helpful conversation summarizer."},
            {"role": "user", "content": summary_prompt},
        ]
        summary_text = summarizer._call_model(summary_messages)
    except Exception as e:
        # If summarization fails, don't break the conversation — just trim history
        summary_text = f"[Summary unavailable: {e}. Earlier conversation has been trimmed.]"

    # Replace old turns with a single summary system message
    summary_message = {
        "role": "system",
        "content": f"[Conversation summary from earlier: {summary_text}]",
    }

    return [summary_message] + recent_turns


def maybe_compress(assistant, summarizer=None) -> None:
    """
    Convenience function: compress assistant.history in-place if needed.
    Call this at the start of BaseAssistant.chat() when summary memory is enabled.

    Args:
        assistant:  Any BaseAssistant instance
        summarizer: The assistant to use for generating summaries.
                    Defaults to the frontier assistant stored on the instance.
    """
    if not getattr(assistant, "_use_summary_memory", False):
        return

    _summarizer = summarizer or getattr(assistant, "_summary_assistant", None)
    if _summarizer is None:
        return

    assistant.history = compress_history(assistant.history, _summarizer)
