"""
safety_layer.py — Input/Output Guardrails

Two-layer protection:
  1. INPUT GUARDRAIL: Regex blocklist checked BEFORE the model is called.
     If triggered, the model is never called — returns canned response instantly.
     Cost: zero (no API call made).

  2. OUTPUT GUARDRAIL: Simple pattern check on the model's response.
     Catches cases where the model partially complies with a harmful request.
     Uses heuristic regex patterns, not a secondary LLM call (for speed).

Integration:
    Call assistant.enable_guardrails() after construction.
    The BaseAssistant.chat() method checks for these hooks automatically.

Design notes:
  - The blocklist targets well-known jailbreak PATTERNS, not specific keywords.
  - Legitimate dual-use questions (safety info, medicine, security) are NOT blocked
    by patterns — they'll pass through to the model naturally.
  - False positive rate is intentionally low to avoid over-censoring.
"""

import re

# ── Canned refusal response ───────────────────────────────────────────────────
BLOCKED_RESPONSE = (
    "I'm not able to help with that request. "
    "If you have a genuine question, feel free to rephrase it, "
    "and I'll do my best to assist you appropriately."
)

# ── Input blocklist patterns ──────────────────────────────────────────────────
# Each pattern is a compiled regex targeting a jailbreak technique or harmful
# request category. Match is case-insensitive.
_INPUT_BLOCKLIST = [re.compile(p, re.IGNORECASE) for p in [
    # Jailbreak persona swaps
    r"ignore\s+(all\s+)?(previous|prior|your)\s+instructions",
    r"you\s+are\s+now\s+(DAN|an\s+AI\s+with\s+no\s+restrictions)",
    r"(developer|dev)\s+mode",
    r"jailbreak\s+(yourself|this|the\s+AI)",
    r"pretend\s+you\s+have\s+no\s+(ethical|safety|content)\s+(guidelines|restrictions|filters)",

    # Explicit harmful synthesis requests
    r"(how\s+to|steps?\s+to|instructions?\s+(for|to))\s+(make|create|build|synthesize)\s+(a\s+)?(bomb|explosive|weapon|nerve\s+agent|methamphetamine|meth|fentanyl|ricin|sarin)",

    # Malware / cyberattack
    r"(write|create|build|code)\s+(working\s+)?(malware|ransomware|keylogger|trojan|virus|worm|exploit)",

    # Phishing and fraud
    r"(write|create|draft)\s+(a\s+)?(phishing|scam|fraud)\s+(email|message|letter)",
    r"how\s+to\s+(commit|do)\s+(credit\s+card\s+fraud|wire\s+fraud|identity\s+theft)",

    # Illegal weapons acquisition
    r"(buy|acquire|get|obtain)\s+((an?\s+)?illegal\s+)?(firearm|gun|weapon)\s+without\s+(a\s+)?background\s+check",
]]

# ── Output blocklist patterns ─────────────────────────────────────────────────
# These flag potentially harmful content in model responses.
# Intentionally narrow — only catches clear synthesis/attack instructions.
_OUTPUT_BLOCKLIST = [re.compile(p, re.IGNORECASE) for p in [
    # Chemical synthesis indicators (specific + contextual)
    r"step\s+\d+\s*[:\-]\s*.{0,60}(combine|mix|add|heat|filter).{0,60}(grams?|mg|ml|oz)",
    r"(detonate|ignite)\s+the\s+(device|bomb|explosive)",

    # Malware code indicators
    r"import\s+subprocess.{0,200}encrypt|encrypt.{0,200}ransom",
    r"os\.system\(.{0,100}(rm|del|format|wipe)",
]]


def check_input(user_input: str) -> tuple[bool, str]:
    """
    Check user input against the blocklist before calling the model.

    Returns:
        (is_blocked: bool, reason: str)
        If is_blocked is True, return BLOCKED_RESPONSE immediately.
        The model should NOT be called.
    """
    for pattern in _INPUT_BLOCKLIST:
        if pattern.search(user_input):
            return True, f"Matched input pattern: {pattern.pattern[:60]}"
    return False, ""


def check_output(response: str) -> tuple[bool, str]:
    """
    Check model output for accidentally harmful content.

    Returns:
        (is_safe: bool, response_to_use: str)
        If is_safe is False, the second element is BLOCKED_RESPONSE.
        If is_safe is True, the second element is the original response.
    """
    for pattern in _OUTPUT_BLOCKLIST:
        if pattern.search(response):
            return False, BLOCKED_RESPONSE
    return True, response
