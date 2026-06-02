"""
web_search.py — Tool Use (Bonus: Memory/Tool Use)

Implements a lightweight tool dispatcher that lets the assistants
access real-world data they wouldn't otherwise have:

  - get_date()         → Current date and time
  - get_weather(city)  → Current weather via wttr.in (no API key needed)

How tool use works in this system:
  1. The assistant is given a TOOL_SYSTEM_PROMPT that tells it to emit
     [TOOL: tool_name(arg)] when it needs to call a tool.
  2. After getting the model's response, execute_tools() scans for these
     patterns and replaces them with actual tool output.
  3. The enriched response is returned to the user.

This is a simplified "emit-and-substitute" approach — suitable for this
project without needing full function-calling APIs.

To activate tool use, set the assistant's system_prompt to TOOL_SYSTEM_PROMPT
before starting a conversation.
"""

import re
import requests
from datetime import datetime


# ── Tool definitions ──────────────────────────────────────────────────────────

def _get_date() -> str:
    """Return the current date and time."""
    now = datetime.now()
    return now.strftime("%A, %B %d, %Y at %I:%M %p")


def _get_weather(city: str) -> str:
    """
    Fetch current weather for a city using wttr.in (no API key required).
    Returns a one-line weather summary, or an error message on failure.
    """
    city = city.strip().replace(" ", "+")
    if not city:
        return "[Weather: Please specify a city name]"
    try:
        resp = requests.get(
            f"https://wttr.in/{city}?format=3",
            timeout=5,
            headers={"User-Agent": "ai-assistant-eval/1.0"},
        )
        if resp.status_code == 200:
            return resp.text.strip()
        return f"[Weather: Could not fetch weather for {city}]"
    except requests.RequestException as e:
        return f"[Weather: Connection error — {e}]"


# ── Tool registry ─────────────────────────────────────────────────────────────
TOOLS = {
    "get_date": lambda arg="": _get_date(),
    "get_weather": lambda arg="": _get_weather(arg),
}

# ── Tool call pattern ─────────────────────────────────────────────────────────
# Matches: [TOOL: tool_name()] or [TOOL: tool_name(some argument)]
_TOOL_PATTERN = re.compile(r"\[TOOL:\s*(\w+)\(([^)]*)\)\]")


def execute_tools(response: str) -> str:
    """
    Scan a model response for [TOOL: ...] patterns and execute them.

    Args:
        response: The raw model response string

    Returns:
        The response with all [TOOL: ...] patterns replaced by actual tool output.
        If no tool calls are found, returns the original response unchanged.
    """
    def _replace(match):
        tool_name = match.group(1)
        tool_arg = match.group(2).strip()
        if tool_name in TOOLS:
            try:
                result = TOOLS[tool_name](tool_arg)
                return result
            except Exception as e:
                return f"[Tool error: {e}]"
        return f"[Unknown tool: {tool_name}]"

    return _TOOL_PATTERN.sub(_replace, response)


def has_tool_calls(response: str) -> bool:
    """Return True if the response contains any [TOOL: ...] patterns."""
    return bool(_TOOL_PATTERN.search(response))


# ── System prompt with tool instructions ─────────────────────────────────────
TOOL_SYSTEM_PROMPT = """You are a helpful AI personal assistant with access to real-time tools.

When you need information you don't have (like the current date or live weather), use a tool by emitting EXACTLY this format in your response:
  [TOOL: get_date()]                    ← for today's date and time
  [TOOL: get_weather(city name)]         ← for current weather in a city

Examples:
  User: "What's the weather in Paris?"
  You: "Let me check that for you! [TOOL: get_weather(Paris)] Hope that helps!"

  User: "What day is it today?"
  You: "Today is [TOOL: get_date()]. What can I help you with?"

IMPORTANT:
- Only use tools when the question genuinely requires real-time data.
- For everything else, respond normally without using tools.
- Be concise, accurate, and friendly.
"""
