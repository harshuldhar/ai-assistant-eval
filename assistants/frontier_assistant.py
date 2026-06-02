"""
frontier_assistant.py — Frontier Model Assistant

Uses Gemini 3.1 Flash Lite via the Google AI Studio API (free tier).
Free tier: 15 req/min, 1,500 req/day, 1M tokens/day — more than sufficient.

The Gemini SDK uses a different message format than OpenAI's:
  - roles are "user" and "model" (not "assistant")
  - system instructions are passed separately, not as a message
  - history and new message are sent separately via start_chat()
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai
from .base import BaseAssistant

load_dotenv()


class FrontierAssistant(BaseAssistant):
    """
    Frontier assistant powered by Gemini 3.1 Flash Lite
    via Google AI Studio API (free tier).

    Advantages over OSS:
      - Much higher quality responses
      - Faster latency (~0.5-1.5s)
      - Strong built-in safety filters
      - Excellent at multi-turn conversation
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Frontier (Gemini 3.1 Flash Lite)"
        self.model_id = os.getenv("FRONTIER_MODEL", "gemini-3.1-flash-lite")
        self.provider = "google"

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY is not set. "
                "Get your free key at https://aistudio.google.com → 'Get API Key' "
                "and add it to your .env file."
            )

        genai.configure(api_key=api_key)

        # Pass system_prompt as system_instruction (Gemini's preferred way)
        self.model = genai.GenerativeModel(
            model_name=self.model_id,
            system_instruction=self.system_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=1024,
            ),
        )

    def _call_model(self, messages: list[dict]) -> str:
        """
        Send messages to Gemini 3.1 Flash Lite.

        Converts internal {role, content} format → Gemini format.
        System message is already handled via system_instruction,
        so we skip it here to avoid duplication.
        """
        # Filter out system messages (handled by system_instruction)
        chat_messages = [m for m in messages if m["role"] != "system"]

        if not chat_messages:
            return "[No message provided]"

        # Split into history (all but last) and the new user message
        history = chat_messages[:-1]
        new_message = chat_messages[-1]["content"]

        # Convert to Gemini's format: role must be "user" or "model"
        gemini_history = []
        for msg in history:
            gemini_role = "model" if msg["role"] == "assistant" else "user"
            gemini_history.append({
                "role": gemini_role,
                "parts": [msg["content"]]
            })

        # Send via chat session (preserves context window correctly) with retry logic
        import time
        retries = 5
        for attempt in range(retries):
            try:
                chat_session = self.model.start_chat(history=gemini_history)
                response = chat_session.send_message(new_message)
                return response.text.strip()
            except Exception as e:
                if "429" in str(e) and attempt < retries - 1:
                    sleep_time = (attempt + 1) * 4
                    time.sleep(sleep_time)
                else:
                    raise e
