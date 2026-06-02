"""
oss_assistant.py — Open-Source Assistant

Uses Qwen2.5-7B-Instruct via HuggingFace Serverless Inference API.
No local model download or GPU required — just a HF token.

The InferenceClient.chat_completion() method is OpenAI-compatible,
so it accepts the same {role, content} message format as base.py produces.
"""

import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from .base import BaseAssistant

load_dotenv()


class OSSAssistant(BaseAssistant):
    """
    Open-Source assistant powered by Qwen/Qwen2.5-7B-Instruct
    via HuggingFace Serverless Inference API (free tier).

    Limitations:
      - Rate limited on free HF tier (~10-30 req/min)
      - Smaller model, so quality/accuracy lower than frontier
      - Latency ~2-5s per response (model cold starts may be longer)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "OSS (Qwen2.5-7B-Instruct)"
        self.model_id = os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
        self.provider = "huggingface"
        self.max_tokens = int(os.getenv("OSS_MAX_TOKENS", "512"))
        self.temperature = float(os.getenv("OSS_TEMPERATURE", "0.7"))

        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise EnvironmentError(
                "HF_TOKEN is not set. "
                "Get your token at https://huggingface.co/settings/tokens "
                "and add it to your .env file."
            )

        self.client = InferenceClient(
            model=self.model_id,
            token=hf_token,
        )

    def _call_model(self, messages: list[dict]) -> str:
        """
        Send messages to Qwen via HF Inference API.
        Uses chat_completion (OpenAI-compatible endpoint).
        """
        response = self.client.chat_completion(
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content.strip()
