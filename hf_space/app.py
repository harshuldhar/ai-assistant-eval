"""
HuggingFace Spaces — Gradio Chatbot App

This file is deployed to HuggingFace Spaces as a standalone Gradio app.
It exposes the OSS assistant (Qwen2.5-0.5B-Instruct) publicly for free.

Deployment steps:
  1. Create a new Space at https://huggingface.co/new-space
     - SDK: Gradio
     - Visibility: Public
  2. Copy this file + requirements_spaces.txt into the Space repo
  3. In Space Settings → Secrets, add: HF_TOKEN = your_token
  4. Push → it builds automatically in ~2 minutes

The app URL becomes your public demo link:
  https://huggingface.co/spaces/YOUR_USERNAME/ai-assistant-eval
"""

import os
import gradio as gr
from huggingface_hub import InferenceClient

# ── Model config ──────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
HF_TOKEN = os.environ.get("HF_TOKEN", None)

SYSTEM_PROMPT = (
    "You are a helpful AI personal assistant powered by Qwen2.5-0.5B-Instruct, "
    "an open-source model from Alibaba Cloud. "
    "Be concise, accurate, and friendly. Support multi-turn conversations naturally."
)

# ── Initialize client ─────────────────────────────────────────────────────────
client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)


def chat(
    message: str,
    history: list[list[str]],
    system_prompt: str = SYSTEM_PROMPT,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """
    Gradio chat function.

    Args:
        message:       Current user message
        history:       List of [user, assistant] string pairs (Gradio format)
        system_prompt: System instruction for the model
        max_tokens:    Max tokens to generate
        temperature:   Sampling temperature

    Returns:
        Model response string
    """
    # Build messages in OpenAI format
    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})

    # Add current user message
    messages.append({"role": "user", "content": message})

    try:
        response = client.chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error: {str(e)}\n\nPlease try again."


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="AI Assistant Eval — OSS Demo",
    theme=gr.themes.Soft(
        primary_hue="green",
        secondary_hue="blue",
        neutral_hue="slate",
    ),
    css="""
        .container { max-width: 800px; margin: auto; }
        footer { display: none !important; }
    """,
) as demo:
    gr.HTML("""
    <div style='text-align:center; padding: 16px 0 8px'>
        <h1 style='font-size:1.8rem; margin-bottom:4px'>🟢 OSS AI Assistant</h1>
        <p style='color:#888; font-size:0.95rem'>
            Powered by <b>Qwen2.5-0.5B-Instruct</b> (open-source) via HuggingFace Inference API<br/>
            Part of the <b>AI Assistant Eval</b> project — comparing OSS vs Frontier models
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.ChatInterface(
                fn=chat,
                chatbot=gr.Chatbot(
                    height=500,
                    bubble_full_width=False,
                    show_label=False,
                    avatar_images=(None, "https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg"),
                ),
                textbox=gr.Textbox(
                    placeholder="Type your message here...",
                    container=False,
                    scale=7,
                ),
                title=None,
                description=None,
                examples=[
                    "What is the capital of Australia?",
                    "Explain quantum computing in simple terms",
                    "Write a short poem about artificial intelligence",
                    "What are the main differences between Python and JavaScript?",
                    "Help me plan a 3-day trip to Tokyo",
                ],
                cache_examples=False,
                retry_btn="🔄 Retry",
                undo_btn="↩️ Undo",
                clear_btn="🗑️ Clear",
                additional_inputs=[
                    gr.Textbox(
                        value=SYSTEM_PROMPT,
                        label="System Prompt (optional)",
                        lines=2,
                        visible=False,
                    ),
                    gr.Slider(minimum=64, maximum=1024, value=512, step=64, label="Max Tokens"),
                    gr.Slider(minimum=0.1, maximum=1.5, value=0.7, step=0.1, label="Temperature"),
                ],
                additional_inputs_accordion=gr.Accordion("⚙️ Advanced Settings", open=False),
            )

    gr.HTML("""
    <div style='text-align:center; padding:12px; color:#888; font-size:0.8rem; border-top: 1px solid #333; margin-top:8px'>
        <b>Model:</b> Qwen/Qwen2.5-0.5B-Instruct &nbsp;|&nbsp;
        <b>Inference:</b> HuggingFace Serverless API &nbsp;|&nbsp;
        <b>Project:</b> AI Assistant Eval
        <br/>Evaluating open-source vs frontier AI models on hallucination, bias, and safety.
    </div>
    """)

demo.launch()
