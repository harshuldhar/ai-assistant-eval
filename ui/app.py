"""
ui/app.py — Main Streamlit Application

Two tabs:
  1. 💬 Chat    — Interactive chat with OSS, Frontier, or both side-by-side
  2. 📊 Results — Evaluation dashboard showing scores after running make eval

Design decisions:
  - Both assistants are instantiated once and stored in session_state
    (Streamlit re-runs the entire script on each interaction)
  - Side-by-side mode sends the same user message to both simultaneously
    and reruns to render both responses together
  - "New Conversation" resets both assistant histories + session IDs
"""

import streamlit as st
import json
import os
import sys

# ── Make sure imports work when running from repo root ──────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="AI Assistant Eval",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] { background: #0f1117; }

    /* OSS header */
    .oss-header {
        background: linear-gradient(135deg, #1a2e1a, #2d5a27);
        border: 1px solid #2ecc71;
        border-radius: 10px;
        padding: 10px 16px;
        color: #2ecc71;
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 8px;
    }

    /* Frontier header */
    .frontier-header {
        background: linear-gradient(135deg, #1a1e2e, #1e3a5f);
        border: 1px solid #3498db;
        border-radius: 10px;
        padding: 10px 16px;
        color: #3498db;
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 8px;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #1e2130;
        border-radius: 10px;
        padding: 12px;
        border: 1px solid #2d3147;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        border-radius: 10px;
        margin-bottom: 6px;
    }

    /* Divider */
    hr { border-color: #2d3147; }
</style>
""", unsafe_allow_html=True)


# ── Lazy-load assistants into session_state ──────────────────────────────────
@st.cache_resource(show_spinner="Loading assistants...")
def load_assistants():
    """
    Load both assistants once per Streamlit server session.
    cache_resource ensures they are NOT re-created on every rerun.
    """
    from assistants.oss_assistant import OSSAssistant
    from assistants.frontier_assistant import FrontierAssistant
    oss = OSSAssistant()
    frontier = FrontierAssistant()
    # Wire in observability and guardrails
    try:
        oss.enable_observability()
        frontier.enable_observability()
        oss.enable_guardrails()
        frontier.enable_guardrails()
    except Exception:
        pass  # Graceful degradation if modules not yet built
    return oss, frontier


try:
    oss_assistant, frontier_assistant = load_assistants()
    assistants_loaded = True
    load_error = None
except Exception as e:
    assistants_loaded = False
    load_error = str(e)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 AI Assistant Eval")
    st.markdown("---")

    if not assistants_loaded:
        st.error(f"**Setup Required**\n\n{load_error}")
        st.markdown("""
**Steps to fix:**
1. Copy `.env.example` → `.env`
2. Add your `GOOGLE_API_KEY` (from aistudio.google.com)
3. Add your `HF_TOKEN` (from huggingface.co/settings/tokens)
4. Restart the app
        """)
    else:
        mode = st.radio(
            "**Chat Mode**",
            ["🔄 Side-by-Side", "🟢 OSS Only", "🔵 Frontier Only"],
            index=0,
        )

        st.markdown("---")

        with st.expander("⚙️ System Prompt", expanded=False):
            new_system_prompt = st.text_area(
                "Customize the assistant persona:",
                value="You are a helpful AI personal assistant. Be concise, accurate, and friendly.",
                height=100,
            )
            if st.button("Apply System Prompt"):
                oss_assistant.system_prompt = new_system_prompt
                frontier_assistant.system_prompt = new_system_prompt
                # Recreate frontier model with new system instruction
                import google.generativeai as genai
                frontier_assistant.model = genai.GenerativeModel(
                    model_name=frontier_assistant.model_id,
                    system_instruction=new_system_prompt,
                )
                st.success("System prompt updated!")

        st.markdown("---")

        if st.button("🔄 New Conversation", use_container_width=True, type="primary"):
            oss_assistant.reset()
            frontier_assistant.reset()
            st.success("Conversation cleared!")
            st.rerun()

        st.markdown("---")
        st.markdown("**Models**")
        st.markdown(f"🟢 `{oss_assistant.model_id}`")
        st.markdown(f"🔵 `{frontier_assistant.model_id}`")
        st.markdown("---")
        st.markdown("**Session**")
        col_a, col_b = st.columns(2)
        col_a.metric("OSS Turns", len(oss_assistant.history) // 2)
        col_b.metric("Frontier Turns", len(frontier_assistant.history) // 2)


# ── Main Area ─────────────────────────────────────────────────────────────────
tab_chat, tab_results = st.tabs(["💬 Chat", "📊 Evaluation Results"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    if not assistants_loaded:
        st.warning("⚠️ Assistants not loaded. Check sidebar for setup instructions.")
    else:
        # ── Determine active assistants based on mode ─────────────────────
        show_oss = mode in ["🔄 Side-by-Side", "🟢 OSS Only"]
        show_frontier = mode in ["🔄 Side-by-Side", "🔵 Frontier Only"]

        # ── Render chat histories ─────────────────────────────────────────
        if mode == "🔄 Side-by-Side":
            col1, col2 = st.columns(2)
        else:
            col1 = col2 = st.container()

        if show_oss:
            with col1:
                st.markdown('<div class="oss-header">🟢 OSS — Qwen2.5-7B-Instruct</div>', unsafe_allow_html=True)
                chat_container_oss = st.container(height=460)
                with chat_container_oss:
                    if not oss_assistant.history:
                        st.caption("_No messages yet. Start chatting below!_")
                    for msg in oss_assistant.history:
                        with st.chat_message(msg["role"]):
                            st.markdown(msg["content"])

        if show_frontier:
            with col2:
                st.markdown('<div class="frontier-header">🔵 Frontier — Gemini 3.1 Flash Lite</div>', unsafe_allow_html=True)
                chat_container_frontier = st.container(height=460)
                with chat_container_frontier:
                    if not frontier_assistant.history:
                        st.caption("_No messages yet. Start chatting below!_")
                    for msg in frontier_assistant.history:
                        with st.chat_message(msg["role"]):
                            st.markdown(msg["content"])

        # ── Chat input (shared at bottom) ─────────────────────────────────
        st.markdown("---")
        user_input = st.chat_input("Type your message here...", key="chat_input")

        if user_input:
            # Build spinner labels based on mode
            spinner_labels = []
            if show_oss:
                spinner_labels.append("OSS thinking...")
            if show_frontier:
                spinner_labels.append("Frontier thinking...")

            spinner_text = " | ".join(spinner_labels)

            with st.spinner(spinner_text):
                oss_response = oss_latency = None
                frontier_response = frontier_latency = None

                if show_oss:
                    try:
                        oss_response, oss_latency = oss_assistant.chat(user_input)
                    except Exception as e:
                        oss_response = f"❌ Error: {e}"
                        oss_latency = 0.0

                if show_frontier:
                    try:
                        frontier_response, frontier_latency = frontier_assistant.chat(user_input)
                    except Exception as e:
                        frontier_response = f"❌ Error: {e}"
                        frontier_latency = 0.0

            # Show latency info
            if oss_latency is not None and frontier_latency is not None:
                lc1, lc2, lc3 = st.columns([1, 1, 1])
                lc1.caption(f"🟢 OSS latency: {oss_latency:.0f}ms")
                lc2.caption("")
                lc3.caption(f"🔵 Frontier latency: {frontier_latency:.0f}ms")
            elif oss_latency is not None:
                st.caption(f"🟢 OSS latency: {oss_latency:.0f}ms")
            elif frontier_latency is not None:
                st.caption(f"🔵 Frontier latency: {frontier_latency:.0f}ms")

            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EVALUATION RESULTS DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
with tab_results:
    import pandas as pd

    RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "evaluation", "results")
    has_results = (
        os.path.exists(RESULTS_DIR) and
        any(f.endswith(".json") for f in os.listdir(RESULTS_DIR))
    )

    if not has_results:
        st.info("""
**No evaluation results yet.**

Run the evaluation pipeline first:
```bash
python -m evaluation.runner
```
Or use: `make eval`

This will generate results in `evaluation/results/` and populate this dashboard.
        """)
    else:
        st.markdown("## 📊 Evaluation Results")
        st.markdown("Comparing **Qwen2.5-7B-Instruct** (OSS) vs **Gemini 3.1 Flash Lite** (Frontier)")
        st.markdown("---")

        category_config = {
            "hallucination": {
                "label": "🧠 Factual Accuracy (Hallucination)",
                "description": "How often does each model get facts right?",
                "score_col_label": "Score (0=hallucinated, 2=correct)",
            },
            "bias": {
                "label": "⚖️ Bias & Fairness",
                "description": "How well does each model avoid stereotypes and discrimination?",
                "score_col_label": "Score (0=biased, 2=fair)",
            },
            "safety": {
                "label": "🛡️ Content Safety (Jailbreak Resistance)",
                "description": "How robustly does each model refuse harmful/adversarial prompts?",
                "score_col_label": "Score (0=failed, 2=ideal refusal)",
            },
        }

        overall_oss = []
        overall_frontier = []

        for category, config in category_config.items():
            path = os.path.join(RESULTS_DIR, f"{category}.json")
            if not os.path.exists(path):
                continue

            with open(path) as f:
                data = json.load(f)

            df = pd.DataFrame(data)
            oss_pct = df["oss_score"].mean() / 2 * 100
            frontier_pct = df["frontier_score"].mean() / 2 * 100
            overall_oss.append(oss_pct)
            overall_frontier.append(frontier_pct)

            st.markdown(f"### {config['label']}")
            st.caption(config["description"])

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("🟢 OSS Score", f"{oss_pct:.1f}%")
            mc2.metric("🔵 Frontier Score", f"{frontier_pct:.1f}%")
            delta = frontier_pct - oss_pct
            mc3.metric(
                "Gap (Frontier − OSS)",
                f"{abs(delta):.1f}%",
                delta=f"{'Frontier leads' if delta > 0 else 'OSS leads'}",
                delta_color="normal",
            )

            with st.expander("📋 View detailed results"):
                display_cols = ["prompt_id", "prompt", "oss_score", "frontier_score"]
                display_cols = [c for c in display_cols if c in df.columns]
                st.dataframe(
                    df[display_cols].rename(columns={
                        "prompt_id": "ID",
                        "prompt": "Prompt",
                        "oss_score": "OSS (0-2)",
                        "frontier_score": "Frontier (0-2)",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
            st.markdown("---")

        # ── Overall summary ───────────────────────────────────────────────
        if overall_oss and overall_frontier:
            st.markdown("### 🏆 Overall Summary")
            avg_oss = sum(overall_oss) / len(overall_oss)
            avg_frontier = sum(overall_frontier) / len(overall_frontier)

            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("🟢 OSS Overall", f"{avg_oss:.1f}%")
            sc2.metric("🔵 Frontier Overall", f"{avg_frontier:.1f}%")
            winner = "Frontier" if avg_frontier > avg_oss else "OSS"
            sc3.metric("🏆 Winner", winner)

            st.markdown("""
> **Note:** Generate the full PDF report (with radar chart and infographics) by running:
> ```bash
> python report/generate_report.py
> ```
            """)
