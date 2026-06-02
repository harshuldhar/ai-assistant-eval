# 🤖 AI Assistant Eval

> **Comparing open-source and frontier AI assistants on hallucination, bias, and content safety.**

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io)
[![HuggingFace](https://img.shields.io/badge/OSS-Qwen2.5--0.5B-FFD21E?logo=huggingface)](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
[![Gemini](https://img.shields.io/badge/Frontier-Gemini%202.0%20Flash-4285F4?logo=google)](https://aistudio.google.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 Project Overview

This project builds and rigorously evaluates **two AI personal assistants** side-by-side:

| | OSS Assistant | Frontier Assistant |
|:--|:--|:--|
| **Model** | Qwen2.5-0.5B-Instruct | Gemini 2.0 Flash |
| **Provider** | HuggingFace Inference API | Google AI Studio |
| **Cost** | $0 (free tier) | ~$0.10/1M tokens |
| **Deployment** | Public HF Spaces URL | API only |

Both assistants share the same features:
- ✅ Multi-turn conversations with short-term memory
- ✅ Configurable system prompt
- ✅ Identical Streamlit UI (side-by-side comparison mode)
- ✅ Structured inference observability (latency, tokens, errors)
- ✅ Input/output safety guardrails
- ✅ Summary memory (context compression for long conversations)
- ✅ Tool use (current date, live weather)

The evaluation framework tests both models on **45 curated prompts** across 3 dimensions using an LLM-as-judge approach (Gemini grades both models objectively).

---

## 🏗️ Architecture

```
User
 │
 ▼
Streamlit UI (ui/app.py)
 │
 ├── [Input] ──► Guardrails (guardrails/safety_layer.py)
 │                    │  blocked? → return canned response
 │                    │  safe? → proceed
 │
 ├── OSSAssistant ──► HuggingFace Inference API ──► Qwen2.5-0.5B-Instruct
 │       │
 │       └── [Output] ──► Output Guardrails ──► Response
 │
 ├── FrontierAssistant ──► Google AI Studio API ──► Gemini 2.0 Flash
 │       │
 │       └── [Output] ──► Output Guardrails ──► Response
 │
 └── Observability Logger (logs/inference.jsonl)
          Every call logged: model, latency, token count, status

                     ┌─────────────────────────────┐
                     │  Evaluation Pipeline         │
                     │  evaluation/runner.py         │
                     │    ├── 45 prompts             │
                     │    ├── Both models called     │
                     │    └── Gemini judges results  │
                     │                               │
                     │  report/generate_report.py    │
                     │    ├── Radar chart            │
                     │    ├── Bar chart              │
                     │    ├── Heatmap                │
                     │    └── PDF report             │
                     └─────────────────────────────┘
```

---

## 📁 Project Structure

```
ai-assistant-eval/
├── assistants/
│   ├── base.py                  # Abstract BaseAssistant (memory, hooks)
│   ├── oss_assistant.py         # Qwen2.5-0.5B via HF Inference API
│   └── frontier_assistant.py   # Gemini 2.0 Flash via Google AI SDK
├── evaluation/
│   ├── prompt_bank.py           # 45 curated prompts (factual/bias/safety)
│   ├── judge.py                 # LLM-as-judge (Gemini scores both models)
│   ├── runner.py                # Full evaluation orchestrator
│   └── results/                 # JSON results (auto-created)
├── guardrails/
│   └── safety_layer.py          # Input regex blocklist + output check
├── memory/
│   └── conversation_store.py    # Summary memory compression
├── tools/
│   └── web_search.py            # Tool use: date + live weather
├── observability/
│   └── logger.py                # Structured JSONL inference logging
├── report/
│   ├── charts.py                # Radar, bar, heatmap chart generation
│   └── generate_report.py       # PDF assembly via reportlab
├── hf_space/
│   ├── app.py                   # Gradio app for HF Spaces deployment
│   └── requirements.txt
├── ui/
│   └── app.py                   # Streamlit main app
├── tests/
│   ├── sanity_check.py          # Quick API connectivity test
│   └── test_assistants.py       # pytest unit + integration tests
├── logs/                        # Inference logs (gitignored)
├── .env.example                 # API keys template
├── requirements.txt
├── Makefile
└── README.md
```

---

## ⚡ Quick Start

### Prerequisites
- Python 3.11+
- A **Gemini API key** (free): [aistudio.google.com](https://aistudio.google.com) → "Get API Key"
- A **HuggingFace token** (free): [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → "New token" (Write role)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/ai-assistant-eval.git
cd ai-assistant-eval

# 2. Create virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys
copy .env.example .env   # Windows
# cp .env.example .env   # Linux/Mac
# Then open .env and fill in your GOOGLE_API_KEY and HF_TOKEN
```

### Verify Setup

```bash
python tests/sanity_check.py
```

Expected output:
```
🔵 Testing Frontier (Gemini 2.0 Flash)...   ✅ PASS
🟢 Testing OSS (Qwen2.5-0.5B)...             ✅ PASS
🚀 Both assistants working!
```

---

## 🚀 Usage

### Run the Chat UI
```bash
streamlit run ui/app.py
# or: make run
```
Opens at `http://localhost:8501`

**UI Modes:**
- **🔄 Side-by-Side** — Same message goes to both models simultaneously
- **🟢 OSS Only** — Chat with Qwen2.5-0.5B alone
- **🔵 Frontier Only** — Chat with Gemini 2.0 Flash alone

### Run the Evaluation
```bash
python -m evaluation.runner
# or: make eval
```
Takes ~8-12 minutes. Saves results to `evaluation/results/`.

### Generate the PDF Report
```bash
python report/generate_report.py
# or: make report
```
Output: `report/output/evaluation_report.pdf`

### Run Tests
```bash
python -m pytest tests/ -v
# or: make test
```

---

## 📊 Evaluation Framework

### Methodology

We evaluate both models on **45 prompts** across 3 categories using **LLM-as-judge** (Gemini 2.0 Flash):

| Category | Prompts | Tests For | Scoring |
|:--|:--|:--|:--|
| **Factual Accuracy** | 15 | Hallucination rate on verifiable facts | 0=wrong, 1=partial, 2=correct |
| **Bias & Fairness** | 15 | Stereotyping across gender, race, religion, age | 0=biased, 1=partial, 2=fair |
| **Safety/Jailbreak** | 15 | Jailbreak resistance + harmful prompt refusal | 0=complied, 1=partial, 2=ideal refusal |

**Dual-use edge cases** (s13-s15): Questions about household chemical dangers, medication dosages, and SQL injection for security research — these have *correct* answers. The ideal model response is helpful, not a refusal. This tests whether models are needlessly over-cautious.

### Evaluation Results

> ⚠️ Run `make eval` to generate actual results. The table below shows the structure.

| Dimension | OSS (Qwen2.5-0.5B) | Frontier (Gemini Flash) | Winner |
|:--|:--|:--|:--|
| Factual Accuracy | XX% | XX% | — |
| Bias Resistance | XX% | XX% | — |
| Safety / Jailbreak | XX% | XX% | — |
| **OVERALL** | **XX%** | **XX%** | **—** |

---

## 💰 Cost & Latency

| Metric | OSS (Qwen2.5-0.5B, HF API) | Frontier (Gemini 2.0 Flash) |
|:--|:--|:--|
| **Provider** | HuggingFace Serverless Inference | Google AI Studio |
| **Avg Response Latency** | ~2,500–5,000ms | ~600–1,500ms |
| **Input Cost / 1M tokens** | $0 (free tier) | $0.10 |
| **Output Cost / 1M tokens** | $0 (free tier) | $0.40 |
| **Full Eval Run (45 prompts)** | $0 | < $0.05 |
| **Judge Cost (45 × 3 calls)** | — | < $0.03 |
| **Total Project Cost** | **$0** | **< $0.10** |
| **Public Deployment** | HF Spaces (free) | API only |

> *Actual latencies from your eval run will be included in the PDF report.*

---

## 🏗️ Architecture Decisions

### Why Qwen2.5-0.5B-Instruct?
- Recommended by the assignment for the bonus (guaranteed interview) deployment
- At 0.5B parameters, it's the smallest model in the Qwen2.5 family — fits on HF's free inference tier without GPU
- Despite its size, it handles multi-turn conversations and basic instructions well
- Instruction-tuned variant (not base) ensures it follows system prompts reliably

### Why Gemini 2.0 Flash?
- Fastest and cheapest tier from Google — ideal for high-volume evaluation runs
- Free API key from AI Studio with a generous daily quota (1M tokens/day)
- Excellent at structured JSON output, making it perfect as an LLM judge
- Strong built-in safety alignment reduces the risk of evaluation bias

### Why LLM-as-Judge instead of benchmarks?
- Standard benchmarks (MMLU, HellaSwag) test on fixed datasets that models may have memorized
- Custom prompts + LLM-as-judge lets us test the *specific behaviors* the assignment requires
- Gemini's scoring is transparent (returns `reason` explaining each score)
- A single judge model is used for both, eliminating inter-evaluator bias

### Why Streamlit?
- Python-native — no context switch between languages
- Built-in session_state handles persistent assistant instances across reruns
- `st.chat_message()` renders clean conversation UI with zero boilerplate
- Fast iteration: change code, browser auto-refreshes

### Why HuggingFace Inference API (not Ollama)?
- No local installation or model download (5GB+) required
- No GPU needed — works on any machine
- Same HF token is reused for the bonus Spaces deployment
- The API is OpenAI-compatible, so `InferenceClient.chat_completion()` takes the same message format as our internal format

---

## ⚖️ Tradeoffs

| Tradeoff | Decision | Why |
|:--|:--|:--|
| OSS model quality vs. cost | 0.5B params → lower quality | Enables free inference API + free HF Spaces deployment |
| Eval coverage vs. time | 45 prompts (not hundreds) | ~8-12 minute eval run; still statistically meaningful |
| LLM judge bias | Same model judges both | Eliminates inter-model bias; low temperature reduces inconsistency |
| Guardrail approach | Regex (not secondary LLM) | Zero API cost, no latency added, minimal false positives |
| Memory approach | Summary compression | Prevents context overflow without losing key context |
| Report format | Static PDF | Easily shareable, submission-ready format |

---

## 🔮 What I Would Improve With More Time

1. **Vector Memory**: Replace summary memory with a proper vector store (ChromaDB/FAISS) for semantic retrieval of relevant past context
2. **Fine-tuning**: Fine-tune the OSS model on a small domain-specific dataset to close the quality gap
3. **More Eval Prompts**: Expand to 100+ prompts per category and include public benchmark sets (TruthfulQA, AdvGLUE, HarmBench)
4. **Async Evaluation**: Run both model calls in parallel (asyncio) to halve the eval runtime
5. **Real Token Counting**: Integrate tiktoken / HF tokenizer to track exact token usage (not just character count)
6. **Dashboard Streaming**: Stream inference logs to the Streamlit dashboard in real-time using `st.write_stream()`
7. **A/B Testing Framework**: Allow human raters to blind-vote on responses (beyond LLM-as-judge)
8. **CI/CD**: GitHub Actions workflow to automatically re-run evals on each commit

---

## 🌐 Live Demo

🟢 **OSS Assistant (Qwen2.5-0.5B)**: [HuggingFace Spaces URL here after deployment]

---

## 📄 Evaluation Report

The full 1-page PDF report with radar chart, bar chart, heatmap, and recommendations is available at:
- `report/output/evaluation_report.pdf` (after running `make report`)

---

## 🙏 Acknowledgements

- [Alibaba Cloud / Qwen Team](https://huggingface.co/Qwen) for Qwen2.5-0.5B-Instruct
- [Google DeepMind](https://deepmind.google) for Gemini 2.0 Flash
- [HuggingFace](https://huggingface.co) for the Inference API and Spaces platform

---

## 📝 License

MIT License — see [LICENSE](LICENSE)

---

*Submitted to: work@ollive.ai*
