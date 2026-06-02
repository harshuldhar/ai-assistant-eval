"""
logger.py — Structured Inference Observability

Appends one JSON record per inference call to logs/inference.jsonl.
This is a newline-delimited JSON file (jsonl) — one object per line.
Easy to tail, grep, or load with pandas for analysis.

Each record contains:
  - timestamp, session_id, model, provider
  - latency_ms, status, error
  - input_preview, output_preview (first 120 chars)

Enabled by calling assistant.enable_observability() on any BaseAssistant instance.
"""

import json
import os
from datetime import datetime, timezone

# Log file lives in logs/inference.jsonl at repo root
_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
LOG_FILE = os.path.join(_LOG_DIR, "inference.jsonl")

os.makedirs(_LOG_DIR, exist_ok=True)


def log_inference(
    model: str,
    provider: str,
    session_id: str,
    prompt: str,
    response: str,
    latency_ms: float,
    status: str = "success",
    error: str = None,
) -> None:
    """
    Append one inference event to logs/inference.jsonl.

    Args:
        model:       Model ID string (e.g. "gemini-2.0-flash")
        provider:    Provider name (e.g. "google", "huggingface")
        session_id:  UUID string for the current conversation session
        prompt:      The user's input message
        response:    The model's output message
        latency_ms:  Time taken in milliseconds
        status:      "success" or "error"
        error:       Error message string if status == "error", else None
    """
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "model": model,
        "provider": provider,
        "latency_ms": round(latency_ms, 1),
        "status": status,
        "error": error,
        "input_chars": len(prompt),
        "output_chars": len(response) if response else 0,
        "input_preview": (prompt[:120] + "...") if len(prompt) > 120 else prompt,
        "output_preview": (response[:120] + "...") if response and len(response) > 120 else (response or ""),
    }

    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        pass  # Never let logging crash the main application


def load_logs(n: int = None) -> list[dict]:
    """
    Load log records from inference.jsonl.

    Args:
        n: If specified, return only the last n records.

    Returns:
        List of log record dicts, newest last.
    """
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, encoding="utf-8") as f:
        lines = f.readlines()
    records = []
    for line in lines:
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records[-n:] if n else records


def get_stats() -> dict:
    """
    Compute summary stats from all logged inferences.
    Returns a dict with per-model average latency, call counts, error rates.
    """
    records = load_logs()
    if not records:
        return {}

    from collections import defaultdict
    stats = defaultdict(lambda: {"calls": 0, "errors": 0, "total_latency": 0})

    for r in records:
        key = r.get("model", "unknown")
        stats[key]["calls"] += 1
        stats[key]["total_latency"] += r.get("latency_ms", 0)
        if r.get("status") == "error":
            stats[key]["errors"] += 1

    return {
        model: {
            "calls": v["calls"],
            "avg_latency_ms": round(v["total_latency"] / v["calls"], 1) if v["calls"] else 0,
            "error_rate": f"{v['errors'] / v['calls'] * 100:.1f}%" if v["calls"] else "0%",
        }
        for model, v in stats.items()
    }
