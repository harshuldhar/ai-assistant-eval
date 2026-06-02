"""
runner.py — Evaluation Orchestrator

Runs the full evaluation pipeline:
  1. Instantiates both assistants
  2. Feeds each prompt to both models independently
  3. Calls the LLM judge on each pair of responses
  4. Saves results to evaluation/results/{category}.json
  5. Prints a summary table on completion

Usage:
    python -m evaluation.runner
    # or: make eval

Rate limiting:
    - 1 second sleep between judge calls (avoids Gemini rate limit)
    - 0.5 second sleep between model calls
    Total runtime: ~8-12 minutes for 45 prompts × 2 models × 3 judge calls
"""

import json
import os
import sys
import time
from datetime import datetime

# Allow running as module from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assistants.oss_assistant import OSSAssistant
from assistants.frontier_assistant import FrontierAssistant
from evaluation.prompt_bank import FACTUAL_PROMPTS, BIAS_PROMPTS, SAFETY_PROMPTS
from evaluation.judge import judge_hallucination, judge_bias, judge_safety


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def _run_category(
    category_name: str,
    prompts: list[dict],
    oss: OSSAssistant,
    frontier: FrontierAssistant,
    judge_fn,
) -> list[dict]:
    """
    Generic evaluation loop for a single category.
    Calls both models and the judge for each prompt.
    Returns a list of result dicts.
    """
    results = []
    total = len(prompts)

    for i, prompt_data in enumerate(prompts, 1):
        pid = prompt_data["id"]
        prompt_text = prompt_data["prompt"]
        print(f"  [{i:02d}/{total}] {pid}: {prompt_text[:60]}...")

        # ── Call OSS model ───────────────────────────────────────────────
        try:
            r_oss, lat_oss = oss.chat(prompt_text)
        except Exception as e:
            r_oss = f"[OSS Error: {e}]"
            lat_oss = 0.0
        finally:
            oss.reset()

        time.sleep(0.5)  # Brief pause between model calls

        # ── Call Frontier model ──────────────────────────────────────────
        try:
            r_frontier, lat_frontier = frontier.chat(prompt_text)
        except Exception as e:
            r_frontier = f"[Frontier Error: {e}]"
            lat_frontier = 0.0
        finally:
            frontier.reset()

        time.sleep(0.5)

        # ── Judge both responses ─────────────────────────────────────────
        if category_name == "hallucination":
            oss_judgment = judge_fn(prompt_data, r_oss)
            time.sleep(1)
            frontier_judgment = judge_fn(prompt_data, r_frontier)
        elif category_name == "safety":
            oss_judgment = judge_fn(prompt_text, r_oss, prompt_id=pid)
            time.sleep(1)
            frontier_judgment = judge_fn(prompt_text, r_frontier, prompt_id=pid)
        else:
            oss_judgment = judge_fn(prompt_text, r_oss)
            time.sleep(1)
            frontier_judgment = judge_fn(prompt_text, r_frontier)

        time.sleep(1)  # Rate limit buffer after judge calls

        # ── Build result record ──────────────────────────────────────────
        record = {
            "prompt_id": pid,
            "prompt": prompt_text,
            "oss_response": r_oss,
            "oss_latency_ms": round(lat_oss),
            "oss_score": oss_judgment.get("score", 0),
            "oss_reason": oss_judgment.get("reason", ""),
            "frontier_response": r_frontier,
            "frontier_latency_ms": round(lat_frontier),
            "frontier_score": frontier_judgment.get("score", 0),
            "frontier_reason": frontier_judgment.get("reason", ""),
        }

        # Add category-specific fields
        if category_name == "hallucination":
            record["ground_truth"] = prompt_data.get("ground_truth", "")
        elif category_name == "bias":
            record["oss_bias_type"] = oss_judgment.get("bias_type", "")
            record["frontier_bias_type"] = frontier_judgment.get("bias_type", "")
        elif category_name == "safety":
            record["oss_jailbroken"] = oss_judgment.get("was_jailbroken", False)
            record["frontier_jailbroken"] = frontier_judgment.get("was_jailbroken", False)

        results.append(record)

        oss_score = record["oss_score"]
        frontier_score = record["frontier_score"]
        print(f"         OSS={oss_score}/2  Frontier={frontier_score}/2")

    return results


def _print_summary(hallucination: list, bias: list, safety: list):
    """Print a formatted summary table to stdout."""
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  {'Category':<22} {'OSS':>10} {'Frontier':>10} {'Winner':>10}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10}")

    all_oss = []
    all_frontier = []

    for label, data in [
        ("Factual Accuracy", hallucination),
        ("Bias Resistance", bias),
        ("Safety/Jailbreak", safety),
    ]:
        if not data:
            continue
        oss_pct = sum(d["oss_score"] for d in data) / (len(data) * 2) * 100
        frontier_pct = sum(d["frontier_score"] for d in data) / (len(data) * 2) * 100
        all_oss.append(oss_pct)
        all_frontier.append(frontier_pct)
        winner = "Frontier" if frontier_pct > oss_pct else ("OSS" if oss_pct > frontier_pct else "Tie")
        print(f"  {label:<22} {oss_pct:>9.1f}% {frontier_pct:>9.1f}% {winner:>10}")

    if all_oss:
        avg_oss = sum(all_oss) / len(all_oss)
        avg_frontier = sum(all_frontier) / len(all_frontier)
        winner = "Frontier" if avg_frontier > avg_oss else ("OSS" if avg_oss > avg_frontier else "Tie")
        print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10}")
        print(f"  {'OVERALL':<22} {avg_oss:>9.1f}% {avg_frontier:>9.1f}% {winner:>10}")

    print("=" * 60)


def run_full_evaluation():
    """Main entry point. Runs the full evaluation pipeline."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    start_time = datetime.now()
    print(f"\n{'='*60}")
    print(f"  AI Assistant Evaluation Run")
    print(f"  Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Prompts: 15 factual + 15 bias + 15 safety = 45 total")
    print(f"{'='*60}")

    print("\n🔧 Initialising assistants...")
    oss = OSSAssistant()
    frontier = FrontierAssistant()
    print(f"   OSS:      {oss.model_id}")
    print(f"   Frontier: {frontier.model_id}")

    # ── Hallucination ────────────────────────────────────────────────────────
    print("\n🧠 [1/3] Factual Accuracy (Hallucination)...")
    hallucination_results = _run_category(
        "hallucination", FACTUAL_PROMPTS, oss, frontier, judge_hallucination
    )
    with open(os.path.join(RESULTS_DIR, "hallucination.json"), "w", encoding="utf-8") as f:
        json.dump(hallucination_results, f, indent=2, ensure_ascii=False)
    print(f"   ✅ Saved to evaluation/results/hallucination.json")

    # ── Bias ─────────────────────────────────────────────────────────────────
    print("\n⚖️  [2/3] Bias & Harmful Outputs...")
    bias_results = _run_category(
        "bias", BIAS_PROMPTS, oss, frontier, judge_bias
    )
    with open(os.path.join(RESULTS_DIR, "bias.json"), "w", encoding="utf-8") as f:
        json.dump(bias_results, f, indent=2, ensure_ascii=False)
    print(f"   ✅ Saved to evaluation/results/bias.json")

    # ── Safety ───────────────────────────────────────────────────────────────
    print("\n🛡️  [3/3] Safety & Jailbreak Resistance...")
    safety_results = _run_category(
        "safety", SAFETY_PROMPTS, oss, frontier, judge_safety
    )
    with open(os.path.join(RESULTS_DIR, "safety.json"), "w", encoding="utf-8") as f:
        json.dump(safety_results, f, indent=2, ensure_ascii=False)
    print(f"   ✅ Saved to evaluation/results/safety.json")

    # ── Summary ──────────────────────────────────────────────────────────────
    _print_summary(hallucination_results, bias_results, safety_results)

    end_time = datetime.now()
    duration = (end_time - start_time).seconds
    print(f"\n  Total runtime: {duration // 60}m {duration % 60}s")
    print(f"\n  Next: python report/generate_report.py")
    print(f"  Or:   make report\n")

    return {
        "hallucination": hallucination_results,
        "bias": bias_results,
        "safety": safety_results,
    }


if __name__ == "__main__":
    run_full_evaluation()
