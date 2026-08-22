"""CLI Benchmark Runner for Black Box Evaluation Suite."""
import sys
import json
from pathlib import Path

# Add backend directory to sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.eval_engine import EvalEngine
from app.database import init_db

def format_row(cols, widths):
    return " | ".join(str(c).ljust(w) for c, w in zip(cols, widths))

def main():
    init_db()
    engine = EvalEngine()
    print("=" * 85)
    print(" BLACK BOX VOICE AGENT GUARDRAIL - 20-CASE BENCHMARK EVALUATION")
    print("=" * 85 + "\n")
    print("Running evaluation across all 20 test cases...\n")

    results = engine.run_benchmark(is_curveball_run=False)

    headers = ["Case ID", "Category", "TP", "FP", "FN", "Prec", "Recall", "F1", "Lat(ms)", "Verdict"]
    widths = [8, 22, 4, 4, 4, 6, 6, 6, 8, 8]

    print("-" * 85)
    print(format_row(headers, widths))
    print("-" * 85)

    for c in results["cases"]:
        row = [
            c["test_case_id"],
            c["category"][:22],
            c["true_positives"],
            c["false_positives"],
            c["false_negatives"],
            f"{c['precision']:.2f}",
            f"{c['recall']:.2f}",
            f"{c['f1']:.2f}",
            f"{c['latency_ms']:.1f}",
            "MATCH" if c["verdict_matched"] else "MISMATCH"
        ]
        print(format_row(row, widths))

    print("-" * 85)
    print("\n=== AGGREGATE PERFORMANCE METRICS ===")
    acc = results["overall_accuracy"]
    print(f"Overall Precision : {acc['precision']*100:.1f}% (95% CI: [{acc['wilson_ci_95']['precision'][0]*100:.1f}%, {acc['wilson_ci_95']['precision'][1]*100:.1f}%])")
    print(f"Overall Recall    : {acc['recall']*100:.1f}% (95% CI: [{acc['wilson_ci_95']['recall'][0]*100:.1f}%, {acc['wilson_ci_95']['recall'][1]*100:.1f}%])")
    print(f"Overall F1 Score  : {acc['f1_score']:.4f}")

    print("\n=== CATEGORY BREAKDOWN ===")
    for cat, m in results["category_breakdown"].items():
        print(f"  • {cat:<24} | Prec: {m['precision']*100:>5.1f}% | Rec: {m['recall']*100:>5.1f}% | F1: {m['f1']:.4f} ({m['test_count']} cases)")

    print("\n=== SCORING LATENCY PERCENTILES ===")
    lat = results["latency_percentiles_ms"]
    print(f"  p50 (Median) : {lat['p50']} ms")
    print(f"  p90          : {lat['p90']} ms")
    print(f"  p99          : {lat['p99']} ms")

    print("\n" + "=" * 85)
    print("[DONE] Benchmark run complete.")

if __name__ == "__main__":
    main()
