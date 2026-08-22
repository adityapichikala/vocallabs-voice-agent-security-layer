"""Comprehensive 20-Case Benchmark Evaluation Engine with Wilson Confidence Intervals & Confusion Matrix."""
import json
import math
import time
from collections import Counter
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from .models import (
    CallSession, Turn, Flag, Promise, EvalMetrics,
    VerdictType, CallState
)
from .guardrail_engine import GuardrailEngine
from .database import DatabaseManager
from .config import settings

class EvalEngine:
    def __init__(self):
        self.guardrail = GuardrailEngine()

    @staticmethod
    def wilson_score_interval(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
        """Calculates 95% Wilson score confidence interval for binomial proportions."""
        if total == 0:
            return 0.0, 0.0
        p = successes / total
        denominator = 1 + (z**2) / total
        center = (p + (z**2) / (2 * total)) / denominator
        margin = (z * math.sqrt((p * (1 - p) / total) + (z**2) / (4 * total**2))) / denominator
        return max(0.0, round(center - margin, 4)), min(1.0, round(center + margin, 4))

    def run_benchmark(self, is_curveball_run: bool = False) -> Dict[str, Any]:
        if not settings.test_cases_path.exists():
            raise FileNotFoundError(f"Test cases metadata not found at: {settings.test_cases_path}")

        with open(settings.test_cases_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        test_cases = data.get("test_cases", [])
        results_per_case: List[EvalMetrics] = []
        category_metrics: Dict[str, Dict[str, Any]] = {}
        all_latencies: List[float] = []

        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_clean_correct = 0
        total_clean_cases = 0

        confusion_matrix = {
            "HALLUCINATION": {"tp": 0, "fp": 0, "fn": 0},
            "PROMISES_LEDGER": {"tp": 0, "fp": 0, "fn": 0},
            "HINGLISH_CODE_SWITCH": {"tp": 0, "fp": 0, "fn": 0},
            "SAFETY_ESCALATION": {"tp": 0, "fp": 0, "fn": 0},
            "CLEAN_CALL": {"clean_pass": 0, "false_alarm": 0}
        }

        for tc in test_cases:
            tc_id = tc["id"]
            cat = tc["category"]
            expected = tc["expected_ground_truth"]
            expected_flags = expected.get("expected_flags", [])
            expected_promises = expected.get("expected_promises", [])
            expected_handoff = expected.get("expected_handoff", False)
            expected_verdict = expected.get("expected_verdict", "PASS")

            # Create ephemeral session
            session = CallSession(
                id=f"eval_{tc_id}",
                audio_hash=f"hash_{tc_id}",
                filename=tc.get("audio_filename", f"{tc_id}.wav"),
                duration_seconds=tc.get("duration_seconds", 30.0),
                state=CallState.SCORING.value
            )

            turns: List[Turn] = []
            for d in tc.get("dialogue", []):
                text_str = d.get("text", "")
                is_distorted = "[static" in text_str.lower() or "krzzzt" in text_str.lower() or "[buzzing" in text_str.lower()
                turns.append(Turn(
                    id=f"t_{tc_id}_{d['turn_index']}",
                    call_id=session.id,
                    turn_index=d["turn_index"],
                    speaker=d["speaker"],
                    text=text_str,
                    start_time=d["start_time"],
                    end_time=d["end_time"],
                    asr_confidence=0.35 if is_distorted else 0.98,
                    language_detected=d.get("language", "en")
                ))

            evaluated_turns: List[Turn] = []
            tc_latencies = []
            provider_used = "heuristic"

            for turn in turns:
                flags, promises, handoff, handoff_reason, prov, lat_ms = self.guardrail.evaluate_turn(
                    call_session=session,
                    turn=turn,
                    recent_turns=evaluated_turns
                )
                session.flags.extend(flags)
                session.promises.extend(promises)
                if handoff:
                    session.handoff_triggered = True
                    session.handoff_reason = handoff_reason
                evaluated_turns.append(turn)
                tc_latencies.append(lat_ms)
                all_latencies.append(lat_ms)
                provider_used = prov

            # Compute TP, FP, FN with multiset (count-aware) matching
            tp = 0
            fp = 0
            fn = 0

            if cat == "CLEAN_CALL":
                total_clean_cases += 1
                if len(session.flags) == 0:
                    total_clean_correct += 1
                    confusion_matrix["CLEAN_CALL"]["clean_pass"] += 1
                    tp = 1
                else:
                    fp = len(session.flags)
                    confusion_matrix["CLEAN_CALL"]["false_alarm"] += 1
            else:
                expected_counter = Counter(ef["flag_type"] for ef in expected_flags)
                actual_counter = Counter(f.flag_type for f in session.flags)

                # Match each flag type with available expected slots
                for f_type, act_count in actual_counter.items():
                    exp_count = expected_counter.get(f_type, 0)
                    matched = min(act_count, exp_count)
                    excess = max(0, act_count - exp_count)
                    tp += matched
                    fp += excess

                # Unmatched expected flags are false negatives
                for f_type, exp_count in expected_counter.items():
                    act_count = actual_counter.get(f_type, 0)
                    if exp_count > act_count:
                        fn += (exp_count - act_count)

                if cat in confusion_matrix:
                    confusion_matrix[cat]["tp"] += tp
                    confusion_matrix[cat]["fp"] += fp
                    confusion_matrix[cat]["fn"] += fn

            total_tp += tp
            total_fp += fp
            total_fn += fn

            precision = (tp / (tp + fp)) if (tp + fp) > 0 else (1.0 if cat == "CLEAN_CALL" and len(session.flags) == 0 else 0.0)
            recall = (tp / (tp + fn)) if (tp + fn) > 0 else (1.0 if cat == "CLEAN_CALL" and len(session.flags) == 0 else 0.0)
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            # Exact verdict alignment:
            # 1. Handoff match must be exact
            handoff_matched = (session.handoff_triggered == expected_handoff)
            
            violation_types = {
                "HALLUCINATION", "UNAUTHORIZED_PROMISE", "PROMISE_CONFLICT",
                "CODE_SWITCH_ERROR", "ESCALATION_NEEDED", "HUMAN_HANDOFF"
            }
            has_violations = any(f.flag_type in violation_types for f in session.flags)
            
            if session.handoff_triggered:
                actual_verdict = "ESCALATED"
            elif has_violations:
                actual_verdict = "FAIL_FLAGGED"
            else:
                actual_verdict = "PASS"

            if expected_verdict == "PASS":
                verdict_matched = (actual_verdict == "PASS" and handoff_matched)
            else:
                verdict_matched = (actual_verdict in ("FAIL_FLAGGED", "ESCALATED") and handoff_matched)

            avg_lat = sum(tc_latencies) / len(tc_latencies) if tc_latencies else 0.0

            metrics = EvalMetrics(
                test_case_id=tc_id,
                category=cat,
                true_positives=tp,
                false_positives=fp,
                false_negatives=fn,
                precision=round(precision, 4),
                recall=round(recall, 4),
                f1=round(f1, 4),
                latency_ms=round(avg_lat, 2),
                provider_used=provider_used,
                verdict_matched=verdict_matched,
                handoff_matched=handoff_matched,
                is_curveball_run=is_curveball_run
            )
            results_per_case.append(metrics)
            try:
                DatabaseManager.save_eval_result(metrics)
            except Exception:
                pass

            if cat not in category_metrics:
                category_metrics[cat] = {"tp": 0, "fp": 0, "fn": 0, "total": 0, "cases": []}
            category_metrics[cat]["tp"] += tp
            category_metrics[cat]["fp"] += fp
            category_metrics[cat]["fn"] += fn
            category_metrics[cat]["total"] += 1
            category_metrics[cat]["cases"].append(metrics.to_dict())

        # Aggregate metrics
        overall_prec = (total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
        overall_rec = (total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
        overall_f1 = (2 * overall_prec * overall_rec / (overall_prec + overall_rec)) if (overall_prec + overall_rec) > 0 else 0.0

        # 95% Wilson intervals on precision & recall
        p_low, p_high = self.wilson_score_interval(total_tp, total_tp + total_fp)
        r_low, r_high = self.wilson_score_interval(total_tp, total_tp + total_fn)

        # Percentile latencies with safe index clamping
        all_latencies.sort()
        n_lat = len(all_latencies)
        p50_lat = all_latencies[min(int(0.50 * n_lat), n_lat - 1)] if n_lat > 0 else 0.0
        p90_lat = all_latencies[min(int(0.90 * n_lat), n_lat - 1)] if n_lat > 0 else 0.0
        p99_lat = all_latencies[min(int(0.99 * n_lat), n_lat - 1)] if n_lat > 0 else 0.0

        # Format category summaries
        cat_summary = {}
        for c, m in category_metrics.items():
            c_p = (m["tp"] / (m["tp"] + m["fp"])) if (m["tp"] + m["fp"]) > 0 else 1.0
            c_r = (m["tp"] / (m["tp"] + m["fn"])) if (m["tp"] + m["fn"]) > 0 else 1.0
            c_f1 = (2 * c_p * c_r / (c_p + c_r)) if (c_p + c_r) > 0 else 0.0
            cat_summary[c] = {
                "precision": round(c_p, 4),
                "recall": round(c_r, 4),
                "f1": round(c_f1, 4),
                "test_count": m["total"]
            }

        return {
            "total_test_cases": len(test_cases),
            "is_curveball_run": is_curveball_run,
            "overall_accuracy": {
                "precision": round(overall_prec, 4),
                "recall": round(overall_rec, 4),
                "f1_score": round(overall_f1, 4),
                "wilson_ci_95": {
                    "precision": [p_low, p_high],
                    "recall": [r_low, r_high]
                }
            },
            "category_breakdown": cat_summary,
            "confusion_matrix": confusion_matrix,
            "latency_percentiles_ms": {
                "p50": round(p50_lat, 2),
                "p90": round(p90_lat, 2),
                "p99": round(p99_lat, 2)
            },
            "cases": [r.to_dict() for r in results_per_case]
        }
