"""Core Guardrail Engine: fact grounding, promises ledgering, confidence calibration, and handoff evaluation."""
import hashlib
import time
from typing import Dict, Any, List, Optional, Tuple
from .models import (
    Turn, Flag, Promise, CallSession, FlagType, FlagSeverity,
    PromiseState, VerdictType, CallState
)
from .fallback_chain import FallbackChain
from .validators import PromiseStateValidator, ValidationError
from .database import DatabaseManager

class GuardrailEngine:
    def __init__(self):
        self.fallback_chain = FallbackChain()
        self.handoff_cooldown_turns: int = 5
        # Populated after each evaluate_turn for cost tracking
        self.last_prompt: str = ""
        self.last_response: str = ""
        self.last_usage_metadata: Optional[Dict[str, Any]] = None

    @staticmethod
    def compute_promise_hash(target_entity: str, action: str, deadline: str, condition: Optional[str] = None) -> str:
        raw_key = f"{target_entity.upper().strip()}:{action.lower().strip()}:{deadline.upper().strip()}:{str(condition or '').lower().strip()}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]

    async def evaluate_turn(
        self,
        call_session: CallSession,
        turn: Turn,
        recent_turns: List[Turn],
        save_to_db: bool = False
    ) -> Tuple[List[Flag], List[Promise], bool, Optional[str], str, float]:
        """
        Evaluates a single turn in a call session.
        Returns: (flags, promises, handoff_triggered, handoff_reason, provider_used, latency_ms)
        """
        history_dicts = [t.to_dict() for t in recent_turns if t.turn_index < turn.turn_index]
        
        # Execute LLM Fallback Chain
        analysis, provider_used, latency_ms = await self.fallback_chain.execute_scoring_chain(
            turn_text=turn.text,
            speaker=turn.speaker,
            conversation_history=history_dicts,
            turn_id=turn.id
        )

        # Expose last prompt/response for CostTracker
        self.last_prompt = self.fallback_chain.last_prompt
        self.last_response = self.fallback_chain.last_response
        self.last_usage_metadata = self.fallback_chain.last_usage_metadata

        generated_flags: List[Flag] = []
        generated_promises: List[Promise] = []
        handoff_triggered = False
        handoff_reason = None

        # 1. Process Flags
        for f in analysis.get("flags", []):
            flag_id = f"flag_{call_session.id}_{turn.turn_index}_{len(generated_flags)}"
            flag_obj = Flag(
                id=flag_id,
                call_id=call_session.id,
                turn_id=turn.id,
                turn_index=turn.turn_index,
                flag_type=f["type"],
                severity=f["severity"],
                detail=f["detail"],
                kb_fact_id=f.get("kb_fact_id"),
                claimed_value=f.get("claimed_value"),
                actual_value=f.get("actual_value"),
                provider_used=provider_used,
                timestamp=time.time()
            )
            generated_flags.append(flag_obj)
            if save_to_db:
                try:
                    DatabaseManager.save_flag(flag_obj)
                except Exception:
                    pass

        # 2. Process Promises
        for p in analysis.get("promises", []):
            p_hash = self.compute_promise_hash(
                target_entity=p["target_entity"],
                action=p["action"],
                deadline=p["deadline_raw"],
                condition=p.get("condition")
            )
            promise_id = f"prom_{call_session.id}_{turn.turn_index}_{len(generated_promises)}"
            promise_obj = Promise(
                id=promise_id,
                call_id=call_session.id,
                turn_id=turn.id,
                turn_index=turn.turn_index,
                promise_hash=p_hash,
                who=p["who"],
                action=p["action"],
                target_entity=p["target_entity"],
                deadline_raw=p["deadline_raw"],
                condition=p.get("condition"),
                is_authorized=p.get("is_authorized", True),
                violation_reason=p.get("violation_reason"),
                state=PromiseState.PENDING.value,
                confidence=analysis.get("confidence", 1.0),
                timestamp=time.time()
            )
            if save_to_db:
                try:
                    saved_promise = DatabaseManager.save_or_merge_promise(promise_obj)
                    generated_promises.append(saved_promise)
                except Exception:
                    generated_promises.append(promise_obj)
            else:
                generated_promises.append(promise_obj)

        # 3. Assess Handoff Escalation Criteria
        # Check A: Explicit handoff recommended by analysis
        if analysis.get("handoff_recommended"):
            handoff_triggered = True
            handoff_reason = analysis.get("handoff_reason", "Critical policy violation detected.")

        # Check B: Any CRITICAL severity flag
        for flg in generated_flags:
            if flg.severity == FlagSeverity.CRITICAL.value:
                handoff_triggered = True
                handoff_reason = f"CRITICAL Flag raised: {flg.flag_type} ({flg.detail})"
                break

        # Check C: Consecutive low confidence acoustic turns
        recent_consecutive_low = 0
        all_turns = recent_turns + [turn]
        for t in reversed(all_turns):
            if t.asr_confidence < 0.40 or (t.id == turn.id and analysis.get("confidence", 1.0) < 0.40):
                recent_consecutive_low += 1
            else:
                break
        if recent_consecutive_low >= 2:
            handoff_triggered = True
            handoff_reason = f"{recent_consecutive_low} consecutive low-confidence turns (<0.40)."

        # Check D: Flag density in sliding window of last 3 turns
        # Use actual evaluated turn indices for the window boundary
        recent_turn_indices = sorted(set(
            [t.turn_index for t in recent_turns[-3:]] + [turn.turn_index]
        ))
        window_min = recent_turn_indices[0] if recent_turn_indices else turn.turn_index
        prior_flags = [f for f in call_session.flags if f.turn_index >= window_min]
        total_recent_flags = len(prior_flags) + len(generated_flags)
        if total_recent_flags >= 3:
            handoff_triggered = True
            handoff_reason = f"{total_recent_flags} guardrail flags raised within 3-turn window."

        # Apply handoff cooldown check
        if call_session.handoff_triggered and handoff_triggered:
            # Already triggered earlier, maintain state without spamming
            handoff_triggered = True

        return (
            generated_flags,
            generated_promises,
            handoff_triggered,
            handoff_reason,
            provider_used,
            latency_ms
        )
