"""Pipeline state machine, timeout guards, and partial failure recovery manager."""
import asyncio
import time
from typing import Dict, Any, List, Optional, AsyncGenerator
from .models import (
    CallSession, Turn, Flag, Promise, CallState,
    VerdictType, PipelineProgress
)
from .validators import PipelineStateValidator, ValidationError
from .database import DatabaseManager
from .asr_pipeline import ASRPipeline
from .guardrail_engine import GuardrailEngine
from .config import settings

class PipelineManager:
    def __init__(self):
        self.asr = ASRPipeline()
        self.guardrail = GuardrailEngine()
        self._active_pipelines: Dict[str, bool] = {}

    def transition_state(self, session: CallSession, target_state: str, error_msg: Optional[str] = None):
        PipelineStateValidator.validate_transition(session.state, target_state)
        session.state = target_state
        if error_msg:
            session.error_message = error_msg
        DatabaseManager.update_call_state(session.id, target_state, error_message=error_msg)

    async def execute_call_pipeline(
        self,
        session: CallSession,
        audio_path_or_name: str,
        speed_multiplier: float = 2.0
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Executes end-to-end pipeline and yields real-time SSE events for frontend streaming.
        Event Types:
        - state_change: stage update
        - progress: current turn / total turns
        - turn_transcribed: new ASR turn
        - turn_evaluated: guardrail results, flags, promises
        - handoff_alert: emergency human handoff trigger
        - complete: final call summary
        - error: failure details
        """
        if self._active_pipelines.get(session.id):
            yield {
                "event": "error",
                "seq_id": 1,
                "data": {"error": f"Call {session.id} is already actively processing."}
            }
            return

        self._active_pipelines[session.id] = True
        seq_id = 0

        def make_event(event_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
            nonlocal seq_id
            seq_id += 1
            return {
                "event": event_type,
                "seq_id": seq_id,
                "timestamp": time.time(),
                "data": data
            }

        try:
            # Stage 1: TRANSCRIBING
            self.transition_state(session, CallState.TRANSCRIBING.value)
            yield make_event("state_change", {
                "call_id": session.id,
                "state": CallState.TRANSCRIBING.value,
                "message": "Initializing acoustic ingestion and transcription..."
            })

            # Retrieve/Transcribe turns
            turns = self.asr.get_simulated_turns_for_call(audio_path_or_name, session.id)
            if not turns:
                turns = self.asr.get_simulated_turns_for_call("TC-01", session.id)
            
            session.turns = turns
            DatabaseManager.save_turns(session.id, turns)

            # Stage 2: SCORING
            self.transition_state(session, CallState.SCORING.value)
            yield make_event("state_change", {
                "call_id": session.id,
                "state": CallState.SCORING.value,
                "message": f"Transcribed {len(turns)} turns. Commencing real-time guardrail scoring..."
            })

            total_turns = len(turns)
            evaluated_turns: List[Turn] = []

            for idx, turn in enumerate(turns):
                # Emit turn transcribed event (simulating real-time speech)
                delta_sleep = max(0.4, (turn.end_time - turn.start_time) / speed_multiplier)
                await asyncio.sleep(min(delta_sleep, 1.2))

                yield make_event("turn_transcribed", {
                    "call_id": session.id,
                    "turn": turn.to_dict(),
                    "progress_pct": round(((idx + 0.5) / total_turns) * 100, 1)
                })

                # Evaluate turn against guardrails
                try:
                    flags, promises, handoff, handoff_reason, provider, lat_ms = self.guardrail.evaluate_turn(
                        call_session=session,
                        turn=turn,
                        recent_turns=evaluated_turns,
                        save_to_db=True
                    )

                    session.flags.extend(flags)
                    session.promises.extend(promises)

                    # Capture prior state BEFORE mutating so handoff_alert can fire
                    was_already_handoff = session.handoff_triggered
                    if handoff:
                        session.handoff_triggered = True
                        session.handoff_reason = handoff_reason

                    evaluated_turns.append(turn)

                    # Emit evaluation result
                    yield make_event("turn_evaluated", {
                        "call_id": session.id,
                        "turn_index": turn.turn_index,
                        "turn_id": turn.id,
                        "flags": [f.to_dict() for f in flags],
                        "promises": [p.to_dict() for p in promises],
                        "provider_used": provider,
                        "latency_ms": round(lat_ms, 1),
                        "handoff_triggered": handoff,
                        "handoff_reason": handoff_reason,
                        "progress_pct": round(((idx + 1) / total_turns) * 100, 1)
                    })

                    # Emit handoff alert on FIRST trigger only
                    if handoff and not was_already_handoff:
                        yield make_event("handoff_alert", {
                            "call_id": session.id,
                            "turn_index": turn.turn_index,
                            "reason": handoff_reason,
                            "action": "TRANSFER_TO_HUMAN_SUPERVISOR"
                        })

                except Exception as eval_err:
                    # Partial failure recovery: record error and continue next turn
                    yield make_event("turn_error", {
                        "call_id": session.id,
                        "turn_index": turn.turn_index,
                        "error": str(eval_err)
                    })

            # Stage 3: COMPLETE
            # Determine overall verdict:
            # ESCALATED = handoff triggered or critical violation
            # FAIL_FLAGGED = policy violation / hallucination flags present
            # PASS = clean call or authorized commitments without violations
            has_critical = any(f.severity == "CRITICAL" for f in session.flags)
            violation_types = {
                "HALLUCINATION", "UNAUTHORIZED_PROMISE", "PROMISE_CONFLICT",
                "CODE_SWITCH_ERROR", "ESCALATION_NEEDED", "HUMAN_HANDOFF"
            }
            has_violations = any(f.flag_type in violation_types for f in session.flags)

            if session.handoff_triggered or has_critical:
                verdict = VerdictType.ESCALATED.value
            elif has_violations:
                verdict = VerdictType.FAIL_FLAGGED.value
            else:
                verdict = VerdictType.PASS.value

            session.verdict = verdict
            # Single atomic DB update for COMPLETE state + verdict + handoff (avoids double-write race)
            DatabaseManager.update_call_state(
                session.id,
                CallState.COMPLETE.value,
                verdict=verdict,
                handoff_triggered=session.handoff_triggered,
                handoff_reason=session.handoff_reason
            )
            session.state = CallState.COMPLETE.value

            yield make_event("complete", {
                "call_id": session.id,
                "state": CallState.COMPLETE.value,
                "verdict": verdict,
                "total_turns": len(session.turns),
                "total_flags": len(session.flags),
                "total_promises": len(session.promises),
                "handoff_triggered": session.handoff_triggered,
                "handoff_reason": session.handoff_reason,
                "flags_summary": [f.to_dict() for f in session.flags],
                "promises_summary": [p.to_dict() for p in session.promises]
            })

        except Exception as e:
            self.transition_state(session, CallState.FAILED.value, error_msg=str(e))
            yield make_event("error", {
                "call_id": session.id,
                "state": CallState.FAILED.value,
                "error": str(e)
            })
        finally:
            self._active_pipelines[session.id] = False
