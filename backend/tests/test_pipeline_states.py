"""Unit tests for Pipeline State transitions, SSE event streaming, and error recovery."""
import asyncio
import time
import unittest
from backend.app.database import init_db, DatabaseManager
from backend.app.models import CallSession, CallState, VerdictType, Flag, FlagSeverity
from backend.app.pipeline_manager import PipelineManager

class TestPipelineStates(unittest.TestCase):
    def setUp(self):
        init_db()
        self.mgr = PipelineManager()

    def test_pipeline_valid_lifecycle(self):
        uid = int(time.time() * 1000)
        session = DatabaseManager.create_or_get_call(
            call_id=f"call_lifecycle_{uid}",
            audio_hash=f"audio_hash_lifecycle_{uid}",
            filename="TC-01.wav",
            duration=28.0
        )
        self.assertEqual(session.state, CallState.UPLOADED.value)

        # Transition UPLOADED -> TRANSCRIBING
        self.mgr.transition_state(session, CallState.TRANSCRIBING.value)
        self.assertEqual(session.state, CallState.TRANSCRIBING.value)

        # Transition TRANSCRIBING -> SCORING
        self.mgr.transition_state(session, CallState.SCORING.value)
        self.assertEqual(session.state, CallState.SCORING.value)

        # Transition SCORING -> COMPLETE
        self.mgr.transition_state(session, CallState.COMPLETE.value)
        self.assertEqual(session.state, CallState.COMPLETE.value)

    def test_pipeline_failure_state(self):
        uid = int(time.time() * 1000)
        session = DatabaseManager.create_or_get_call(
            call_id=f"call_fail_{uid}",
            audio_hash=f"audio_hash_fail_{uid}",
            filename="bad_audio.wav",
            duration=10.0
        )
        self.mgr.transition_state(session, CallState.FAILED.value, error_msg="Simulated acoustic failure")
        self.assertEqual(session.state, CallState.FAILED.value)
        self.assertEqual(session.error_message, "Simulated acoustic failure")

    def test_pipeline_streaming_emits_events(self):
        uid = int(time.time() * 1000)
        async def run_stream():
            session = DatabaseManager.create_or_get_call(
                call_id=f"call_stream_tc05_{uid}",
                audio_hash=f"hash_stream_tc05_{uid}",
                filename="TC-05.wav",
                duration=30.0
            )
            events = []
            async for ev in self.mgr.execute_call_pipeline(session, "TC-05", speed_multiplier=100.0):
                events.append(ev)
            return events, session

        events, session = asyncio.run(run_stream())
        event_types = [e["event"] for e in events]
        self.assertIn("state_change", event_types)
        self.assertIn("turn_transcribed", event_types)
        self.assertIn("turn_evaluated", event_types)
        self.assertIn("handoff_alert", event_types)
        self.assertIn("complete", event_types)
        self.assertEqual(session.state, CallState.COMPLETE.value)
        self.assertEqual(session.verdict, VerdictType.ESCALATED.value)

if __name__ == "__main__":
    unittest.main()
