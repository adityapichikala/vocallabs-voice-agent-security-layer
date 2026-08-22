"""Unit tests for Promise deduplication, hashing, and state management."""
import unittest
import time
from backend.app.guardrail_engine import GuardrailEngine
from backend.app.models import Promise, PromiseState, Turn
from backend.app.database import init_db, DatabaseManager

class TestPromiseState(unittest.TestCase):
    def setUp(self):
        init_db()

    def test_promise_hash_deduplication(self):
        h1 = GuardrailEngine.compute_promise_hash("CALLBACK", "personal phone callback", "5:00 PM")
        h2 = GuardrailEngine.compute_promise_hash("CALLBACK", "personal phone callback", "5:00 PM")
        self.assertEqual(h1, h2)

        # Minor whitespace/casing differences should normalize to same hash
        h3 = GuardrailEngine.compute_promise_hash("callback  ", " Personal Phone Callback ", "5:00 pm")
        self.assertEqual(h1, h3)

    def test_promise_save_and_merge_duplicate(self):
        unique_suffix = int(time.time() * 1000)
        call_id = f"call_test_prom_{unique_suffix}"
        audio_hash = f"hash_test_prom_{unique_suffix}"
        
        # 1. Create Call
        DatabaseManager.create_or_get_call(call_id, audio_hash, "test.wav", 20.0)
        
        # 2. Create Turns for Foreign Key consistency
        turn1_id = f"t1_{unique_suffix}"
        turn3_id = f"t3_{unique_suffix}"
        DatabaseManager.save_turns(call_id, [
            Turn(id=turn1_id, call_id=call_id, turn_index=1, speaker="agent", text="I will call you back today.", start_time=0.0, end_time=2.0),
            Turn(id=turn3_id, call_id=call_id, turn_index=3, speaker="agent", text="As promised, calling back today.", start_time=4.0, end_time=6.0)
        ])

        p_hash = GuardrailEngine.compute_promise_hash("CALLBACK", "callback", "today")
        p1 = Promise(
            id=f"p1_{unique_suffix}",
            call_id=call_id,
            turn_id=turn1_id,
            turn_index=1,
            promise_hash=p_hash,
            who="agent",
            action="callback",
            target_entity="CALLBACK",
            deadline_raw="today",
            state=PromiseState.PENDING.value
        )
        
        saved1 = DatabaseManager.save_or_merge_promise(p1)
        self.assertEqual(saved1.mention_count, 1)

        # Insert second identical promise with same hash
        p2 = Promise(
            id=f"p2_{unique_suffix}",
            call_id=call_id,
            turn_id=turn3_id,
            turn_index=3,
            promise_hash=p_hash,
            who="agent",
            action="callback",
            target_entity="CALLBACK",
            deadline_raw="today",
            state=PromiseState.PENDING.value
        )
        saved2 = DatabaseManager.save_or_merge_promise(p2)
        self.assertEqual(saved2.mention_count, 2)
        self.assertEqual(saved2.state, PromiseState.DUPLICATE.value)

if __name__ == "__main__":
    unittest.main()
