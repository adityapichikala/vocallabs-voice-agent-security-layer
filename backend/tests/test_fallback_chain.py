"""Unit tests for fallback chain and circuit breaker mechanics."""
import unittest
import time
from backend.app.circuit_breaker import CircuitBreaker, CircuitBreakerRegistry
from backend.app.fallback_chain import FallbackChain
from backend.app.models import CircuitState

class TestFallbackChain(unittest.TestCase):
    def setUp(self):
        CircuitBreakerRegistry.reset_instance()

    def test_circuit_breaker_trips_on_consecutive_failures(self):
        cb = CircuitBreaker("test_provider", failure_threshold=2, cooldown_seconds=10.0)
        self.assertEqual(cb.state, CircuitState.CLOSED)
        self.assertTrue(cb.is_available())

        # First failure
        cb.record_failure("HTTP 500")
        self.assertEqual(cb.state, CircuitState.CLOSED)

        # Second failure -> trips to OPEN
        cb.record_failure("HTTP 503")
        self.assertEqual(cb.state, CircuitState.OPEN)
        self.assertFalse(cb.is_available())

    def test_circuit_breaker_cooldown_to_half_open(self):
        cb = CircuitBreaker("test_cooldown", failure_threshold=1, cooldown_seconds=0.05)
        cb.record_failure("Timeout")
        self.assertEqual(cb.state, CircuitState.OPEN)
        self.assertFalse(cb.is_available())

        # Wait for cooldown
        time.sleep(0.06)
        self.assertTrue(cb.is_available())
        self.assertEqual(cb.state, CircuitState.HALF_OPEN)

        # Successful probe recovers to CLOSED
        cb.record_success(20.0)
        self.assertEqual(cb.state, CircuitState.CLOSED)

    def test_circuit_breaker_simulated_outage(self):
        cb = CircuitBreaker("test_gemini")
        cb.set_simulated_outage(True)
        self.assertFalse(cb.is_available())
        
        cb.set_simulated_outage(False)
        self.assertTrue(cb.is_available())

    def test_fallback_chain_heuristic_execution(self):
        chain = FallbackChain()
        res, prov, lat = chain.execute_scoring_chain(
            turn_text="I will give you a 50% lifetime student discount.",
            speaker="agent",
            conversation_history=[],
            turn_id="turn_test_1"
        )
        self.assertIsNotNone(res)
        self.assertIn("flags", res)
        self.assertTrue(any(f["type"] == "HALLUCINATION" for f in res["flags"]))
        self.assertTrue(res["handoff_recommended"])

    def test_heuristic_operator_precedence_no_false_positive(self):
        chain = FallbackChain()
        # Text with 800 and credit in separate context should not trigger ₹800 unauthorized credit
        res, prov, lat = chain.execute_scoring_chain(
            turn_text="Your balance has 800 loyalty reward points available for store credit.",
            speaker="customer",
            conversation_history=[],
            turn_id="turn_test_fp"
        )
        unauth_flags = [f for f in res.get("flags", []) if f["type"] == "UNAUTHORIZED_PROMISE"]
        self.assertEqual(len(unauth_flags), 0)

    def test_heuristic_unauthorized_credit_triggers(self):
        chain = FallbackChain()
        res, prov, lat = chain.execute_scoring_chain(
            turn_text="I will apply a goodwill credit ₹800 to your current bill.",
            speaker="agent",
            conversation_history=[],
            turn_id="turn_test_unauth"
        )
        unauth_flags = [f for f in res.get("flags", []) if f["type"] == "UNAUTHORIZED_PROMISE"]
        self.assertGreaterEqual(len(unauth_flags), 1)
        self.assertTrue(res["handoff_recommended"])

if __name__ == "__main__":
    unittest.main()
