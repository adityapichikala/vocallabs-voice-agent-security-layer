"""Unit tests for Knowledge Base repository and search."""
import unittest
from backend.app.knowledge_base import KnowledgeBaseRepository

class TestKnowledgeBase(unittest.TestCase):
    def setUp(self):
        self.kb = KnowledgeBaseRepository.get_instance()

    def test_kb_has_15_facts(self):
        facts = self.kb.get_all_facts()
        self.assertEqual(len(facts), 15)

    def test_kb_fact_lookup_by_id(self):
        fact = self.kb.get_fact_by_id("KB-PRC-001")
        self.assertIsNotNone(fact)
        self.assertEqual(fact["category"], "PRICING")
        self.assertEqual(fact["canonical_value"]["value"], 699)

    def test_kb_keyword_search(self):
        results = self.kb.search_facts("gigabit 1 gbps speed")
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "KB-PRC-002")

    def test_kb_prompt_context_generation(self):
        ctx = self.kb.get_prompt_context()
        self.assertIn("OFFICIAL CORPORATE KNOWLEDGE BASE", ctx)
        self.assertIn("KB-PRC-001", ctx)
        self.assertIn("KB-SLA-015", ctx)

if __name__ == "__main__":
    unittest.main()
