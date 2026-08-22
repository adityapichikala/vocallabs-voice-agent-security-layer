"""Knowledge Base repository with validation, indexing, thread safety, and semantic/keyword search."""
import json
import re
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional
from .config import settings
from .validators import KnowledgeBaseValidator

class KnowledgeBaseRepository:
    _instance: Optional['KnowledgeBaseRepository'] = None
    _lock = threading.Lock()

    def __init__(self, kb_path: Optional[Path] = None):
        self.kb_path = kb_path or settings.kb_file_path
        self.metadata: Dict[str, Any] = {}
        self.facts: List[Dict[str, Any]] = []
        self.facts_by_id: Dict[str, Dict[str, Any]] = {}
        self.load_and_validate()

    @classmethod
    def get_instance(cls, kb_path: Optional[Path] = None) -> 'KnowledgeBaseRepository':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = KnowledgeBaseRepository(kb_path=kb_path)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Allows test fixtures to reset singleton state."""
        with cls._lock:
            cls._instance = None

    def load_and_validate(self):
        if not self.kb_path.exists():
            raise FileNotFoundError(f"Knowledge Base JSON not found at: {self.kb_path}")

        with open(self.kb_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        warnings = KnowledgeBaseValidator.validate_kb(data)
        self.metadata = {
            "version": data.get("version", "1.0"),
            "company_name": data.get("company_name", "Apex Telecom"),
            "last_updated": data.get("last_updated", ""),
            "description": data.get("description", ""),
            "validation_warnings": warnings
        }
        self.facts = data.get("facts", [])
        self.facts_by_id = {f["id"]: f for f in self.facts if "id" in f}

    def get_all_facts(self) -> List[Dict[str, Any]]:
        return self.facts

    def get_fact_by_id(self, fact_id: str) -> Optional[Dict[str, Any]]:
        return self.facts_by_id.get(fact_id)

    def search_facts(self, query: str, category: Optional[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            return self.facts[:top_k]

        query_words = set(re.findall(r"\w+", query.lower()))
        if not query_words:
            return self.facts[:top_k]

        scored_facts = []

        for fact in self.facts:
            if category and fact.get("category") != category:
                continue

            score = 0
            # Match title
            title_words = set(re.findall(r"\w+", str(fact.get("claim_title", "")).lower()))
            score += len(query_words & title_words) * 3

            # Match tags
            tag_words = set(str(t).lower() for t in fact.get("tags", []))
            score += len(query_words & tag_words) * 2

            # Match statement
            stmt_words = set(re.findall(r"\w+", str(fact.get("official_statement", "")).lower()))
            score += len(query_words & stmt_words) * 1

            if score > 0:
                scored_facts.append((score, fact))

        scored_facts.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in scored_facts[:top_k]]

    def get_prompt_context(self) -> str:
        """Formats the entire KB into a compact, bulleted reference for LLM prompts with defensive key access."""
        company = self.metadata.get("company_name", "Apex Telecom")
        lines = [f"=== OFFICIAL CORPORATE KNOWLEDGE BASE: {company} ==="]
        for f in self.facts:
            canon = f.get("canonical_value", {})
            val = canon.get("value", "N/A")
            unit = canon.get("unit", "")
            val_str = f"{val} {unit}".strip()
            
            authority = f.get("allowed_agent_authority", {})
            max_waiver = authority.get("max_waiver_value", 0)
            
            fact_id = f.get("id", "UNKNOWN")
            cat = f.get("category", "GENERAL")
            title = f.get("claim_title", "Untitled")
            stmt = f.get("official_statement", "")

            lines.append(
                f"- [{fact_id}] ({cat}) {title}: {stmt} "
                f"[Canonical Value: {val_str}, Max Waiver Authority: {max_waiver}]"
            )
        return "\n".join(lines)
