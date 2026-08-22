"""Configuration settings for Black Box Voice Agent Guardrail."""
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Dict, Any

# Root project paths
BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
DATA_DIR = BACKEND_DIR / "data"
SCHEMAS_DIR = DATA_DIR / "schemas"
AUDIO_DIR = DATA_DIR / "audio"   # backend/data/audio — WAVs live here
DB_PATH = BACKEND_DIR / "blackbox.db"

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 3
    cooldown_seconds: float = 60.0
    timeout_seconds: float = 10.0
    max_retries: int = 1

@dataclass
class Settings:
    # App Info
    app_name: str = "Black Box for Voice Agents"
    app_version: str = "1.0.0"
    debug: bool = os.getenv("DEBUG", "False").lower() in ("true", "1")

    # Database
    db_path: Path = DB_PATH

    # ── Tier 1A: Grok (xAI) ─────────────────────────────────────────────────
    grok_api_key: str = os.getenv("GROK_API_KEY", "")
    grok_model: str = os.getenv("GROK_MODEL", "grok-beta")
    grok_endpoint: str = "https://api.x.ai/v1/chat/completions"

    # ── Tier 1B: OpenRouter ──────────────────────────────────────────────────
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_model: str = os.getenv(
        "OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"
    )
    openrouter_endpoint: str = "https://openrouter.ai/api/v1/chat/completions"

    # ── Tier 2: Gemini ───────────────────────────────────────────────────────
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    # ── Tier 3: Groq (kept for legacy / backup) ──────────────────────────────
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # ── Tier 4: Local Ollama ─────────────────────────────────────────────────
    ollama_enabled: bool = os.getenv("OLLAMA_ENABLED", "False").lower() in ("true", "1")
    ollama_endpoint: str = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

    # Outage Simulation Switches (runtime-mutable for demo)
    simulate_grok_outage: bool = False
    simulate_openrouter_outage: bool = False
    simulate_gemini_outage: bool = False
    simulate_groq_outage: bool = False
    simulate_ollama_outage: bool = False

    # Pipeline Timeouts (seconds)
    pipeline_timeout_seconds: float = 120.0
    asr_timeout_seconds: float = 60.0
    scoring_turn_timeout_seconds: float = 15.0

    # Guardrail Thresholds
    confidence_handoff_threshold: float = 0.40
    consecutive_low_conf_turns: int = 2
    max_promises_per_call: int = 5
    fuzzy_price_tolerance_pct: float = 5.0

    # Circuit Breakers per provider — strict 3s timeout for all cloud LLMs
    circuit_breakers: Dict[str, CircuitBreakerConfig] = field(default_factory=lambda: {
        "grok":        CircuitBreakerConfig(failure_threshold=3, cooldown_seconds=60.0, timeout_seconds=3.0),
        "openrouter":  CircuitBreakerConfig(failure_threshold=3, cooldown_seconds=60.0, timeout_seconds=3.0),
        "gemini":      CircuitBreakerConfig(failure_threshold=3, cooldown_seconds=60.0, timeout_seconds=3.0),
        "groq":        CircuitBreakerConfig(failure_threshold=3, cooldown_seconds=60.0, timeout_seconds=3.0),
        "ollama":      CircuitBreakerConfig(failure_threshold=5, cooldown_seconds=30.0, timeout_seconds=10.0),
        "heuristic":   CircuitBreakerConfig(failure_threshold=999, cooldown_seconds=0.0, timeout_seconds=0.1),
    })

    # Data paths (all canonical — under backend/data/)
    kb_file_path: Path = DATA_DIR / "knowledge_base.json"
    test_cases_path: Path = DATA_DIR / "test_cases_metadata.json"
    ground_truth_path: Path = DATA_DIR / "ground_truth.json"
    generate_tests_path: Path = DATA_DIR / "generate_tests.py"
    schemas_dir: Path = SCHEMAS_DIR
    audio_dir: Path = AUDIO_DIR

settings = Settings()
