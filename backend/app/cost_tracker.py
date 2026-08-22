"""
CostTracker — Per-call token counting and cost projection for the Black Box pipeline.

Pricing (as of 2024-Q4, conservative estimates for hackathon budget projection):
  Gemini 1.5 Flash   : $0.075 / 1M input tokens  +  $0.30  / 1M output tokens
  Groq Llama-3.3-70B : $0.59  / 1M input tokens  +  $0.79  / 1M output tokens
  Ollama (local)     : $0.00  (zero cloud cost)
  Heuristic (local)  : $0.00  (zero cloud cost)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

# ---------------------------------------------------------------------------
# Provider price table  (USD per million tokens)
# ---------------------------------------------------------------------------
_PRICE_TABLE: Dict[str, Dict[str, float]] = {
    "gemini":    {"input": 0.075,  "output": 0.300},
    "groq":      {"input": 0.590,  "output": 0.790},
    "ollama":    {"input": 0.000,  "output": 0.000},
    "heuristic": {"input": 0.000,  "output": 0.000},
    "error":     {"input": 0.000,  "output": 0.000},
}


@dataclass
class TurnCost:
    provider: str
    tokens_in: int
    tokens_out: int
    cost_usd: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider":   self.provider,
            "tokens_in":  self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_usd":   round(self.cost_usd, 6),
        }


@dataclass
class CallCostSummary:
    total_turns: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_cost_usd: float = 0.0
    cost_per_turn: list = field(default_factory=list)   # List[TurnCost]

    # Extrapolation helpers
    CALLS_PER_HOUR_TYPICAL: int = 60          # BPO estimate
    HOURS_PER_DAY: int = 24

    def add_turn(self, turn_cost: TurnCost):
        self.total_turns += 1
        self.total_tokens_in  += turn_cost.tokens_in
        self.total_tokens_out += turn_cost.tokens_out
        self.total_cost_usd   += turn_cost.cost_usd
        self.cost_per_turn.append(turn_cost)

    @property
    def cost_per_call(self) -> float:
        return round(self.total_cost_usd, 6)

    @property
    def projected_cost_per_100_calls(self) -> float:
        return round(self.total_cost_usd * 100, 4)

    @property
    def projected_daily_cost(self) -> float:
        return round(self.total_cost_usd * self.CALLS_PER_HOUR_TYPICAL * self.HOURS_PER_DAY, 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_turns":                self.total_turns,
            "total_tokens_in":            self.total_tokens_in,
            "total_tokens_out":           self.total_tokens_out,
            "total_cost_usd":             round(self.total_cost_usd, 6),
            "cost_per_100_calls_usd":     self.projected_cost_per_100_calls,
            "projected_daily_cost_usd":   self.projected_daily_cost,
            "budget_ok":                  self.projected_cost_per_100_calls < 0.50,
        }


class CostTracker:
    """
    Stateless utility — call `record_turn()` with LLM usage metadata and
    accumulate into a `CallCostSummary`.
    """

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token estimator: ~4 chars per token (GPT/Gemini average)."""
        return max(1, len(text) // 4)

    @staticmethod
    def compute_turn_cost(
        provider: str,
        tokens_in: int,
        tokens_out: int,
    ) -> TurnCost:
        prices = _PRICE_TABLE.get(provider.lower(), _PRICE_TABLE["heuristic"])
        cost = (tokens_in * prices["input"] + tokens_out * prices["output"]) / 1_000_000
        return TurnCost(
            provider=provider,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
        )

    @staticmethod
    def record_turn(
        summary: CallCostSummary,
        provider: str,
        prompt_text: str,
        response_text: str,
        usage_metadata: Optional[Dict[str, Any]] = None,
    ) -> TurnCost:
        """
        Compute and accumulate cost for one scored turn.

        If the LLM response included usage metadata (tokens_in / tokens_out),
        use that for precision. Otherwise fall back to char-based estimation.
        """
        if usage_metadata:
            tokens_in  = usage_metadata.get("prompt_tokens",     usage_metadata.get("input_tokens",  CostTracker.estimate_tokens(prompt_text)))
            tokens_out = usage_metadata.get("completion_tokens",  usage_metadata.get("output_tokens", CostTracker.estimate_tokens(response_text)))
        else:
            tokens_in  = CostTracker.estimate_tokens(prompt_text)
            tokens_out = CostTracker.estimate_tokens(response_text)

        turn_cost = CostTracker.compute_turn_cost(provider, tokens_in, tokens_out)
        summary.add_turn(turn_cost)
        return turn_cost


class GlobalMetrics:
    """In-memory accumulator for all processed calls across the session."""
    total_calls: int = 0
    total_turns: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0

    # New detailed tracking
    model_counts: Dict[str, int] = {}
    total_errors: int = 0
    total_fallbacks: int = 0  # tracked as turns where we didn't use grok/openrouter
    _recent_latencies: List[float] = []
    _MAX_LATENCY_HISTORY = 1000

    @classmethod
    def record_turn(cls, turn_cost: TurnCost, latency_ms: float):
        cls.total_turns += 1
        cls.total_tokens_in += turn_cost.tokens_in
        cls.total_tokens_out += turn_cost.tokens_out
        cls.total_cost_usd += turn_cost.cost_usd
        cls.total_latency_ms += latency_ms

        provider = turn_cost.provider.lower()
        cls.model_counts[provider] = cls.model_counts.get(provider, 0) + 1

        if provider not in ("grok", "openrouter"):
            cls.total_fallbacks += 1
            if provider == "error" or provider == "sentinel":
                cls.total_errors += 1

        cls._recent_latencies.append(latency_ms)
        if len(cls._recent_latencies) > cls._MAX_LATENCY_HISTORY:
            cls._recent_latencies.pop(0)

    @classmethod
    def record_call(cls, summary: CallCostSummary):
        cls.total_calls += 1

    @classmethod
    def get_summary(cls) -> Dict[str, Any]:
        avg_latency = cls.total_latency_ms / cls.total_turns if cls.total_turns > 0 else 0
        avg_cost_per_call = cls.total_cost_usd / cls.total_calls if cls.total_calls > 0 else 0

        p95_latency_ms = 0.0
        if cls._recent_latencies:
            sorted_lat = sorted(cls._recent_latencies)
            idx = int(len(sorted_lat) * 0.95)
            p95_latency_ms = sorted_lat[min(idx, len(sorted_lat) - 1)]

        error_rate = cls.total_errors / cls.total_turns if cls.total_turns > 0 else 0.0
        fallback_rate = cls.total_fallbacks / cls.total_turns if cls.total_turns > 0 else 0.0

        return {
            "total_requests": cls.total_calls,
            "total_turns": cls.total_turns,
            "total_tokens": cls.total_tokens_in + cls.total_tokens_out,
            "total_cost_usd": round(cls.total_cost_usd, 6),
            "average_latency_ms": round(avg_latency, 2),
            "p95_latency_ms": round(p95_latency_ms, 2),
            "average_cost_per_call_usd": round(avg_cost_per_call, 6),
            "error_rate": round(error_rate, 4),
            "fallback_rate": round(fallback_rate, 4),
            "model_distribution": cls.model_counts,
        }
