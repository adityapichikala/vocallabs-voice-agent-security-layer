"""Pydantic schemas and dataclasses for Black Box data models."""
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional, Dict, Any, Union
import json
import time

class CallState(str, Enum):
    UPLOADED = "UPLOADED"
    TRANSCRIBING = "TRANSCRIBING"
    SCORING = "SCORING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"

class SpeakerType(str, Enum):
    AGENT = "agent"
    CUSTOMER = "customer"

class FlagType(str, Enum):
    HALLUCINATION = "HALLUCINATION"
    PROMISE_MADE = "PROMISE_MADE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    CODE_SWITCH_ERROR = "CODE_SWITCH_ERROR"
    ESCALATION_NEEDED = "ESCALATION_NEEDED"
    ASR_UNCERTAIN = "ASR_UNCERTAIN"
    UNVERIFIABLE = "UNVERIFIABLE"
    PROMISE_CONFLICT = "PROMISE_CONFLICT"
    UNAUTHORIZED_PROMISE = "UNAUTHORIZED_PROMISE"
    EXCESSIVE_PROMISES = "EXCESSIVE_PROMISES"
    SCOPE_MISMATCH = "SCOPE_MISMATCH"
    HUMAN_HANDOFF = "HUMAN_HANDOFF"

class FlagSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class PromiseState(str, Enum):
    PENDING = "PENDING"
    FULFILLED = "FULFILLED"
    BROKEN = "BROKEN"
    EXPIRED = "EXPIRED"
    DUPLICATE = "DUPLICATE"

class VerdictType(str, Enum):
    PASS = "PASS"
    FAIL_FLAGGED = "FAIL_FLAGGED"
    ESCALATED = "ESCALATED"

class CircuitState(str, Enum):
    CLOSED = "CLOSED"      # Healthy
    OPEN = "OPEN"          # Outage / tripped
    HALF_OPEN = "HALF_OPEN"# Probing recovery

@dataclass
class Turn:
    id: str
    call_id: str
    turn_index: int
    speaker: str
    text: str
    start_time: float
    end_time: float
    asr_confidence: float = 1.0
    language_detected: Optional[str] = "en"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class Flag:
    id: str
    call_id: str
    turn_id: str
    turn_index: int
    flag_type: str
    severity: str
    detail: str
    kb_fact_id: Optional[str] = None
    claimed_value: Optional[str] = None
    actual_value: Optional[str] = None
    provider_used: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class Promise:
    id: str
    call_id: str
    turn_id: str
    turn_index: int
    promise_hash: str
    who: str
    action: str
    target_entity: str
    deadline_raw: str
    deadline_parsed_iso: Optional[str] = None
    condition: Optional[str] = None
    is_authorized: bool = True
    violation_reason: Optional[str] = None
    state: str = PromiseState.PENDING.value
    mention_count: int = 1
    merged_into: Optional[str] = None
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class CallSession:
    id: str
    audio_hash: str
    filename: str
    duration_seconds: float
    state: str = CallState.UPLOADED.value
    turns: List[Turn] = field(default_factory=list)
    flags: List[Flag] = field(default_factory=list)
    promises: List[Promise] = field(default_factory=list)
    verdict: str = VerdictType.PASS.value
    handoff_triggered: bool = False
    handoff_reason: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "audio_hash": self.audio_hash,
            "filename": self.filename,
            "duration_seconds": self.duration_seconds,
            "state": self.state,
            "turns": [t.to_dict() for t in self.turns],
            "flags": [f.to_dict() for f in self.flags],
            "promises": [p.to_dict() for p in self.promises],
            "verdict": self.verdict,
            "handoff_triggered": self.handoff_triggered,
            "handoff_reason": self.handoff_reason,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error_message": self.error_message
        }

@dataclass
class EvalMetrics:
    test_case_id: str
    category: str
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    latency_ms: float
    provider_used: str
    verdict_matched: bool
    handoff_matched: bool
    is_curveball_run: bool = False
    cost_usd: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ProviderHealthStatus:
    provider: str
    state: str
    failure_count: int
    last_failure_time: Optional[float]
    last_success_time: Optional[float]
    average_latency_ms: float
    is_simulated_outage: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class OutageModeRequest:
    simulate_gemini: Optional[bool] = None
    simulate_groq: Optional[bool] = None
    simulate_ollama: Optional[bool] = None

@dataclass
class PipelineProgress:
    call_id: str
    stage: str
    current_turn: int
    total_turns: int
    pct: float
    provider: str
    status_message: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
