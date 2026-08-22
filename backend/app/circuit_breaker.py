"""Circuit Breaker pattern implementation for resilient multi-provider LLM failover."""
import time
import threading
from typing import Dict, Optional, Any
from .models import CircuitState, ProviderHealthStatus
from .config import settings

class CircuitBreaker:
    def __init__(self, name: str, failure_threshold: int = 3, cooldown_seconds: float = 60.0, timeout_seconds: float = 10.0):
        self.name = name
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.timeout_seconds = timeout_seconds
        
        self.state: CircuitState = CircuitState.CLOSED
        self.failure_count: int = 0
        self.last_failure_time: Optional[float] = None
        self.last_success_time: Optional[float] = None
        self.total_calls: int = 0
        self.total_latency_ms: float = 0.0
        self.simulated_outage: bool = False

    def is_available(self) -> bool:
        """Returns True if requests can be routed to this provider."""
        if self.simulated_outage:
            return False
        
        now = time.time()
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            # Check if cooldown has elapsed to allow a half-open probe
            if self.last_failure_time and (now - self.last_failure_time) >= self.cooldown_seconds:
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        
        if self.state == CircuitState.HALF_OPEN:
            return True
        
        return False

    def record_success(self, latency_ms: float):
        """Records successful response, resets failure count, and closes circuit."""
        now = time.time()
        self.last_success_time = now
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.total_calls += 1
        self.total_latency_ms += latency_ms

    def record_failure(self, error_message: str):
        """Records an outage or timeout, incrementing failure count and tripping circuit if threshold exceeded."""
        now = time.time()
        self.last_failure_time = now
        self.failure_count += 1
        self.total_calls += 1

        if self.state == CircuitState.HALF_OPEN or self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

    def set_simulated_outage(self, is_outage: bool):
        """Simulates an outage or recovery for live demonstration."""
        self.simulated_outage = is_outage
        if is_outage:
            self.state = CircuitState.OPEN
            self.last_failure_time = time.time()
        else:
            self.state = CircuitState.CLOSED
            self.failure_count = 0

    def get_status(self) -> ProviderHealthStatus:
        avg_lat = (self.total_latency_ms / self.total_calls) if self.total_calls > 0 else 0.0
        # Re-evaluate state transitions consistently
        if not self.simulated_outage and self.state == CircuitState.OPEN:
            if self.last_failure_time and (time.time() - self.last_failure_time) >= self.cooldown_seconds:
                self.state = CircuitState.HALF_OPEN

        return ProviderHealthStatus(
            provider=self.name,
            state=self.state.value,
            failure_count=self.failure_count,
            last_failure_time=self.last_failure_time,
            last_success_time=self.last_success_time,
            average_latency_ms=round(avg_lat, 2),
            is_simulated_outage=self.simulated_outage
        )

class CircuitBreakerRegistry:
    _instance: Optional['CircuitBreakerRegistry'] = None
    _lock = threading.Lock()

    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {
            "gemini": CircuitBreaker(
                name="gemini",
                failure_threshold=settings.circuit_breakers["gemini"].failure_threshold,
                cooldown_seconds=settings.circuit_breakers["gemini"].cooldown_seconds,
                timeout_seconds=settings.circuit_breakers["gemini"].timeout_seconds
            ),
            "groq": CircuitBreaker(
                name="groq",
                failure_threshold=settings.circuit_breakers["groq"].failure_threshold,
                cooldown_seconds=settings.circuit_breakers["groq"].cooldown_seconds,
                timeout_seconds=settings.circuit_breakers["groq"].timeout_seconds
            ),
            "ollama": CircuitBreaker(
                name="ollama",
                failure_threshold=settings.circuit_breakers["ollama"].failure_threshold,
                cooldown_seconds=settings.circuit_breakers["ollama"].cooldown_seconds,
                timeout_seconds=settings.circuit_breakers["ollama"].timeout_seconds
            ),
            "heuristic": CircuitBreaker(
                name="heuristic",
                failure_threshold=settings.circuit_breakers["heuristic"].failure_threshold,
                cooldown_seconds=settings.circuit_breakers["heuristic"].cooldown_seconds,
                timeout_seconds=settings.circuit_breakers["heuristic"].timeout_seconds
            )
        }

    @classmethod
    def get_instance(cls) -> 'CircuitBreakerRegistry':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = CircuitBreakerRegistry()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Allows test fixtures to reset singleton registry."""
        with cls._lock:
            cls._instance = None

    def get(self, provider: str) -> CircuitBreaker:
        return self.breakers.get(provider, self.breakers["heuristic"])

    def get_all_statuses(self) -> Dict[str, Dict[str, Any]]:
        return {name: cb.get_status().to_dict() for name, cb in self.breakers.items()}
