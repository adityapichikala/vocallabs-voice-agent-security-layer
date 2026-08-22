"""FastAPI Application with Real-Time SSE Streams, Outage Simulation, and Guardrail APIs."""
import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from .config import settings
from .database import init_db, DatabaseManager
from .models import CallSession, OutageModeRequest, CallState
from .validators import AudioValidator, ValidationError
from .pipeline_manager import PipelineManager
from .circuit_breaker import CircuitBreakerRegistry
from .knowledge_base import KnowledgeBaseRepository
from .eval_engine import EvalEngine

# Lazy-initialized global instances (created at startup, not import time)
pipeline_mgr: Optional[PipelineManager] = None
cb_registry: Optional[CircuitBreakerRegistry] = None
kb_repo: Optional[KnowledgeBaseRepository] = None
eval_engine: Optional[EvalEngine] = None

# In-memory replay buffer for SSE reconnection (last 100 events per call)
# Capped at 50 call buffers to prevent unbounded memory growth
_MAX_REPLAY_BUFFERS = 50
_MAX_EVENTS_PER_BUFFER = 100
sse_replay_buffers: Dict[str, List[Dict[str, Any]]] = {}

def _evict_oldest_replay_buffer():
    """Evict the oldest replay buffer when the cap is exceeded."""
    if len(sse_replay_buffers) > _MAX_REPLAY_BUFFERS:
        oldest_key = next(iter(sse_replay_buffers))
        del sse_replay_buffers[oldest_key]

# Import FastAPI and sse-starlette separately for clear error messages
try:
    from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import JSONResponse, StreamingResponse
    from fastapi.responses import FileResponse
except ImportError as e:
    raise ImportError(
        f"FastAPI is required but not installed. Run: pip install fastapi python-multipart uvicorn. Error: {e}"
    ) from e

try:
    from sse_starlette.sse import EventSourceResponse
except ImportError as e:
    raise ImportError(
        f"sse-starlette is required but not installed. Run: pip install sse-starlette. Error: {e}"
    ) from e

app = FastAPI(
    title="Black Box for Voice Agents",
    description="Real-time reliability, fact-grounding, promise-tracking, and guardrail layer.",
    version=settings.app_version
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    """Initialize DB and global singletons at server startup, not module import time."""
    global pipeline_mgr, cb_registry, kb_repo, eval_engine
    init_db()
    pipeline_mgr = PipelineManager()
    cb_registry = CircuitBreakerRegistry.get_instance()
    kb_repo = KnowledgeBaseRepository.get_instance()
    eval_engine = EvalEngine()

@app.get("/")
async def root():
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "status": "online",
        "active_providers": cb_registry.get_all_statuses() if cb_registry else {}
    }

@app.get("/api/kb")
async def get_knowledge_base():
    return {
        "metadata": kb_repo.metadata,
        "facts": kb_repo.get_all_facts()
    }

@app.get("/api/test-cases")
async def get_test_cases():
    if not settings.test_cases_path.exists():
        raise HTTPException(status_code=404, detail="Test cases file not found.")
    with open(settings.test_cases_path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/api/providers/health")
async def get_provider_health():
    return cb_registry.get_all_statuses()

@app.post("/api/providers/simulate-outage")
async def simulate_provider_outage(payload: OutageModeRequest):
    if payload.simulate_gemini is not None:
        cb_registry.get("gemini").set_simulated_outage(payload.simulate_gemini)
    if payload.simulate_groq is not None:
        cb_registry.get("groq").set_simulated_outage(payload.simulate_groq)
    if payload.simulate_ollama is not None:
        cb_registry.get("ollama").set_simulated_outage(payload.simulate_ollama)
    return {
        "message": "Outage simulation updated.",
        "providers": cb_registry.get_all_statuses()
    }

@app.post("/api/calls/upload")
async def upload_call_audio(
    file: Optional[UploadFile] = File(None),
    test_case_id: Optional[str] = Form(None)
):
    """Uploads a .wav file or selects a benchmark test case to start a new analysis session."""
    if test_case_id:
        filename = f"{test_case_id}.wav"
        audio_hash = f"hash_{test_case_id}_{int(time.time()*1000)}"
        duration = 30.0
        call_id = f"call_{test_case_id}_{int(time.time())}"
    elif file:
        content = await file.read()
        val = AudioValidator.validate_file(file.filename, content)
        filename = file.filename
        audio_hash = val["sha256"]
        duration = val["estimated_duration_seconds"]
        call_id = f"call_{int(time.time())}_{audio_hash[:8]}"
    else:
        raise HTTPException(status_code=400, detail="Must provide an audio file or a test_case_id.")

    session = DatabaseManager.create_or_get_call(
        call_id=call_id,
        audio_hash=audio_hash,
        filename=filename,
        duration=duration
    )
    return session.to_dict()

@app.get("/api/calls")
async def list_calls():
    return DatabaseManager.get_all_calls()

@app.get("/api/calls/{call_id}")
async def get_call_details(call_id: str):
    call = DatabaseManager.get_call(call_id)
    if not call:
        raise HTTPException(status_code=404, detail=f"Call {call_id} not found.")
    return call.to_dict()

@app.get("/api/calls/{call_id}/stream")
async def stream_call_analysis(call_id: str, request: Request, speed: float = Query(2.0)):
    """Server-Sent Events (SSE) stream yielding real-time transcript turns and guardrail telemetry.
    
    If call_id looks like a test case ID (e.g. 'TC-05') and no session exists in DB,
    auto-creates a session so the frontend can stream test cases directly without a
    separate upload step.
    """
    session = DatabaseManager.get_call(call_id)
    if not session:
        # Auto-create session for test case IDs (e.g. TC-01, TC-05, etc.)
        if call_id.startswith("TC-") or call_id.startswith("tc-"):
            test_case_id = call_id.upper()
            filename = f"{test_case_id}.wav"
            audio_hash = f"hash_{test_case_id}_{int(time.time() * 1000)}"
            real_call_id = f"call_{test_case_id}_{int(time.time())}"
            session = DatabaseManager.create_or_get_call(
                call_id=real_call_id,
                audio_hash=audio_hash,
                filename=filename,
                duration=30.0
            )
        else:
            raise HTTPException(status_code=404, detail=f"Call {call_id} not found.")

    # Replay buffer check
    last_event_id = request.headers.get("Last-Event-ID")
    replay_events = sse_replay_buffers.get(session.id, [])

    async def event_generator():
        # If reconnecting with Last-Event-ID, replay buffered events
        if last_event_id:
            try:
                last_id_num = int(last_event_id)
                for ev in replay_events:
                    if ev.get("seq_id", 0) > last_id_num:
                        yield {
                            "id": str(ev["seq_id"]),
                            "event": ev["event"],
                            "data": json.dumps(ev["data"])
                        }
            except ValueError:
                pass

        # Create/reset replay buffer for this call
        if session.id not in sse_replay_buffers:
            sse_replay_buffers[session.id] = []
            _evict_oldest_replay_buffer()

        # Stream real-time pipeline events
        async for ev in pipeline_mgr.execute_call_pipeline(session, session.filename, speed_multiplier=speed):
            sse_replay_buffers[session.id].append(ev)
            if len(sse_replay_buffers[session.id]) > _MAX_EVENTS_PER_BUFFER:
                sse_replay_buffers[session.id].pop(0)

            yield {
                "id": str(ev["seq_id"]),
                "event": ev["event"],
                "data": json.dumps(ev["data"])
            }

    return EventSourceResponse(event_generator())

@app.post("/api/eval/run")
async def run_evaluation(curveball: bool = Query(False)):
    results = await eval_engine.run_benchmark(is_curveball_run=curveball)
    return results

@app.get("/api/eval/latest")
async def get_latest_evaluation():
    results = await eval_engine.run_benchmark(is_curveball_run=False)
    return results

@app.post("/api/eval/simulate-timeout")
async def simulate_timeout(turn_text: str = Query("Hello, I need help with my bill.")):
    """
    Live demo endpoint: trips Gemini + Groq circuit breakers, runs the scoring chain,
    and returns a structured proof that the 3-second timeout + safe sentinel path works.
    Circuit breakers are automatically restored after the test.
    """
    return await eval_engine.simulate_timeout_scenario(turn_text=turn_text)

@app.post("/api/eval/all-providers-dead")
async def all_providers_dead():
    """
    Extreme failure test: trips ALL provider circuit breakers.
    Returns a structured proof that the pipeline gracefully degradates to the 
    SAFE_RESPONSE_SENTINEL instantly.
    """
    return await eval_engine.simulate_all_providers_dead()

# ==============================================================================
#  BRIDGE ROUTES — backward-compatible with the Next.js frontend
#  These mirror the old root main.py API surface so page.tsx needs zero changes.
# ==============================================================================

from fastapi.responses import FileResponse as _FileResponse
from fastapi.staticfiles import StaticFiles

@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    """Serve WAV files from backend/data/audio/ for browser playback."""
    # Try new canonical location first, then legacy fallback
    for candidate in [settings.audio_dir / filename, settings.root_test_calls_dir / filename]:
        if candidate.exists():
            return _FileResponse(candidate)
    raise HTTPException(status_code=404, detail=f"Audio file not found: {filename}")

@app.get("/api/metrics")
async def get_metrics():
    """
    Returns aggregate cost, token usage, and latency metrics.
    """
    from .cost_tracker import GlobalMetrics
    return GlobalMetrics.get_summary()

# Mount the static frontend.
# Vite builds to frontend/dist
FRONTEND_DIR = PROJECT_ROOT / "frontend" / "dist"

if FRONTEND_DIR.exists():
    @app.get("/")
    async def serve_frontend_index():
        return FileResponse(FRONTEND_DIR / "index.html")
    
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

@app.get("/health")
async def health_check():
    """Lightweight health + provider availability endpoint."""
    audio_files = sorted(
        f.name for f in settings.audio_dir.glob("*.wav")
    ) if settings.audio_dir.exists() else []
    return {
        "status": "ok",
        "gemini_configured": bool(settings.gemini_api_key),
        "groq_configured": bool(settings.groq_api_key),
        "ollama_enabled": settings.ollama_enabled,
        "ollama_model": settings.ollama_model,
        "audio_files": audio_files,
        "providers": cb_registry.get_all_statuses() if cb_registry else {},
    }


@app.get("/analyze")
@app.post("/analyze")
async def analyze_bridge(
    filename: str = Query(..., description="WAV filename, e.g. test_2_hallucination.wav"),
    force_outage: bool = Query(False, description="Simulate Gemini outage to demo cascade"),
    speed: float = Query(2.0, description="Playback speed multiplier for turn delays"),
):
    """
    SSE bridge route — compatible with the existing Next.js frontend.

    Auto-creates a lightweight call session and streams pipeline events.
    The force_outage flag maps to the circuit-breaker simulation switches
    so the 🔥 toggle in the UI correctly cascades Gemini → Groq → Heuristic.
    """
    if force_outage:
        cb_registry.get("gemini").set_simulated_outage(True)
        cb_registry.get("groq").set_simulated_outage(False)  # keep Groq alive for visible cascade
    else:
        cb_registry.get("gemini").set_simulated_outage(False)
        cb_registry.get("groq").set_simulated_outage(False)

    import time as _time
    call_id = f"bridge_{filename.replace('.wav','').replace('.','_')}_{int(_time.time())}"
    session = DatabaseManager.create_or_get_call(
        call_id=call_id,
        audio_hash=f"bridge_{call_id}",
        filename=filename,
        duration=60.0,
    )

    async def event_generator():
        async for ev in pipeline_mgr.execute_call_pipeline(session, filename, speed_multiplier=speed):
            yield {
                "id": str(ev["seq_id"]),
                "event": ev["event"],
                "data": json.dumps(ev["data"]),
            }

    return EventSourceResponse(event_generator())
