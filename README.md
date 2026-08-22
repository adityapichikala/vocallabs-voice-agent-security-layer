# Black Box for Voice Agents
### Real-Time Reliability, Fact-Grounding, Promise Tracking & Guardrail Layer for AI Voice Agents
**Built for:** VocalLabs AI Hackathon | **Track:** Developer Tooling / Agents & Automation

---

## 1. The Problem & Solution

Voice AI companies don't struggle to make agents talk — they struggle to **trust** agents in production. An agent can hallucinate a discount, misquote an early termination penalty, fail to recognize a Hindi-English code-switched escalation, or silently make a promise nobody tracks. 

**Black Box** is an enterprise-grade reliability layer that wraps any voice agent's live call stream:
1. **Transcribes and Confidence-Gates Audio**: Calibrates acoustic confidence from ASR streams.
2. **Real-Time Fact-Grounding**: Cross-references every factual assertion against an authoritative corporate Knowledge Base.
3. **Automated Promises Ledger**: Detects, extracts, deduplicates, and tracks every commitment (callbacks, refunds, credits, hardware replacements) with SHA-256 state hashing.
4. **Emergency Simulated Human Handoff**: Triggers instant supervisor rerouting on critical policy violations, hallucinations, or regulatory threats (e.g. TRAI complaints).
5. **Resilient 4-Tier Fallback Chain**: Operates across Gemini Flash → Groq Llama-3 → Local Ollama → Offline Heuristics with autonomous circuit breakers and zero-latency failover.

---

## 2. Visual Interface & Telemetry HUD

Black Box features an ultra-futuristic **Neon Terminal** cyber-HUD designed with **Google Stitch**, featuring dual WebGL background & acoustic energy shaders, glassmorphism cards, and real-time SSE stream telemetry.

### 🔴 Live Turn-by-Turn Monitor & Emergency Handoff
![Live Monitor & Emergency Handoff Alert](docs/assets/live_monitor_guardrail.png)
*Live call stream capturing Turn 1 of TC-05 ("50% lifetime student discount") with real-time `[HALLUCINATION]` and `[PROMISE]` badges, pulsing emergency handoff banner, and live acoustic telemetry visualizer.*

---

### 🟣 Authoritative Promises Ledger
![Promises Ledger View](docs/assets/promises_ledger_view.png)
*Structured commitment audit table tracking agent action, target entity, deadlines, authorization limits, and deduplication state with instant search and CSV export.*

---

### 📊 20-Case Benchmark Evaluation Harness
![20-Case Benchmark Suite](docs/assets/eval_benchmark_suite.png)
*Automated benchmark evaluation across all 20 test cases with 95% Wilson score confidence intervals, per-category breakdown, and sub-millisecond scoring latencies.*

---

### 📚 Ground Truth Knowledge Base
![Knowledge Base Explorer](docs/assets/knowledge_base_explorer.png)
*Canonical repository of 15 corporate policies spanning rates, SLA limits, hardware warranties, agent goodwill caps, and regulatory protocols.*

---

## 3. System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│ FRONTEND — Stitch "Neon Terminal" Cyber-HUD (Live SSE Stream)          │
│  - Dual WebGL Shaders: Ambient grid background + Acoustic energy bars  │
│  - Turn-by-turn stream: [FACT_CHECK] [HALLUCINATION] [PROMISE] badges  │
│  - Multi-Tier Circuit Breaker Outage Toggles (Kill Gemini/Groq/Ollama) │
│  - Automated Promises Ledger with search filter & CSV export           │
│  - 20-Case Benchmark Engine with 95% Wilson Confidence Intervals       │
└────────────────────────────────────────────────────────────────────────┘
                                    │  SSE (Server-Sent Events)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ BACKEND — FastAPI Single-Service ASGI Engine (`backend/app/main.py`)    │
│  - Pipeline State Machine: UPLOADED -> TRANSCRIBING -> SCORING -> DONE │
│  - Audio validation (44-byte WAV header, RMS energy silence detection) │
│  - Replay buffer (50 sessions, 100 events) + automatic session init    │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 1 — ASR Pipeline (`backend/app/asr_pipeline.py`)                 │
│  - Whisper acoustic confidence calibration: math.exp(avg_logprob)      │
│  - Speaker turn parsing & word-level confidence gating                 │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 2 — Multi-Provider Fallback & Guardrail Engine                   │
│  - Tier 1: Gemini 2.5 Flash (Google AI Studio)                         │
│  - Tier 2: Groq Llama-3 70B (Fast sub-second LLM inference)           │
│  - Tier 3: Local Ollama (Local on-device fallback)                     │
│  - Tier 4: Zero-Crash Heuristic Scorer (<1ms latency)                  │
│                                                                        │
│  Circuit Breaker Lifecycle per tier: CLOSED -> OPEN -> HALF_OPEN       │
│  Validations executed per turn:                                        │
│    ✓ Fact Grounding vs 15-fact JSON Knowledge Base                     │
│    ✓ Promise Extraction & Deduplication with SHA-256 state machine     │
│    ✓ Hinglish & dialect code-switching intent preservation            │
│    ✓ Emergency Human Handoff escalation with cooldown suppression      │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STORAGE — SQLite in WAL Mode (`backend/blackbox.db`)                   │
│  - Tables: calls, turns, flags, promises, eval_results                 │
│  - PRAGMA foreign_keys = ON, PRAGMA busy_timeout = 5000                │
│  - Atomic INSERT OR IGNORE promise merge & transactional integrity     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Key Validations & Reliability Safeguards

1. **Audio & Header Validation**: Validates RIFF/WAVE header, 44-byte minimums, size limits (50MB), audio duration bounds, and RMS silence detection (`AudioValidator`).
2. **Calibrated ASR Confidence**: Converts Whisper `avg_logprob` to calibrated `[0.0, 1.0]` scores using `math.exp(min(0.0, avg_logprob))`.
3. **Knowledge Base Schema & Contradiction Detection**: Schema-enforces required fields, canonical value types, and checks pairwise contradictions at load time.
4. **LLM Output Schema Enforcement**: Retries and sanitizes raw LLM responses to conform to `llm_response_schema.json`, clamping confidence to `[0.0, 1.0]`.
5. **Thread-Safe Circuit Breakers**: Double-checked locking with `threading.Lock` across `CircuitBreakerRegistry` and `KnowledgeBaseRepository`.
6. **Promise State Machine & Deduplication**: State transitions (`PENDING -> FULFILLED | BROKEN | EXPIRED | DUPLICATE`), normalizing semantic duplicates via SHA-256 hashes.
7. **Negation & Fuzzy Numeric Fact Grounding**: Differentiates between assertions vs hedges and applies 5% tolerance on pricing matches.
8. **Severity-Weighted Handoff Escalation**: Triggers `[HUMAN HANDOFF]` on any CRITICAL flag or consecutive low-confidence turns, with a 5-turn cooldown to prevent notification spam.
9. **SSE Heartbeat & Sequence Replay**: Reconnects with `Last-Event-ID` replaying missed events from an in-memory ring buffer.
10. **Database Concurrency & Idempotency**: Foreign key cascades, `PRAGMA busy_timeout = 5000`, atomic `INSERT OR IGNORE` + re-`SELECT` for promise concurrency.
11. **Pipeline State Machine**: Validates lifecycle `UPLOADED -> TRANSCRIBING -> SCORING -> COMPLETE / FAILED` with single-write atomic state transitions.
12. **20-Case Benchmark with Wilson 95% Intervals**: Multiset count-aware matching for TP/FP/FN, reporting precision, recall, F1, and p50/p90/p99 latencies.
13. **Zero-Crash Standalone Fallback**: Runs 100% offline out-of-the-box using standard library Python and modern ES6 JavaScript.

---

## 5. 20-Case Benchmark Performance

Executed via `python backend/scripts/run_eval_cli.py`:

| Category | Cases | True Positives | False Positives | False Negatives | Precision | Recall | F1 Score |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **CLEAN_CALL** | 4 | 4 | 0 | 0 | **100.0%** | **100.0%** | **1.0000** |
| **HALLUCINATION** | 4 | 7 | 2 | 0 | **77.8%** | **100.0%** | **0.8750** |
| **PROMISES_LEDGER** | 4 | 7 | 0 | 0 | **100.0%** | **100.0%** | **1.0000** |
| **HINGLISH_CODE_SWITCH** | 4 | 2 | 0 | 0 | **100.0%** | **100.0%** | **1.0000** |
| **SAFETY_ESCALATION** | 4 | 8 | 3 | 0 | **72.7%** | **100.0%** | **0.8421** |
| **OVERALL BENCHMARK** | **20** | **28** | **5** | **0** | **84.9%** | **100.0%** | **0.9180** |

- **Verdict Match Accuracy**: **20 / 20 (100.0% Exact Ground Truth Match)**
- **95% Wilson Score Confidence Intervals**:
  - Precision: `[69.1%, 93.3%]`
  - Recall: `[87.9%, 100.0%]`
- **Scoring Latency**:
  - **p50 (Median)**: `0.04 ms`
  - **p90**: `0.05 ms`
  - **p99**: `0.53 ms`

---

## 6. Answers to the Five Questions

1. **Who exactly has this problem?**
   A QA lead / Compliance Director at a voice-AI or BPO company who currently samples and listens to calls manually to catch agent hallucinations, unauthorized commitments, and churn risks.
2. **Non-obvious hard part:**
   Telling apart a genuine hallucination from a correct answer phrased unusually or in a code-switched dialect (Hinglish) — requiring grounding against structured KB canonical values rather than fragile string matching.
3. **Built vs. API gave you:**
   The API gives raw transcription and generative tokens. We built the real-time fact-grounding logic, the promise-tracking ledger, the 4-tier circuit breaker fallback chain, the acoustic/semantic confidence gating, and the emergency handoff engine.
4. **Breaks without AI?**
   Completely — detecting subtly hallucinated commitments, code-switched intent, or negotiated concessions mid-conversation is impossible with rigid regex patterns alone.
5. **Breaks at 10k users?**
   Real-time scoring cost per call-minute and free-tier LLM rate limits. Black Box solves this with local small-model caching (Ollama) and tiered circuit-breaker routing that fails over to sub-millisecond heuristics.

---

## 7. How to Run

### Offline Standalone Mode (Zero Downloads / API Keys Required)
1. **Frontend**: Open `frontend/index.html` directly in any web browser.
2. **CLI Benchmark Runner**:
   ```bash
   python backend/scripts/run_eval_cli.py
   ```
3. **Validate Dataset Pre-Flight**:
   ```bash
   python backend/scripts/validate_dataset.py
   ```
4. **Run Complete Unit Test Suite (25 Tests)**:
   ```bash
   python -m unittest discover -s backend/tests -p "test_*.py" -v
   ```

### Full Live Mode (With FastAPI Server & Live Streaming)
1. Set optional API keys:
   ```bash
   export GEMINI_API_KEY="your_gemini_api_key"
   export GROQ_API_KEY="your_groq_api_key"
   ```
2. Start the backend server:
   ```bash
   python -m uvicorn backend.app.main:app --reload --port 8000
   ```
3. Open `frontend/index.html` in your browser.
