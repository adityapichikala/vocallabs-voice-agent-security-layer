"""
main.py
===============================================================================
Black Box -- Voice Agent Guardrail Layer
FastAPI backend  *  SSE streaming  *  Two-call LLM scoring  *  Fallback chain
===============================================================================

Pipeline per Agent turn
  Call 1 (Fact + Promise):  Is the claim grounded? Did agent make a commitment?
  Call 2 (Confidence):      Given Call 1 result, what is the risk level?
  Merge both JSON payloads -> single SSE event per turn

LLM fallback order
  Gemini 1.5 Flash  -->  Groq llama-3.1-8b-instant  -->  Ollama llama3.2:3b

force_outage behaviour
  When force_outage=True, the FIRST Call 1 of the first Agent turn deliberately
  skips Gemini (with a printed message) so the cascade is visible in the UI.
  Call 2 is never affected -- confidence scoring always uses best available LLM.

Run
  uvicorn main:app --reload --port 8000
"""

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import AsyncGenerator

# Force UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import google.genai as genai
import ollama
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from sse_starlette.sse import EventSourceResponse

# ==============================================================================
#  ENVIRONMENT & CLIENT SETUP
# ==============================================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY",   "")

# Gemini client (google.genai SDK -- replaces deprecated google-generativeai)
_gemini_client: "genai.Client | None" = None
if GEMINI_API_KEY:
    _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    print("[WARN] GEMINI_API_KEY not set -- Gemini provider will be skipped.")

# Groq client
groq_client: "Groq | None" = None
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    print("[WARN] GROQ_API_KEY not set -- Groq provider will be skipped.")

# ==============================================================================
#  CONSTANTS & KNOWLEDGE BASE
# ==============================================================================

TEST_CALLS_DIR      = Path("test_calls")
KNOWLEDGE_BASE_PATH = Path("knowledge_base.json")

GEMINI_MODEL   = "gemini-1.5-flash"
GROQ_LLM_MODEL = "llama-3.1-8b-instant"
OLLAMA_MODEL   = "llama3.2:3b"
WHISPER_MODEL  = "whisper-large-v3"

# Load KB at startup -- fail loudly if missing
if not KNOWLEDGE_BASE_PATH.exists():
    raise FileNotFoundError(
        f"knowledge_base.json not found at {KNOWLEDGE_BASE_PATH.resolve()}. "
        "Run project setup first."
    )

with KNOWLEDGE_BASE_PATH.open(encoding="utf-8") as _f:
    KNOWLEDGE_BASE: dict = json.load(_f)

_KB_RULES_TEXT = "\n".join(
    f"  [{r['rule_id']}] {r['topic']}: {r['rule']}"
    for r in KNOWLEDGE_BASE.get("rules", [])
)

# ==============================================================================
#  FASTAPI APP & CORS
# ==============================================================================

app = FastAPI(
    title       = "Black Box -- Voice Agent Guardrail Layer",
    description = "Real-time SSE streaming analysis of AI voice-agent calls.",
    version     = "1.0.0",
)

# Allow all origins for the hackathon demo
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ==============================================================================
#  LLM FALLBACK CHAIN
# ==============================================================================

async def call_llm(
    prompt: str,
    force_outage: bool = False,
    _is_first_call: bool = True,
) -> tuple[str, str]:
    """
    Call LLMs in priority order: Gemini -> Groq -> Ollama (local).

    Parameters
    ----------
    prompt         : Full prompt text.
    force_outage   : When True AND _is_first_call=True, Gemini is intentionally
                     skipped with a printed simulation message so the cascade is
                     visible in the UI. Call 2 always passes _is_first_call=False
                     so only the first Call-1 of the first Agent turn triggers it.
    _is_first_call : Internal flag managed by score_turn; leave as True for
                     direct calls.

    Returns
    -------
    (response_text, provider_name)   provider: "gemini" | "groq" | "local"

    Raises
    ------
    RuntimeError if all three providers fail.
    """

    # -- Provider 1: Gemini ----------------------------------------------------
    simulate_gemini_failure = force_outage and _is_first_call

    if not simulate_gemini_failure and _gemini_client:
        try:
            t0 = time.perf_counter()
            response = await asyncio.to_thread(
                _gemini_client.models.generate_content,
                model    = GEMINI_MODEL,
                contents = prompt,
            )
            elapsed = time.perf_counter() - t0
            print(f"[LLM] Gemini responded in {elapsed:.2f}s")
            return response.text.strip(), "gemini"
        except Exception as exc:
            print(f"[LLM] Gemini failed ({type(exc).__name__}: {exc}). "
                  "Falling back to Groq...")
    elif simulate_gemini_failure:
        print("[LLM] force_outage=True -> simulating Gemini failure on Call 1.")

    # -- Provider 2: Groq (llama-3.1-8b-instant) -------------------------------
    if groq_client:
        try:
            t0 = time.perf_counter()
            completion = await asyncio.to_thread(
                groq_client.chat.completions.create,
                messages    = [{"role": "user", "content": prompt}],
                model       = GROQ_LLM_MODEL,
                temperature = 0.1,
                max_tokens  = 512,
            )
            elapsed = time.perf_counter() - t0
            text = completion.choices[0].message.content.strip()
            print(f"[LLM] Groq responded in {elapsed:.2f}s")
            return text, "groq"
        except Exception as exc:
            print(f"[LLM] Groq failed ({type(exc).__name__}: {exc}). "
                  "Falling back to local Ollama...")

    # -- Provider 3: Ollama (local llama3.2:3b) --------------------------------
    try:
        t0 = time.perf_counter()
        response = await asyncio.to_thread(
            ollama.chat,
            model    = OLLAMA_MODEL,
            messages = [{"role": "user", "content": prompt}],
        )
        elapsed = time.perf_counter() - t0
        text = response["message"]["content"].strip()
        print(f"[LLM] Ollama responded in {elapsed:.2f}s")
        return text, "local"
    except Exception as exc:
        raise RuntimeError(
            f"All LLM providers exhausted. Last Ollama error: {exc}"
        ) from exc

# ==============================================================================
#  TRANSCRIPTION
# ==============================================================================

async def transcribe_audio(wav_path: Path) -> str:
    """
    Transcribe a WAV file to text.

    Strategy
    --------
    1. Groq Whisper (whisper-large-v3) -- primary.
    2. Same-name .txt file in test_calls/ -- fallback for offline dev / demos.
    """
    if groq_client:
        try:
            print(f"[Transcription] Sending {wav_path.name} to Groq Whisper...")
            with wav_path.open("rb") as audio_file:
                transcription = await asyncio.to_thread(
                    groq_client.audio.transcriptions.create,
                    file            = (wav_path.name, audio_file, "audio/wav"),
                    model           = WHISPER_MODEL,
                    response_format = "text",
                    language        = "en",
                )
            print(f"[Transcription] Groq Whisper OK ({len(transcription)} chars)")
            return transcription
        except Exception as exc:
            print(f"[Transcription] Groq Whisper failed: {exc}. Trying .txt fallback...")

    # Fallback: pre-saved transcript file (same stem, .txt)
    txt_path = wav_path.with_suffix(".txt")
    if txt_path.exists():
        text = txt_path.read_text(encoding="utf-8")
        print(f"[Transcription] Loaded fallback from {txt_path.name} ({len(text)} chars)")
        return text

    raise HTTPException(
        status_code = 500,
        detail      = (
            f"Cannot transcribe '{wav_path.name}': Groq Whisper unavailable "
            f"and no fallback '{wav_path.stem}.txt' found in test_calls/."
        ),
    )

# ==============================================================================
#  TRANSCRIPT PARSING
# ==============================================================================

def parse_transcript(raw_text: str) -> list[dict]:
    """
    Parse raw transcript into a list of turn dicts.

    Formats supported
    -----------------
    A) Labelled:   "Customer: ..." / "Agent: ..."
    B) Unlabelled: alternate Customer/Agent (Whisper output)
    """
    lines   = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    labeled = any(
        re.match(r"^(customer|agent)\s*:", ln, re.IGNORECASE)
        for ln in lines
    )

    turns: list[dict] = []

    if labeled:
        for line in lines:
            m = re.match(r"^(customer|agent)\s*:\s*(.+)$", line, re.IGNORECASE)
            if m:
                turns.append({
                    "speaker": m.group(1).capitalize(),
                    "text"   : m.group(2).strip(),
                })
    else:
        speakers = ["Customer", "Agent"]
        for i, line in enumerate(lines):
            turns.append({"speaker": speakers[i % 2], "text": line})

    return turns

# ==============================================================================
#  SCORING SYSTEM -- TWO CALLS PER AGENT TURN
# ==============================================================================
#
#  Call 1 (Fact + Promise merged)
#    System: "You are a monitor. Check this agent turn against the Knowledge
#             Base (JSON). Check for factual claims (GROUNDED/CONTRADICTED/
#             UNVERIFIABLE) AND check for commitments/promises."
#    Returns: has_claim, claim_status, matched_fact_id, promise_made,
#             promise_action, timeframe, conflicts_with_policy,
#             policy_conflict_detail
#
#  Call 2 (Confidence Scorer)
#    System: "You are a risk-assessor. Given the agent text and the Call 1
#             JSON result, score confidence. Set LOW if fact is CONTRADICTED/
#             UNVERIFIABLE, OR promise conflicts with policy, OR agent uses
#             hedging."
#    Returns: confidence, route_to_human, reason
#
#  Both results are merged into one SSE payload per Agent turn.
# ==============================================================================


def _extract_json(raw: str) -> dict:
    """
    Robustly extract the first {...} JSON object from an LLM response.
    Strips markdown fences, surrounding prose, and common single-quote issues.
    """
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON object found in LLM output: {raw[:120]}")
    candidate = m.group()
    # Replace Python-style single quotes with double quotes (cautiously)
    candidate = re.sub(r"(?<![\\])'", '"', candidate)
    return json.loads(candidate)


def _build_fact_promise_prompt(
    agent_text    : str,
    context_turns : list[dict],
) -> str:
    """
    Call 1 system prompt: Fact-checking + promise detection combined.

    You are a monitor for a telecom call centre.
    Check for factual claims (GROUNDED/CONTRADICTED/UNVERIFIABLE/NO_CLAIM)
    AND check for commitments/promises simultaneously.
    """
    context_str = "\n".join(
        f"  {t['speaker']}: {t['text']}" for t in context_turns
    ) or "  (start of call)"

    return f"""You are a monitor for a telecom call centre.
Check the Agent's response against the FastNet Knowledge Base below.
Do TWO checks simultaneously: (1) factual claims, (2) promises/commitments.

=== FastNet Knowledge Base ===
{json.dumps(KNOWLEDGE_BASE, ensure_ascii=False, indent=2)}

=== Recent Conversation Context ===
{context_str}

=== Agent Response ===
"{agent_text}"

=== Instructions ===
Return ONLY valid JSON. No markdown. No prose. No extra keys.

{{
  "has_claim"              : false,
  "claim_status"           : "NO_CLAIM",
  "matched_fact_id"        : null,
  "promise_made"           : false,
  "promise_action"         : null,
  "timeframe"              : null,
  "conflicts_with_policy"  : false,
  "policy_conflict_detail" : null
}}

Field rules:
- has_claim             : true if the agent states any factual claim about FastNet policy, speeds, fees, timelines, or discounts.
- claim_status          : "GROUNDED" (claim exactly matches KB) | "CONTRADICTED" (claim contradicts KB) | "UNVERIFIABLE" (claim not in KB) | "NO_CLAIM" (no factual claim made).
- matched_fact_id       : The rule_id from the KB that was cited or violated (e.g. "R01"), or null.
- promise_made          : true if agent makes a concrete commitment (callback time, refund, escalation, resolution, discount).
- promise_action        : short description of what was promised, or null.
- timeframe             : any time commitment stated by the agent (e.g. "30 minutes", "5-7 days"), or null.
- conflicts_with_policy : true if the promise or claim directly contradicts a KB rule.
- policy_conflict_detail: one sentence explaining the conflict, or null."""


def _build_confidence_prompt(
    agent_text  : str,
    call1_result: dict,
) -> str:
    """
    Call 2 system prompt: Confidence + risk scorer.

    You are a risk-assessor. Given the agent text and the Call 1 JSON result,
    score confidence. Set LOW if fact is CONTRADICTED/UNVERIFIABLE, OR promise
    conflicts with policy, OR agent uses hedging language.
    """
    return f"""You are a risk-assessor for a telecom call centre.
Given the agent's response and the Fact+Promise analysis below, score the confidence level and escalation need.

=== Agent Response ===
"{agent_text}"

=== Fact + Promise Analysis (from previous check) ===
{json.dumps(call1_result, ensure_ascii=False, indent=2)}

=== Instructions ===
Return ONLY valid JSON. No markdown. No prose. No extra keys.

{{
  "confidence"     : "HIGH",
  "route_to_human" : false,
  "reason"         : ""
}}

Field rules:
- confidence     : Set "LOW" if claim_status is CONTRADICTED or UNVERIFIABLE, OR if conflicts_with_policy is true, OR if agent used hedging words ("I think", "probably", "maybe", "I'm not sure", "I believe").
                   Set "MEDIUM" if agent was slightly vague but not incorrect.
                   Set "HIGH" only if agent was factually correct, spoke clearly, and made no policy-violating promises.
- route_to_human : true if confidence is LOW, OR if a policy was directly contradicted, OR if the customer seems very upset.
- reason         : Empty string if no issues. Otherwise one concise sentence explaining the primary problem."""


async def score_turn(
    agent_text   : str,
    context_turns: list[dict],
    force_outage : bool,
) -> tuple[dict, str]:
    """
    Run the two-call scoring pipeline for a single Agent turn.

    Parameters
    ----------
    agent_text    : The Agent's utterance to evaluate.
    context_turns : Recent turns (up to 4) before this one, for context.
    force_outage  : Passed through to call_llm; only affects Call 1.

    Returns
    -------
    (merged_scores, provider)
      merged_scores contains all fields from Call 1 + Call 2 + derived flag_type.
      provider is the LLM that serviced Call 1.
    """

    # ---- Call 1: Fact + Promise check ----------------------------------------
    prompt1 = _build_fact_promise_prompt(agent_text, context_turns)
    try:
        raw1, provider = await call_llm(
            prompt1,
            force_outage   = force_outage,
            _is_first_call = True,          # may trigger Gemini cascade demo
        )
        call1 = _extract_json(raw1)
        print(f"[Score/C1] provider={provider}  "
              f"claim_status={call1.get('claim_status')}  "
              f"promise={call1.get('promise_made')}  "
              f"conflict={call1.get('conflicts_with_policy')}")
    except Exception as exc:
        print(f"[Score/C1] Failed: {exc}")
        call1 = {
            "has_claim"             : False,
            "claim_status"          : "UNVERIFIABLE",
            "matched_fact_id"       : None,
            "promise_made"          : False,
            "promise_action"        : None,
            "timeframe"             : None,
            "conflicts_with_policy" : False,
            "policy_conflict_detail": f"Call 1 failed: {exc}",
        }
        provider = "error"

    # ---- Call 2: Confidence + Risk -------------------------------------------
    # Never uses force_outage -- accurate confidence scoring even during demo.
    prompt2 = _build_confidence_prompt(agent_text, call1)
    try:
        raw2, _ = await call_llm(
            prompt2,
            force_outage   = False,
            _is_first_call = False,         # skip force_outage simulation
        )
        call2 = _extract_json(raw2)
        print(f"[Score/C2] confidence={call2.get('confidence')}  "
              f"route_to_human={call2.get('route_to_human')}")
    except Exception as exc:
        print(f"[Score/C2] Failed: {exc}")
        call2 = {
            "confidence"     : "LOW",
            "route_to_human" : True,
            "reason"         : f"Confidence scoring failed: {exc}",
        }

    # ---- Derive flag_type from merged results ---------------------------------
    flag_type = ""
    claim_status       = call1.get("claim_status", "NO_CLAIM")
    conflicts_policy   = call1.get("conflicts_with_policy", False)
    confidence         = call2.get("confidence", "MEDIUM")

    if claim_status == "CONTRADICTED":
        flag_type = "hallucination"
    elif conflicts_policy:
        flag_type = "promise_conflict"
    elif confidence == "LOW":
        flag_type = "low_confidence"

    # ---- Merge ----------------------------------------------------------------
    merged = {
        # Call 1 fields
        "has_claim"             : call1.get("has_claim",              False),
        "claim_status"          : claim_status,
        "matched_fact_id"       : call1.get("matched_fact_id",        None),
        "promise_made"          : call1.get("promise_made",           False),
        "promise_action"        : call1.get("promise_action",         None),
        "timeframe"             : call1.get("timeframe",              None),
        "conflicts_with_policy" : conflicts_policy,
        "policy_conflict_detail": call1.get("policy_conflict_detail", None),
        # Call 2 fields
        "confidence"            : confidence,
        "route_to_human"        : call2.get("route_to_human", False),
        "reason"                : call2.get("reason",         ""),
        # Derived
        "flag_type"             : flag_type,
    }
    return merged, provider

# ==============================================================================
#  SSE EVENT BUILDER
# ==============================================================================

def make_turn_event(
    turn_index             : int,
    speaker                : str,
    text                   : str,
    provider               : str       = "n/a",
    # Call 1 fields
    has_claim              : bool      = False,
    claim_status           : str       = "NO_CLAIM",
    matched_fact_id        : "str|None" = None,
    promise_made           : bool      = False,
    promise_action         : "str|None" = None,
    timeframe              : "str|None" = None,
    conflicts_with_policy  : bool      = False,
    policy_conflict_detail : "str|None" = None,
    # Call 2 fields
    confidence             : str  = "n/a",
    route_to_human         : bool = False,
    reason                 : str  = "",
    # Derived
    flag_type              : str  = "",
) -> dict:
    """Construct the full merged SSE event payload."""
    return {
        # Identity
        "turn_index"             : turn_index,
        "speaker"                : speaker,
        "text"                   : text,
        "provider"               : provider,
        # Call 1 -- Fact + Promise
        "has_claim"              : has_claim,
        "claim_status"           : claim_status,
        "matched_fact_id"        : matched_fact_id,
        "promise_made"           : promise_made,
        "promise_action"         : promise_action,
        "timeframe"              : timeframe,
        "conflicts_with_policy"  : conflicts_with_policy,
        "policy_conflict_detail" : policy_conflict_detail,
        # Call 2 -- Confidence + Risk
        "confidence"             : confidence,
        "route_to_human"         : route_to_human,
        "reason"                 : reason,
        # Derived
        "flag_type"              : flag_type,
    }

# ==============================================================================
#  CORE STREAMING GENERATOR
# ==============================================================================

async def analyze_stream(
    filename    : str,
    force_outage: bool,
) -> AsyncGenerator[str | dict, None]:
    """
    Full pipeline: transcribe -> parse -> score (2 LLM calls) -> stream.
    Yields dicts (which EventSourceResponse handles) or raw strings.
    A keep-alive ping is sent every 5 seconds if the LLM/ASR blocks.
    """

    queue = asyncio.Queue()

    async def worker():
        try:
            wav_path = TEST_CALLS_DIR / filename
            if not wav_path.exists():
                await queue.put({"data": json.dumps({"error": f"File not found: {filename}"})})
                return

            # -- Step 1: Transcribe ----------------------------------------------------
            await queue.put({"data": json.dumps({"status": "transcribing", "file": filename})})
            try:
                raw_transcript = await transcribe_audio(wav_path)
            except HTTPException as exc:
                await queue.put({"data": json.dumps({"error": exc.detail})})
                return
            except Exception as exc:
                await queue.put({"data": json.dumps({"error": f"Transcription error: {exc}"})})
                return

            await queue.put({"data": json.dumps({"status": "transcribed", "chars": len(raw_transcript)})})

            # -- Step 2: Parse turns ---------------------------------------------------
            turns = parse_transcript(raw_transcript)
            await queue.put({"data": json.dumps({"status": "parsed", "turn_count": len(turns)})})

            # -- Step 3: Score each Agent turn & stream --------------------------------
            first_agent_turn = True

            for i, turn in enumerate(turns):
                speaker = turn["speaker"]
                text    = turn["text"]

                # Customer turns: pass through immediately with no scoring
                if speaker == "Customer":
                    await queue.put({"data": json.dumps(make_turn_event(
                        turn_index = i,
                        speaker    = "Customer",
                        text       = text,
                    ))})
                    continue

                # Agent turn: run two-call scoring pipeline
                context_turns = turns[max(0, i - 4): i]
                use_outage    = force_outage and first_agent_turn
                first_agent_turn = False

                try:
                    scores, provider = await score_turn(
                        agent_text    = text,
                        context_turns = context_turns,
                        force_outage  = use_outage,
                    )
                except RuntimeError as exc:
                    await queue.put({"data": json.dumps(make_turn_event(
                        turn_index     = i,
                        speaker        = "Agent",
                        text           = text,
                        provider       = "error",
                        confidence     = "LOW",
                        reason         = f"All LLM providers failed: {exc}",
                        route_to_human = True,
                    ))})
                    continue

                # Emit fully merged event
                await queue.put({"data": json.dumps(make_turn_event(
                    turn_index             = i,
                    speaker                = "Agent",
                    text                   = text,
                    provider               = provider,
                    has_claim              = scores["has_claim"],
                    claim_status           = scores["claim_status"],
                    matched_fact_id        = scores["matched_fact_id"],
                    promise_made           = scores["promise_made"],
                    promise_action         = scores["promise_action"],
                    timeframe              = scores["timeframe"],
                    conflicts_with_policy  = scores["conflicts_with_policy"],
                    policy_conflict_detail = scores["policy_conflict_detail"],
                    confidence             = scores["confidence"],
                    route_to_human         = scores["route_to_human"],
                    reason                 = scores["reason"],
                    flag_type              = scores["flag_type"],
                ))})

            # -- Done ------------------------------------------------------------------
            await queue.put({"data": json.dumps({"status": "done", "total_turns": len(turns)})})

        finally:
            await queue.put(None)  # Sentinel

    # Start the worker in the background
    asyncio.create_task(worker())

    # Yield from queue, emitting a keep-alive if 5 seconds pass with no items
    while True:
        try:
            item = await asyncio.wait_for(queue.get(), timeout=5.0)
            if item is None:
                break
            yield item
        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'ping': 'keep-alive'})}\n\n"



# ==============================================================================
#  ROUTES
# ==============================================================================

@app.get("/analyze")
@app.post("/analyze")
async def analyze(
    filename    : str  = Query(...,   description="WAV filename inside test_calls/ e.g. test_2_hallucination.wav"),
    force_outage: bool = Query(False, description="Simulate Gemini failure on first Call 1 to demo cascade"),
):
    """
    Stream real-time guardrail analysis via Server-Sent Events.

    Two LLM calls are made per Agent turn and merged into one event:

    **Call 1 -- Fact + Promise** (system: "You are a monitor...")
    ```json
    {
      "has_claim"              : true,
      "claim_status"           : "CONTRADICTED",
      "matched_fact_id"        : "R01",
      "promise_made"           : true,
      "promise_action"         : "full refund",
      "timeframe"              : null,
      "conflicts_with_policy"  : true,
      "policy_conflict_detail" : "Refunds only apply for outages > 24h; this outage was 6h."
    }
    ```

    **Call 2 -- Confidence + Risk** (system: "You are a risk-assessor...")
    ```json
    {
      "confidence"     : "LOW",
      "route_to_human" : true,
      "reason"         : "Agent promised a refund that violates policy R01."
    }
    ```

    **Merged SSE payload (one event per turn)**
    ```json
    {
      "turn_index"             : 3,
      "speaker"                : "Agent",
      "text"                   : "I can process a full refund for you.",
      "provider"               : "groq",
      "has_claim"              : true,
      "claim_status"           : "CONTRADICTED",
      "matched_fact_id"        : "R01",
      "promise_made"           : true,
      "promise_action"         : "full refund",
      "timeframe"              : null,
      "conflicts_with_policy"  : true,
      "policy_conflict_detail" : "Refunds only apply for outages > 24h.",
      "confidence"             : "LOW",
      "route_to_human"         : true,
      "reason"                 : "Agent promised a refund that violates policy R01.",
      "flag_type"              : "hallucination"
    }
    ```

    Status frames emitted: `transcribing` -> `transcribed` -> `parsed` -> `done`.

    Set `force_outage=true` to simulate a Gemini outage and watch Groq take over.
    """
    return EventSourceResponse(analyze_stream(filename, force_outage))
    
@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """Serve the audio file for frontend playback."""
    from fastapi.responses import FileResponse
    path = TEST_CALLS_DIR / filename
    if not path.exists():
        raise HTTPException(404, "Audio file not found")
    return FileResponse(path)

@app.get("/")
async def root():
    """Service metadata and quick-start reference."""
    return {
        "service" : "Black Box -- Voice Agent Guardrail Layer",
        "version" : "1.0.0",
        "status"  : "ok",
        "usage"   : {
            "analyze"      : "GET /analyze?filename=test_1_clean.wav",
            "force_outage" : "GET /analyze?filename=test_2_hallucination.wav&force_outage=true",
            "health"       : "GET /health",
            "docs"         : "GET /docs",
        },
    }


@app.get("/health")
async def health():
    """Diagnostic endpoint -- provider availability and discovered test files."""
    test_files = sorted(f.name for f in TEST_CALLS_DIR.glob("*.wav"))
    return {
        "gemini_configured" : bool(GEMINI_API_KEY),
        "groq_configured"   : bool(GROQ_API_KEY),
        "ollama_model"      : OLLAMA_MODEL,
        "knowledge_rules"   : len(KNOWLEDGE_BASE.get("rules", [])),
        "test_calls_found"  : test_files,
        "test_calls_dir"    : str(TEST_CALLS_DIR.resolve()),
    }
