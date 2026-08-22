"""SQLite Database connection, table schemas, and transactional CRUD."""
import sqlite3
import json
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from .config import settings
from .models import (
    CallSession, Turn, Flag, Promise, CallState,
    VerdictType, PromiseState, EvalMetrics
)

# Default busy timeout (ms) to let SQLite retry internally on SQLITE_BUSY
_BUSY_TIMEOUT_MS = 5000

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(
        str(settings.db_path),
        check_same_thread=False,
        timeout=_BUSY_TIMEOUT_MS / 1000.0
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS};")
    return conn

def init_db():
    conn = get_connection()
    with conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS calls (
            id TEXT PRIMARY KEY,
            audio_hash TEXT UNIQUE NOT NULL,
            filename TEXT NOT NULL,
            duration_seconds REAL CHECK(duration_seconds > 0),
            state TEXT NOT NULL DEFAULT 'UPLOADED'
                CHECK(state IN ('UPLOADED','TRANSCRIBING','SCORING','COMPLETE','FAILED')),
            verdict TEXT NOT NULL DEFAULT 'PASS'
                CHECK(verdict IN ('PASS','FAIL_FLAGGED','ESCALATED')),
            handoff_triggered INTEGER NOT NULL DEFAULT 0,
            handoff_reason TEXT,
            created_at REAL NOT NULL,
            completed_at REAL,
            error_message TEXT
        );

        CREATE TABLE IF NOT EXISTS turns (
            id TEXT PRIMARY KEY,
            call_id TEXT NOT NULL REFERENCES calls(id) ON DELETE CASCADE,
            turn_index INTEGER NOT NULL,
            speaker TEXT NOT NULL CHECK(speaker IN ('agent','customer')),
            text TEXT NOT NULL,
            start_time REAL NOT NULL,
            end_time REAL NOT NULL CHECK(end_time >= start_time),
            asr_confidence REAL CHECK(asr_confidence BETWEEN 0.0 AND 1.0),
            language_detected TEXT,
            UNIQUE(call_id, turn_index)
        );

        CREATE TABLE IF NOT EXISTS flags (
            id TEXT PRIMARY KEY,
            call_id TEXT NOT NULL REFERENCES calls(id) ON DELETE CASCADE,
            turn_id TEXT NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
            turn_index INTEGER NOT NULL,
            flag_type TEXT NOT NULL,
            severity TEXT NOT NULL CHECK(severity IN ('LOW','MEDIUM','HIGH','CRITICAL')),
            detail TEXT NOT NULL,
            kb_fact_id TEXT,
            claimed_value TEXT,
            actual_value TEXT,
            provider_used TEXT,
            timestamp REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS promises (
            id TEXT PRIMARY KEY,
            call_id TEXT NOT NULL REFERENCES calls(id) ON DELETE CASCADE,
            turn_id TEXT NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
            turn_index INTEGER NOT NULL,
            promise_hash TEXT NOT NULL,
            who TEXT NOT NULL CHECK(who IN ('agent','customer')),
            action TEXT NOT NULL,
            target_entity TEXT NOT NULL,
            deadline_raw TEXT,
            deadline_parsed_iso TEXT,
            condition TEXT,
            is_authorized INTEGER NOT NULL DEFAULT 1,
            violation_reason TEXT,
            state TEXT NOT NULL DEFAULT 'PENDING'
                CHECK(state IN ('PENDING','FULFILLED','BROKEN','EXPIRED','DUPLICATE')),
            mention_count INTEGER NOT NULL DEFAULT 1,
            merged_into TEXT REFERENCES promises(id),
            confidence REAL NOT NULL DEFAULT 1.0,
            timestamp REAL NOT NULL,
            UNIQUE(call_id, promise_hash)
        );

        CREATE TABLE IF NOT EXISTS eval_results (
            id TEXT PRIMARY KEY,
            test_case_id TEXT NOT NULL,
            category TEXT NOT NULL,
            true_positives INTEGER NOT NULL DEFAULT 0,
            false_positives INTEGER NOT NULL DEFAULT 0,
            false_negatives INTEGER NOT NULL DEFAULT 0,
            precision REAL,
            recall REAL,
            f1 REAL,
            latency_ms REAL,
            provider_used TEXT,
            verdict_matched INTEGER NOT NULL DEFAULT 1,
            handoff_matched INTEGER NOT NULL DEFAULT 1,
            is_curveball_run INTEGER NOT NULL DEFAULT 0,
            run_timestamp REAL NOT NULL
        );
        """)
    conn.close()

class DatabaseManager:
    @staticmethod
    def create_or_get_call(call_id: str, audio_hash: str, filename: str, duration: float) -> CallSession:
        conn = get_connection()
        try:
            with conn:
                existing = conn.execute("SELECT * FROM calls WHERE audio_hash = ?", (audio_hash,)).fetchone()
                if existing:
                    return DatabaseManager.get_call(existing["id"])
                
                now = time.time()
                conn.execute(
                    """INSERT INTO calls (id, audio_hash, filename, duration_seconds, state, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (call_id, audio_hash, filename, duration, CallState.UPLOADED.value, now)
                )
                return CallSession(
                    id=call_id,
                    audio_hash=audio_hash,
                    filename=filename,
                    duration_seconds=duration,
                    state=CallState.UPLOADED.value,
                    created_at=now
                )
        finally:
            conn.close()

    @staticmethod
    def update_call_state(call_id: str, state: str, error_message: Optional[str] = None,
                          verdict: Optional[str] = None, handoff_triggered: Optional[bool] = None,
                          handoff_reason: Optional[str] = None):
        conn = get_connection()
        try:
            with conn:
                # SAFETY: `updates` list is built exclusively from hardcoded column names below.
                # All user-supplied values go through parameterized `?` placeholders in `params`.
                updates = ["state = ?"]
                params: list = [state]
                if error_message is not None:
                    updates.append("error_message = ?")
                    params.append(error_message)
                if verdict is not None:
                    updates.append("verdict = ?")
                    params.append(verdict)
                if handoff_triggered is not None:
                    updates.append("handoff_triggered = ?")
                    params.append(1 if handoff_triggered else 0)
                if handoff_reason is not None:
                    updates.append("handoff_reason = ?")
                    params.append(handoff_reason)
                if state in (CallState.COMPLETE.value, CallState.FAILED.value):
                    updates.append("completed_at = ?")
                    params.append(time.time())
                
                params.append(call_id)
                conn.execute(f"UPDATE calls SET {', '.join(updates)} WHERE id = ?", params)
        finally:
            conn.close()

    @staticmethod
    def save_turns(call_id: str, turns: List[Turn]):
        conn = get_connection()
        try:
            with conn:
                for t in turns:
                    conn.execute(
                        """INSERT OR REPLACE INTO turns
                           (id, call_id, turn_index, speaker, text, start_time, end_time, asr_confidence, language_detected)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (t.id, call_id, t.turn_index, t.speaker, t.text, t.start_time, t.end_time, t.asr_confidence, t.language_detected)
                    )
        finally:
            conn.close()

    @staticmethod
    def save_flag(flag: Flag):
        conn = get_connection()
        try:
            with conn:
                conn.execute(
                    """INSERT OR REPLACE INTO flags
                       (id, call_id, turn_id, turn_index, flag_type, severity, detail, kb_fact_id, claimed_value, actual_value, provider_used, timestamp)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (flag.id, flag.call_id, flag.turn_id, flag.turn_index, flag.flag_type, flag.severity, flag.detail,
                     flag.kb_fact_id, flag.claimed_value, flag.actual_value, flag.provider_used, flag.timestamp)
                )
        finally:
            conn.close()

    @staticmethod
    def save_or_merge_promise(promise: Promise) -> Promise:
        """Saves a new promise or merges into an existing one by promise_hash.
        Uses INSERT OR IGNORE + re-SELECT to avoid the TOCTOU race where two concurrent
        turns with the same hash both see existing=None and both try to INSERT."""
        conn = get_connection()
        try:
            with conn:
                # Attempt insert first; IGNORE silently skips on UNIQUE(call_id, promise_hash) conflict
                conn.execute(
                    """INSERT OR IGNORE INTO promises
                       (id, call_id, turn_id, turn_index, promise_hash, who, action, target_entity,
                        deadline_raw, deadline_parsed_iso, condition, is_authorized, violation_reason,
                        state, mention_count, merged_into, confidence, timestamp)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (promise.id, promise.call_id, promise.turn_id, promise.turn_index, promise.promise_hash,
                     promise.who, promise.action, promise.target_entity, promise.deadline_raw,
                     promise.deadline_parsed_iso, promise.condition, 1 if promise.is_authorized else 0,
                     promise.violation_reason, promise.state, promise.mention_count, promise.merged_into,
                     promise.confidence, promise.timestamp)
                )

                # Check if this was a new insert or a conflict (existing row)
                existing = conn.execute(
                    "SELECT * FROM promises WHERE call_id = ? AND promise_hash = ?",
                    (promise.call_id, promise.promise_hash)
                ).fetchone()

                if existing and existing["id"] != promise.id:
                    # Row existed before our insert — merge by incrementing mention_count
                    new_count = existing["mention_count"] + 1
                    conn.execute(
                        "UPDATE promises SET mention_count = ? WHERE id = ?",
                        (new_count, existing["id"])
                    )
                    promise.id = existing["id"]
                    promise.mention_count = new_count
                    promise.state = PromiseState.DUPLICATE.value

                return promise
        finally:
            conn.close()

    @staticmethod
    def get_call(call_id: str) -> Optional[CallSession]:
        conn = get_connection()
        try:
            row = conn.execute("SELECT * FROM calls WHERE id = ?", (call_id,)).fetchone()
            if not row:
                return None
            
            turns_rows = conn.execute("SELECT * FROM turns WHERE call_id = ? ORDER BY turn_index ASC", (call_id,)).fetchall()
            flags_rows = conn.execute("SELECT * FROM flags WHERE call_id = ? ORDER BY turn_index ASC", (call_id,)).fetchall()
            promises_rows = conn.execute("SELECT * FROM promises WHERE call_id = ? ORDER BY turn_index ASC", (call_id,)).fetchall()
            
            turns = [
                Turn(
                    id=t["id"],
                    call_id=t["call_id"],
                    turn_index=t["turn_index"],
                    speaker=t["speaker"],
                    text=t["text"],
                    start_time=t["start_time"],
                    end_time=t["end_time"],
                    asr_confidence=t["asr_confidence"] if t["asr_confidence"] is not None else 1.0,
                    language_detected=t["language_detected"]
                ) for t in turns_rows
            ]
            
            flags = [
                Flag(
                    id=f["id"],
                    call_id=f["call_id"],
                    turn_id=f["turn_id"],
                    turn_index=f["turn_index"],
                    flag_type=f["flag_type"],
                    severity=f["severity"],
                    detail=f["detail"],
                    kb_fact_id=f["kb_fact_id"],
                    claimed_value=f["claimed_value"],
                    actual_value=f["actual_value"],
                    provider_used=f["provider_used"],
                    timestamp=f["timestamp"]
                ) for f in flags_rows
            ]
            
            promises = [
                Promise(
                    id=p["id"],
                    call_id=p["call_id"],
                    turn_id=p["turn_id"],
                    turn_index=p["turn_index"],
                    promise_hash=p["promise_hash"],
                    who=p["who"],
                    action=p["action"],
                    target_entity=p["target_entity"],
                    deadline_raw=p["deadline_raw"],
                    deadline_parsed_iso=p["deadline_parsed_iso"],
                    condition=p["condition"],
                    is_authorized=bool(p["is_authorized"]),
                    violation_reason=p["violation_reason"],
                    state=p["state"],
                    mention_count=p["mention_count"],
                    merged_into=p["merged_into"],
                    confidence=p["confidence"],
                    timestamp=p["timestamp"]
                ) for p in promises_rows
            ]
            
            return CallSession(
                id=row["id"],
                audio_hash=row["audio_hash"],
                filename=row["filename"],
                duration_seconds=row["duration_seconds"],
                state=row["state"],
                turns=turns,
                flags=flags,
                promises=promises,
                verdict=row["verdict"],
                handoff_triggered=bool(row["handoff_triggered"]),
                handoff_reason=row["handoff_reason"],
                created_at=row["created_at"],
                completed_at=row["completed_at"],
                error_message=row["error_message"]
            )
        finally:
            conn.close()

    @staticmethod
    def get_all_calls() -> List[Dict[str, Any]]:
        conn = get_connection()
        try:
            rows = conn.execute("SELECT * FROM calls ORDER BY created_at DESC").fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    @staticmethod
    def save_eval_result(res: EvalMetrics):
        conn = get_connection()
        try:
            with conn:
                rec_id = f"eval_{res.test_case_id}_{int(time.time()*1000)}"
                conn.execute(
                    """INSERT INTO eval_results
                       (id, test_case_id, category, true_positives, false_positives, false_negatives,
                        precision, recall, f1, latency_ms, provider_used, verdict_matched, handoff_matched,
                        is_curveball_run, run_timestamp)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (rec_id, res.test_case_id, res.category, res.true_positives, res.false_positives,
                     res.false_negatives, res.precision, res.recall, res.f1, res.latency_ms,
                     res.provider_used, 1 if res.verdict_matched else 0, 1 if res.handoff_matched else 0,
                     1 if res.is_curveball_run else 0, time.time())
                )
        finally:
            conn.close()

    @staticmethod
    def get_latest_eval_results(curveball_only: bool = False) -> List[Dict[str, Any]]:
        conn = get_connection()
        try:
            query = "SELECT * FROM eval_results WHERE is_curveball_run = ? ORDER BY run_timestamp DESC"
            rows = conn.execute(query, (1 if curveball_only else 0,)).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()
