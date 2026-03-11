"""
session_manager.py
──────────────────
Redis-backed session layer for the Streamlit KG-RAG app.

Key schema (all under session:{sid}:*)
  meta          — JSON: { user, created_at }
  summary       — Rolling LLM summary of full conversation history
  turns         — Redis List of JSON turn objects, newest at index 0
                  Each turn: { question, rewritten_query, answer, lab }
  last_lab      — Last lab text shown (enables !expand / scenario flow)
  last_question — Question that triggered the last lab

Sessions expire after 7 days of inactivity — the TTL is reset on every
save_turn() call so active sessions never expire mid-conversation.
Falls back to an in-memory dict if Redis is unavailable.
"""

import os
import json
import uuid
from datetime import datetime, timezone
from openai import OpenAI
import redis as _redis

# ── Config ────────────────────────────────────────────────────────────────────

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASS = os.getenv("REDIS_PASSWORD", None)
MAX_RAW_TURNS    = 5       # raw turns kept per session
SUMMARY_TURN_CAP = 20      # max turns fed to summary LLM
SESSION_TTL      = 7 * 24 * 60 * 60   # 7 days in seconds — reset on every write

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Redis connection ──────────────────────────────────────────────────────────

def _connect() -> _redis.Redis | None:
    try:
        r = _redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASS,
            decode_responses=True,
        )
        r.ping()
        return r
    except Exception as e:
        print(f"⚠️  Redis unavailable ({e}) — using in-memory fallback")
        return None


_r = _connect()

# ── In-memory fallback ────────────────────────────────────────────────────────

_mem: dict[str, dict] = {}


def _empty(user: str) -> dict:
    return {
        "user": user,
        "summary": "",
        "turns": [],
        "last_lab": "",
        "last_question": "",
    }


# ── Key helpers ───────────────────────────────────────────────────────────────

def _k(sid: str, field: str) -> str:
    return f"session:{sid}:{field}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _refresh_ttl(sid: str) -> None:
    """Reset the 7-day expiry on all keys for this session.
    Called on create and every save_turn so active sessions never expire.
    Inactive sessions (no activity for 7 days) are cleaned up automatically.
    """
    if _r:
        for field in ("meta", "summary", "turns", "last_lab", "last_question"):
            _r.expire(_k(sid, field), SESSION_TTL)


# ── Public API ────────────────────────────────────────────────────────────────

def create_session(user: str = "anonymous") -> str:
    """
    Create a new session. Returns the session ID (UUID string).
    Store this in st.session_state so it survives Streamlit reruns.
    """
    sid = str(uuid.uuid4())

    if _r:
        _r.set(_k(sid, "meta"),          json.dumps({"user": user, "created_at": _now()}))
        _r.set(_k(sid, "summary"),       "")
        _r.set(_k(sid, "last_lab"),      "")
        _r.set(_k(sid, "last_question"), "")
        _refresh_ttl(sid)   # start the 7-day clock
    else:
        _mem[sid] = _empty(user)

    return sid


def session_exists(sid: str) -> bool:
    if _r:
        return bool(_r.exists(_k(sid, "meta")))
    return sid in _mem


def logout(sid: str) -> None:
    """Delete all Redis keys for this session (call on explicit logout)."""
    if _r:
        keys = _r.keys(f"session:{sid}:*")
        if keys:
            _r.delete(*keys)
    else:
        _mem.pop(sid, None)


def get_context(sid: str) -> dict:
    """
    Return everything the agent needs to answer in context:
      summary       — full conversation summary paragraph
      recent_turns  — last MAX_RAW_TURNS as list of dicts (oldest first)
      last_lab      — last lab text
      last_question — question that triggered last lab
    """
    if _r:
        summary       = _r.get(_k(sid, "summary"))       or ""
        last_lab      = _r.get(_k(sid, "last_lab"))      or ""
        last_question = _r.get(_k(sid, "last_question")) or ""
        raw           = _r.lrange(_k(sid, "turns"), 0, MAX_RAW_TURNS - 1)
        recent_turns  = list(reversed([json.loads(t) for t in raw]))  # oldest first
    else:
        mem           = _mem.get(sid, _empty(""))
        summary       = mem["summary"]
        last_lab      = mem["last_lab"]
        last_question = mem["last_question"]
        recent_turns  = mem["turns"][-MAX_RAW_TURNS:]

    return {
        "summary":       summary,
        "recent_turns":  recent_turns,
        "last_lab":      last_lab,
        "last_question": last_question,
    }


def save_turn(
    sid:             str,
    question:        str,
    rewritten_query: str,
    answer:          str,
    lab:             str = "",
) -> None:
    """
    Persist a completed turn and regenerate the rolling summary.
    Call this after the full response has been assembled.
    """
    turn = json.dumps({
        "question":        question,
        "rewritten_query": rewritten_query,
        "answer":          answer,
        "lab":             lab,
    })

    if _r:
        turns_key = _k(sid, "turns")
        _r.lpush(turns_key, turn)
        _r.ltrim(turns_key, 0, MAX_RAW_TURNS - 1)

        if lab:
            _r.set(_k(sid, "last_lab"),      lab)
            _r.set(_k(sid, "last_question"), question)

        all_raw   = _r.lrange(turns_key, 0, -1)
        all_turns = [json.loads(t) for t in all_raw]
    else:
        mem = _mem.setdefault(sid, _empty(""))
        mem["turns"].append({
            "question": question,
            "rewritten_query": rewritten_query,
            "answer": answer,
            "lab": lab,
        })
        if lab:
            mem["last_lab"]      = lab
            mem["last_question"] = question
        all_turns = mem["turns"]

    new_summary = _regenerate_summary(all_turns)

    if _r:
        _r.set(_k(sid, "summary"), new_summary)
        _refresh_ttl(sid)   # reset 7-day clock on every activity
    else:
        _mem[sid]["summary"] = new_summary


def update_lab(sid: str, lab: str, question: str) -> None:
    """Standalone update for last_lab (used by scenario expansion)."""
    if _r:
        _r.set(_k(sid, "last_lab"),      lab)
        _r.set(_k(sid, "last_question"), question)
    else:
        mem = _mem.get(sid, {})
        if mem:
            mem["last_lab"]      = lab
            mem["last_question"] = question


# ── Summary generator ─────────────────────────────────────────────────────────

def _regenerate_summary(turns: list[dict]) -> str:
    if not turns:
        return ""

    turns_text = "\n".join(
        f"Q: {t['question']}\nA: {t['answer'][:400]}"
        for t in turns[-SUMMARY_TURN_CAP:]
    )

    prompt = f"""
Summarise this conversation in 3-5 sentences.
Capture: the main topics discussed, what the user already understands,
their apparent skill level, and any labs or scenarios they have worked through.
Be concise — this summary is injected into future LLM prompts as context.

Conversation:
{turns_text}
"""
    try:
        resp = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Summary generation error: {e}")
        return ""