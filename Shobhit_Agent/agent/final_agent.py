import os
import json
import uuid
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from openai import OpenAI
import redis

from tool_guidance.recommend_tools import recommend_tools
from tool_guidance.progress_tracker import log_progress


# =============================
# CONFIG
# =============================

MILVUS_HOST = "localhost"
MILVUS_PORT = "29530"
COLLECTION_NAME = "copilot_studio_docs"

NEO4J_URI  = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "password"

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASS = os.getenv("REDIS_PASSWORD", None)

# How many raw turns to keep in Redis per session.
# Older turns are dropped once this limit is hit.
MAX_RAW_TURNS = 5

client      = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


# =============================
# REDIS CONNECTION
# =============================

def connect_redis() -> redis.Redis | None:
    try:
        r = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASS,
            decode_responses=True   # always return str, never bytes
        )
        r.ping()
        print("✅ Redis connected")
        return r
    except Exception as e:
        print(f"⚠️  Redis unavailable: {e} — falling back to in-memory session")
        return None


redis_client = connect_redis()


# =============================
# REDIS KEY SCHEMA
# =============================
# All keys are namespaced under the session_id (a UUID).
#
#   session:{sid}:meta          — JSON: { user, persona, created_at }
#   session:{sid}:summary       — str : rolling LLM summary of full history
#   session:{sid}:turns         — Redis List of JSON strings, newest at index 0
#                                 Each turn: { question, answer, lab? }
#   session:{sid}:last_lab      — str : last lab text shown (for !expand)
#   session:{sid}:last_question — str : question that triggered that lab
#
# No TTL is set — sessions persist until logout() is called.
# =============================

def _key(sid: str, field: str) -> str:
    return f"session:{sid}:{field}"


# =============================
# SESSION LIFECYCLE
# =============================

def create_session(user: str) -> str:
    """
    Generate a new UUID session ID, initialise all keys in Redis,
    and return the session ID to the caller (frontend stores this).
    """
    sid = str(uuid.uuid4())

    if redis_client:
        meta = json.dumps({
            "user":       user,
            "persona":    None,   # filled in on first question
            "created_at": _now()
        })
        redis_client.set(_key(sid, "meta"),          meta)
        redis_client.set(_key(sid, "summary"),       "")
        redis_client.set(_key(sid, "last_lab"),      "")
        redis_client.set(_key(sid, "last_question"), "")
        # turns list is created lazily on first lpush
        print(f"🆕 Session created: {sid}")
    else:
        # In-memory fallback — store in module-level dict
        _mem_sessions[sid] = _empty_mem_session(user)

    return sid


def logout_session(sid: str) -> None:
    """
    Delete all Redis keys for this session.
    Call this when the user explicitly logs out.
    """
    if redis_client:
        keys = redis_client.keys(f"session:{sid}:*")
        if keys:
            redis_client.delete(*keys)
        print(f"🚪 Session {sid} deleted (logout)")
    else:
        _mem_sessions.pop(sid, None)


def session_exists(sid: str) -> bool:
    """Check whether a session ID is valid (used on return visits)."""
    if redis_client:
        return redis_client.exists(_key(sid, "meta")) == 1
    return sid in _mem_sessions


# =============================
# SESSION READ / WRITE
# =============================

def get_session_context(sid: str) -> dict:
    """
    Load everything the agent needs to answer in context:
      - summary     : full conversation summary so far
      - recent_turns: last MAX_RAW_TURNS as list of dicts
      - persona     : last detected persona
      - last_lab    : last lab text (for scenario expansion)
      - last_question: question that triggered that lab
    """
    if redis_client:
        summary      = redis_client.get(_key(sid, "summary"))      or ""
        last_lab     = redis_client.get(_key(sid, "last_lab"))     or ""
        last_question= redis_client.get(_key(sid, "last_question"))or ""

        raw_turns = redis_client.lrange(_key(sid, "turns"), 0, MAX_RAW_TURNS - 1)
        recent_turns = [json.loads(t) for t in raw_turns]

        meta_raw = redis_client.get(_key(sid, "meta"))
        persona  = json.loads(meta_raw).get("persona") if meta_raw else None

    else:
        mem = _mem_sessions.get(sid, _empty_mem_session(""))
        summary       = mem["summary"]
        last_lab      = mem["last_lab"]
        last_question = mem["last_question"]
        recent_turns  = mem["turns"][-MAX_RAW_TURNS:]
        persona       = mem["persona"]

    return {
        "summary":       summary,
        "recent_turns":  recent_turns,
        "persona":       persona,
        "last_lab":      last_lab,
        "last_question": last_question,
    }


def save_turn(sid: str, question: str, answer: str, persona: str, lab: str = "") -> None:
    """
    After every response:
      1. Prepend the new turn to the turns list (trim to MAX_RAW_TURNS)
      2. Update the persona in meta
      3. Update last_lab / last_question if a lab was generated
      4. Regenerate the rolling summary
    """
    turn = json.dumps({ "question": question, "answer": answer, "lab": lab })

    if redis_client:
        turns_key = _key(sid, "turns")
        redis_client.lpush(turns_key, turn)           # newest at index 0
        redis_client.ltrim(turns_key, 0, MAX_RAW_TURNS - 1)

        # Update persona in meta
        meta_raw = redis_client.get(_key(sid, "meta"))
        if meta_raw:
            meta = json.loads(meta_raw)
            meta["persona"] = persona
            redis_client.set(_key(sid, "meta"), json.dumps(meta))

        # Update lab state
        if lab:
            redis_client.set(_key(sid, "last_lab"),      lab)
            redis_client.set(_key(sid, "last_question"), question)

        # Regenerate summary from all stored turns
        all_turns_raw = redis_client.lrange(turns_key, 0, -1)
        all_turns = [json.loads(t) for t in all_turns_raw]

    else:
        mem = _mem_sessions.setdefault(sid, _empty_mem_session(""))
        mem["turns"].append({"question": question, "answer": answer, "lab": lab})
        mem["persona"] = persona
        if lab:
            mem["last_lab"]      = lab
            mem["last_question"] = question
        all_turns = mem["turns"]

    # Regenerate summary (LLM call — runs async in production, sync here)
    new_summary = _generate_summary(all_turns)

    if redis_client:
        redis_client.set(_key(sid, "summary"), new_summary)
    else:
        _mem_sessions[sid]["summary"] = new_summary


def update_session_lab(sid: str, lab: str, question: str) -> None:
    """Standalone update for last_lab — used by scenario expansion handler."""
    if redis_client:
        redis_client.set(_key(sid, "last_lab"),      lab)
        redis_client.set(_key(sid, "last_question"), question)
    else:
        mem = _mem_sessions.get(sid, {})
        mem["last_lab"]      = lab
        mem["last_question"] = question


# =============================
# SUMMARY GENERATOR
# =============================
# Called after every turn to keep a rolling plain-English summary
# of the entire conversation — used as long-range context in prompts.

def _generate_summary(turns: list[dict]) -> str:
    if not turns:
        return ""

    turns_text = "\n".join(
        f"Q: {t['question']}\nA: {t['answer']}"
        for t in turns[-20:]   # cap at last 20 turns to keep prompt short
    )

    prompt = f"""
Summarise this conversation in 3-5 sentences.
Capture: the user's main topics of interest, their skill level, what they have already learned,
and any labs or scenarios they have worked through.
Be concise — this summary will be injected into future prompts as context.

Conversation:
{turns_text}
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return resp.choices[0].message.content.strip()
    except:
        return ""


# =============================
# IN-MEMORY FALLBACK
# =============================
# Used when Redis is unavailable. Same structure as Redis keys,
# stored as a plain dict so the rest of the code doesn't branch.

_mem_sessions: dict[str, dict] = {}


def _empty_mem_session(user: str) -> dict:
    return {
        "user":          user,
        "persona":       None,
        "summary":       "",
        "turns":         [],
        "last_lab":      "",
        "last_question": "",
    }


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


# =============================
# PRACTICAL INTENT DETECTION
# =============================

def detect_practical_intent(question: str) -> bool:
    prompt = f"""
Does this question require the user to actually BUILD or IMPLEMENT something hands-on?

Return ONLY YES or NO.

YES — the user wants to create, build, implement, design, or configure something hands-on:
  Examples:
  - "How do I build a RAG pipeline?"
  - "How do I set up a vector database?"
  - "How do I implement authentication in my app?"
  - "How do I create a chatbot with Copilot Studio?"
  - "How do I design a scalable architecture?"
  - "How do I design a microservices system?"
  - "How do I architect a Copilot solution?"
  - "How do I structure my Power Automate flows?"
  - "How do I deploy this to Azure?"
  - "How do I configure a Copilot agent?"

NO — the user is asking for advice, explanation, comparison, or decision guidance only:
  Examples:
  - "What tools would you use to make this?"
  - "How do you decide which SDK to use?"
  - "What is the difference between X and Y?"
  - "Which language is best for this project?"
  - "What are best practices for X?"
  - "Can you explain how X works?"
  - "What should I consider when choosing X?"
  - "What is microservices architecture?"
  - "Why would I use an API gateway?"

The key test: Is the user about to BUILD, DESIGN, or CONFIGURE something RIGHT NOW?
  "How do I design X"    → YES  (they are actively designing something)
  "How do I implement X" → YES  (they are actively building something)
  "What is X"            → NO   (they are learning a concept)
  "Which X should I use" → NO   (they are making a decision, not building yet)

Question: {question}
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return resp.choices[0].message.content.strip() == "YES"
    except:
        return False


# =============================
# SCENARIO EXPANSION INTENT DETECTION
# =============================

_SCENARIO_KEYWORDS = [
    "scenario", "give me a scenario", "walk me through",
    "input data", "input database", "dataset", "sample data",
    "expected output", "expected result", "how do i perform",
    "perform the lab", "run the lab", "try it out",
    "step by step", "steps to perform", "hands on", "hands-on",
    "show me how", "example data", "practice this",
]


def detect_scenario_intent(question: str, explicit_trigger: bool = False) -> bool:
    if explicit_trigger:
        return True

    q_lower = question.lower()
    keyword_hit = any(kw in q_lower for kw in _SCENARIO_KEYWORDS)

    if keyword_hit:
        prompt = f"""
A user has an active lab from a previous turn.
Does this follow-up message mean they want to:
- Get a real-world scenario for the lab
- Get sample input data / a dataset to use
- Get step-by-step instructions to actually run the lab
- See the expected output of the lab

Return ONLY YES or NO.

Message: {question}
"""
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return resp.choices[0].message.content.strip() == "YES"
        except:
            return keyword_hit

    return False


# =============================
# LAB SCENARIO GENERATOR
# =============================

def generate_lab_scenario(original_question: str, lab_text: str) -> str:
    prompt = f"""
You are an expert technical trainer creating a complete hands-on lab experience.

The user was previously shown this lab:
---
{lab_text}
---

Their original question was: {original_question}

Generate a FULL LAB SCENARIO using EXACTLY this structure:

────────────────────────────────────────
🌍 SCENARIO
────────────────────────────────────────
Write a realistic real-world story (3-5 lines) that explains WHY this lab
matters in a business or technical context. Name the company, team, or role
involved so the user can picture themselves doing this.

────────────────────────────────────────
📦 INPUT DATA
────────────────────────────────────────
Provide the sample data the user will actually use to perform the lab.
Choose the BEST FORMAT for this specific topic:
  - JSON   if the topic is APIs, RAG pipelines, LLM workflows, or NoSQL
  - SQL    if the topic is databases, analytics, or reporting
  - CSV    if the topic is data science, ML, or pandas-based work
  - Python if the topic is scripting, automation, or SDK usage
  - Plain structured text if none of the above fits

Start the block with a one-line note explaining your format choice.
Include at least 5-10 meaningful records — enough to make the lab realistic.

────────────────────────────────────────
🪜 STEPS TO PERFORM
────────────────────────────────────────
Numbered step-by-step instructions (minimum 5 steps) for executing the lab
using the input data above.
Be specific: include exact commands, code snippets, config values, or
API calls where relevant. Do not write vague instructions.

────────────────────────────────────────
✅ EXPECTED OUTPUT
────────────────────────────────────────
Show exactly what the user should see or receive if they completed the lab
correctly. Include sample output text, result data, or console responses.
Explain what each part of the output means and how to verify success.

────────────────────────────────────────
⚠️ COMMON MISTAKES
────────────────────────────────────────
List 2-3 specific mistakes users commonly make in this type of lab
and exactly how to avoid or fix each one.
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    return resp.choices[0].message.content


# =============================
# CUSTOM LAB GENERATOR
# =============================

def generate_custom_lab(question: str) -> str:
    prompt = f"""
Create a practical learning mini lab.

FORMAT EXACTLY:

📚 Title

🎯 Goal:
1-2 lines objective.

👉 Tasks:
• 3-5 practical steps

✅ Learning:
Skills gained.

Topic:
{question}
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    return resp.choices[0].message.content


# =============================
# MILVUS CONNECTION
# =============================

def connect_milvus():
    try:
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
        collection = Collection(COLLECTION_NAME)
        collection.load()
        print("✅ Milvus connected")
        return collection
    except Exception as e:
        print("Milvus error:", e)
        return None


collection = connect_milvus()


# =============================
# NEO4J CONNECTION + SEARCH
# =============================

def connect_kg():
    try:
        return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    except:
        return None


kg_driver = connect_kg()


def search_kg(question: str) -> list[str]:
    if not kg_driver:
        return []
    try:
        with kg_driver.session() as session:
            result = session.run("""
                MATCH (n)
                WHERE ANY(key IN keys(n)
                    WHERE toLower(toString(n[key]))
                    CONTAINS toLower($q))
                RETURN n LIMIT 5
            """, q=question)
            return [str(dict(r["n"])) for r in result]
    except Exception as e:
        print("Neo4j search error:", e)
        return []


# =============================
# PERSONA DETECTION
# =============================

def detect_persona(question: str, previous_persona: str | None = None) -> str:
    """
    Detect persona from the current question.
    Falls back to previous_persona if the current question is ambiguous
    (e.g. a short follow-up that doesn't reveal skill level).
    """
    q = question.lower()
    if "architecture" in q or "architect" in q or "design" in q:
        return "Architect"
    if "python" in q or "code" in q or "implement" in q or "build" in q:
        return "Developer"
    if "what is" in q or "explain" in q or "how does" in q:
        return "Beginner"
    # Ambiguous — reuse last known persona if available
    return previous_persona or "Developer"


# =============================
# VECTOR SEARCH
# =============================

def search_milvus(question: str) -> list[str]:
    if not collection:
        return []
    try:
        emb = embed_model.encode(question).tolist()
        results = collection.search(
            data=[emb],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=5,
            output_fields=["content", "url"]
        )
        return [hit.entity.get("content", "") for hit in results[0]]
    except Exception as e:
        print("Milvus search error:", e)
        return []


# =============================
# ANSWER GENERATION
# =============================
# Now receives session context (summary + recent turns) so the LLM
# knows what the user has already asked and learned.

def generate_answer(
    question: str,
    persona: str,
    context: str,
    session_summary: str,
    recent_turns: list[dict]
) -> str:

    # Build a short recent-history block from raw turns
    history_block = ""
    if recent_turns:
        lines = []
        for t in reversed(recent_turns):   # oldest first
            lines.append(f"User: {t['question']}")
            lines.append(f"Agent: {t['answer'][:300]}...")  # truncate long answers
        history_block = "\n".join(lines)

    prompt = f"""
You are a Microsoft Copilot Studio expert having a natural conversation.

Persona: {persona}

{'--- Conversation summary (what this user has discussed before) ---' if session_summary else ''}
{session_summary}

{'--- Recent exchanges ---' if history_block else ''}
{history_block}

--- Retrieved knowledge context ---
{context}

Rules:
- Answer conversationally and directly — like a knowledgeable colleague, not a documentation page.
- If the conversation summary shows the user already knows something, do not re-explain it.
- Match the length to the question. Simple questions get short answers. Complex questions get more detail.
- Only use bullet points or numbered lists if the question genuinely asks for a list.
- Never pad the answer with intros like "Great question!" or outros like "I hope this helps!".
- Do NOT suggest labs, tutorials, or practice exercises — that is handled separately by the system.
- Stick to what was actually asked. Do not add unrequested sections.
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user",   "content": question},
        ]
    )
    return resp.choices[0].message.content


# =============================
# FOLLOW-UP GENERATION
# =============================

def generate_followups(question: str, answer: str) -> str:
    prompt = f"""
Generate 3 smart followup questions.

Question:
{question}

Answer:
{answer}
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content


# =============================
# SCENARIO EXPANSION HANDLER
# =============================

def handle_scenario_expansion(question: str, sid: str) -> None:
    ctx = get_session_context(sid)
    last_lab      = ctx["last_lab"]
    last_question = ctx["last_question"]

    if not last_lab:
        print("\n⚠️  No active lab found in your session.")
        print("    Ask a technical question first to generate a lab,")
        print("    then ask for a scenario to expand it.\n")
        return

    print("\n" + "═" * 55)
    print("📚 YOUR ACTIVE LAB (from previous turn)")
    print("═" * 55)
    print(last_lab)

    print("\n" + "═" * 55)
    print("🔬 LAB SCENARIO EXPANSION")
    print("═" * 55 + "\n")
    scenario = generate_lab_scenario(last_question, last_lab)
    print(scenario)

    print("\n" + "═" * 55)
    print("💬 FOLLOW-UP QUESTIONS")
    print("═" * 55 + "\n")
    followups = generate_followups(question, scenario)
    print(followups)

    # Scenario becomes the new last_lab so chained follow-ups work
    update_session_lab(sid, scenario, question)


# =============================
# MAIN PIPELINE
# =============================

def ask_question(
    question:        str,
    sid:             str,
    expand_lab:      bool = False
) -> None:
    """
    Main entry point.

    Parameters
    ----------
    question   : The user's message.
    sid        : Session ID (UUID). Create with create_session() on first visit.
                 On return visits, pass the same SID — agent will load history.
    expand_lab : Explicit lab expansion trigger (UI button or !expand CLI).
    """

    # ------------------------------------------------------------------
    # BRANCH A: Scenario expansion
    # ------------------------------------------------------------------
    if detect_scenario_intent(question, explicit_trigger=expand_lab):
        handle_scenario_expansion(question, sid)
        return

    # ------------------------------------------------------------------
    # BRANCH B: Standard RAG pipeline
    # ------------------------------------------------------------------

    # Load full session context from Redis
    ctx = get_session_context(sid)

    # Detect persona — reuse previous if current question is ambiguous
    persona = detect_persona(question, previous_persona=ctx["persona"])
    print(f"\n🎯 Persona: {persona}")

    # If this is a return visit and there's history, acknowledge it
    if ctx["summary"]:
        print(f"\n📖 Session context loaded — continuing from previous conversation")

    # Retrieval
    docs    = search_milvus(question)
    kg_docs = search_kg(question)
    context = "\n".join(docs + kg_docs)

    # Generate answer with full session context injected
    answer = generate_answer(
        question        = question,
        persona         = persona,
        context         = context,
        session_summary = ctx["summary"],
        recent_turns    = ctx["recent_turns"],
    )
    followups = generate_followups(question, answer)

    # ------------------------------------------------------------------
    # LAB SELECTION
    # ------------------------------------------------------------------
    labs:               list[str] = []
    generated_lab_text: str       = ""

    if detect_practical_intent(question):
        official = recommend_tools(question)

        if official:
            lab_text    = official[0].lower()
            q           = question.lower()
            rag_keywords = ["rag", "knowledge graph", "embedding", "vector", "retrieval"]

            if (any(k in q for k in rag_keywords)
                    and not any(k in lab_text for k in rag_keywords)):
                generated_lab_text = generate_custom_lab(question)
                labs = [generated_lab_text]
            else:
                labs               = official
                generated_lab_text = official[0]
        else:
            generated_lab_text = generate_custom_lab(question)
            labs               = [generated_lab_text]
    # else: no practical intent — no lab, just answer

    # ------------------------------------------------------------------
    # Persist turn to Redis BEFORE printing
    # ------------------------------------------------------------------
    save_turn(
        sid      = sid,
        question = question,
        answer   = answer,
        persona  = persona,
        lab      = generated_lab_text,
    )

    log_progress(sid, question, [], labs)

    # ------------------------------------------------------------------
    # PRINT OUTPUT
    # ------------------------------------------------------------------
    print("\n" + "═" * 55)
    print("💬 ANSWER")
    print("═" * 55 + "\n")
    print(answer)

    if labs:
        print("\n" + "═" * 55)
        print("📚 SUGGESTED LAB")
        print("═" * 55 + "\n")
        for lab in labs:
            print(lab)

        print("\n" + "─" * 55)
        print("💡 Want to practice this?")
        print('   Say: "give me a scenario to perform this lab"')
        print('   Or:  "show me the input data and expected output"')
        print("   Or just type: !expand")
        print("─" * 55)

    print("\n" + "═" * 55)
    print("🔁 FOLLOW-UP QUESTIONS")
    print("═" * 55 + "\n")
    print(followups)


# =============================
# CLI LOOP
# =============================

if __name__ == "__main__":
    print("\n💬 Hybrid Copilot AI Agent Ready")
    print("   Commands: 'exit' | '!expand' | '!logout' | '!session'\n")

    # In a real app the SID comes from the frontend (cookie/localStorage).
    # In CLI we generate one per run and print it so you can reuse it.
    cli_user = "shobhit"

    # Check if a SID was passed as env var for testing return-visit flow
    existing_sid = os.getenv("SESSION_ID")
    if existing_sid and session_exists(existing_sid):
        sid = existing_sid
        print(f"🔄 Resuming session: {sid}")
        ctx = get_session_context(sid)
        if ctx["summary"]:
            print(f"\n📖 Previous context:\n{ctx['summary']}\n")
    else:
        sid = create_session(cli_user)
        print(f"🆕 New session: {sid}")
        print(f"   (Set SESSION_ID={sid} to resume this session later)\n")

    while True:
        q = input("Ask: ").strip()

        if not q:
            continue

        if q.lower() == "exit":
            break

        if q.lower() == "!logout":
            logout_session(sid)
            print("✅ Logged out. Session deleted.")
            break

        if q.lower() == "!session":
            ctx = get_session_context(sid)
            print(f"\n📋 Session ID : {sid}")
            print(f"   Persona    : {ctx['persona']}")
            print(f"   Turns      : {len(ctx['recent_turns'])}")
            print(f"   Summary    : {ctx['summary'] or '(none yet)'}\n")
            continue

        if q.lower().startswith("!expand"):
            ask_question("expand my last lab", sid=sid, expand_lab=True)
        else:
            ask_question(q, sid=sid)