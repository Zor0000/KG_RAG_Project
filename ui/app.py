"""
app.py  (updated)
─────────────────
Multi-Product KG-RAG Streamlit app.

Added on top of original:
  ✅ Session ID — UUID created on load, stored in st.session_state
  ✅ Redis session management — summary + last 5 turns, no TTL, logout deletes
  ✅ Query rewriting — LLM rewrites raw input before unified_search is called
  ✅ Redis summary injected into generate_answer / generate_guided_preview
  ✅ Lab detection + generation — shown below answer when practical intent detected
  ✅ Follow-up question buttons — 3 clickable buttons after every answer
  ✅ Scenario expansion — user can expand last lab into full scenario + input data
"""

import sys
from pathlib import Path

# ROOT_DIR = KG_RAG_PROJECT/ (parent of ui/)
# Insert at 0 so project root is found before anything else.
# session_manager, query_rewriter, lab_engine must live at project root.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from openai import OpenAI
import os

from retrieval.unified_retriever  import unified_search
from retrieval.answer_generator   import generate_answer, generate_guided_preview

# New modules — place these 3 files at KG_RAG_PROJECT/ (project root):
#   KG_RAG_PROJECT/session_manager.py
#   KG_RAG_PROJECT/query_rewriter.py
#   KG_RAG_PROJECT/lab_engine.py
from session_manager import (
    create_session, session_exists, logout,
    get_context, save_turn, update_lab,
)
from query_rewriter import rewrite_query
from lab_engine     import detect_practical_intent, get_or_generate_lab, generate_followups

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Multi-Product KG-RAG", layout="wide")
st.title("🧠 Multi-Product Conversational KG-RAG")


# ── Session bootstrap ─────────────────────────────────────────────────────────
# On every Streamlit run (page load, rerun, interaction), ensure a valid
# session ID is in st.session_state. If the user has no session or the
# Redis session expired, create a fresh one.

if "sid" not in st.session_state or not session_exists(st.session_state["sid"]):
    st.session_state["sid"] = create_session(user="streamlit_user")

sid = st.session_state["sid"]


# ── Standard session state init ───────────────────────────────────────────────

for key, default in {
    "chat_history":           [],
    "pending_clarification":  False,
    "clarification_options":  [],
    "last_query":             None,
    "followup_questions":     [],    # list[str] — rendered as buttons
    "expand_lab_pending":     False, # True when user clicked "Expand Lab"
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.header("🔧 Filters")

persona = st.sidebar.selectbox(
    "Select Persona",
    ["All", "NoCode", "LowCode", "ProDeveloper", "Admin", "Architect"]
)

product = st.sidebar.selectbox(
    "Select Product",
    ["All", "copilot_studio", "azure_bot_service", "autogen"]
)

top_k = st.sidebar.slider("Top K Results", 3, 10, 5)

st.sidebar.divider()
st.sidebar.markdown(f"**Session ID**")
st.sidebar.code(sid[:18] + "...", language=None)

if st.sidebar.button("🚪 Logout / Clear Session"):
    logout(sid)
    # Force a new session on next rerun
    del st.session_state["sid"]
    st.session_state["chat_history"] = []
    st.rerun()

# Show session summary in sidebar if it exists
ctx_preview = get_context(sid)
if ctx_preview["summary"]:
    with st.sidebar.expander("📖 Session Memory"):
        st.write(ctx_preview["summary"])


# ── Lab scenario expander ─────────────────────────────────────────────────────

def _generate_lab_scenario(original_question: str, lab_text: str) -> str:
    """Expand a lab into a full scenario with input data, steps, expected output."""
    prompt = f"""
You are an expert technical trainer creating a complete hands-on lab experience.

The user was shown this lab:
---
{lab_text}
---

Their original question: {original_question}

Generate a FULL LAB SCENARIO with EXACTLY these sections:

🌍 SCENARIO
A realistic real-world story (3-5 lines) explaining WHY this lab matters.
Name the company, team, or role involved.

📦 INPUT DATA
Sample data the user will use. Choose the BEST FORMAT automatically:
  JSON   → APIs, RAG pipelines, LLM workflows, NoSQL
  SQL    → databases, analytics, reporting
  CSV    → data science, ML, pandas
  Python → scripting, automation, SDK usage
  Plain  → config-based or conceptual labs
Start with one line explaining the format choice.
Include 5-10 realistic records.

🪜 STEPS TO PERFORM
Minimum 5 numbered steps. Be specific — include exact commands,
code snippets, config values, or API calls. No vague instructions.

✅ EXPECTED OUTPUT
Exactly what the user should see on success.
Include sample output text or result data and explain how to verify.

⚠️ COMMON MISTAKES
2-3 specific mistakes users make and how to avoid them.
"""
    try:
        resp = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating scenario: {e}"


# ── Helper: render a full answer block ───────────────────────────────────────
# Extracted so it can be called from both the normal flow and the
# clarification-resolution flow without code duplication.

def _render_answer_block(
    query:           str,
    raw_question:    str,
    rewritten_query: str,
    results:         list[dict],
    session_ctx:     dict,
    show_evidence:   bool = True,
    persona:         str = "All",
    product:         str = "All",
) -> None:
    """
    Renders: answer → sources → evidence expanders → lab → follow-ups.
    Also saves the completed turn to Redis.
    """

    # ── Generate answer with session context injected ─────────────────
    answer_data = generate_answer(
        query           = rewritten_query,
        results         = results,
        session_context = session_ctx,
        persona         = persona,
        product         = product,
    )

    with st.chat_message("assistant"):
        st.markdown(answer_data["answer"])

        # Sources
        if answer_data["sources"]:
            st.markdown("### 🔗 Sources")
            for s in answer_data["sources"]:
                st.write(s)

        # Evidence expanders
        if show_evidence and results:
            st.markdown("### 📂 Retrieved Evidence")
            for i, r in enumerate(results, 1):
                score = r.get("minilm_score", r.get("final_score", r.get("score", 0)))
                with st.expander(f"{i}. {r['canonical_topic']} (Score: {score:.4f})"):
                    st.write(r["text"])
                    st.caption(
                        f"Persona: {r.get('persona','—')} | "
                        f"Intent: {r.get('intent','—')} | "
                        f"Product: {r.get('product','—')}"
                    )

        # ── Lab ───────────────────────────────────────────────────────
        lab_text = ""
        if detect_practical_intent(raw_question):
            with st.spinner("🔬 Checking for relevant lab..."):
                lab_text = get_or_generate_lab(raw_question)

            if lab_text:
                st.divider()
                st.markdown("### 📚 Suggested Lab")
                st.markdown(lab_text)
                update_lab(sid, lab_text, raw_question)

                # Expand Lab button
                if st.button("🔬 Expand into full scenario + input data",
                             key=f"expand_{len(st.session_state.chat_history)}"):
                    st.session_state["expand_lab_pending"] = True
                    st.rerun()

        # ── Follow-up question buttons ────────────────────────────────
        with st.spinner("💬 Generating follow-up questions..."):
            followups = generate_followups(raw_question, answer_data["answer"])

        if followups:
            st.divider()
            st.markdown("**💡 You might also want to ask:**")
            cols = st.columns(len(followups))
            for i, (col, fq) in enumerate(zip(cols, followups)):
                with col:
                    if st.button(fq, key=f"fq_{len(st.session_state.chat_history)}_{i}"):
                        # Inject the follow-up as the next user message
                        st.session_state["injected_query"] = fq
                        st.rerun()

    # ── Save to chat history and Redis ────────────────────────────────
    st.session_state.chat_history.append({
        "role":    "assistant",
        "content": answer_data["answer"],
    })

    save_turn(
        sid             = sid,
        question        = raw_question,
        rewritten_query = rewritten_query,
        answer          = answer_data["answer"],
        lab             = lab_text,
    )


# ── Display existing chat history ─────────────────────────────────────────────

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ── Handle lab scenario expansion (triggered by button) ──────────────────────

if st.session_state.get("expand_lab_pending"):
    st.session_state["expand_lab_pending"] = False
    ctx = get_context(sid)

    if ctx["last_lab"]:
        with st.chat_message("assistant"):
            st.markdown("### 📚 Your Active Lab")
            st.markdown(ctx["last_lab"])
            st.divider()
            st.markdown("### 🔬 Full Lab Scenario")
            with st.spinner("Building scenario..."):
                scenario = _generate_lab_scenario(ctx["last_question"], ctx["last_lab"])
            st.markdown(scenario)

        st.session_state.chat_history.append({
            "role":    "assistant",
            "content": f"**Lab Scenario**\n\n{scenario}",
        })
        update_lab(sid, scenario, ctx["last_question"])
    else:
        st.info("No active lab found. Ask a practical question first to generate a lab.")


# ── Resolve any injected follow-up query ──────────────────────────────────────

if "injected_query" in st.session_state:
    injected = st.session_state.pop("injected_query")
    with st.chat_message("user"):
        st.markdown(injected)
    st.session_state.chat_history.append({"role": "user", "content": injected})

    # Run it through the full pipeline
    ctx              = get_context(sid)
    rewritten        = rewrite_query(injected, ctx["summary"], ctx["recent_turns"])
    search_product   = product if product != "All" else "All"

    with st.spinner("🔎 Retrieving..."):
        response = unified_search(
            query   = rewritten,
            persona = persona,
            product = search_product,
            top_k   = top_k,
        )

    if response["mode"] == "answer":
        _render_answer_block(
            query           = injected,
            raw_question    = injected,
            rewritten_query = rewritten,
            results         = response["results"],
            session_ctx     = ctx,
        )
    elif response["mode"] == "empty":
        with st.chat_message("assistant"):
            st.markdown("No relevant information found for that follow-up.")


# ── Main chat input ───────────────────────────────────────────────────────────

user_input = st.chat_input("Ask your question...")

if user_input:

    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.chat_history.append({
        "role":    "user",
        "content": user_input,
    })

    # Load full Redis session context
    ctx = get_context(sid)

    # ── QUERY REWRITING ───────────────────────────────────────────────
    # Rewrite before retrieval — resolves references, cleans phrasing.
    # Show the rewritten query so the user can see what was searched.
    with st.spinner("✏️ Optimising query..."):
        rewritten_query = rewrite_query(
            raw_question  = user_input,
            summary       = ctx["summary"],
            recent_turns  = ctx["recent_turns"],
        )

    if rewritten_query.lower() != user_input.lower():
        st.caption(f"🔍 Searching for: *{rewritten_query}*")

    # ── HANDLE CLARIFICATION RESPONSE ────────────────────────────────
    if st.session_state.pending_clarification:

        chosen_product = user_input.strip()

        if chosen_product in st.session_state.clarification_options:

            st.session_state.pending_clarification = False

            with st.spinner("🔎 Retrieving..."):
                response = unified_search(
                    query   = rewritten_query,
                    persona = persona,
                    product = chosen_product,
                    top_k   = top_k,
                )

            if response["mode"] == "answer":
                _render_answer_block(
                    query           = user_input,
                    raw_question    = user_input,
                    rewritten_query = rewritten_query,
                    results         = response["results"],
                    session_ctx     = ctx,
                    persona         = persona,
                    product         = chosen_product,
                )
        else:
            msg = (
                "Please select one of the following products:\n\n"
                + "\n".join(f"- {p}" for p in st.session_state.clarification_options)
            )
            with st.chat_message("assistant"):
                st.markdown(msg)
            st.session_state.chat_history.append({"role": "assistant", "content": msg})

    # ── NORMAL QUERY FLOW ─────────────────────────────────────────────
    else:

        with st.spinner("🔎 Retrieving..."):
            response = unified_search(
                query   = rewritten_query,
                persona = persona,
                product = product,
                top_k   = top_k,
            )

        # ── Empty ────────────────────────────────────────────────────
        if response["mode"] == "empty":
            msg = "No relevant information found. Try rephrasing or selecting a different product."
            with st.chat_message("assistant"):
                st.markdown(msg)
            st.session_state.chat_history.append({"role": "assistant", "content": msg})

        # ── Guided clarification ──────────────────────────────────────
        elif response["mode"] == "guided_clarification":

            st.session_state.pending_clarification = True
            st.session_state.clarification_options = response["options"]
            st.session_state.last_query            = user_input

            preview_data = generate_guided_preview(
                query           = rewritten_query,
                preview_chunks  = response["preview_chunks"],
                product_options = response["options"],
                session_context = ctx,
            )

            with st.chat_message("assistant"):
                st.markdown(preview_data["answer"])

                if preview_data["sources"]:
                    st.markdown("### 🔗 Sources (Preview)")
                    for s in preview_data["sources"]:
                        st.write(s)

                st.markdown("### 📂 Retrieved Evidence (Preview)")
                for i, r in enumerate(response["preview_chunks"], 1):
                    score = r.get("minilm_score", r.get("score", 0))
                    with st.expander(
                        f"{i}. {r['canonical_topic']} (Score: {score:.4f})"
                    ):
                        st.write(r["text"])
                        st.caption(
                            f"Persona: {r.get('persona','—')} | "
                            f"Intent: {r.get('intent','—')} | "
                            f"Product: {r.get('product','—')}"
                        )

            st.session_state.chat_history.append({
                "role":    "assistant",
                "content": preview_data["answer"],
            })

        # ── Direct answer ─────────────────────────────────────────────
        elif response["mode"] == "answer":
            _render_answer_block(
                query           = user_input,
                raw_question    = user_input,
                rewritten_query = rewritten_query,
                results         = response["results"],
                session_ctx     = ctx,
                persona         = persona,
                product         = product,
            )