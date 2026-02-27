import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import streamlit as st
from retrieval.unified_retriever import unified_search
from retrieval.answer_generator import generate_answer, generate_guided_preview

st.set_page_config(page_title="Multi-Product KG-RAG", layout="wide")

st.title("🧠 Multi-Product Conversational KG-RAG")

# ============================================================
# 🔹 Sidebar Filters
# ============================================================

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

# ============================================================
# 🔹 Session State Initialization
# ============================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pending_clarification" not in st.session_state:
    st.session_state.pending_clarification = False

if "clarification_options" not in st.session_state:
    st.session_state.clarification_options = []

if "last_query" not in st.session_state:
    st.session_state.last_query = None


# ============================================================
# 🔹 Display Chat History
# ============================================================

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# 🔹 User Input (Chat)
# ============================================================

user_input = st.chat_input("Ask your question...")

if user_input:

    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })

    # ========================================================
    # 🔹 Handle Clarification Response
    # ========================================================

    if st.session_state.pending_clarification:

        chosen_product = user_input.strip()

        if chosen_product in st.session_state.clarification_options:

            st.session_state.pending_clarification = False

            response = unified_search(
                query=st.session_state.last_query,
                persona=persona,
                product=chosen_product,
                top_k=top_k
            )

            if response["mode"] == "answer":

                answer_data = generate_answer(
                    st.session_state.last_query,
                    response["results"]
                )

                with st.chat_message("assistant"):
                    st.markdown(answer_data["answer"])

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer_data["answer"]
                })

        else:

            msg = (
                "Please select one of the following products:\n\n"
                + "\n".join(f"- {p}" for p in st.session_state.clarification_options)
            )

            with st.chat_message("assistant"):
                st.markdown(msg)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": msg
            })

    # ========================================================
    # 🔹 Normal Query Flow
    # ========================================================

    else:

        with st.spinner("🔎 Retrieving..."):
            response = unified_search(
                query=user_input,
                persona=persona,
                product=product,
                top_k=top_k
            )

        # ===============================
        # Empty
        # ===============================
        if response["mode"] == "empty":

            msg = "No relevant information found."

            with st.chat_message("assistant"):
                st.markdown(msg)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": msg
            })

        # ===============================
        # Guided Clarification Mode
        # ===============================
        elif response["mode"] == "guided_clarification":

            st.session_state.pending_clarification = True
            st.session_state.clarification_options = response["options"]
            st.session_state.last_query = user_input

            preview_data = generate_guided_preview(
                query=user_input,
                preview_chunks=response["preview_chunks"],
                product_options=response["options"]
            )

            with st.chat_message("assistant"):
                st.markdown(preview_data["answer"])
                if preview_data["sources"]:
                    st.markdown("### 🔗 Sources (Preview)")
                    for s in preview_data["sources"]:
                        st.write(s)

                st.markdown("### 📂 Retrieved Evidence (Preview)")

                for i, r in enumerate(response["preview_chunks"], 1):
                    with st.expander(
                        f"{i}. {r['canonical_topic']} "
                        f"(Score: {r.get('minilm_score', r.get('score', 0)):.4f})"
                    ):
                        st.write(r["text"])
                        st.caption(
                            f"Persona: {r['persona']} | "
                            f"Intent: {r['intent']} | "
                            f"Product: {r['product']}"
                        )                

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": preview_data["answer"]
            })

        # ===============================
        # Direct Answer Mode
        # ===============================
        elif response["mode"] == "answer":

            answer_data = generate_answer(
                user_input,
                response["results"]
            )

            with st.chat_message("assistant"):

                st.markdown(answer_data["answer"])

                if answer_data["sources"]:
                    st.markdown("### 🔗 Sources")
                    for s in answer_data["sources"]:
                        st.write(s)

                st.markdown("### 📂 Retrieved Evidence")

                for i, r in enumerate(response["results"], 1):
                    with st.expander(
                        f"{i}. {r['canonical_topic']} "
                        f"(Score: {r.get('minilm_score', r.get('final_score', r.get('score', 0))):.4f})"
                    ):
                        st.write(r["text"])
                        st.caption(
                            f"Persona: {r['persona']} | "
                            f"Intent: {r['intent']} | "
                            f"Product: {r['product']}"
                        )

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer_data["answer"]
            })