import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import streamlit as st
from retrieval.graph_guided_retriever import graph_guided_search


st.set_page_config(page_title="Graph-Guided KG-RAG", layout="wide")

st.title("🧠 Graph-Guided KG-RAG")


# ============================================================
# Sidebar Filters
# ============================================================

st.sidebar.header("🔧 Retrieval Settings")

persona = st.sidebar.selectbox(
    "Persona",
    ["All", "NoCode", "LowCode", "ProDeveloper", "Admin", "Architect"]
)

product = st.sidebar.selectbox(
    "Product",
    ["All", "copilot_studio", "azure_bot_service", "autogen"]
)

top_k = st.sidebar.slider("Top K Results", 3, 10, 5)


# ============================================================
# Session State
# ============================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ============================================================
# Display History
# ============================================================

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ============================================================
# Input
# ============================================================

user_input = st.chat_input("Ask your question...")

if user_input:

    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })

    with st.spinner("🔎 Retrieving..."):

        debug = graph_guided_search(
            query=user_input,
            persona=persona,
            product=product,
            top_k=top_k
        )

    results = debug["results"]

    # ========================================================
    # Debug Sidebar
    # ========================================================

    st.sidebar.markdown("### 🧠 Query Structure")
    st.sidebar.json(debug["structure"])

    st.sidebar.markdown("### 🔎 Detected Topics")
    st.sidebar.write(debug["detected_topics"])

    st.sidebar.markdown("### 🌐 Expanded Topics")
    st.sidebar.write(debug["expanded_topics"])


    # ========================================================
    # No Results
    # ========================================================

    if not results:

        msg = "No relevant information found."

        with st.chat_message("assistant"):
            st.markdown(msg)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": msg
        })


    # ========================================================
    # Show Results
    # ========================================================

    else:

        with st.chat_message("assistant"):

            st.markdown("### 📂 Retrieved Evidence")

            for i, r in enumerate(results, 1):

                with st.expander(
                    f"{i}. {r['topic']} (Score: {r['rerank_score']:.4f})"
                ):

                    st.write(r["text"])

                    st.caption(
                        f"Persona: {r.get('persona')} | "
                        f"Intent: {r.get('intent')} | "
                        f"Product: {r.get('product')}"
                    )

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": "Retrieved evidence shown above."
        })