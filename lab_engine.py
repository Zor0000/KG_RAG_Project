"""
lab_engine.py
─────────────
Handles everything lab-related that gets bolted onto the main answer flow:
  1. detect_practical_intent()  — should this response include a lab?
  2. get_or_generate_lab()      — check recommend_tools() first, generate if no match
  3. generate_followups()       — 3 smart follow-up questions as a clean list

All LLM calls use gpt-4o-mini to keep cost low.
These functions are stateless — session persistence is handled by session_manager.
"""

import os
from openai import OpenAI

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Import your existing lab recommender — path may vary depending on project layout
try:
    from tool_guidance.recommend_tools import recommend_tools as _recommend
    _HAS_RECOMMENDER = True
except ImportError:
    _HAS_RECOMMENDER = False
    print("⚠️  recommend_tools not found — will always generate custom labs")


# ── Practical intent detection ────────────────────────────────────────────────

def detect_practical_intent(question: str) -> bool:
    """
    Returns True if the question suggests the user wants to BUILD,
    DESIGN, IMPLEMENT, or CONFIGURE something — i.e. a lab is appropriate.
    Returns False for concept questions, comparisons, and advice requests.
    """
    prompt = f"""
Does this question require the user to actually BUILD or IMPLEMENT something hands-on?

Return ONLY YES or NO.

YES — the user wants to create, build, implement, design, or configure something:
  - "How do I build a RAG pipeline?"
  - "How do I set up a vector database?"
  - "How do I implement authentication in my app?"
  - "How do I design a scalable architecture?"
  - "How do I configure a Copilot agent?"
  - "How do I deploy this to Azure?"
  - "How do I structure my Power Automate flows?"

NO — the user is asking for advice, explanation, comparison, or decision guidance:
  - "What tools would you use to make this?"
  - "How do you decide which SDK to use?"
  - "What is the difference between X and Y?"
  - "What are best practices for X?"
  - "Can you explain how X works?"
  - "What is microservices architecture?"
  - "Why would I use an API gateway?"

Key test: Is the user about to BUILD, DESIGN, or CONFIGURE something RIGHT NOW?
  "How do I design X"    → YES
  "How do I implement X" → YES
  "What is X"            → NO
  "Which X should I use" → NO

Question: {question}
"""
    try:
        resp = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return resp.choices[0].message.content.strip().upper() == "YES"
    except:
        return False


# ── Lab retrieval + generation ────────────────────────────────────────────────

def get_or_generate_lab(question: str) -> str:
    """
    1. Try recommend_tools() — returns official lab if score ≥ threshold.
    2. If no official lab matches (or recommender unavailable), generate a
       custom lab with the LLM.
    Returns the lab text string, or "" if no lab is appropriate.
    """
    if _HAS_RECOMMENDER:
        try:
            official = _recommend(question)

            if official:
                lab_text = official[0].lower()
                q        = question.lower()

                rag_keywords = [
                    "rag", "knowledge graph", "embedding",
                    "vector", "retrieval", "semantic search",
                ]

                # If the question is RAG-specific but the official lab isn't,
                # the official lab is irrelevant — generate custom instead
                if (any(k in q for k in rag_keywords)
                        and not any(k in lab_text for k in rag_keywords)):
                    return _generate_custom_lab(question)

                return official[0]   # official lab is relevant

        except Exception as e:
            print(f"recommend_tools error: {e} — falling back to custom lab")

    # No official lab found or recommender unavailable
    return _generate_custom_lab(question)


def _generate_custom_lab(question: str) -> str:
    prompt = f"""
You are an expert Microsoft technical trainer.
Create a practical hands-on mini lab for the topic below.

FORMAT EXACTLY — do not add extra sections:

📚 [Title]

🎯 Goal:
1-2 sentence objective.

👉 Tasks:
• Task 1
• Task 2
• Task 3
• Task 4
• Task 5

✅ What You'll Learn:
Specific skills gained.

Topic: {question}
"""
    try:
        resp = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Custom lab generation error: {e}")
        return ""


# ── Follow-up question generator ──────────────────────────────────────────────

def generate_followups(question: str, answer: str) -> list[str]:
    """
    Returns exactly 3 follow-up questions as a clean Python list of strings.
    Designed to render as clickable buttons in Streamlit.
    """
    prompt = f"""
Based on this question and answer, generate exactly 3 smart follow-up questions
a business user learning Microsoft tools would naturally ask next.

Rules:
- Each question must be on its own line, numbered 1. 2. 3.
- Questions should go deeper, not repeat what was already answered.
- Keep each question under 15 words.
- Do not add any intro text or explanation — just the 3 numbered questions.

Question: {question}
Answer: {answer[:600]}
"""
    try:
        resp = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        raw = resp.choices[0].message.content.strip()

        # Parse "1. ...\n2. ...\n3. ..." into a clean list
        lines = [
            line.lstrip("123. ").strip()
            for line in raw.splitlines()
            if line.strip() and line.strip()[0].isdigit()
        ]
        return lines[:3]

    except Exception as e:
        print(f"Follow-up generation error: {e}")
        return []