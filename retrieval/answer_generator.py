"""
answer_generator.py  (updated)
───────────────────────────────
Generates grounded answers from retrieved chunks.

Changes from original:
  - generate_answer() accepts an optional `session_context` dict
    (keys: summary, recent_turns) so the LLM has full conversation
    history when forming its response.
  - generate_guided_preview() also accepts `session_context` for
    consistency across both answer modes.
  - Answer quality prompt significantly upgraded:
      • Conversational tone, no documentation-page formatting
      • Explicitly told NOT to suggest labs (handled separately)
      • Told to skip re-explaining things the summary shows user knows
      • Cites sources naturally in-line rather than dumping a list

All other behaviour (sources extraction, return shape) is unchanged
so the Streamlit app's rendering code needs no edits.
"""

import os
from openai import OpenAI

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_history_block(recent_turns: list[dict], max_turns: int = 3) -> str:
    """Format last N turns as a compact readable block for the prompt."""
    if not recent_turns:
        return ""
    lines = []
    for t in recent_turns[-max_turns:]:
        lines.append(f"User: {t['question']}")
        lines.append(f"Agent: {t['answer'][:300]}...")
    return "\n".join(lines)


def _build_context_block(chunks: list[dict]) -> str:
    """Concatenate retrieved chunk texts into a single context string."""
    return "\n\n".join(
        f"[{i+1}] {c.get('text', '')}"
        for i, c in enumerate(chunks)
    )


def _extract_sources(chunks: list[dict]) -> list[str]:
    """Pull unique source URLs / titles from retrieved chunks."""
    seen   = set()
    sources = []
    for c in chunks:
        src = c.get("source") or c.get("url") or c.get("canonical_topic", "")
        if src and src not in seen:
            seen.add(src)
            sources.append(src)
    return sources


# ── Main answer generator ─────────────────────────────────────────────────────

def generate_answer(
    query:           str,
    results:         list[dict],
    session_context: dict | None = None,
    persona:         str = "All",
    product:         str = "All",
) -> dict:
    """
    Generate a grounded answer from retrieved chunks.

    Parameters
    ----------
    query           : The (rewritten) user question.
    results         : List of retrieved chunk dicts from unified_search().
    session_context : Optional dict with keys:
                        'summary'      — rolling conversation summary from Redis
                        'recent_turns' — list of last N turn dicts
    persona         : Selected persona from sidebar (NoCode / LowCode /
                      ProDeveloper / Admin / Architect / All).
                      Shapes tone, depth, and language of the response.
    product         : Selected product from sidebar (copilot_studio /
                      azure_bot_service / autogen / All).
                      Focuses the answer on the relevant product context.

    Returns
    -------
    { "answer": str, "sources": list[str] }
    """
    ctx = session_context or {}
    summary      = ctx.get("summary", "")
    recent_turns = ctx.get("recent_turns", [])

    history_block  = _build_history_block(recent_turns)
    context_block  = _build_context_block(results)
    sources        = _extract_sources(results)

    # ── Persona instruction ───────────────────────────────────────────────────
    # Each persona gets a specific tone, depth, and vocabulary instruction
    # so the same retrieved knowledge reads differently for a no-code user
    # vs a professional developer vs an architect.
    persona_instructions = {
        "NoCode": (
            "The user has no coding background. Use plain English only. "
            "Avoid all technical jargon, code snippets, and acronyms unless you explain them. "
            "Use simple analogies and step-by-step language a business user can follow."
        ),
        "LowCode": (
            "The user is comfortable with basic configuration and simple formulas "
            "but is not a programmer. Use clear language, explain concepts briefly, "
            "and include simple examples or UI steps where helpful. Avoid deep code."
        ),
        "ProDeveloper": (
            "The user is an experienced developer. Be technical and precise. "
            "Include code snippets, API references, and implementation details. "
            "Skip basic explanations — get straight to the technical answer."
        ),
        "Admin": (
            "The user is an IT administrator focused on configuration, governance, "
            "security, licensing, and deployment. Focus on admin settings, policies, "
            "tenant-level controls, and operational concerns. Avoid end-user fluff."
        ),
        "Architect": (
            "The user is a solution architect. Focus on design patterns, scalability, "
            "integration points, trade-offs, and enterprise architecture decisions. "
            "Be strategic and assume deep technical knowledge."
        ),
        "All": (
            "Adapt your tone to the complexity of the question. "
            "Be clear and direct without assuming a specific skill level."
        ),
    }
    persona_instruction = persona_instructions.get(persona, persona_instructions["All"])

    # ── Product instruction ───────────────────────────────────────────────────
    product_names = {
        "copilot_studio":    "Microsoft Copilot Studio",
        "azure_bot_service": "Azure Bot Service",
        "autogen":           "AutoGen (Microsoft multi-agent framework)",
        "All":               "Microsoft Copilot Studio, Azure Bot Service, and AutoGen",
    }
    product_name = product_names.get(product, product)
    product_instruction = (
        f"Focus your answer specifically on {product_name}. "
        f"If the retrieved context covers other products, prioritise {product_name} content."
        if product != "All"
        else "Cover whichever product is most relevant to the question."
    )

    system_prompt = f"""
You are an expert AI assistant specialising in Microsoft Copilot Studio,
Microsoft 365, Power Platform, and Azure AI / Azure OpenAI.

--- Persona ---
{persona_instruction}

--- Product focus ---
{product_instruction}

{"--- What this user has discussed before (do NOT re-explain things they already know) ---" if summary else ""}
{summary}

{"--- Recent conversation ---" if history_block else ""}
{history_block}

--- Retrieved knowledge (use this as your primary source of truth) ---
{context_block}

Answer rules:
1. Adapt tone and depth to the persona instruction above — this is the most important rule.
2. Only use bullet points or numbered lists if the question genuinely asks for a list.
3. Match length to complexity — short questions deserve short answers.
4. If the retrieved context contains the answer, ground your response in it.
   If it does not, say so honestly rather than guessing.
5. Never pad with "Great question!", "I hope this helps!", or similar filler.
6. Do NOT suggest labs or tutorials — that is handled separately by the system.
7. Do NOT re-explain concepts the conversation summary shows the user already knows.
8. Cite sources naturally inline when relevant (e.g. "According to the Copilot Studio
   docs...") — do not dump a raw list of URLs inside your answer text.
"""

    try:
        resp = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": query},
            ],
            temperature=0.3,
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        answer = f"Sorry, I encountered an error generating the answer: {e}"

    return {"answer": answer, "sources": sources}


# ── Guided preview generator ──────────────────────────────────────────────────

def generate_guided_preview(
    query:           str,
    preview_chunks:  list[dict],
    product_options: list[str],
    session_context: dict | None = None,
) -> dict:
    """
    Generate a preview answer shown during guided clarification mode,
    where results span multiple products and the user needs to pick one.

    Parameters
    ----------
    query           : The (rewritten) user question.
    preview_chunks  : Subset of top chunks shown before product is chosen.
    product_options : List of product names the user can pick from.
    session_context : Same optional context dict as generate_answer().

    Returns
    -------
    { "answer": str, "sources": list[str] }
    """
    ctx = session_context or {}
    summary = ctx.get("summary", "")

    context_block   = _build_context_block(preview_chunks)
    sources         = _extract_sources(preview_chunks)
    products_listed = "\n".join(f"- {p}" for p in product_options)

    system_prompt = f"""
You are an expert AI assistant for Microsoft Copilot Studio, Microsoft 365,
Power Platform, and Azure AI.

{"--- Previous conversation context ---" if summary else ""}
{summary}

--- Retrieved preview context ---
{context_block}

The user's question touches multiple Microsoft products.
Give a brief (2-4 sentence) overview answer using the preview context,
then naturally ask the user to specify which product they want to focus on.

Available products:
{products_listed}

Keep the tone conversational. Do not use headers or bullet lists.
Do not suggest labs or tutorials.
"""

    try:
        resp = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": query},
            ],
            temperature=0.3,
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        answer = f"I found relevant information across multiple products. Please select one: {', '.join(product_options)}"

    return {"answer": answer, "sources": sources}