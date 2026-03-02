def rerank(candidates: list[str], intent: str, persona: str, top_k=8):
    """
    Simple heuristic reranker.
    Later you can replace with cross-encoder.
    """
    scored = []

    for text in candidates:
        score = 0

        if intent.lower() in text.lower():
            score += 1

        if persona.lower() in text.lower():
            score += 0.5

        score += min(len(text) / 500, 1)

        scored.append((score, text))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [t for _, t in scored[:top_k]]
