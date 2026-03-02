def topic_based_chunking(text):
    """
    Simple topic-aware chunking:
    Splits on headings + keeps semantic size.
    """

    import re

    sections = re.split(r"\n#{1,3}\s", text)

    chunks = []

    for sec in sections:
        words = sec.split()

        chunk_size = 400
        overlap = 80

        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if len(chunk) > 50:
                chunks.append(chunk)

    return chunks
