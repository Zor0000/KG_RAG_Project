import os
from pymilvus import Collection, connections
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# =========================
# CONFIG
# =========================

MILVUS_HOST = "localhost"
MILVUS_PORT = "29530"
COLLECTION_NAME = "copilot_studio_docs"

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# =========================
# CONNECT TO MILVUS
# =========================

def get_collection():
    try:
        connections.connect(
            alias="default",
            host=MILVUS_HOST,
            port=MILVUS_PORT
        )

        collection = Collection(COLLECTION_NAME)
        collection.load()
        return collection

    except Exception as e:
        print("Milvus error:", e)
        return None


# =========================
# LLM LAB STRUCTURING
# =========================

def extract_lab_with_llm(question, content, url):

    content = content[:4000]

    prompt = f"""
You are a Microsoft Copilot training assistant.

From the lab content below, extract:

1. Lab Title
2. Why this lab is relevant to the user's question
3. What the user will learn (bullet points)

Ignore:
- difficulty level
- estimated time
- total labs
- download links
- repeated titles
- metadata

User Question:
{question}

Lab Content:
{content}

Return ONLY in this format:

📚 Lab Recommendation

🔹 Title:
<Title>

🔹 Why this lab:
<Reason in 2-3 lines>

🔹 What you'll learn:
• point
• point
• point

🔗 Lab Link:
{url}
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        return resp.choices[0].message.content

    except Exception as e:
        print("LLM extraction error:", e)
        return None


# =========================
# MAIN FUNCTION
# =========================

def recommend_tools(question):

    collection = get_collection()
    if not collection:
        return []

    try:
        emb = embed_model.encode(question).tolist()

        results = collection.search(
            data=[emb],
            anns_field="embedding",
            param={
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            },
            limit=5,
            expr='url like "%labs%"',
            output_fields=["content", "url"]
        )

        labs = []

        for hit in results[0]:

            # similarity threshold
            if hit.score < 0.30:
                continue

            content = hit.entity.get("content", "")
            url = hit.entity.get("url", "")

            structured_output = extract_lab_with_llm(
                question,
                content,
                url
            )

            if structured_output:
                labs.append(structured_output)
                break  # return top 1

        return labs

    except Exception as e:
        print("Search error:", e)
        return []