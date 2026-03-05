from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def rewrite_query(query, intent, persona):

    prompt = f"""
Rewrite this query for best document retrieval.

Intent: {intent}
Persona: {persona}

Original query:
{query}

Rewritten query:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()
