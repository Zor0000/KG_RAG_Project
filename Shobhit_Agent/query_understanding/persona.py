from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PERSONAS = [
    "Beginner",
    "Developer",
    "BusinessUser",
    "Consultant",
    "EnterpriseArchitect"
]


def detect_persona(query: str):

    prompt = f"""
Identify the most likely persona:

{", ".join(PERSONAS)}

Query:
{query}

Return only the persona.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content.strip()
