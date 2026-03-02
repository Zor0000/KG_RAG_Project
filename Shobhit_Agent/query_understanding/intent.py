from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INTENTS = [
    "WhatIs",
    "HowTo",
    "CanI",
    "Troubleshooting",
    "Comparison",
    "Recommendation",
    "Explanation",
    "SetupGuide",
    "BestPractices",
    "PricingLicensing",
    "Integration",
    "CapabilitiesLimits",
    "GreetingSmalltalk"
]


def classify_intent(query: str) -> str:

    prompt = f"""
Classify the user's intent into ONE category:

{", ".join(INTENTS)}

User query:
{query}

Return only the intent name.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content.strip()
