import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load .env from project root
load_dotenv()

# Hard check (fails fast, industry standard)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY not found. "
        "Ensure .env exists in project root and contains OPENAI_API_KEY."
    )

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    timeout=60,
    max_retries=5,
)

def classify(text: str) -> str:
    prompt = f"""
Classify the following documentation into one category:
Concept, HowTo, Tutorial, Reference, API, BestPractice, Troubleshooting.

Text:
{text[:2000]}

Return only the category name.
"""
    return llm.invoke(prompt).content.strip()
