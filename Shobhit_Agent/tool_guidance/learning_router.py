from recommend_tools import recommend_tools
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# -------- Intent Detection --------
def detect_learning_intent(question):

    prompt = f"""
Does this question require practical hands-on learning?

Return ONLY:

YES
NO

YES if:
- building
- implementing
- designing systems
- projects/labs help understanding

NO if:
- definitions
- explanations
- conceptual only

Question:
{question}
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return resp.choices[0].message.content.strip()


# -------- Custom Lab Generator --------
def generate_custom_lab(question):

    prompt = f"""
Create a concise practical mini lab.

Format EXACTLY:

📚 Title

🎯 Goal:
2 lines objective.

👉 Tasks:
3–4 practical steps.

✅ Learning Outcomes:
Skills gained.

Topic:
{question}
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )

    return resp.choices[0].message.content


# -------- Main Router --------
def get_learning_help(question):

    intent = detect_learning_intent(question)

    if intent == "NO":
        return None

    # Try official labs first
    official_lab = recommend_tools(question)

    if official_lab:
        return official_lab[0]

    # Generate custom lab if none exists
    return generate_custom_lab(question)


# -------- Test --------
if __name__ == "__main__":

    q = input("Ask: ")
    result = get_learning_help(q)

    if result:
        print("\n" + result)
    else:
        print("\nNo lab needed.")