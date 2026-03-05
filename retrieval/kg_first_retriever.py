# retrieval/kg_first_retriever.py

from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import os


# ============================================================
# 🔹 Neo4j Config
# ============================================================

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Neer@j080105"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ============================================================
# 🔹 Initialize Graph
# ============================================================

graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USER,
    password=NEO4J_PASSWORD,
)


# ============================================================
# 🔹 LLM
# ============================================================

llm = ChatOpenAI(
    model="gpt-5.2",
    temperature=0,
    openai_api_key=OPENAI_API_KEY
)


# ============================================================
# 🔹 Custom Cypher Prompt
# ============================================================

CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Cypher generator.

Use ONLY the provided schema.

Schema:
{schema}

Rules:
- Only use existing labels and relationships.
- Never invent properties.
- Return Chunk nodes.
- Always LIMIT 15.

User Question:
{question}

Generate ONLY the Cypher query.
"""

cypher_prompt = PromptTemplate(
    input_variables=["schema", "question"],
    template=CYPHER_GENERATION_TEMPLATE,
)


# ============================================================
# 🔹 Chain
# ============================================================

chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    cypher_prompt=cypher_prompt,
    verbose=True,
    return_intermediate_steps=True,
    allow_dangerous_requests=True,
)


# ============================================================
# 🔹 KG-First Search
# ============================================================

def kg_first_search(query: str):

    result = chain.invoke({"query": query})

    answer = result["result"]

    intermediate = result.get("intermediate_steps", [])

    generated_cypher = None
    for step in intermediate:
        if isinstance(step, dict) and "query" in step:
            generated_cypher = step["query"]

    return {
        "answer": answer,
        "generated_cypher": generated_cypher,
    }


# ============================================================
# 🔹 CLI Test
# ============================================================

if __name__ == "__main__":

    while True:
        q = input("\n💬 Ask (or exit): ")

        if q.lower() == "exit":
            break

        res = kg_first_search(q)

        print("\n🧠 Generated Cypher:\n")
        print(res["generated_cypher"])

        print("\n📘 Answer:\n")
        print(res["answer"])