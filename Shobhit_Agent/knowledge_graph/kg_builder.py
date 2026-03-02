import os
import json
import re
from neo4j import GraphDatabase


RAW_DIR = "data/raw_pages"

driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "password")
)


def extract_concepts(text):
    # Simple but effective concept extraction
    return list(set(
        re.findall(r"\b[A-Z][a-zA-Z0-9\-]{3,}\b", text)
    ))


def split_sections(text):
    sections = []
    current = None

    for line in text.splitlines():
        if line.startswith("#"):
            if current:
                sections.append(current)
            current = {"title": line.strip("# "), "content": []}
        elif current:
            current["content"].append(line)

    if current:
        sections.append(current)

    return sections


def save_to_kg(tx, url, title, content, concepts):

    tx.run("""
        MERGE (d:Document {url:$url})
        MERGE (s:Section {title:$title, url:$url})
        SET s.content=$content
        MERGE (d)-[:HAS_SECTION]->(s)
    """, url=url, title=title, content=content)

    for concept in concepts:
        tx.run("""
            MATCH (s:Section {title:$title, url:$url})
            MERGE (c:Concept {name:$concept})
            MERGE (s)-[:MENTIONS]->(c)
        """, title=title, url=url, concept=concept)


def build_kg():

    files = os.listdir(RAW_DIR)

    with driver.session() as session:

        for file in files:

            with open(os.path.join(RAW_DIR, file),
                      encoding="utf-8") as f:

                data = json.load(f)

            url = data["url"]
            text = data["content"]

            sections = split_sections(text)

            for sec in sections:

                content = "\n".join(sec["content"])
                concepts = extract_concepts(content)

                session.execute_write(
                    save_to_kg,
                    url,
                    sec["title"],
                    content,
                    concepts
                )

    print("KG build complete.")


if __name__ == "__main__":
    build_kg()
