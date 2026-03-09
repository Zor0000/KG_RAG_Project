from fastapi import FastAPI
from langserve import add_routes
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from retrieval.graph_guided_retriever import retrieve_answer


app = FastAPI(
    title="GraphRAG API",
    version="1.0",
    description="Graph-guided RAG with Milvus + Neo4j"
)


# Define schema so playground can render input box
class QueryInput(BaseModel):
    input: str


# LangServe runnable
def rag_fn(data: dict):
    question = data.get("input")
    return retrieve_answer(question)


rag_chain = RunnableLambda(rag_fn).with_types(input_type=QueryInput)

add_routes(
    app,
    rag_chain,
    path="/rag"
)