# KG-RAG Project Setup Guide

This guide provides step-by-step instructions for the backend team to set up and run the **Graph-Guided KG-RAG** platform securely.

## Prerequisites

1. **Docker & Docker Compose**: Verify that Docker is installed and running on your system.
2. **Python 3.9+**: For running the backend services and ingestion pipeline.
3. **OpenAI API Key**: Required for embeddings and LLM generation.

## ⚙️ Step 1: Configure Environment Variables

Create a `.env` file in the root of the project (`d:\KG_RAG_Project_V2\.env`) or export the following environment variables in your terminal:

```env
OPENAI_API_KEY="your-openai-api-key-here"

# (Optional) Redis settings if you change the defaults:
# REDIS_HOST="localhost"
# REDIS_PORT=6379
# REDIS_PASSWORD="your-redis-password"
```

## 🐳 Step 2: Start the Core Infrastructure

The project uses Docker Compose to run **PostgreSQL, Redis, and Milvus**. 

Run the following command from the project root:
```bash
docker-compose up -d
```

### ⚠️ Important: Neo4j Requirement
The codebase currently relies on a **Neo4j** graph database that is *not* included in `docker-compose.yml`. Based on the application's configuration, it expects Neo4j at `bolt://localhost:7687` with the following credentials:
- **User**: `neo4j`
- **Password**: `Neer@j080105`

You can start a compatible Neo4j instance using Docker:
```bash
docker run -d \
  --name neo4j-kg_rag \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/Neer@j080105 \
  neo4j:5.20.0
```

## 📦 Step 3: Install Python Dependencies

It is highly recommended to use a virtual environment.

```bash
# Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# Linux/Mac
# source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

## 🔄 Step 4: Run Data Ingestion (First Time Run)

If you need to populate the Milvus Vector Database and the Neo4j Knowledge Graph with data, run the ingestion pipeline.

*(Note: You can tweak the target URLs and product config directly inside `ingestion/run_pipeline.py` under the `PRODUCT CONFIG` section).*

```bash
python -m ingestion.run_pipeline
```
This will sequentially execute crawling, cleaning, chunking, enriching, embedding, and KG ingestion.

## 🚀 Step 5: Start the Backend API Server

The backend runs a **FastAPI** application utilizing LangServe.

```bash
uvicorn api.server:app --reload --host 0.0.0.0 --port 8000
```
- API is accessible at: `http://localhost:8000`
- Swagger Documentation: `http://localhost:8000/docs`
- LangServe Playground: `http://localhost:8000/rag/playground`

## 🖥️ Step 6: Start the Frontend UI (Optional)

To interact with the application via the UI, start the **Streamlit** application:

```bash
streamlit run ui/app_graph_rag.py
```
- The web interface will open in your browser automatically at `http://localhost:8501`.
