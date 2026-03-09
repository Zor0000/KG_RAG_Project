import psycopg2

def get_connection():
    return psycopg2.connect(
        dbname="kg_rag_platform",
        user="kg_rag_user",
        password="yourpassword",
        host="localhost",
        port="5433"
    )