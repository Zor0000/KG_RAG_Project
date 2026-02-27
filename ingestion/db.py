import psycopg2


def get_connection():
    return psycopg2.connect(
        dbname="kg_rag_platform",   # same DB you already have
        user="kg_rag_user",            # change if needed
        password="Neer@j080105",   # put your local postgres password
        host="localhost",
        port="5433"                 # use 5432 unless you changed it
    )