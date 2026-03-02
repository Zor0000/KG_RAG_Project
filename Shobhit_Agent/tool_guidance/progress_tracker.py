import json
from datetime import datetime
from pathlib import Path

FILE = Path(__file__).parent / "progress.json"


def log_progress(user, query, tools, labs):

    # Create file if missing or empty
    if not FILE.exists():
        FILE.write_text("[]")

    text = FILE.read_text().strip()

    if not text:
        data = []
    else:
        try:
            data = json.loads(text)
        except:
            data = []

    # Append new entry
    data.append({
        "user": user,
        "query": query,
        "tools": tools,
        "labs": labs,
        "timestamp": str(datetime.now())
    })

    FILE.write_text(json.dumps(data, indent=2))
