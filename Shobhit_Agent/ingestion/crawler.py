import os
import json
import asyncio
import hashlib
from collections import deque
from urllib.parse import urljoin, urlparse

from crawl4ai import AsyncWebCrawler


# ================= CONFIG =================

START_URL = "https://learn.microsoft.com/en-us/microsoft-copilot-studio/"
ALLOWED_PREFIX = "https://learn.microsoft.com/en-us/microsoft-copilot-studio"

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_PAGES_DIR = os.path.join(DATA_DIR, "raw_pages")
VISITED_FILE = os.path.join(DATA_DIR, "visited_urls.json")

MAX_PAGES = 3000  # safety limit

os.makedirs(RAW_PAGES_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# ================= HELPERS =================

def safe_filename(url: str):
    return hashlib.sha256(url.encode("utf-8")).hexdigest() + ".json"


def load_visited():
    if not os.path.exists(VISITED_FILE):
        return set()
    try:
        with open(VISITED_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    except Exception:
        return set()


def save_visited(visited):
    with open(VISITED_FILE, "w", encoding="utf-8") as f:
        json.dump(list(visited), f, indent=2)


def extract_internal_links(base_url, result):
    urls = []

    if not result.links or "internal" not in result.links:
        return urls

    for link in result.links["internal"]:
        href = link.get("href")
        if not href:
            continue

        full = urljoin(base_url, href)
        parsed = urlparse(full)

        # stay inside Copilot docs only
        if not full.startswith(ALLOWED_PREFIX):
            continue

        clean = parsed._replace(fragment="").geturl()
        urls.append(clean)

    return urls


# ================= MAIN CRAWLER =================

async def crawl_site():

    visited = load_visited()
    queue = deque([START_URL])
    new_pages = 0

    print("🚀 Starting deep crawl...")

    async with AsyncWebCrawler() as crawler:

        while queue and len(visited) < MAX_PAGES:

            url = queue.popleft()

            if url in visited:
                continue

            try:
                result = await crawler.arun(url)
            except Exception as e:
                print(f"❌ Failed: {url} → {e}")
                continue

            visited.add(url)
            save_visited(visited)

            markdown = getattr(result, "markdown", None)

            # Skip empty pages
            if not markdown or len(markdown.strip()) < 200:
                continue

            out_path = os.path.join(RAW_PAGES_DIR, safe_filename(url))

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "url": url,
                        "content": markdown
                    },
                    f,
                    ensure_ascii=False,
                    indent=2
                )

            new_pages += 1

            # Extract next links
            new_links = extract_internal_links(url, result)

            for link in new_links:
                if link not in visited:
                    queue.append(link)

            print(f"✅ Saved: {url} | Queue: {len(queue)}")

    print("\n🎉 Crawl Complete")
    print(f"📄 New pages saved: {new_pages}")
    print(f"🧠 Total unique pages: {len(visited)}")


if __name__ == "__main__":
    asyncio.run(crawl_site())
