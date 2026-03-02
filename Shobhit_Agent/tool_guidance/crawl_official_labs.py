import requests
import json
import re
from bs4 import BeautifulSoup
from pathlib import Path

BASE_URL = "https://pratapladhani.github.io/mcs-labs/labs/"
OUTPUT = Path(__file__).parent / "official_labs.json"


def normalize_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s([?.!,"])', r'\1', text)
    text = text.replace('\xa0', ' ')
    return text.strip()


def clean_html(html):
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    return normalize_text(text)


def get_lab_links():
    res = requests.get(BASE_URL, timeout=30)
    soup = BeautifulSoup(res.text, "html.parser")

    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]

        if "/labs/" in href and href != "/labs/":
            full = "https://pratapladhani.github.io" + href
            if full not in links:
                links.append(full)

    return links


def crawl():
    labs = []

    links = get_lab_links()
    print(f"Found {len(links)} lab links")

    for url in links:
        try:
            r = requests.get(url, timeout=30)
            cleaned = clean_html(r.text)

            soup = BeautifulSoup(r.text, "html.parser")
            title = soup.find("h1")

            labs.append({
                "title": normalize_text(title.text) if title else "Copilot Lab",
                "url": url,
                "content": cleaned
            })

            print("Added:", url)

        except Exception as e:
            print("Error:", e)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(labs, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(labs)} labs → official_labs.json")


if __name__ == "__main__":
    crawl()