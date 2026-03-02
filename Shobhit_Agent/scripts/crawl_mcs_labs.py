import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urljoin

BASE_URL = "https://pratapladhani.github.io/mcs-labs/labs/"
OUTPUT_FILE = "tool_guidance/official_labs.json"


def get_lab_links():
    response = requests.get(BASE_URL)
    soup = BeautifulSoup(response.text, "html.parser")

    links = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/labs/" in href and href.endswith(".html"):
            full_url = urljoin(BASE_URL, href)
            links.append(full_url)

    return list(set(links))


def scrape_lab(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    title = soup.find("h1").text.strip() if soup.find("h1") else "No Title"

    paragraphs = soup.find_all("p")
    content = "\n".join([p.text.strip() for p in paragraphs[:10]])

    return {
        "title": title,
        "url": url,
        "content": content
    }


def main():
    labs = []
    links = get_lab_links()

    for link in links:
        try:
            lab = scrape_lab(link)
            labs.append(lab)
            print("Scraped:", lab["title"])
        except Exception as e:
            print("Error:", e)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(labs, f, indent=2)

    print("Labs saved to official_labs.json")


if __name__ == "__main__":
    main()