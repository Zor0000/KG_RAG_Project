from bs4 import BeautifulSoup

def clean_html(html: str):
    soup = BeautifulSoup(html, "html.parser")

    # Remove useless UI elements
    for tag in soup(["nav", "footer", "aside", "script", "style"]):
        tag.decompose()

    # Microsoft docs specific noise
    for div in soup.select(".feedback, .breadcrumbs, .buttons"):
        div.decompose()

    return soup
