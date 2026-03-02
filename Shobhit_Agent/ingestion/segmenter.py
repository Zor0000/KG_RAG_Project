def segment_by_headings(soup, url):
    segments = []
    current = None

    for tag in soup.find_all(["h1", "h2", "h3", "p", "li", "pre"]):

        if tag.name in ["h1", "h2", "h3"]:
            if current:
                segments.append(current)

            current = {
                "title": tag.get_text(strip=True),
                "content": [],
                "url": url
            }

        else:
            if current:
                current["content"].append(tag.get_text(strip=True))

    if current:
        segments.append(current)

    return segments
