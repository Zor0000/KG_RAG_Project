import asyncio

from ingestion.crawler import crawl_site
from ingestion.html_cleaner import clean_html
from ingestion.segmenter import segment_by_headings
from knowledge_graph.neo4j_client import Neo4jClient
from knowledge_graph.kg_builder import save_section


async def main():
    pages = await crawl_site()

    kg = Neo4jClient()

    with kg.driver.session() as session:

        for page in pages:

            if not page.html:
                continue

            print(f"\nProcessing: {page.url}")

            soup = clean_html(page.html)
            segments = segment_by_headings(soup, page.url)

            print(f"Segments found: {len(segments)}")

            for seg in segments:
                session.execute_write(save_section, seg)

            print(f"Saved {len(segments)} sections")

    kg.close()


asyncio.run(main())
