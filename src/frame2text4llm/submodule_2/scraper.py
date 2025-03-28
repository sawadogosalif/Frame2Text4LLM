import requests
from bs4 import BeautifulSoup
import urllib
from loguru import logger
import pandas as pd
from .utils import clean_text


def scrape_single_page(url: str) -> tuple:
    """Scrape verses from a single page and return verses and next URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Error accessing {url}: {str(e)}")
        return [], None

    soup = BeautifulSoup(response.content, "html.parser")
    verses = []
    verse_elements = soup.find_all("span", class_="verse")

    for verse in verse_elements:
        verse_id = verse.get("id", "Unknown ID")
        verse_text = verse.get_text(separator=" ", strip=True)
        verse_text = clean_text(verse_text)
        verses.append((verse_id, verse_text))

    next_page = soup.find("div", class_="navLinkNext")
    next_url = None

    if next_page:
        next_button = next_page.find(
            "a", class_="primaryButton"
        )
        if next_button:
            next_url = next_button.get("href")
            if next_url:
                next_url = urllib.parse.urljoin(
                    url, next_url.split("#")[0]
                )  # Remove fragment

    return verses, next_url


def save_verses_to_parquet(
    verses: list, current_url: str, output_dir: str, page_sep: str = "books"
):
    """Save scraped verses to a parquet file."""
    verses_with_metadata = [
        {"verse_id": verse[0], "verse_text": verse[1], "source_url": current_url}
        for verse in verses
    ]
    root = urllib.parse.unquote(current_url.split(page_sep)[-1].strip("/"))
    df = pd.DataFrame(verses_with_metadata).assign(page=root)
    df.to_parquet(output_dir, partition_cols=["page"], engine="pyarrow")


def scrape_multi_page(
    start_url: str, output_dir: str, max_pages: int = 10000, page_sep: str = "books"
):
    """Scrape multiple pages starting from the given URL and store the data."""
    current_url = start_url
    i = 0
    all_verses = []

    while current_url and i < max_pages:
        logger.info(f"Scraping {current_url}")
        verses, next_url = scrape_single_page(current_url)

        if verses:
            save_verses_to_parquet(verses, current_url, output_dir, page_sep)
            all_verses.extend(verses)

        current_url = next_url
        i += 1

    logger.info(f"Scraping complete. Processed {i} pages.")
    return all_verses