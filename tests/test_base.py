import requests
from unittest.mock import patch, MagicMock
from jwsoup.text import scrape_single_page, scrape_multi_page

# Sample mock data for testing
BASE_URL = (
    "https://www.jw.org/fr/biblioth%C3%A8que/bible/bible-d-etude/livres/Gen%C3%A8se/"
)
MOCK_URL = f"{BASE_URL}1/"
MOCK_VERSES = [
    ("verse_1", "In the beginning God created the heavens and the earth."),
    ("verse_2", "Now the earth was formless and empty."),
]
MOCK_NEXT_URL = f"{BASE_URL}2/"


def mock_response(content: str, status_code: int = 200) -> MagicMock:
    """Helper to create a mock HTTP response."""
    response = MagicMock()
    response.status_code = status_code
    response.content = content.encode("utf-8")
    return response


@patch("requests.get")
def test_scrape_single_page(mock_get):
    """Test that scrape_single_page correctly scrapes verses from a single page."""
    # Arrange
    mock_get.return_value = mock_response(
        """
        <html>
            <body>
                <span class='verse' id='verse_1'>In the beginning God created the heavens and the earth.</span>
            </body>
        </html>
    """
    )

    # Act
    verses, next_url = scrape_single_page(MOCK_URL)

    # Assert
    assert len(verses) == 1, "Should scrape one verse from the page."
    assert verses[0] == MOCK_VERSES[0], "Scraped verse should match the expected data."
    assert next_url is None, "No next URL should be found on this mock page."


@patch("requests.get")
def test_scrape_multi_page(mock_get):
    """Test that scrape_multi_page correctly scrapes multiple pages."""
    # Arrange
    mock_get.side_effect = [
        mock_response(
            f"""
            <html>
                <body>
                    <span class='verse' id='verse_1'>In the beginning God created the heavens and the earth.</span>
                    <div class='navLinkNext'>
                        <a class='primaryButton articleNavButton jsBibleChapter' href='/fr/biblioth%C3%A8que/bible/bible-d-etude/livres/Gen%C3%A8se/2/'>Next</a>
                    </div>
                </body>
            </html>
        """
        ),
        mock_response(
            """
            <html>
                <body>
                    <span class='verse' id='verse_2'>Now the earth was formless and empty.</span>
                </body>
            </html>
        """
        ),
    ]

    # Act
    all_verses = scrape_multi_page(MOCK_URL, max_pages=2, page_sep="livres")

    # Assert
    assert len(all_verses) == 2, "Should scrape two verses across multiple pages."
    assert all_verses == MOCK_VERSES, "Scraped verses should match the expected data."


@patch("requests.get")
def test_scrape_single_page_error(mock_get):
    """Test that scrape_single_page gracefully handles request errors."""
    # Arrange
    mock_get.side_effect = requests.exceptions.RequestException("Network error")

    # Act
    verses, next_url = scrape_single_page(MOCK_URL)

    # Assert
    assert verses == [], "Should return an empty list for verses on request error."
    assert next_url is None, "Next URL should be None on request error."


@patch("requests.get")
def test_scrape_empty_page(mock_get):
    """Test that scrape_single_page handles empty page content gracefully."""
    # Arrange
    mock_get.return_value = mock_response("<html><body></body></html>")

    # Act
    verses, next_url = scrape_single_page(MOCK_URL)

    # Assert
    assert verses == [], "Should return an empty list for a page with no verse content."
    assert next_url is None, "Next URL should be None for a page with no navigation."


@patch("requests.get")
def test_scrape_multi_page_max_limit(mock_get):
    """Test that scrape_multi_page respects the max_pages limit."""
    # Arrange
    mock_get.side_effect = [
        mock_response(
            f"""
            <html>
                <body>
                    <span class='verse' id='verse_1'>In the beginning God created the heavens and the earth.</span>
                    <div class='navLinkNext'>
                        <a class='primaryButton articleNavButton jsBibleChapter' href='/fr/biblioth%C3%A8que/bible/bible-d-etude/livres/Gen%C3%A8se/2/'>Next</a>
                    </div>
                </body>
            </html>
        """
        ),
        mock_response(
            """
            <html>
                <body>
                    <span class='verse' id='verse_2'>Now the earth was formless and empty.</span>
                </body>
            </html>
        """
        ),
    ]

    # Act
    all_verses = scrape_multi_page(MOCK_URL, max_pages=1, page_sep="livres")

    # Assert
    assert (
        len(all_verses) == 1
    ), "Should scrape only one page when max_pages is set to 1."
    assert all_verses == [
        MOCK_VERSES[0]
    ], "Scraped verses should match the first page data."
