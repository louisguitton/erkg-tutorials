import requests
import spacy
from bs4 import BeautifulSoup, SoupStrainer
from spacy.tokens import DocBin

SPACY_MODEL: str = "en_core_web_md"
SCRAPE_HEADERS: dict[str, str] = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36",
}

URLS = [
     "https://www.icij.org/investigations/pandora-papers/secret-real-estate-purchases-are-a-driving-force-behind-the-offshore-economy/",
     "https://www.icij.org/investigations/pandora-papers/former-czech-leaders-secret-french-estate-revealed-in-pandora-papers-listed-for-sale/"
]


def search(tag, attrs):
    """Parsing only part of the document, using the fact that an ICIJ article has a header and a body.

    References:
    - https://stackoverflow.com/a/34536484/3823815
    - https://www.crummy.com/software/BeautifulSoup/bs4/doc/#parsing-only-part-of-a-document
    - https://www.crummy.com/software/BeautifulSoup/bs4/doc/#a-function
    """
    if tag == "header" and "post-header" in attrs.get("class", []):
        return True

    if tag == "div" and "post-body" in attrs.get("class", []):
        return True
    
    # TODO: stretch goal: demo advanced search based scraping to filter out news recommendations
    """   
    if tag == "div" and any(forbidden_class in attrs.get("class", []) for forbidden_class in ["newsrelated-widget", "banner-donation-default", "responsive-iframe-widget"]):
        return False
    """


scrape_nlp: spacy.Language = spacy.load(SPACY_MODEL)

# ref: https://spacy.io/api/docbin
doc_bin = DocBin()

for url in URLs:

    response: requests.Response = requests.get(
        url,
        headers=SCRAPE_HEADERS,
    )

    soup: BeautifulSoup = BeautifulSoup(
        response.text, features="html.parser", parse_only=SoupStrainer(search)
    )
    text_contents = soup.find_all(
        ["h1", "p", "figcaption"],
    )

    scrape_doc: spacy.tokens.doc.Doc = scrape_nlp(
        "\n".join(
            [
                text_content.text.strip() + "." * (idx == 0)
                for idx, text_content in enumerate(text_contents)
            ]
        )
    )
    doc_bin.add(scrape_doc)

doc_bin.to_disk("./data.spacy")
