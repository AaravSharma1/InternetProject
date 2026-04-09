"""
HTML parser and link extractor.

parse_page(html, base_url) -> ParsedPage

Each extracted link carries:
  - resolved absolute URL
  - anchor text (cleaned)
  - ~200-char context window of surrounding text taken from the
    nearest block-level ancestor of the <a> tag.
"""

import re
from dataclasses import dataclass, field
from typing import List
from urllib.parse import urljoin, urldefrag, urlparse

from bs4 import BeautifulSoup, Tag


_ALLOWED_SCHEMES = frozenset(("http", "https"))
_BLOCK_TAGS = frozenset((
    "p", "div", "li", "td", "th", "article", "section",
    "blockquote", "dd", "dt", "figcaption", "h1", "h2",
    "h3", "h4", "h5", "h6",
))
_CONTEXT_WINDOW = 200   # chars on each side of the anchor text


@dataclass
class ExtractedLink:
    url: str
    anchor_text: str
    context: str          # ~200-char window from surrounding block text


@dataclass
class ParsedPage:
    url: str
    title: str
    text: str             # full visible text, whitespace-normalised
    links: List[ExtractedLink] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _nearest_block(tag: Tag) -> Tag:
    """Walk up the DOM until we hit a block-level element (or <body>)."""
    node = tag.parent
    while node and node.name not in _BLOCK_TAGS:
        if node.name in ("body", "[document]", None):
            return node
        node = node.parent
    return node or tag.parent


def _context_around(a_tag: Tag, window: int = _CONTEXT_WINDOW) -> str:
    """
    Extract roughly (2 * window) chars of text centred on the anchor.

    Strategy:
      1. Take the text of the nearest block ancestor.
      2. Find where the anchor text sits inside it.
      3. Slice a symmetric window around that position.
    """
    block = _nearest_block(a_tag)
    if block is None:
        return a_tag.get_text(" ", strip=True)[:window * 2]

    block_text = block.get_text(" ", strip=True)
    anchor_text = a_tag.get_text(" ", strip=True)

    idx = block_text.find(anchor_text)
    if idx == -1:
        return block_text[: window * 2]

    start = max(0, idx - window)
    end = min(len(block_text), idx + len(anchor_text) + window)
    snippet = block_text[start:end]
    # Trim partial words at boundaries
    if start > 0 and " " in snippet:
        snippet = snippet[snippet.index(" ") + 1:]
    return snippet


def _normalise_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_page(html: str, base_url: str) -> ParsedPage:
    """
    Parse *html* fetched from *base_url*.

    Returns a ParsedPage with full text and all outbound links
    (deduplicated by URL within this page, fragment stripped).
    """
    soup = BeautifulSoup(html, "lxml")

    # --- Title ---
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    # --- Visible text (remove boilerplate tags first) ---
    for tag in soup(["script", "style", "noscript", "head",
                     "meta", "link", "svg", "iframe"]):
        tag.decompose()
    text = _normalise_whitespace(soup.get_text(separator=" "))

    # --- Links ---
    seen_urls: set = set()
    links: List[ExtractedLink] = []

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].strip()
        if not href or href.startswith("javascript:") or href == "#":
            continue

        full_url, _ = urldefrag(urljoin(base_url, href))
        parsed = urlparse(full_url)

        if parsed.scheme not in _ALLOWED_SCHEMES:
            continue
        if not parsed.netloc:
            continue
        if full_url in seen_urls:
            continue
        seen_urls.add(full_url)

        anchor_text = _normalise_whitespace(a_tag.get_text(" "))
        context = _context_around(a_tag)

        links.append(ExtractedLink(
            url=full_url,
            anchor_text=anchor_text,
            context=context,
        ))

    return ParsedPage(url=base_url, title=title, text=text, links=links)
