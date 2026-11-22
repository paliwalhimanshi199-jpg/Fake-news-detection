"""Live headlines from the Google News RSS search feed. No API key is needed."""
from __future__ import annotations

import html
import re
import urllib.request
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

FEED_URL = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
TIMEOUT_SECONDS = 15
MAX_ITEMS = 50
TAG_RE = re.compile(r"<[^>]+>")


def build_url(query: str) -> str:
    return FEED_URL.format(query=quote_plus(query))


def fetch_feed(query: str, limit: int = MAX_ITEMS) -> list[dict]:
    request = urllib.request.Request(
        build_url(query),
        headers={"User-Agent": "Mozilla/5.0 (compatible; FakeNewsDetector/1.0)"},
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        payload = response.read()
    return parse_feed(payload, limit)


def parse_feed(payload: bytes, limit: int = MAX_ITEMS) -> list[dict]:
    root = ET.fromstring(payload)
    items = []
    for item in root.iter("item"):
        title = _clean(item.findtext("title"))
        source = _clean(item.findtext("source"))
        # Google appends " - Source" to every headline. Remove it so the classifier sees only the story.
        suffix = f" - {source}"
        if source and title.endswith(suffix):
            title = title[: -len(suffix)]
        items.append({
            "headline": title,
            "source": source,
            "published": _format_date(item.findtext("pubDate")),
            "link": _clean(item.findtext("link")),
        })
        if len(items) >= limit:
            break
    return items


def _clean(value: str | None) -> str:
    text = TAG_RE.sub(" ", value or "")
    return " ".join(html.unescape(text).split())


def _format_date(value: str | None) -> str:
    if not value:
        return ""
    try:
        return parsedate_to_datetime(value).strftime("%Y-%m-%d %H:%M %Z").strip()
    except (TypeError, ValueError):
        return value
