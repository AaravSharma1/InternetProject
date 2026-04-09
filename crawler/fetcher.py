"""
Async HTTP fetcher with robots.txt compliance and per-domain rate limiting.

FetchResult  — plain dataclass returned by Fetcher.fetch()
RobotsCache  — async-friendly robots.txt cache (one parser per origin)
RateLimiter  — token-bucket-style delay per domain
Fetcher      — async context manager; open once per crawl run
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import aiohttp


_HTML_CONTENT_TYPES = ("text/html", "application/xhtml+xml")
_DEFAULT_UA = "InternetSysCrawler/1.0 (+https://github.com/your-org/internetsys)"


# ---------------------------------------------------------------------------
# Data transfer object
# ---------------------------------------------------------------------------

@dataclass
class FetchResult:
    url: str
    html: Optional[str]           # None on non-HTML or error
    status_code: Optional[int]    # None on connection error
    byte_size: int
    fetch_time: float             # Unix epoch at start of request
    latency_ms: float
    error: Optional[str] = None   # human-readable reason on failure


# ---------------------------------------------------------------------------
# robots.txt cache
# ---------------------------------------------------------------------------

class RobotsCache:
    """
    Fetches and caches robots.txt per origin.

    Missing or unreachable robots.txt → allow all (standard convention).
    """

    def __init__(self, session: aiohttp.ClientSession,
                 user_agent: str = _DEFAULT_UA) -> None:
        self._session = session
        self._user_agent = user_agent
        self._cache: Dict[str, RobotFileParser] = {}

    def _origin(self, url: str) -> str:
        p = urlparse(url)
        return f"{p.scheme}://{p.netloc}"

    async def is_allowed(self, url: str) -> bool:
        origin = self._origin(url)
        if origin not in self._cache:
            await self._fetch_robots(origin)
        return self._cache[origin].can_fetch(self._user_agent, url)

    async def _fetch_robots(self, origin: str) -> None:
        parser = RobotFileParser()
        robots_url = f"{origin}/robots.txt"
        parser.set_url(robots_url)
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with self._session.get(robots_url, timeout=timeout) as resp:
                text = await resp.text(errors="replace")
            parser.parse(text.splitlines())
        except Exception:
            # Can't fetch → treat as allow-all
            parser.parse([])
        self._cache[origin] = parser


# ---------------------------------------------------------------------------
# Per-domain rate limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """
    Enforces a minimum inter-request delay per domain.

    Uses an asyncio.Lock per domain so concurrent coroutines targeting the
    same host queue up rather than racing.
    """

    def __init__(self, delay: float = 1.0) -> None:
        self.delay = delay
        self._last: Dict[str, float] = {}
        self._locks: Dict[str, asyncio.Lock] = {}

    def _lock_for(self, domain: str) -> asyncio.Lock:
        if domain not in self._locks:
            self._locks[domain] = asyncio.Lock()
        return self._locks[domain]

    async def acquire(self, url: str) -> None:
        domain = urlparse(url).netloc
        async with self._lock_for(domain):
            now = time.monotonic()
            wait = self.delay - (now - self._last.get(domain, 0.0))
            if wait > 0:
                await asyncio.sleep(wait)
            self._last[domain] = time.monotonic()


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------

class Fetcher:
    """
    Async HTTP client.  Use as an async context manager:

        async with Fetcher(...) as fetcher:
            result = await fetcher.fetch("https://example.com")

    Parameters
    ----------
    user_agent       : Crawler User-Agent string sent in every request.
    rate_limit_delay : Minimum seconds between requests to the same domain.
    timeout          : Per-request timeout in seconds.
    max_retries      : Number of retry attempts on transient errors.
    max_content_size : Hard cap on bytes read per page (default 5 MB).
                       Pages exceeding this are truncated, not failed.
    """

    def __init__(
        self,
        user_agent: str = _DEFAULT_UA,
        rate_limit_delay: float = 1.0,
        timeout: float = 30.0,
        max_retries: int = 2,
        max_content_size: int = 5 * 1024 * 1024,
    ) -> None:
        self.user_agent = user_agent
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_content_size = max_content_size
        self._rate_limiter = RateLimiter(delay=rate_limit_delay)
        self._session: Optional[aiohttp.ClientSession] = None
        self._robots: Optional[RobotsCache] = None

    async def __aenter__(self) -> "Fetcher":
        connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
        headers = {"User-Agent": self.user_agent}
        self._session = aiohttp.ClientSession(
            connector=connector,
            headers=headers,
            trust_env=True,
        )
        self._robots = RobotsCache(self._session, self.user_agent)
        return self

    async def __aexit__(self, *_) -> None:
        if self._session:
            await self._session.close()

    async def fetch(self, url: str) -> FetchResult:
        """
        Download *url* and return a FetchResult.

        Sequence:
          1. Check robots.txt (cached per origin).
          2. Enforce per-domain rate limit.
          3. GET with retries on transient errors.
          4. Discard non-HTML content types.
          5. Read body (capped at max_content_size).
        """
        # robots.txt check
        if not await self._robots.is_allowed(url):
            return FetchResult(
                url=url, html=None, status_code=None, byte_size=0,
                fetch_time=time.time(), latency_ms=0.0,
                error="robots.txt disallowed",
            )

        # Rate limit
        await self._rate_limiter.acquire(url)

        fetch_time = time.time()
        t0 = time.monotonic()

        for attempt in range(self.max_retries + 1):
            try:
                return await self._do_get(url, fetch_time, t0)
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
                if attempt == self.max_retries:
                    return FetchResult(
                        url=url, html=None, status_code=None, byte_size=0,
                        fetch_time=fetch_time,
                        latency_ms=(time.monotonic() - t0) * 1000,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                await asyncio.sleep(2.0 ** attempt)   # exponential back-off

        # Should never reach here
        return FetchResult(url=url, html=None, status_code=None, byte_size=0,
                           fetch_time=fetch_time, latency_ms=0.0, error="unknown")

    async def _do_get(self, url: str, fetch_time: float,
                      t0: float) -> FetchResult:
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with self._session.get(
            url,
            timeout=timeout,
            allow_redirects=True,
            max_redirects=10,
        ) as resp:
            content_type = resp.headers.get("Content-Type", "").lower()
            is_html = any(ct in content_type for ct in _HTML_CONTENT_TYPES)

            # Stream body with size cap regardless of content type
            # (we need the byte count even for non-HTML)
            chunks = []
            total = 0
            async for chunk in resp.content.iter_chunked(65_536):
                total += len(chunk)
                if total <= self.max_content_size:
                    chunks.append(chunk)

            byte_size = total
            latency_ms = (time.monotonic() - t0) * 1000

            if not is_html:
                return FetchResult(
                    url=url, html=None, status_code=resp.status,
                    byte_size=byte_size, fetch_time=fetch_time,
                    latency_ms=latency_ms, error="non-HTML content-type",
                )

            html = b"".join(chunks).decode("utf-8", errors="replace")
            return FetchResult(
                url=url, html=html, status_code=resp.status,
                byte_size=byte_size, fetch_time=fetch_time,
                latency_ms=latency_ms,
            )
