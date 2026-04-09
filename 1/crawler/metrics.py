"""
Metrics logging harness.

Writes one CSV row per crawled URL.  The CSV is the shared evaluation
artefact — everyone's experiment plots are built from it.

Columns
-------
timestamp           Unix epoch (float)
url                 Fetched URL
bytes_downloaded    Raw bytes received (0 on error)
fetch_latency_ms    Wall-clock time from connect to last byte
status_code         HTTP status (0 if connection failed)
is_relevant         Boolean flag set by the caller's relevance function
cumulative_pages    Running total of successfully fetched pages
cumulative_relevant Running total of relevant pages
error               Error message if fetch failed, else empty
"""

import csv
import io
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


FIELDNAMES = [
    "timestamp",
    "url",
    "bytes_downloaded",
    "fetch_latency_ms",
    "status_code",
    "is_relevant",
    "cumulative_pages",
    "cumulative_relevant",
    "error",
]


@dataclass
class CrawlEvent:
    timestamp: float
    url: str
    bytes_downloaded: int
    fetch_latency_ms: float
    status_code: int
    is_relevant: bool
    cumulative_pages: int
    cumulative_relevant: int
    error: str = ""          # empty string instead of None so CSV stays clean


class MetricsLogger:
    """
    Append-only CSV logger.  Flushed after every row so a crashed run
    still produces a usable partial log.
    """

    def __init__(self, output_path: str = "crawl_metrics.csv") -> None:
        self.output_path = output_path
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self._file = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDNAMES)
        self._writer.writeheader()
        self._file.flush()

    def log(self, event: CrawlEvent) -> None:
        self._writer.writerow(asdict(event))
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
