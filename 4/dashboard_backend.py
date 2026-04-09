"""
dashboard_backend.py — Flask backend for the live crawl monitoring dashboard
Person 4: Live Demo Dashboard

Reads from Redis where Person 2's crawler nodes publish per-node stats.

Redis key schema (published by crawler nodes):
  crawler:stats:{node_id}   → JSON hash with fields:
      pages_crawled, pages_per_sec, frontier_size, relevant_count,
      bytes_downloaded, last_updated, mode

  crawler:scores:{node_id}  → Redis list of recent relevance scores (floats)
  crawler:global            → JSON: total_pages, start_time, mode

Run:
    pip install flask flask-cors redis
    python dashboard_backend.py

The dashboard HTML/JS frontend is served at http://localhost:5050/
API endpoints:
    GET /api/stats           → per-node stats + global totals
    GET /api/scores          → relevance score distribution (last 200 per node)
    GET /api/history         → cumulative harvest rate over time
    GET /api/health          → liveness check
"""

import json
import os
import time
import math
import random
from collections import defaultdict, deque
from datetime import datetime, timezone

from flask import Flask, jsonify, render_template_string
from flask_cors import CORS

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
POLL_INTERVAL_MS = 1000
HISTORY_WINDOW = 120  # seconds of history to retain

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
# Redis client (with mock fallback for demo)
# ─────────────────────────────────────────────
_r = None

def get_redis():
    global _r
    if _r is None and REDIS_AVAILABLE:
        try:
            _r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT,
                             decode_responses=True, socket_timeout=1)
            _r.ping()
        except Exception:
            _r = None
    return _r


# ─────────────────────────────────────────────
# Mock data generator (used when Redis unavailable)
# ─────────────────────────────────────────────
_mock_state = {
    "t0": time.time(),
    "nodes": {f"node_{i}": {"pages": 0, "relevant": 0, "bytes": 0} for i in range(1, 5)},
}

def _mock_stats():
    elapsed = time.time() - _mock_state["t0"]
    nodes = {}
    total_pages = 0
    total_relevant = 0
    for nid, state in _mock_state["nodes"].items():
        # Simulate ~25 pages/sec with variance
        delta = int((25 + random.gauss(0, 5)) * POLL_INTERVAL_MS / 1000)
        rel_delta = int(delta * random.uniform(0.4, 0.7))  # semantic mode ~55% relevant
        state["pages"] += max(0, delta)
        state["relevant"] += max(0, rel_delta)
        state["bytes"] += delta * random.randint(30_000, 120_000)
        nodes[nid] = {
            "pages_crawled": state["pages"],
            "pages_per_sec": round(max(0, delta / (POLL_INTERVAL_MS / 1000)), 1),
            "frontier_size": random.randint(500, 3000),
            "relevant_count": state["relevant"],
            "bytes_downloaded": state["bytes"],
            "harvest_rate": round(state["relevant"] / max(state["pages"], 1), 3),
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "mode": "semantic",
        }
        total_pages += state["pages"]
        total_relevant += state["relevant"]

    global_stats = {
        "total_pages": total_pages,
        "total_relevant": total_relevant,
        "harvest_rate": round(total_relevant / max(total_pages, 1), 3),
        "elapsed_sec": round(elapsed, 1),
        "mode": "semantic",
        "redis_connected": False,
    }
    return {"nodes": nodes, "global": global_stats}


def _mock_scores():
    """Return a simulated relevance score histogram."""
    # Semantic mode: bimodal — most pages score high or very low
    high = [round(random.betavariate(5, 2), 3) for _ in range(140)]
    low  = [round(random.betavariate(1, 4), 3) for _ in range(60)]
    return high + low


_history_buffer = deque(maxlen=HISTORY_WINDOW)

def _append_history(stats):
    _history_buffer.append({
        "ts": time.time(),
        "harvest_rate": stats["global"]["harvest_rate"],
        "total_pages": stats["global"]["total_pages"],
    })


# ─────────────────────────────────────────────
# Redis readers
# ─────────────────────────────────────────────

def _redis_stats(r):
    node_keys = r.keys("crawler:stats:*")
    nodes = {}
    for key in node_keys:
        nid = key.split(":")[-1]
        raw = r.get(key)
        if raw:
            nodes[nid] = json.loads(raw)
    raw_global = r.get("crawler:global")
    global_stats = json.loads(raw_global) if raw_global else {}
    global_stats["redis_connected"] = True
    return {"nodes": nodes, "global": global_stats}


def _redis_scores(r):
    scores = []
    for key in r.keys("crawler:scores:*"):
        vals = r.lrange(key, -200, -1)
        scores.extend(float(v) for v in vals)
    return scores


# ─────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────

@app.route("/api/stats")
def api_stats():
    r = get_redis()
    stats = _redis_stats(r) if r else _mock_stats()
    _append_history(stats)
    return jsonify(stats)


@app.route("/api/scores")
def api_scores():
    r = get_redis()
    scores = _redis_scores(r) if r else _mock_scores()
    # Bin into 20 buckets [0,1]
    buckets = [0] * 20
    for s in scores:
        idx = min(int(s * 20), 19)
        buckets[idx] += 1
    total = max(sum(buckets), 1)
    return jsonify({
        "labels": [f"{i*5}–{(i+1)*5}%" for i in range(20)],
        "counts": buckets,
        "pct": [round(b / total * 100, 1) for b in buckets],
        "n": len(scores),
    })


@app.route("/api/history")
def api_history():
    now = time.time()
    return jsonify([
        {
            "t": round(entry["ts"] - _mock_state["t0"], 1),
            "harvest_rate": entry["harvest_rate"],
            "pages": entry["total_pages"],
        }
        for entry in _history_buffer
    ])


@app.route("/api/health")
def api_health():
    r = get_redis()
    return jsonify({"ok": True, "redis": r is not None})


# ─────────────────────────────────────────────
# Serve Dashboard HTML
# ─────────────────────────────────────────────

DASHBOARD_HTML = open(
    os.path.join(os.path.dirname(__file__), "dashboard_frontend.html")
).read() if os.path.exists(
    os.path.join(os.path.dirname(__file__), "dashboard_frontend.html")
) else "<h1>Dashboard frontend not found. See dashboard_frontend.html</h1>"


@app.route("/")
def index():
    return DASHBOARD_HTML


if __name__ == "__main__":
    print("=" * 50)
    print("  Semantic Crawler Live Dashboard")
    print(f"  Open: http://localhost:5050")
    r = get_redis()
    print(f"  Redis: {'connected' if r else 'NOT available — using mock data'}")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5050, debug=True)
