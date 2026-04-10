import json
import os
import random
import time
from collections import deque
from datetime import datetime, timezone

from flask import Flask, jsonify
from flask_cors import CORS

try:
    import redis as redis_lib
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Config
REDIS_HOST      = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT      = int(os.getenv("REDIS_PORT", 6379))
POLL_INTERVAL_S = 1.0
HISTORY_WINDOW  = 120  # data points to keep for the harvest-rate chart

STATS_KEY_PREFIX  = "crawler:stats:"    # + node_id  → JSON
SCORES_KEY_PREFIX = "crawler:scores:"   # + node_id  → Redis list of floats
GLOBAL_KEY        = "crawler:global"    # JSON

app = Flask(__name__)
CORS(app)

# Redis client (lazy, cached, re-tested each poll)
_redis_client = None
_redis_ok     = False

def get_redis():
    global _redis_client, _redis_ok
    if not REDIS_AVAILABLE:
        return None, False
    try:
        if _redis_client is None:
            _redis_client = redis_lib.Redis(
                host=REDIS_HOST, port=REDIS_PORT,
                decode_responses=True, socket_timeout=0.5,
            )
        _redis_client.ping()
        _redis_ok = True
        return _redis_client, True
    except Exception:
        _redis_ok = False
        return None, False



# Redis readers 

def _redis_read_stats(r) -> dict:
    """Read per-node stats and global state from Person 2's Redis keys."""
    nodes = {}
    for key in r.keys(f"{STATS_KEY_PREFIX}*"):
        node_id = key[len(STATS_KEY_PREFIX):]
        raw = r.get(key)
        if raw:
            try:
                nodes[node_id] = json.loads(raw)
            except json.JSONDecodeError:
                pass

    raw_global = r.get(GLOBAL_KEY)
    global_stats = {}
    if raw_global:
        try:
            global_stats = json.loads(raw_global)
        except json.JSONDecodeError:
            pass

    # Compute aggregate totals from per-node data
    total_pages   = sum(n.get("pages_crawled",  0) for n in nodes.values())
    total_relevant = sum(n.get("relevant_count", 0) for n in nodes.values())
    start_time    = global_stats.get("start_time", time.time())

    global_stats.update({
        "total_pages":    total_pages,
        "total_relevant": total_relevant,
        "harvest_rate":   round(total_relevant / max(total_pages, 1), 4),
        "elapsed_sec":    round(time.time() - start_time, 1),
        "mode":           global_stats.get("mode", "unknown"),
        "redis_connected": True,
    })

    # Mark the dashboard's last poll time
    r.set("dashboard:last_poll", time.time())

    return {"nodes": nodes, "global": global_stats}


def _redis_read_scores(r) -> list:
    """Read recent relevance scores from all nodes' score lists."""
    scores = []
    for key in r.keys(f"{SCORES_KEY_PREFIX}*"):
        for val in r.lrange(key, -200, -1):
            try:
                scores.append(float(val))
            except (ValueError, TypeError):
                pass
    return scores


# Mock data (used when Redis is unreachable)

_mock = {
    "t0":    time.time(),
    "nodes": {f"node_{i}": {"pages": 0, "relevant": 0, "bytes": 0}
              for i in range(1, 5)},
}


def _mock_stats() -> dict:
    elapsed = time.time() - _mock["t0"]
    nodes = {}
    total_pages = total_rel = 0

    for nid, s in _mock["nodes"].items():
        # Simulate ~25 pages/sec with variance, semantic-mode harvest ~58%
        delta     = max(0, int(25 + random.gauss(0, 4)))
        rel_delta = max(0, int(delta * random.uniform(0.45, 0.70)))
        s["pages"]   += delta
        s["relevant"] += rel_delta
        s["bytes"]    += delta * random.randint(30_000, 120_000)
        nodes[nid] = {
            "pages_crawled":   s["pages"],
            "pages_per_sec":   round(delta / POLL_INTERVAL_S, 1),
            "frontier_size":   random.randint(500, 3000),
            "relevant_count":  s["relevant"],
            "bytes_downloaded": s["bytes"],
            "harvest_rate":    round(s["relevant"] / max(s["pages"], 1), 4),
            "mode":            "semantic",
            "last_updated":    datetime.now(timezone.utc).isoformat(),
        }
        total_pages += s["pages"]
        total_rel   += s["relevant"]

    return {
        "nodes": nodes,
        "global": {
            "total_pages":    total_pages,
            "total_relevant": total_rel,
            "harvest_rate":   round(total_rel / max(total_pages, 1), 4),
            "elapsed_sec":    round(elapsed, 1),
            "mode":           "semantic",
            "redis_connected": False,
        },
    }


def _mock_scores() -> list:
    """Simulate bimodal score distribution typical of semantic mode."""
    high = [round(random.betavariate(5, 2), 3) for _ in range(140)]
    low  = [round(random.betavariate(1, 4), 3) for _ in range(60)]
    return high + low


# History buffer for harvest-rate chart
_history: deque = deque(maxlen=HISTORY_WINDOW)

def _record_history(stats: dict):
    _history.append({
        "ts":           time.time() - _mock["t0"],
        "harvest_rate": stats["global"]["harvest_rate"],
        "total_pages":  stats["global"]["total_pages"],
    })


# Flask API endpoints

@app.route("/api/stats")
def api_stats():
    r, ok = get_redis()
    stats = _redis_read_stats(r) if ok else _mock_stats()
    _record_history(stats)
    return jsonify(stats)


@app.route("/api/scores")
def api_scores():
    r, ok = get_redis()
    scores = _redis_read_scores(r) if ok else _mock_scores()

    # Bin into 20 buckets spanning [0, 1]
    buckets = [0] * 20
    for s in scores:
        idx = min(19, max(0, int(s * 20)))
        buckets[idx] += 1
    total = max(sum(buckets), 1)

    return jsonify({
        "labels": [f"{i * 5}–{(i + 1) * 5}%" for i in range(20)],
        "counts": buckets,
        "pct":    [round(b / total * 100, 1) for b in buckets],
        "n":      len(scores),
    })


@app.route("/api/history")
def api_history():
    return jsonify([
        {"t": round(e["ts"], 1), "harvest_rate": e["harvest_rate"], "pages": e["total_pages"]}
        for e in _history
    ])


@app.route("/api/health")
def api_health():
    _, ok = get_redis()
    return jsonify({"ok": True, "redis": ok})


# Serve the frontend HTML

_frontend_path = os.path.join(os.path.dirname(__file__), "dashboard_frontend.html")

@app.route("/")
def index():
    if os.path.exists(_frontend_path):
        with open(_frontend_path, encoding="utf-8") as f:
            return f.read()
    return (
        "<h2>dashboard_frontend.html not found</h2>"
        "<p>Make sure dashboard_frontend.html is in the same directory.</p>",
        404,
    )


# Main

if __name__ == "__main__":
    _, ok = get_redis()
    print("=" * 56)
    print("  Semantic Crawler — Live Dashboard")
    print(f"  Open: http://localhost:5050")
    print(f"  Redis ({REDIS_HOST}:{REDIS_PORT}): {'connected' if ok else 'NOT available — mock data mode'}")
    if ok:
        print(f"  Reading keys: {STATS_KEY_PREFIX}*  |  {SCORES_KEY_PREFIX}*  |  {GLOBAL_KEY}")
    print("=" * 56)
    app.run(host="0.0.0.0", port=5050, debug=True)
