"""
Flask coordinator service for the distributed crawler.

Responsibilities:
  - Node registration and partition assignment
  - Heartbeat monitoring with automatic failover
  - Cluster status endpoint for monitoring / demo
"""

import json
import logging
import threading
import time

import redis
from flask import Flask, jsonify, request

from config import (
    ASSIGNMENT_CHANNEL,
    ASSIGNMENT_KEY,
    COORDINATOR_DEFAULT_PORT,
    HEARTBEAT_INTERVAL,
    HEARTBEAT_MISS_THRESHOLD,
    NODE_REGISTRY_KEY,
    REDIS_URL,
)
from coordinator.hash_ring import HashRing

logger = logging.getLogger(__name__)

# ── Shared state (module-level so the health-check thread can access) ──
_ring = HashRing()
_node_info: dict = {}   # node_id → {host, port, last_seen, pages_crawled, frontier_size}
_lock = threading.Lock()
_redis_client: redis.Redis = None


def _publish_assignment() -> None:
    """Push current assignment to Redis so nodes can read it."""
    assignment = _ring.get_assignment()
    payload = json.dumps({
        "nodes": {
            nid: {
                "partitions": assignment.get(nid, []),
                "host": _node_info[nid]["host"],
                "port": _node_info[nid]["port"],
            }
            for nid in _node_info
            if nid in assignment
        }
    })
    _redis_client.set(ASSIGNMENT_KEY, payload)
    _redis_client.publish(ASSIGNMENT_CHANNEL, payload)


def _health_check_loop() -> None:
    """Background thread: detect dead nodes and rebalance."""
    while True:
        time.sleep(HEARTBEAT_INTERVAL)
        now = time.time()
        dead = []
        with _lock:
            for nid, info in list(_node_info.items()):
                elapsed = now - info["last_seen"]
                if elapsed > HEARTBEAT_INTERVAL * HEARTBEAT_MISS_THRESHOLD:
                    dead.append(nid)
            for nid in dead:
                logger.warning("Node %s missed %d heartbeats — removing",
                               nid, HEARTBEAT_MISS_THRESHOLD)
                del _node_info[nid]
                _ring.remove_node(nid)
            if dead:
                _publish_assignment()


def create_app(redis_url: str = REDIS_URL) -> Flask:
    global _redis_client
    app = Flask(__name__)
    _redis_client = redis.Redis.from_url(redis_url, decode_responses=True)

    # Start background health checker
    t = threading.Thread(target=_health_check_loop, daemon=True)
    t.start()

    # ── Endpoints ──────────────────────────────────────────────────

    @app.post("/register")
    def register():
        data = request.get_json(force=True)
        node_id = data["node_id"]
        host = data["host"]
        port = data["port"]

        with _lock:
            _node_info[node_id] = {
                "host": host,
                "port": port,
                "last_seen": time.time(),
                "pages_crawled": 0,
                "frontier_size": 0,
            }
            assignment = _ring.register_node(node_id)
            _publish_assignment()

        logger.info("Registered node %s (%s:%s) — %d partitions",
                     node_id, host, port, len(assignment.get(node_id, [])))
        return jsonify({
            "node_id": node_id,
            "partitions": assignment.get(node_id, []),
            "all_nodes": assignment,
        })

    @app.post("/heartbeat")
    def heartbeat():
        data = request.get_json(force=True)
        node_id = data["node_id"]

        with _lock:
            if node_id not in _node_info:
                return jsonify({"error": "unknown node"}), 404
            _node_info[node_id]["last_seen"] = time.time()
            _node_info[node_id]["pages_crawled"] = data.get("pages_crawled", 0)
            _node_info[node_id]["frontier_size"] = data.get("frontier_size", 0)
            assignment = _ring.get_assignment()

        return jsonify({
            "status": "ok",
            "assignment": assignment,
        })

    @app.get("/status")
    def status():
        now = time.time()
        with _lock:
            nodes = {}
            for nid, info in _node_info.items():
                nodes[nid] = {
                    **info,
                    "alive": (now - info["last_seen"])
                             < HEARTBEAT_INTERVAL * HEARTBEAT_MISS_THRESHOLD,
                    "partitions": _ring.get_assignment().get(nid, []),
                }
            total_pages = sum(n.get("pages_crawled", 0) for n in _node_info.values())
        return jsonify({
            "node_count": len(nodes),
            "total_pages_crawled": total_pages,
            "nodes": nodes,
        })

    @app.get("/assignment")
    def assignment():
        with _lock:
            a = _ring.get_assignment()
            result = {}
            for nid, parts in a.items():
                info = _node_info.get(nid, {})
                result[nid] = {
                    "partitions": parts,
                    "host": info.get("host", ""),
                    "port": info.get("port", 0),
                }
        return jsonify({"nodes": result})

    @app.post("/deregister")
    def deregister():
        data = request.get_json(force=True)
        node_id = data["node_id"]
        with _lock:
            _node_info.pop(node_id, None)
            _ring.remove_node(node_id)
            _publish_assignment()
        logger.info("Deregistered node %s", node_id)
        return jsonify({"status": "ok"})

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = create_app()
    app.run(host="0.0.0.0", port=COORDINATOR_DEFAULT_PORT)
