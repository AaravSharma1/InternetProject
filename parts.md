Here's a clean 4-person split designed to minimize coordination bottlenecks. Each person owns a vertical slice they can build and evaluate mostly independently.

Person 1 — Single-Node Crawler & BFS Baseline (Rishab)
What you build: The foundational single-node crawler that everyone else's work plugs into. This is the "engine" — the async downloader, HTML parser/link extractor, local URL frontier (queue), and the Bloom filter for deduplication. You also own the BFS crawling mode, which becomes the primary baseline for all experiments.
Detailed responsibilities:
    •    Implement the async fetch loop using asyncio + aiohttp. Each iteration pops a URL from the frontier, downloads the page (respecting robots.txt and rate limits), parses it with BeautifulSoup, extracts outbound links with their anchor text and surrounding context (a ~200-char window around each <a> tag), and pushes new URLs back into the frontier.
    •    Build the local URL frontier as a pluggable priority queue. In BFS mode it's a plain FIFO. Expose a simple interface (push(url, priority, metadata), pop()) so Person 3 can swap in semantic scoring and Person 2 can swap in link-count scoring without touching your code.
    •    Implement a Bloom filter (use pybloom_live or roll your own) for URL deduplication. Track stats on false-positive rate.
    •    Build the content store — write each fetched page's HTML, extracted text, outbound links, and metadata (fetch time, status code, byte size) to disk or a lightweight SQLite DB.
    •    Implement the metrics logging harness: every crawl run logs a CSV with columns like timestamp, url, bytes_downloaded, fetch_latency_ms, is_relevant, cumulative_pages, cumulative_relevant. This CSV is what everyone uses for evaluation plots.
Code walkthroughs (for the demo):
    1    Walk through the async fetch pipeline — show how aiohttp.ClientSession is used with a semaphore to cap concurrency, how robots.txt is cached per domain, and how retries/timeouts work. Show a best-case fetch (~50ms, clean HTML) vs. worst-case (timeout, redirect chain, malformed HTML).
    2    Walk through the Bloom filter dedup — show the hash functions, the bit-array, and demonstrate the false-positive rate at 10K vs. 50K URLs.
    3    Walk through the pluggable frontier interface — show how BFS mode and priority mode use the same push/pop API with different internal data structures (deque vs. heapq).
Experiments you own:
    1    Throughput vs. concurrency — Run BFS crawl with concurrency limits of 5, 10, 25, 50, 100 on a single node. Plot pages/sec vs. concurrency. This shows diminishing returns and identifies the sweet spot.
    2    Crawl budget vs. harvest rate (BFS) — Run BFS at budgets of 1K, 5K, 10K, 25K, 50K pages from the same seeds. Plot cumulative relevant pages over total pages crawled. This is the baseline curve that Person 3's semantic mode should beat.
    3    Bloom filter sizing vs. false positive rate — Vary the Bloom filter size parameter and show how dedup accuracy changes at different crawl scales.

Person 2 — Distributed Coordination & Link-Priority Baseline (Aarav)
What you build: The multi-node layer. You take Person 1's single-node crawler and make it run on 2–8 nodes in parallel, coordinated by a lightweight Flask service. You also own the link-priority baseline (URLs ranked by inbound link count discovered during the crawl).
Detailed responsibilities:
    •    Build the Coordinator Service in Flask. It handles node registration (each node sends a POST on startup with its hostname/port), assigns each node a hash range (e.g., node 0 gets URLs whose hash(url) % N falls in [0, N/4)), and runs a health-check heartbeat every 10 seconds. If a node goes down, its hash range is reassigned.
    •    Implement the lateral URL forwarding protocol: when a crawler node discovers a URL that hashes to another node's partition, it pushes it to that node via Redis pub/sub (one Redis channel per node). Build a small receiver thread on each node that listens on its channel and pushes incoming URLs into the local frontier.
    •    Implement the link-priority scoring mode: maintain a dictionary of url → inbound_link_count across the crawl. When a URL is discovered from multiple pages, its count increments. The frontier uses this count as priority (higher = fetched sooner). This count is local to each node's partition — no need for global aggregation, which keeps coordination minimal.
    •    Implement the distributed Bloom filter using Redis bitmap (SETBIT/GETBIT). Before any node enqueues a URL, it checks the Redis bitmap. This replaces Person 1's local Bloom filter when running in distributed mode.
    •    Track cross-node communication overhead: log bytes sent/received via Redis per node per minute.
Code walkthroughs (for the demo):
    1    Walk through the hash-partitioning and lateral forwarding — show how a URL discovered on Node A gets hashed, determined to belong to Node C, serialized with its metadata (anchor text, context), and pushed via Redis. Show the receiver side deserializing and enqueuing. Demonstrate best case (URL stays local, no forwarding needed) vs. worst case (high cross-partition chatter when seeds produce many off-partition links).
    2    Walk through the coordinator's node registration and failover — show the Flask endpoints, the heartbeat loop, and what happens when a node misses 3 heartbeats (hash range reassignment). Show the average case where all nodes are healthy vs. the worst case of a node crash mid-crawl.
    3    Walk through the Redis-based distributed Bloom filter — show the bit-setting logic, how multiple nodes check/set atomically, and compare false-positive rates against Person 1's local Bloom filter at scale.
Experiments you own:
    1    Scalability: throughput vs. node count — Run BFS mode (no semantic scoring, to isolate coordination overhead) with 1, 2, 4, and 8 nodes on the same seed set and budget. Plot total pages/sec vs. node count. Ideal is linear; show where coordination overhead bends the curve.
    2    Link-priority vs. BFS harvest rate — Run both modes on identical seeds/budget/node count. Plot cumulative relevance curves. This is the second baseline that Person 3's semantic mode should beat.
    3    Cross-node communication overhead — Plot bytes exchanged between nodes vs. crawl progress for 2, 4, and 8 nodes. Show that overhead grows sub-linearly (or identify if it doesn't).

Person 3 — Semantic Prioritization Module (Maanas)
What you build: The core novelty of the project — the embedding-based URL scorer. You build it as a standalone module that plugs into Person 1's frontier interface. You don't need to touch the distributed layer at all; you develop and test on a single node, and Person 2 integrates it into multi-node mode by just enabling it on each node.
Detailed responsibilities:
    •    Build the SemanticPrioritizer class. It loads all-MiniLM-L6-v2 via sentence-transformers on init (~80MB model). It exposes a score(url_context: str, topic_centroid: np.ndarray) → float method that embeds the URL context string and returns cosine similarity to the topic centroid.
    •    Define what "URL context" means and build the context extractor: concatenate the anchor text, the surrounding ~200 characters of the referring page around the <a> tag, and the URL path tokens (e.g., /cs/machine-learning/ → "cs machine learning"). This string is what gets embedded.
    •    Implement topic centroid initialization and incremental update. At crawl start, embed the seed descriptions (e.g., "machine learning research papers") to create the initial centroid vector. As pages are fetched and deemed relevant (score above a threshold), blend their embeddings into the centroid with exponential moving average: centroid = 0.95 * centroid + 0.05 * new_embedding. This lets the topic representation drift as the crawl discovers sub-topics.
    •    Implement the relevance classifier for evaluation: take a sample of ~500 crawled pages, manually label them as relevant/irrelevant, then train a simple logistic regression on their embeddings. This classifier is used by all team members to compute harvest rate automatically across the full crawl.
    •    Benchmark inference latency — measure embedding time per URL on CPU to confirm it stays under 10ms and doesn't bottleneck the crawler.
Code walkthroughs (for the demo):
    1    Walk through the score() method end-to-end — show the context string construction from anchor text + surrounding text + URL tokens, the embedding call, and the cosine similarity computation. Demonstrate best case (anchor text says "deep learning survey paper" for an ML-focused crawl → score ~0.85), average case (generic "click here" anchor with some relevant surrounding text → score ~0.45), and worst case (completely off-topic ad link → score ~0.1).
    2    Walk through the centroid update mechanism — show the initial centroid from seed descriptions, then show how it shifts after crawling 100, 500, and 1000 relevant pages. Visualize the drift using t-SNE or PCA on the centroid snapshots plus sampled page embeddings.
    3    Walk through the relevance classifier training — show the labeling scheme, the feature extraction (page embedding), the train/test split, and the precision/recall on the held-out set. This classifier is the ground truth for all harvest rate measurements.
Experiments you own:
    1    Semantic vs. BFS vs. Link-Priority harvest rate — The headline result. Run all three modes on identical conditions (same seeds, same budget of 10K and 50K pages, same single node). Plot relevance-at-depth curves. The semantic curve should rise steeply then plateau early, while BFS should be roughly linear.
    2    Impact of context richness on scoring quality — Ablation study: run semantic mode with (a) anchor text only, (b) anchor text + surrounding text, (c) anchor text + surrounding text + URL tokens. Plot harvest rate for each. This shows which context signals matter most.
    3    Semantic scoring latency vs. batch size — Measure inference time per URL when scoring 1, 10, 50, 100 URLs in a batch. Show that batching amortizes overhead and that the scoring never becomes the bottleneck compared to network fetch time.

Person 4 — Evaluation Pipeline, Visualization & Demo Presentation (Anish)
What you build: The end-to-end evaluation framework, all figures/tables for the report, and the demo presentation. You consume the CSV logs and content stores that Persons 1–3 produce, and you turn them into polished results. You also build a simple live dashboard for the demo.
Detailed responsibilities:
    •    Build the evaluation pipeline as a Python script that reads crawl logs and computes all metrics: harvest rate, relevance-at-depth, redundancy rate (using MinHash via datasketch library for near-duplicate detection), bandwidth efficiency (cumulative bytes per relevant page), throughput, and cross-node communication cost. Output standardized tables and plots (use matplotlib or plotly).
    •    Implement MinHash-based redundancy detection: for each crawled page, compute a MinHash signature over its text shingles (5-word shingles). Compare against all previously seen signatures; if Jaccard similarity > 0.8, flag as near-duplicate. Report redundancy rate per crawl mode.
    •    Build a live monitoring dashboard using Flask + a simple React or HTML/JS frontend. It reads from Redis (where Person 2's nodes publish stats) and shows real-time pages/sec per node, frontier size, relevance score distribution, and cumulative harvest rate. This is what you show during the live demo.
    •    Create all comparison tables and figures: at minimum, a summary table of all metrics across BFS / Link-Priority / Semantic modes, relevance-at-depth line charts, throughput scaling bar charts, and a system architecture diagram.
    •    Assemble the demo slide deck: system architecture, code walkthrough highlights from each person, the 3+ evaluation figures/tables, and a link to the live demo video.
    •    Record or coordinate the demo video showing the dashboard running during a live crawl.
Code walkthroughs (for the demo):
    1    Walk through the MinHash redundancy detector — show the shingling of a page's text, the MinHash signature computation, and the comparison against the signature store. Demonstrate best case (unique pages, 0% redundancy), average case (~5–10% near-duplicates from mirror sites), and worst case (crawling a forum with many template pages → 30%+ redundancy).
    2    Walk through the evaluation pipeline — show how it ingests a crawl log CSV, joins it with the relevance classifier's labels, and computes harvest rate and bandwidth efficiency. Show how it handles the three repeated runs and computes mean ± std.
    3    Walk through the live dashboard — show the Flask backend reading from Redis, the WebSocket or polling mechanism pushing updates to the frontend, and how the charts update in real time.
Experiments you own:
    1    Redundancy rate across modes — Run BFS, Link-Priority, and Semantic at 50K page budget. Report near-duplicate percentage for each. Hypothesis: semantic mode fetches fewer duplicates because it avoids low-quality link farms.
    2    Bandwidth efficiency across modes — Plot cumulative MB downloaded per relevant page discovered, over crawl progress, for all three modes. Semantic should show a flatter (better) curve.
    3    Scalability + semantic combined — Work with Person 2 to run semantic mode at 1, 2, 4, 8 nodes. Plot throughput and harvest rate. This is the "full system" evaluation showing that semantic scoring doesn't kill scalability.

Coordination Interfaces (keep these minimal and agree on them upfront)
The whole split works because of three shared contracts:
    1    Frontier interface — push(url, priority, metadata) / pop() → (url, priority, metadata). Person 1 defines it, everyone uses it.
    2    Crawl log CSV schema — timestamp, node_id, url, bytes, fetch_ms, relevance_score, is_relevant, mode. Person 1 defines it, Person 4 consumes it.
    3    Semantic scorer interface — score(url_context, topic_centroid) → float. Person 3 defines it, Person 1/2 call it when semantic mode is enabled.
Each person can develop and test independently against mock implementations of the other interfaces. Integration happens in weeks 5–6 per the existing schedule.