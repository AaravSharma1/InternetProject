# CS 6675 Project - Person 3: Semantic Prioritization Module

This is the semantic prioritization component of our distributed focused web crawler. The idea is that instead of crawling in BFS order or by inbound link count, we use sentence embeddings to score each candidate URL by how relevant it looks to our target topic, and we always fetch the highest-scoring URL next.

## Files

**semantic_prioritizer.py**
The core module. Contains two classes:
- `URLContextExtractor` - builds a context string for each candidate link by combining the anchor text, ~200 characters of surrounding page text, and URL path tokens (e.g. `/cs/machine-learning/` becomes `"cs machine learning"`).
- `SemanticPrioritizer` - loads `all-MiniLM-L6-v2` and exposes the shared interface `score(url_context, topic_centroid) -> float`. Also handles centroid initialization from seed descriptions and EMA-based centroid updates as the crawl discovers relevant pages.

**relevance_classifier.py**
Trains a logistic regression classifier on top of page embeddings. Used as the evaluation ground truth to label pages as relevant or irrelevant. All four team members use this to compute harvest rate from crawl logs. Supports saving/loading the trained model.

**mock_frontier.py**
Implements Person 1's frontier interface (`push`/`pop`) so this module can be developed and tested without waiting for Person 1's code. Supports `mode="priority"` (max-heap by score) and `mode="bfs"` (FIFO).

**benchmark.py**
Measures embedding latency to confirm scoring stays under 10ms per URL on CPU. Also tests batch sizes 1, 10, 50, 100 to show that batching amortizes the overhead.

**demo.py**
Runs the four walkthroughs for the project demo:
1. `score()` end-to-end - shows best/average/worst case scoring
2. Centroid drift - shows how the centroid shifts after 100/500/1000 relevant pages, with a PCA visualization
3. Relevance classifier training and evaluation on synthetic labeled data
4. Frontier integration - scores a set of candidate URLs and shows pop order

## Setup

Install dependencies:

```
pip install -r requirements.txt
```

The first time you run anything, sentence-transformers will download `all-MiniLM-L6-v2` (~80MB). It will be cached after that.

## Running

**Run all demo walkthroughs:**
```
python demo.py
```

**Run a specific walkthrough:**
```
python demo.py --walkthrough 1
python demo.py --walkthrough 2
python demo.py --walkthrough 3
python demo.py --walkthrough 4
```

**Skip saving plots:**
```
python demo.py --no-plot
```

**Run the latency benchmark:**
```
python benchmark.py
```

**Benchmark with custom settings:**
```
python benchmark.py --trials 200 --batch-trials 50 --batch-sizes 1 10 50 100 200
python benchmark.py --no-plot
```

## Output Files

Running the demo produces these plots:
- `centroid_drift.png` - PCA visualization of centroid drift across checkpoints (walkthrough 2)
- `classifier_eval.png` - bar chart of precision/recall/F1/accuracy/ROC-AUC (walkthrough 3)

Running the benchmark produces:
- `benchmark_single.png` - histogram of single-URL latency over 100 trials
- `benchmark_batch.png` - per-URL and total latency vs. batch size

## Integration with Other Components

This module plugs into the shared interfaces defined by the team:

**Frontier interface (Person 1 defines):**
```python
frontier.push(url, priority, metadata)
url, priority, metadata = frontier.pop()
```
In semantic mode, `priority` is the score returned by `SemanticPrioritizer.score()`.

**Semantic scorer interface (this module defines):**
```python
score = prioritizer.score(url_context, topic_centroid)
```
Person 1 and Person 2 call this when semantic mode is enabled. The `url_context` string is built by `URLContextExtractor.extract(anchor_text, surrounding_text, url)`.

**Crawl log CSV schema (Person 1 defines, Person 4 consumes):**
```
timestamp, node_id, url, bytes, fetch_ms, relevance_score, is_relevant, mode
```
The `is_relevant` column is filled using `RelevanceClassifier.predict()` on the crawled page text.

## Quick Usage Example

```python
from semantic_prioritizer import SemanticPrioritizer, URLContextExtractor

prioritizer = SemanticPrioritizer()
centroid = prioritizer.init_centroid(["machine learning research papers"])

extractor = URLContextExtractor()
context = extractor.extract(
    anchor_text="deep learning survey",
    surrounding_text="We present a survey of recent advances in deep learning...",
    url="https://arxiv.org/abs/cs/dl-survey"
)

score = prioritizer.score(context, centroid)
print(score)  # e.g. 0.82

# after crawling a relevant page, update the centroid
centroid = prioritizer.update_centroid(centroid, page_text="...", score=score)
```
