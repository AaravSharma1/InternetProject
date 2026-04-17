from __future__ import annotations

import re
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer


class URLContextExtractor:
    def __init__(self, context_window: int = 200):
        self.context_window = context_window

    def extract(self, anchor_text: str, surrounding_text: str, url: str) -> str:
        url_tokens = self._tokenize_url_path(url)
        parts = [
            anchor_text.strip(),
            surrounding_text.strip()[: self.context_window * 2],
            url_tokens,
        ]
        return " ".join(p for p in parts if p)

    @staticmethod
    def extract_surrounding(page_text: str, anchor_text: str, window: int = 200) -> str:
        idx = page_text.find(anchor_text)
        if idx == -1:
            return page_text[:window]
        start = max(0, idx - window)
        end = min(len(page_text), idx + len(anchor_text) + window)
        return page_text[start:end]

    @staticmethod
    def _tokenize_url_path(url: str) -> str:
        path = re.sub(r"https?://[^/]+", "", url)
        tokens = re.split(r"[^a-zA-Z0-9]+", path)
        return " ".join(t.lower() for t in tokens if len(t) > 1)


class SemanticPrioritizer:
    # how much a new relevant page shifts the centroid
    EMA_ALPHA = 0.05

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", relevance_threshold: float = 0.5):
        self.model = SentenceTransformer(model_name)
        self.relevance_threshold = relevance_threshold
        self.context_extractor = URLContextExtractor()

    def score(self, url_context: str, topic_centroid: np.ndarray) -> float:
        embedding = self._embed_single(url_context)
        return self._cosine(embedding, topic_centroid)

    def score_batch(self, url_contexts: list[str], topic_centroid: np.ndarray) -> list[float]:
        if not url_contexts:
            return []
        embeddings = self.model.encode(url_contexts, convert_to_numpy=True, show_progress_bar=False)
        c_norm = self._normalize(topic_centroid)
        e_norms = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
        return (e_norms @ c_norm).tolist()

    def init_centroid(self, seed_descriptions: list[str]) -> np.ndarray:
        embeddings = self.model.encode(seed_descriptions, convert_to_numpy=True, show_progress_bar=False)
        return self._normalize(embeddings.mean(axis=0))

    def update_centroid(self, centroid: np.ndarray, new_page_text: str, score: Optional[float] = None) -> np.ndarray:
        if score is not None and score < self.relevance_threshold:
            return centroid
        new_emb = self._embed_single(new_page_text)
        updated = (1.0 - self.EMA_ALPHA) * centroid + self.EMA_ALPHA * new_emb
        return self._normalize(updated)

    def update_centroid_from_embedding(self, centroid: np.ndarray, embedding: np.ndarray, score: Optional[float] = None) -> np.ndarray:
        if score is not None and score < self.relevance_threshold:
            return centroid
        updated = (1.0 - self.EMA_ALPHA) * centroid + self.EMA_ALPHA * embedding
        return self._normalize(updated)

    def build_url_context(self, anchor_text: str, surrounding_text: str, url: str) -> str:
        return self.context_extractor.extract(anchor_text, surrounding_text, url)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def _embed_single(self, text: str) -> np.ndarray:
        return self.model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0]

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        a_n = a / (np.linalg.norm(a) + 1e-10)
        b_n = b / (np.linalg.norm(b) + 1e-10)
        return float(np.dot(a_n, b_n))

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        return v / (np.linalg.norm(v) + 1e-10)
