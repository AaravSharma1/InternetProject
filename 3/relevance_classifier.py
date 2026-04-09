import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split

from semantic_prioritizer import SemanticPrioritizer


class RelevanceClassifier:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", C: float = 1.0, max_iter: int = 1000, random_state: int = 42):
        self._prioritizer = SemanticPrioritizer(model_name=model_name)
        self._lr: Optional[LogisticRegression] = None
        self._is_fitted = False
        self._C = C
        self._max_iter = max_iter
        self._random_state = random_state

    def fit(self, page_texts: list[str], labels: list[int], test_size: float = 0.2, verbose: bool = True) -> dict:
        if len(page_texts) != len(labels):
            raise ValueError("page_texts and labels must have the same length.")

        print(f"  Embedding {len(page_texts)} pages...")
        embeddings = self._embed(page_texts)

        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=test_size, random_state=self._random_state, stratify=labels
        )

        self._lr = LogisticRegression(C=self._C, max_iter=self._max_iter, random_state=self._random_state)
        self._lr.fit(X_train, y_train)
        self._is_fitted = True

        return self._report(X_test, y_test, verbose=verbose)

    def predict(self, page_texts: list[str]) -> list[int]:
        self._check_fitted()
        return self._lr.predict(self._embed(page_texts)).tolist()

    def predict_proba(self, page_texts: list[str]) -> list[float]:
        self._check_fitted()
        return self._lr.predict_proba(self._embed(page_texts))[:, 1].tolist()

    def predict_from_embeddings(self, embeddings: np.ndarray) -> list[int]:
        self._check_fitted()
        return self._lr.predict(embeddings).tolist()

    def predict_proba_from_embeddings(self, embeddings: np.ndarray) -> list[float]:
        self._check_fitted()
        return self._lr.predict_proba(embeddings)[:, 1].tolist()

    def evaluate(self, page_texts: list[str], labels: list[int], verbose: bool = True) -> dict:
        self._check_fitted()
        embeddings = self._embed(page_texts)
        return self._report(embeddings, labels, verbose=verbose)

    def save(self, path: str | Path) -> None:
        self._check_fitted()
        with open(path, "wb") as fh:
            pickle.dump({"lr": self._lr, "C": self._C, "max_iter": self._max_iter}, fh)
        print(f"  Classifier saved to {path}")

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        self._lr = obj["lr"]
        self._C = obj.get("C", self._C)
        self._max_iter = obj.get("max_iter", self._max_iter)
        self._is_fitted = True
        print(f"  Classifier loaded from {path}")

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self._prioritizer.embed(texts)

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Classifier is not fitted. Call fit() or load() first.")

    def _report(self, X: np.ndarray, y: list[int], verbose: bool = True) -> dict:
        preds = self._lr.predict(X)
        probas = self._lr.predict_proba(X)[:, 1]

        precision, recall, f1, _ = precision_recall_fscore_support(y, preds, average="binary", zero_division=0)
        accuracy = float(np.mean(np.array(preds) == np.array(y)))
        try:
            roc_auc = float(roc_auc_score(y, probas))
        except ValueError:
            roc_auc = float("nan")

        metrics = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": accuracy,
            "roc_auc": roc_auc,
        }

        if verbose:
            print("\n=== Relevance Classifier Evaluation ===")
            print(classification_report(y, preds, target_names=["irrelevant", "relevant"]))
            print(f"  Precision : {precision:.4f}")
            print(f"  Recall    : {recall:.4f}")
            print(f"  F1        : {f1:.4f}")
            print(f"  Accuracy  : {accuracy:.4f}")
            print(f"  ROC-AUC   : {roc_auc:.4f}")

        return metrics
