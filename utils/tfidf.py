"""TF-IDF variants (Task A-3)."""
import math, numpy as np
from typing import List, Dict, Tuple

def tfidf_variants(
        docs: List[List[str]],
        tf_mode: str = "raw",
        k: float = 1.2
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Build a TF-IDF matrix with selectable TF variants.

    Args:
        docs: tokenized documents (each doc is a list of strings).
        tf_mode: 'raw' | 'log' | 'bm25'
            - raw  : tf/|d|
            - log  : 1 + ln(tf) if tf>0 else 0
            - bm25 : ((k+1)*tf)/(k+tf)
        k: BM25 saturation parameter (only used when tf_mode='bm25').

    Returns:
        (X, vocab):
          - X: np.ndarray of shape [num_docs, num_terms]
          - vocab: Dict[str, int] term -> column index (first occurrence order)
    """
    # Input validation
    if not isinstance(docs, list):
        raise TypeError(f"'docs' must be a list of lists, got {type(docs).__name__}")
    for idx, d in enumerate(docs):
        if not isinstance(d, list):
            raise TypeError(f"each document must be a list of strings; got {type(d).__name__} at index {idx}")
        if not all(isinstance(t, str) for t in d):
            raise ValueError(f"all tokens must be strings (document index {idx})")
    if not isinstance(tf_mode, str):
        raise TypeError(f"'tf_mode' must be a string, got {type(tf_mode).__name__}")
    if tf_mode not in ("raw", "log", "bm25"):
        raise ValueError("tf_mode must be one of: 'raw', 'log', 'bm25'")
    if not isinstance(k, (int, float)):
        raise TypeError(f"'k' must be a number, got {type(k).__name__}")
    if tf_mode == "bm25" and k <= 0:
        raise ValueError("For bm25, 'k' must be > 0")

    try:
        N = len(docs)
        if N == 0:
            return np.zeros((0, 0), dtype=float), {}

        # Build vocabulary
        vocab: Dict[str, int] = {}
        for doc in docs:
            for tok in doc:
                if tok not in vocab:
                    vocab[tok] = len(vocab)

        V = len(vocab)
        if V == 0:
            return np.zeros((N, 0), dtype=float), vocab

        # Document frequency
        df = np.zeros(V, dtype=int)
        for doc in docs:
            seen = set()
            for tok in doc:
                j = vocab[tok]
                if j not in seen:
                    df[j] += 1
                    seen.add(j)

        idf = np.log(N / df.astype(float))

        # Allocate matrix
        X = np.zeros((N, V), dtype=float)

        # Fill TF-IDF rows
        for i, doc in enumerate(docs):
            if not doc:
                continue

            # term counts
            counts: Dict[str, int] = {}
            for tok in doc:
                counts[tok] = counts.get(tok, 0) + 1

            L = float(len(doc))  # for raw tf

            for tok, tf in counts.items():
                j = vocab[tok]

                if tf_mode == "raw":
                    tf_w = tf / L
                elif tf_mode == "log":
                    tf_w = 1.0 + math.log(tf) if tf > 0 else 0.0
                else:  # 'bm25'
                    tf_w = ((k + 1.0) * tf) / (k + tf)

                X[i, j] = tf_w * idf[j]

        return X, vocab

    except Exception as e:
        # Prevent crashes with a clear error context
        raise RuntimeError(f"tfidf_variants failed: {e}") from e