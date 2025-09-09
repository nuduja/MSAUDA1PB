"""Light-weight GloVe loader & semantic aggregation (Task A-4)."""
import io, zipfile, urllib.request, pathlib
from typing import Dict, List

import numpy as np

# ----------------------------------------------------------------------
# 1.  Load a 200-d slice of GloVe (or from cache) and add a random <unk>
# ----------------------------------------------------------------------
_GLOVE_URL  = ("http://nlp.stanford.edu/data/glove.6B.zip", "glove.6B.200d.txt")
_CACHE      = pathlib.Path.home() / ".cache" / "ir_glove_200.txt"


def _ensure_glove() -> Dict[str, np.ndarray]:
    if not _CACHE.exists():
        url, fname = _GLOVE_URL
        with urllib.request.urlopen(url) as resp:
            with zipfile.ZipFile(io.BytesIO(resp.read())) as zf:
                _CACHE.write_text(zf.read(fname).decode("utf-8"), encoding="utf-8", newline="\n")

    vocab = {}
    with _CACHE.open("r", encoding="utf-8", newline="\n") as f:
        for line in f:
            word, *vec = line.strip().split()
            vocab[word] = np.asarray(vec, dtype=float)

    # deterministic random <unk> – keeps load fast
    if "<unk>" not in vocab:
        rng = np.random.default_rng(seed=42)
        dim = len(next(iter(vocab.values())))
        vocab["<unk>"] = rng.normal(0.0, 0.05, size=dim)

    return vocab


_WORD_VEC: Dict[str, np.ndarray] = _ensure_glove()
_DIM: int = next(iter(_WORD_VEC.values())).shape[0]


# ----------------------------------------------------------------------
# 2.  Public helper
# ----------------------------------------------------------------------
def _key(tok: str) -> str:
    """Return token itself if in vocab, else '<unk>'."""
    return tok if tok in _WORD_VEC else "<unk>"


# ----------------------------------------------------------------------
# 3.  Main entry
# ----------------------------------------------------------------------
def semantic_vector(docs: List[List[str]], method: str = "mean") -> np.ndarray:

    # Input validation
    if not isinstance(docs, list):
        raise TypeError(f"'docs' must be a list of lists, got {type(docs).__name__}")
    for i, d in enumerate(docs):
        if not isinstance(d, list):
            raise TypeError(f"each document must be a list, got {type(d).__name__} at index {i}")
        if not all(isinstance(t, str) for t in d):
            raise ValueError(f"all tokens must be strings (document index {i})")

    method = (method or "").lower()
    if method not in {"mean", "max", "sum", "tfidf_weighted", "meanmax"}:
        raise ValueError("method must be one of: mean, max, sum, tfidf_weighted, meanmax")

    N = len(docs)
    if N == 0:
        # For meanmax, width is 2*_DIM; for others, _DIM.
        width = (2 * _DIM) if method == "meanmax" else _DIM
        return np.zeros((0, width), dtype=float)

    # Baseline aggregations
    if method in {"mean", "max", "sum", "meanmax"}:
        rows = []
        for doc in docs:
            if not doc:
                mean_vec = np.zeros(_DIM, dtype=float)
                max_vec  = np.zeros(_DIM, dtype=float)
                sum_vec  = np.zeros(_DIM, dtype=float)
            else:
                # Map tokens to known keys (OOV -> '<unk>') and stack vectors
                vecs = [_WORD_VEC[_key(t)] for t in doc]
                M = np.vstack(vecs).astype(float)     # (len(doc), _DIM)
                sum_vec  = M.sum(axis=0)
                mean_vec = sum_vec / M.shape[0]
                max_vec  = M.max(axis=0)

            if method == "mean":
                rows.append(mean_vec)
            elif method == "max":
                rows.append(max_vec)
            elif method == "sum":
                rows.append(sum_vec)
            else:  # "meanmax"
                rows.append(np.concatenate([mean_vec, max_vec], axis=0))

        return np.vstack(rows)

    # TF-IDF weighted average
    # Replace OOV tokens with <unk> when computing TF-IDF.
    try:
        
        mapped_docs: List[List[str]] = [[_key(t) for t in doc] for doc in docs]

        df: Dict[str, int] = {}
        for doc in mapped_docs:
            for tok in set(doc):
                df[tok] = df.get(tok, 0) + 1

        # IDF = ln(N / df)
        idf: Dict[str, float] = {tok: float(np.log(N / df_val)) for tok, df_val in df.items()}

        # Build weighted vectors per doc
        out = np.zeros((N, _DIM), dtype=float)

        for i, doc in enumerate(mapped_docs):
            if not doc:
                continue

            # raw term counts
            counts: Dict[str, int] = {}
            for tok in doc:
                counts[tok] = counts.get(tok, 0) + 1

            num = np.zeros(_DIM, dtype=float)
            den = 0.0
            for tok, tf in counts.items():
                w = tf * idf.get(tok, 0.0)  # tf-idf weight
                if w <= 0.0:
                    continue
                num += w * _WORD_VEC[tok]   # tok guaranteed via _key()
                den += w

            out[i] = (num / den) if den > 0.0 else np.zeros(_DIM, dtype=float)

        return out

    except Exception as e:
        raise RuntimeError(f"semantic_vector(tfidf_weighted) failed: {e}") from e