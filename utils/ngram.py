"""Word- and char-level n-gram helpers (Task A-2)."""
from typing import List, Tuple

__all__ = ["make_ngrams_tokens", "make_ngrams_chars"]

def make_ngrams_tokens(tokens: List[str], n: int) -> List[Tuple[str, ...]]:

    # Validate arguments
    if not isinstance(tokens, list):
        raise TypeError(f"tokens must be a list of strings, got {type(tokens).__name__}")
    if not all(isinstance(t, str) for t in tokens):
        raise ValueError("all items in tokens must be strings")
    if not isinstance(n, int):
        raise TypeError(f"n must be int, got {type(n).__name__}")
    if n <= 0:
        raise ValueError("n must be >= 1")

    # Handle unigrams: no padding
    if n == 1:
        return [(t,) for t in tokens]

    # Pad start and end
    pad = ["<s>"] * (n - 1)
    end = ["</s>"] * (n - 1)
    seq = pad + list(tokens) + end
    L = len(seq)

    ngrams = []

    for i in range(0, L - n + 1):
        window = seq[i : i + n]     # take slice of length n
        ngram = tuple(window)       # convert to tuple
        ngrams.append(ngram)

    return ngrams

def make_ngrams_chars(text: str, n: int) -> List[str]:

    # Validate arguments
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")
    if not isinstance(n, int):
        raise TypeError(f"n must be an int, got {type(n).__name__}")
    if n <= 0:
        raise ValueError("n must be >= 1")

    if not text.strip():
        return []

    grams: List[str] = []
    for w in text.split():
        if not w.strip():
            continue
        padded = f"${w}$"

        # Skip words too short for this n-gram size
        if len(padded) < n:
            continue

        if len(padded) >= n:
            grams.extend(padded[i : i + n] for i in range(len(padded) - n + 1))
    return grams