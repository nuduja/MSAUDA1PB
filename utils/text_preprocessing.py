"""Robust HTML→tokens cleaning pipeline (Task A-1)."""
import re, html, unicodedata
from typing import List
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def make_positions(tokens: List[str], n: int = 1) -> Dict[Union[str, Tuple[str, ...]], List[int]]:

    # Validate arguments
    if not isinstance(tokens, list):
        raise TypeError(f"tokens must be a list, got {type(tokens).__name__}")
    if not all(isinstance(t, str) for t in tokens):
        raise ValueError("all items in tokens must be strings")
    if not isinstance(n, int):
        raise TypeError(f"n must be int, got {type(n).__name__}")
    if n <= 0:
        raise ValueError("n must be >= 1")
    
    L = len(tokens)
    if L == 0 or L < n:
        return {}

    positions: Dict[Union[str, Tuple[str, ...]], List[int]] = defaultdict(list)

    if n == 1:
        for i, tok in enumerate(tokens):
            positions[tok].append(i)
    else:
        for i in range(L - n + 1):
            gram = tuple(tokens[i : i + n])
            positions[gram].append(i)

    return dict(positions)