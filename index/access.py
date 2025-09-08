# index/access.py
"""
Access functions for the unified Task 1 index package.
Provides O(1) average-case lookups after the package is loaded.
"""

import os
from typing import List, Dict, Any, Union, Tuple

TokGram = Union[str, Tuple[str, ...]]

# cache so we only load each index_path once
_INDEX_CACHE: Dict[str, Dict[str, Any]] = {}


# ----------------- internal helpers -----------------

def _load_index(index_path: str) -> Dict[str, Any]:
    """
    Load and cache the unified index package from disk.
    Returns the cached dict if already loaded.
    """
    if index_path in _INDEX_CACHE:
        return _INDEX_CACHE[index_path]

    from .io import load  # type: ignore

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index file not found: {index_path}")

    package = load(index_path)
    if not isinstance(package, dict):
        raise ValueError("Loaded index package is not a dict")

    # cache it
    _INDEX_CACHE[index_path] = package
    return package


# ----------------- public API -----------------

def get_posting_list(term: TokGram, index_path: str) -> List[int]:
    """
    Return sorted list of document IDs containing `term` (unigram, bigram, trigram).
    If not found, return [].
    """
    package = _load_index(index_path)
    unified = package.get("unified", {})
    posting = unified.get(term)
    if posting is None:
        return []
    # Already stored as sorted list
    return list(posting)


def find_wildcard_matches(pattern: str, index_path: str) -> List[str]:
    """
    Return sorted list of terms that contain the given char n-gram (wildcard pattern).
    If not found, return [].

    Example:
        "$cl" -> ["class", "climate", "clear"]
        "te$" -> ["climate", "update"]
    """
    package = _load_index(index_path)
    wildcard = package.get("wildcard", {})
    terms = wildcard.get(pattern)
    if terms is None:
        return []
    # Already stored as lexicographic list
    return list(terms)


def get_term_positions(term: TokGram, doc_id: int, index_path: str) -> List[int]:
    """
    Return sorted list of positions for `term` in document `doc_id`.
    If not found, return [].
    """
    package = _load_index(index_path)
    proximity = package.get("proximity", {})
    docmap = proximity.get(term)
    if docmap is None:
        return []
    positions = docmap.get(doc_id)
    if positions is None:
        return []
    # Already stored as sorted list
    return list(positions)
