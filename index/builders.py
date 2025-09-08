# index/builders.py
"""
Unified Index Package Builder - Task 1
Creates a single on-disk package containing all three sub-indexes.
"""

from typing import List, Dict, Union, Tuple, Any, Optional

TokGram = Union[str, Tuple[str, ...]]


# ----------------- helpers (no lambdas, no default factories) -----------------

def _token_ngrams(tokens: List[str], n_max: int = 3):
    """Yield (key, pos) for token n-grams n=1..n_max (inclusive)."""
    L = len(tokens)
    for n in range(1, n_max + 1):
        if L < n:
            break
        for i in range(L - n + 1):
            if n == 1:
                key: TokGram = tokens[i]
            else:
                key = tuple(tokens[i : i + n])
            yield key, i


def _char_ngrams(term: str, n_max: int = 3):
    """
    Yield character n-grams (length 1..n_max) from the boundary-augmented term `$<term>$`,
    excluding the bare '$' unigram.
    Includes both interior grams (e.g., 'mat') and boundary grams ('$cl', 'te$').
    """
    s = f"${term}$"
    L = len(s)
    for n in range(1, n_max + 1):
        for i in range(L - n + 1):
            cg = s[i : i + n]
            if cg == "$":
                continue
            yield cg


# ------------------------------------------------------------------------------

def create_all_indexes(
    tokenized_docs: List[List[str]],
    index_path: str,
    doc_ids: Optional[List[int]] = None
) -> None:
    """
    Build a unified index package containing all three sub-indexes in a single pass.

    Args:
        tokenized_docs: List of tokenized documents, each document is a list of tokens
        index_path: Path where the unified index package will be saved
        doc_ids: Optional list of document IDs. If None, uses sequential IDs (0, 1, 2, ...)
                 Must be same length as tokenized_docs if provided.
    """
    # --- import here to comply with your project structure ---
    from .io import dump  # type: ignore

    if doc_ids is None:
        doc_ids = list(range(len(tokenized_docs)))
    if len(doc_ids) != len(tokenized_docs):
        raise ValueError("doc_ids and tokenized_docs must be the same length")

    N = len(tokenized_docs)
    NGRAMS_MAX = 3
    CHAR_NGRAMS_MAX = 3

    # In-memory builders (plain dicts/sets/lists only; final package will convert sets to lists)
    unified_sets: Dict[TokGram, set] = {}                     # tok/ngram -> set(doc_id)
    proximity: Dict[TokGram, Dict[int, List[int]]] = {}       # tok/ngram -> {doc_id: [positions]}
    wildcard_sets: Dict[str, set] = {}                        # char_ngram -> set(terms)

    doc_lengths: Dict[int, int] = {}
    total_len = 0

    # ---- Single pass over documents ------------------------------------------------
    for tokens, did in zip(tokenized_docs, doc_ids):
        L = len(tokens)
        doc_lengths[did] = L
        total_len += L

        # token n-grams (1..3): unified + proximity
        for key, pos in _token_ngrams(tokens, n_max=NGRAMS_MAX):
            # unified postings (set during build; list after post-process)
            bucket = unified_sets.get(key)
            if bucket is None:
                bucket = set()
                unified_sets[key] = bucket
            bucket.add(did)

            # proximity positions (plain dict-of-lists)
            docmap = proximity.get(key)
            if docmap is None:
                docmap = {}
                proximity[key] = docmap
            pos_list = docmap.get(did)
            if pos_list is None:
                pos_list = []
                docmap[did] = pos_list
            pos_list.append(int(pos))

        # wildcard char n-grams for unique terms (avoid redundant adds within the same doc)
        for term in set(tokens):
            for cg in _char_ngrams(term, n_max=CHAR_NGRAMS_MAX):
                wb = wildcard_sets.get(cg)
                if wb is None:
                    wb = set()
                    wildcard_sets[cg] = wb
                wb.add(term)

    avgdl = float(total_len / N) if N > 0 else 0.0

    # ---- Deterministic post-processing (convert sets -> sorted lists) --------------
    # 1) unified postings: doc IDs strictly ascending
    unified: Dict[TokGram, List[int]] = {k: sorted(v) for k, v in unified_sets.items()}

    # 2) proximity positions: strictly ascending per (term, doc_id)
    for key, docmap in proximity.items():
        for did, pos_list in docmap.items():
            pos_list.sort()

    # 3) wildcard terms: strictly lexicographic (per spec)
    wildcard: Dict[str, List[str]] = {cg: sorted(terms) for cg, terms in wildcard_sets.items()}

    # ---- Final package (pickle-safe) ----------------------------------------------
    package: Dict[str, Any] = {
        "__META__": {
            "N": N,
            "doc_lengths": doc_lengths,
            "avgdl": avgdl,
            "version": "1.0",
            "ngrams_max": NGRAMS_MAX,
            "char_ngrams_max": CHAR_NGRAMS_MAX,
        },
        "unified": unified,        # tok/ngram -> [doc_id,...] (asc)
        "wildcard": wildcard,      # char_ngram -> [term,...] (lex asc)
        "proximity": proximity,    # tok/ngram -> {doc_id: [pos,...] (asc)}
    }

    # Save the unified package to disk (single file)
    dump(package, index_path)
