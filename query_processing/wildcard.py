from typing import Set, List
import re
from index.access import find_wildcard_matches, get_posting_list

def _pattern_to_ngrams(pat: str) -> List[str]:
    """
    Derive <=3-length character n-grams (with $ boundaries for anchored sides)
    from a single-token wildcard pattern containing '*'.
    """
    s = pat
    grams: List[str] = []

    # Prefix anchored? (no leading *)
    if not s.startswith('*'):
        stem = s.split('*', 1)[0]
        # boundary grams total length must be <= 3 → only 1 or 2 letters with '$'
        for L in (1, 2):
            if len(stem) >= L:
                grams.append("$" + stem[:L])

    # Suffix anchored? (no trailing *)
    if not s.endswith('*'):
        stem = s.rsplit('*', 1)[-1]
        for L in (1, 2):
            if len(stem) >= L:
                grams.append(stem[-L:] + "$")

    # Internal grams from the core (remove all '*'), lengths 3,2,1
    core = s.replace('*', '')
    for L in (3, 2, 1):
        for i in range(0, max(0, len(core) - L + 1)):
            grams.append(core[i:i+L])

    # Deduplicate while preserving order; exclude meaningless "$"
    seen = set()
    out: List[str] = []
    for g in grams:
        if g and g != "$" and g not in seen:
            seen.add(g)
            out.append(g)
    return out

def _expand_terms(pat: str, index_path: str) -> List[str]:
    grams = _pattern_to_ngrams(pat)
    if not grams:
        return []

    first = True
    candidates = set()
    for g in grams:
        terms = set(find_wildcard_matches(g, index_path))
        if first:
            candidates = terms
            first = False
        else:
            candidates &= terms
        if not candidates:
            break

    rx = re.compile("^" + re.escape(pat).replace("\\*", ".*") + "$")
    return sorted(t for t in candidates if rx.match(t))

def process_wildcard_query(pattern: str, index_path: str) -> Set[int]:
    if not isinstance(pattern, str) or "*" not in pattern or pattern.strip() == "":
        return set()

    terms = _expand_terms(pattern, index_path)
    results: Set[int] = set()
    for t in terms:
        results |= set(get_posting_list(t, index_path))
    return results
