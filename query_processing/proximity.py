# query_processing/proximity.py
from typing import Set, List, Tuple, Union
import re
from index.access import get_term_positions, get_posting_list

TokGram = Union[str, Tuple[str, ...]]
_NEAR_RE = re.compile(r'\bNEAR/(\d+)\b')  # strict, case-sensitive

def _parse_near(query: str) -> Tuple[str, int, str]:
    """
    Extract left operand, k, right operand from a strict NEAR/k query.
    Raises ValueError on malformed inputs.
    """
    m = _NEAR_RE.search(query)
    if not m:
        raise ValueError("Malformed NEAR/k: missing or non-strict NEAR/<int>")
    k = int(m.group(1))
    if k < 0:
        raise ValueError("Malformed NEAR/k: k must be non-negative")
    left = query[:m.start()].strip()
    right = query[m.end():].strip()
    if not left or not right:
        raise ValueError("Malformed NEAR/k: missing left or right operand")
    return left, k, right

def _as_key(operand: str) -> TokGram:
    """
    Convert operand string into a Task-1 lookup key:
      - quoted phrase => tuple[str, ...] (<= 3 tokens)
      - single term   => str
    """
    if operand.startswith('"') and operand.endswith('"'):
        inside = operand[1:-1].strip()
        toks = inside.split()
        if len(toks) == 1:
            return toks[0]
        return tuple(toks)
    return operand

def _span_positions(key: TokGram, doc_id: int, index_path: str) -> List[Tuple[int, int]]:
    """
    Return contiguous spans (start, end) for the operand in doc.
    For a unigram: start=end=position.
    For a phrase of m tokens: positions are the starting offsets of that m-gram,
    end = start + m - 1.
    """
    if isinstance(key, tuple):
        m = len(key)
        starts = get_term_positions(key, doc_id, index_path)
        return [(s, s + m - 1) for s in starts]
    else:
        pos = get_term_positions(key, doc_id, index_path)
        return [(p, p) for p in pos]

def _edge_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """
    Edge-to-edge distance (order-insensitive):
      D = min(|q_start - p_end|, |p_start - q_end|)
    Overlapping spans yield 0.
    """
    (as_, ae) = a
    (bs_, be) = b
    # overlap -> 0
    if not (ae < bs_ or be < as_):
        return 0
    # disjoint: minimal edge gap
    return min(abs(bs_ - ae), abs(as_ - be))

def process_proximity_query(query: str, index_path: str) -> Set[int]:
    """
    Process proximity queries with NEAR/k semantics.

    Args:
        query: Proximity query (e.g., 'climate NEAR/3 change',
               '"machine learning" NEAR/2 algorithms')
        index_path: Path to the unified index package

    Returns:
        Set of document IDs where operands satisfy NEAR/k distance constraint

    NEAR/k semantics:
    - Distance D = min(|q_start - p_end|, |p_start - q_end|)
    - Document satisfies NEAR/k iff D <= k for some occurrence pair
    - Edge-to-edge distance, order-insensitive
    - Same token span cannot satisfy both operands (i.e., identical spans are disallowed)
    """
    left_str, k, right_str = _parse_near(query)
    left_key: TokGram = _as_key(left_str)
    right_key: TokGram = _as_key(right_str)

    # Candidate docs are those containing BOTH operands
    left_docs = set(get_posting_list(left_key, index_path))
    right_docs = set(get_posting_list(right_key, index_path))
    candidates = left_docs & right_docs
    if not candidates:
        return set()

    hits: Set[int] = set()
    for did in candidates:
        left_spans = _span_positions(left_key, did, index_path)
        right_spans = _span_positions(right_key, did, index_path)
        if not left_spans or not right_spans:
            continue

        # Check any pair of spans; disallow identical spans (distinct-span requirement)
        ok = False
        for a in left_spans:
            for b in right_spans:
                if a == b:
                    # same token span cannot satisfy both operands
                    continue
                if _edge_distance(a, b) <= k:
                    ok = True
                    break
            if ok:
                break

        if ok:
            hits.add(did)

    return hits
