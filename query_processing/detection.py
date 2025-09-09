import re

# Case-sensitive detectors
_RE_NEAR = re.compile(r'NEAR/(\d+)')              # strict NEAR/<int>
_RE_BOOL_OP = re.compile(r'\b(?:AND|OR|NOT)\b')   # only these operators
_RE_QUOTES = re.compile(r'"[^"]*"')               # matched quotes only
_RE_ANY_QUOTE = re.compile(r'"')
_RE_UPPER_OPLIKE = re.compile(r'\b[A-Z]{2,}\b')   # XOR, ANDAND, etc.


def _balanced_parens(s: str) -> bool:
    c = 0
    for ch in s:
        if ch == '(':
            c += 1
        elif ch == ')':
            c -= 1
            if c < 0:
                return False
    return c == 0


def _has_unmatched_quotes(s: str) -> bool:
    return (s.count('"') % 2) == 1


def _bad_phrase_lengths(s: str) -> bool:
    for m in _RE_QUOTES.finditer(s):
        inside = m.group(0)[1:-1].strip()
        if not inside:
            return True  # empty phrase ""
        if len(inside.split()) > 3:
            return True
    return False


def _has_mixed_types(query: str) -> bool:
    """
    Mixing wildcard with boolean OR proximity is malformed.
    Mixing proximity with boolean *operators* is malformed.
    NOTE: quoted phrases are allowed as NEAR/k operands and do NOT count as 'boolean'.
    """
    has_star = '*' in query
    has_near = bool(_RE_NEAR.search(query)) or ('NEAR' in query)
    has_bool_ops = bool(_RE_BOOL_OP.search(query))
    has_quotes = bool(_RE_ANY_QUOTE.search(query))

    if has_star and (has_near or has_bool_ops or has_quotes):
        return True
    # allow quotes with NEAR/k
    if has_near and has_bool_ops:
        return True
    return False


def _invalid_near(query: str) -> bool:
    if "NEAR" in query and not _RE_NEAR.search(query):
        return True

    matches = list(_RE_NEAR.finditer(query))
    if len(matches) == 0:
        return False
    if len(matches) > 1:
        return True

    m = matches[0]
    try:
        k = int(m.group(1))
        if k < 0:
            return True
    except Exception:
        return True

    left = query[:m.start()].strip()
    right = query[m.end():].strip()
    if not left or not right:
        return True
    return False


def _invalid_wildcard(query: str) -> bool:
    if '*' not in query:
        return False
    if '"' in query:
        return True
    if any(ch.isspace() for ch in query):
        return True
    return set(query) == {'*'}  # patterns like * or **


def _invalid_boolean_structure(query: str) -> bool:
    if _has_unmatched_quotes(query):
        return True
    if not _balanced_parens(query):
        return True
    if _bad_phrase_lengths(query):
        return True

    toks = query.strip().split()
    if not toks:
        return False

    # Unknown ALL-CAPS operator-like tokens
    for t in toks:
        if _RE_UPPER_OPLIKE.fullmatch(t) and t not in ("AND", "OR", "NOT"):
            return True

    # Leading/trailing operator (allow leading NOT)
    if toks[-1] in ("AND", "OR", "NOT"):
        return True
    if toks[0] in ("AND", "OR"):
        return True

    # Operators adjacency rules:
    # - Disallow AND/OR directly followed by AND/OR
    # - Allow AND NOT / OR NOT (NOT is unary)
    # - Disallow NOT followed by AND/OR/NOT
    for i in range(len(toks) - 1):
        a, b = toks[i], toks[i + 1]
        if a in ("AND", "OR") and b in ("AND", "OR"):
            return True
        if a == "NOT" and b in ("AND", "OR", "NOT"):
            return True

    # NOT must be followed by operand or '(' (already covered above)
    return False


def detect_query_type(query: str) -> str:
    """
    Return one of: "proximity" | "wildcard" | "boolean" | "natural_language".
    Raise ValueError on malformed inputs per spec.
    """
    if not isinstance(query, str):
        raise ValueError("query must be a string")

    # Global malformed checks that apply to any structured form
    if _has_unmatched_quotes(query):
        raise ValueError("Unmatched quotes")
    if not _balanced_parens(query):
        raise ValueError("Unbalanced parentheses")
    if _bad_phrase_lengths(query):
        raise ValueError("Phrases exceed max length 3 or empty phrase")

    # NEW: unknown operator check (e.g., XOR) even if no AND/OR/NOT present
    toks = query.strip().split()
    for t in toks:
        # allow NEAR/<int> (handled below), AND/OR/NOT; flag other ALL-CAPS tokens (len>=2)
        if _RE_UPPER_OPLIKE.fullmatch(t) and t not in ("AND", "OR", "NOT"):
            # if it's a raw NEAR token without /n, it will be caught by _invalid_near below
            if not t.startswith("NEAR/"):
                raise ValueError("Unknown operator token")

    # Disallow mixed types
    if _has_mixed_types(query):
        raise ValueError("Mixed query types (wildcard/boolean/proximity) are not supported")

    # Proximity first
    if "NEAR" in query or _RE_NEAR.search(query):
        if _invalid_near(query):
            raise ValueError("Malformed NEAR/k")
        return "proximity"

    # Wildcard next
    if '*' in query:
        if _invalid_wildcard(query):
            raise ValueError("Malformed wildcard query")
        return "wildcard"

    # Boolean if contains AND/OR/NOT or any quotes (matched)
    if _RE_BOOL_OP.search(query) or _RE_ANY_QUOTE.search(query):
        if _invalid_boolean_structure(query):
            raise ValueError("Malformed boolean query")
        return "boolean"

    # Otherwise NL
    return "natural_language"

