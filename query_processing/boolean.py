from typing import Set, List, Union, Tuple
from index.access import get_posting_list
import re

TokGram = Union[str, Tuple[str, ...]]

# Keep quoted phrases, operators, parens as separate tokens
RE_TOKEN = re.compile(r'"[^"]*"|\(|\)|\bAND\b|\bOR\b|\bNOT\b|[^()\s]+')


def _as_key(token_or_phrase: str) -> TokGram:
    """
    Convert a token to a postings key:
      - quoted phrase => tuple of tokens (<=3), or single string if only 1 token
      - bare token    => string
    """
    if token_or_phrase.startswith('"') and token_or_phrase.endswith('"'):
        inside = token_or_phrase[1:-1].strip()
        if not inside:
            return ""  # empty: no matches
        toks = inside.split()
        if len(toks) == 1:
            return toks[0]
        return tuple(toks)
    return token_or_phrase


def _postings_for_key(k: TokGram, index_path: str) -> Set[int]:
    """Get postings as a set for a term or n-gram key."""
    if k == "" or k is None:
        return set()
    return set(get_posting_list(k, index_path))


def _collect_universe(tokens: List[str], index_path: str) -> Set[int]:
    """
    Build the 'query universe' U as the union of postings of all operands
    (terms/phrases) that appear in the query. This lets us define NOT deterministically
    as U \ A without needing the whole collection doc ID set.
    """
    U: Set[int] = set()
    for t in tokens:
        if t in ("AND", "OR", "NOT", "(", ")"):
            continue
        U |= _postings_for_key(_as_key(t), index_path)
    return U


def _to_rpn(tokens: List[str]) -> List[Union[str, TokGram]]:
    """
    Convert infix boolean expression to Reverse Polish Notation (RPN)
    using shunting-yard with precedence NOT > AND > OR.
    """
    prec = {"NOT": 3, "AND": 2, "OR": 1}
    right_assoc = {"NOT"}

    output: List[Union[str, TokGram]] = []
    ops: List[str] = []

    for t in tokens:
        if t == "(":
            ops.append(t)
        elif t == ")":
            while ops and ops[-1] != "(":
                output.append(ops.pop())
            if not ops:
                raise ValueError("Unbalanced parentheses")
            ops.pop()  # discard '('
        elif t in ("AND", "OR", "NOT"):
            p = prec[t]
            if t in right_assoc:
                # right-associative: pop strictly-higher precedence
                while ops and ops[-1] in prec and prec[ops[-1]] > p:
                    output.append(ops.pop())
            else:
                # left-associative: pop >= precedence
                while ops and ops[-1] in prec and prec[ops[-1]] >= p:
                    output.append(ops.pop())
            ops.append(t)
        else:
            # operand
            output.append(_as_key(t))

    while ops:
        op = ops.pop()
        if op in ("(", ")"):
            raise ValueError("Unbalanced parentheses")
        output.append(op)

    return output


def _eval_rpn(rpn: List[Union[str, TokGram]], index_path: str, universe: Set[int]) -> Set[int]:
    """
    Evaluate RPN with set semantics.
    - Operand => push postings set
    - NOT A   => push (U \ A)
    - A AND B => push (A ∩ B)
    - A OR  B => push (A ∪ B)
    """
    stack: List[Set[int]] = []

    for token in rpn:
        if token == "NOT":
            if not stack:
                raise ValueError("NOT without operand")
            a = stack.pop()
            stack.append(universe - a)
        elif token == "AND":
            if len(stack) < 2:
                raise ValueError("AND needs two operands")
            b = stack.pop()
            a = stack.pop()
            stack.append(a & b)
        elif token == "OR":
            if len(stack) < 2:
                raise ValueError("OR needs two operands")
            b = stack.pop()
            a = stack.pop()
            stack.append(a | b)
        else:
            # operand postings
            stack.append(_postings_for_key(token, index_path))

    if len(stack) != 1:
        raise ValueError("Malformed boolean expression")
    return stack[0]


def process_boolean_query(query: str, index_path: str) -> Set[int]:
    """
    Process Boolean queries with AND/OR/NOT operators, parentheses, and quoted phrases.

    Precedence: NOT > AND > OR
    Phrases use n-gram lookups against the unified index.
    """
    # 1) Tokenize (operators are case-sensitive and already enforced by detection)
    tokens = RE_TOKEN.findall(query)

    # 2) Build query universe (union of all operand postings)
    U = _collect_universe(tokens, index_path)

    # 3) Convert to RPN and evaluate over sets
    rpn = _to_rpn(tokens)
    return _eval_rpn(rpn, index_path, U)
