"""Token-position mapping (Task A-2)."""
from collections import defaultdict
from typing import Dict, List, Union, Tuple

def make_positions(tokens: List[str], n: int = 1) -> Dict[Union[str, Tuple[str, ...]], List[int]]:
    """
    Returns a dictionary mapping each unique n-gram to a list of its starting positions (0-indexed).
    Unigrams are returned as strings, n-grams (n > 1) as tuples of strings.
    """
    pass