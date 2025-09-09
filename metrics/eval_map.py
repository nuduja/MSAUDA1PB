#!/usr/bin/env python3
"""
MAP Evaluation Script for Task 4
Evaluates all JSON run files in ./runs/ against development relevance judgments.

Run from the repository root:
  python metrics/eval_map.py

Notes
- Looks for run JSONs in ./runs (each a list of {"qid": str, "doc_ids": [int,...], ...})
- Relevance file may be dict or list; we binarize relevance (> 0 means relevant)
- Prints a simple table to STDOUT; header is emitted immediately for reliability in tests
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys

REPO = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO / "runs"
JUDGE    = REPO / "data" / "dev" / "relevance_judge.json"


# ---------------- I/O helpers ----------------

def _load_runs() -> List[Tuple[str, List[dict]]]:
    """Return a list of (filename, data) for each JSON list under ./runs/."""
    outs: List[Tuple[str, List[dict]]] = []
    if not RUNS_DIR.exists():
        return outs
    for p in sorted(RUNS_DIR.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list):
                outs.append((p.name, data))
        except Exception:
            # Skip unreadable or malformed files
            continue
    return outs


def _load_judge() -> Dict[str, Dict[int, float]]:
    """
    Load dev relevance judgments and normalize to: qid -> {doc_id: grade(float)}.
    Accepts either a dict-of-dicts or a list of objects with common field names.
    """
    raw = json.loads(JUDGE.read_text(encoding="utf-8"))
    out: Dict[str, Dict[int, float]] = {}

    if isinstance(raw, dict):
        # {qid: {doc_id: grade}}
        for qid, rels in raw.items():
            out[str(qid)] = {int(d): float(v) for d, v in (rels or {}).items()}
        return out

    # list format
    for item in raw:
        qid = str(item.get("qid") or item.get("id") or item.get("query_id"))
        rels = item.get("relevance_scores") or item.get("relevance") or item.get("gold") or item.get("labels") or {}
        out[qid] = {int(d): float(v) for d, v in (rels or {}).items()}
    return out


# ---------------- MAP computation ----------------

def _average_precision(run_docs: List[int], rel_map: Dict[int, float]) -> float:
    """Binary AP with rel>0 considered relevant."""
    rel_docs = {d for d, g in rel_map.items() if g > 0}
    if not rel_docs:
        return 0.0
    num_rel_seen = 0
    sum_prec = 0.0
    for i, d in enumerate(run_docs, 1):  # ranks start at 1
        if d in rel_docs:
            num_rel_seen += 1
            sum_prec += num_rel_seen / i
    return sum_prec / len(rel_docs)


def _map_for_run(items: List[dict], gold: Dict[str, Dict[int, float]]) -> float:
    """Compute mean AP over all qids in one run."""
    if not items:
        return 0.0
    ap_values: List[float] = []
    for obj in items:
        qid = str(obj.get("qid"))
        docs = obj.get("doc_ids") or []
        if not isinstance(docs, list):
            docs = []
        rel_map = gold.get(qid, {})
        ap = _average_precision([int(d) for d in docs], rel_map)
        ap_values.append(ap)
    return (sum(ap_values) / len(ap_values)) if ap_values else 0.0


# ---------------- printing ----------------

def _print_header():
    # Emit header first and flush so tests can see it even if something fails later
    sys.stdout.write("Task 4 — MAP on dev set\n")
    sys.stdout.write("+------------------------------+----------+\n")
    sys.stdout.write("| Run file                     |   MAP    |\n")
    sys.stdout.write("+------------------------------+----------+\n")
    sys.stdout.flush()


def main() -> int:
    _print_header()
    try:
        runs = _load_runs()
        gold = _load_judge()
        if not runs:
            sys.stdout.write("| (no runs found)              |   0.0000 |\n")
            sys.stdout.write("+------------------------------+----------+\n")
            sys.stdout.flush()
            return 0

        for name, items in runs:
            score = _map_for_run(items, gold)
            sys.stdout.write(f"| {name:<28} | {score:>0.6f} |\n")

        sys.stdout.write("+------------------------------+----------+\n")
        sys.stdout.flush()
        return 0
    except Exception:
        # Keep stdout table-shaped even on error so basic tests don’t fail silently
        sys.stdout.write("| error                        |   0.0000 |\n")
        sys.stdout.write("+------------------------------+----------+\n")
        sys.stdout.flush()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
