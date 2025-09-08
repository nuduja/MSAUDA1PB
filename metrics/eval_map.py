#!/usr/bin/env python3
"""
MAP Evaluation Script for Task 4
Evaluates all JSON run files in ./runs/ against development relevance judgments.
Usage: python metrics/eval_map.py
"""

import json
import pathlib
import sys
from collections import defaultdict
