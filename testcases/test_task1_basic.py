"""
Basic correctness tests for Task 1 (Indexing and Access).
Run with:  python -m unittest testcases/test_task1_basic.py
"""

import os
import unittest
from index.builders import create_all_indexes
from index.access import get_posting_list, find_wildcard_matches, get_term_positions

TEST_INDEX_PATH = "testcases/tmp_index.pkl"

class TestTask1Basic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Tiny mock corpus
        docs = [
            ["climate", "change", "is", "real"],
            ["machine", "learning", "climate", "models"],
            ["deep", "learning", "for", "climate", "change"],
        ]
        doc_ids = [10, 20, 30]
        create_all_indexes(docs, TEST_INDEX_PATH, doc_ids)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(TEST_INDEX_PATH):
            os.remove(TEST_INDEX_PATH)

    def test_posting_list_unigram(self):
        postings = get_posting_list("climate", TEST_INDEX_PATH)
        self.assertEqual(postings, [10, 20, 30])

    def test_posting_list_bigram(self):
        postings = get_posting_list(("climate", "change"), TEST_INDEX_PATH)
        self.assertEqual(postings, [10, 30])

    def test_wildcard(self):
        matches = find_wildcard_matches("$cl", TEST_INDEX_PATH)
        self.assertIn("climate", matches)

    def test_positions(self):
        pos = get_term_positions("climate", 10, TEST_INDEX_PATH)
        # "climate" is at index 0 in doc 10
        self.assertEqual(pos, [0])

    def test_oov(self):
        self.assertEqual(get_posting_list("doesnotexist", TEST_INDEX_PATH), [])
        self.assertEqual(find_wildcard_matches("zzz", TEST_INDEX_PATH), [])
        self.assertEqual(get_term_positions("change", 999, TEST_INDEX_PATH), [])

if __name__ == "__main__":
    unittest.main()