import os
import math
import unittest
from typing import List
from ranking.rankers import rank_documents
from index.builders import create_all_indexes

IDX = "testcases/tmp_task3_basic.pkl"

class TestTask3BasicRanking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # d1 has both terms, with higher tf for "change"
        # d2 has only "climate"
        # d3 has none of the query terms
        cls.docs: List[List[str]] = [
            ["climate", "change", "change"],  # id=1
            ["climate", "policy"],            # id=2
            ["machine", "learning"],          # id=3
        ]
        cls.ids = [1, 2, 3]
        create_all_indexes(cls.docs, IDX, cls.ids)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(IDX):
            os.remove(IDX)

    def test_bm25_default_full_permutation(self):
        q = ["climate", "change"]
        ranked, scores = rank_documents(q, self.docs, self.ids, IDX, method="default")
        # shape checks
        self.assertEqual(set(ranked), set(self.ids))
        self.assertEqual(len(ranked), len(scores))
        for s in scores:
            self.assertTrue(math.isfinite(s))

        # relevance ordering: doc 1 (both terms) > doc 2 (one term) > doc 3 (zero)
        self.assertGreater(scores[ranked.index(1)], scores[ranked.index(2)])
        self.assertGreater(scores[ranked.index(2)], scores[ranked.index(3)])

    def test_oov_all_zero_tie_break_by_docid(self):
        q = ["nonexistent"]
        ranked, scores = rank_documents(q, self.docs, self.ids, IDX, method="bm25")
        # all scores zero -> order must be ascending doc_id
        self.assertEqual(ranked, sorted(self.ids))
        self.assertTrue(all(abs(s) < 1e-12 for s in scores))

if __name__ == "__main__":
    unittest.main()