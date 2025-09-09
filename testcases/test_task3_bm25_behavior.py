import os
import unittest
from typing import List
from ranking.rankers import rank_documents
from index.builders import create_all_indexes

IDX = "testcases/tmp_task3_bm25.pkl"

class TestTask3BM25Behavior(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Same tf for "t" (1) but very different lengths:
        # shorter doc should score higher under BM25 normalization.
        cls.docs: List[List[str]] = [
            ["t"],                                  # id=10, dl=1
            ["t"] + ["f"] * 100,                    # id=20, dl=101
        ]
        cls.ids = [10, 20]
        create_all_indexes(cls.docs, IDX, cls.ids)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(IDX):
            os.remove(IDX)

    def test_length_normalization(self):
        q = ["t"]
        ranked, scores = rank_documents(q, self.docs, self.ids, IDX, method="bm25")
        # shorter doc (10) should beat longer (20)
        self.assertEqual(ranked[0], 10)
        self.assertGreater(scores[0], scores[1])

    def test_default_equals_bm25(self):
        q = ["t"]
        r1, s1 = rank_documents(q, self.docs, self.ids, IDX, method="default")
        r2, s2 = rank_documents(q, self.docs, self.ids, IDX, method="bm25")
        self.assertEqual(r1, r2)
        self.assertEqual(s1, s2)

if __name__ == "__main__":
    unittest.main()
