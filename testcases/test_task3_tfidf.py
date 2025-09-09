import os
import unittest
from typing import List
from ranking.rankers import rank_documents
from index.builders import create_all_indexes

IDX = "testcases/tmp_task3_tfidf.pkl"

class TestTask3TFIDF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # tfidf: doc 21 has tf=3 for "a"; doc 22 has tf=1
        cls.docs: List[List[str]] = [
            ["a", "a", "a"],  # id=21
            ["a"],            # id=22
            ["b", "b", "b"],  # id=23 (distractor)
        ]
        cls.ids = [21, 22, 23]
        create_all_indexes(cls.docs, IDX, cls.ids)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(IDX):
            os.remove(IDX)

    def test_tfidf_prefers_higher_tf(self):
        q = ["a"]
        ranked, scores = rank_documents(q, self.docs, self.ids, IDX, method="tfidf")
        # doc 21 should outrank doc 22; doc 23 should be last (no "a")
        self.assertLess(ranked.index(21), ranked.index(22))
        self.assertLess(ranked.index(22), ranked.index(23))

if __name__ == "__main__":
    unittest.main()
