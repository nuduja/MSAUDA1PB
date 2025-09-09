import os
import unittest
from typing import List
from ranking.rankers import rank_documents
from index.builders import create_all_indexes

IDX = "testcases/tmp_task3_fallback.pkl"

class TestTask3MethodsAndFallback(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs: List[List[str]] = [
            ["x", "y", "x"],  # id=101
            ["x"],            # id=102
            ["z"],            # id=103
        ]
        cls.ids = [101, 102, 103]
        create_all_indexes(cls.docs, IDX, cls.ids)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(IDX):
            os.remove(IDX)

    def test_unknown_method_falls_back_to_bm25(self):
        q = ["x"]
        r_default, s_default = rank_documents(q, self.docs, self.ids, IDX, method="default")
        r_bad, s_bad = rank_documents(q, self.docs, self.ids, IDX, method="i_do_not_exist")
        self.assertEqual(r_bad, r_default)
        self.assertEqual(s_bad, s_default)

    def test_scores_for_all_candidates(self):
        q = ["x"]
        ranked, scores = rank_documents(q, self.docs, self.ids, IDX, method="bm25")
        # Must score every candidate
        self.assertEqual(set(ranked), set(self.ids))
        self.assertEqual(len(ranked), len(self.ids))
        self.assertEqual(len(scores), len(self.ids))

if __name__ == "__main__":
    unittest.main()