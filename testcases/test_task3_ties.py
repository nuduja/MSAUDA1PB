import os
import unittest
from typing import List
from ranking.rankers import rank_documents
from index.builders import create_all_indexes

IDX = "testcases/tmp_task3_ties.pkl"

class TestTask3Ties(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Two identical docs; tie should break by ascending doc_id
        cls.docs: List[List[str]] = [
            ["x"],  # id=20
            ["x"],  # id=10
        ]
        cls.ids = [20, 10]  # intentionally unsorted
        create_all_indexes(cls.docs, IDX, cls.ids)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(IDX):
            os.remove(IDX)

    def test_bm25_tie_break_by_docid(self):
        ranked, scores = rank_documents(["x"], self.docs, self.ids, IDX, method="bm25")
        self.assertEqual(ranked, [10, 20])  # ascending doc_id on ties

    def test_tfidf_tie_break_by_docid(self):
        ranked, scores = rank_documents(["x"], self.docs, self.ids, IDX, method="tfidf")
        self.assertEqual(ranked, [10, 20])

if __name__ == "__main__":
    unittest.main()
