import os
import unittest
from typing import List
from ranking.rankers import rank_documents
from index.builders import create_all_indexes
from index.io import load as load_pkg, dump as dump_pkg

IDX = "testcases/tmp_task3_edge.pkl"
IDX_MISS_META = "testcases/tmp_task3_edge_missing_meta.pkl"

class TestTask3EdgeCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Doc with 1 'x'; Doc with 2 'x'; One distractor with 'y'
        cls.docs: List[List[str]] = [
            ["x"],            # id=5
            ["x", "x"],       # id=1
            ["y"],            # id=3
        ]
        cls.ids = [5, 1, 3]  # deliberately unsorted
        create_all_indexes(cls.docs, IDX, cls.ids)

        # Write a copy with some __META__ fields removed to test fallbacks
        pkg = load_pkg(IDX)
        meta = pkg.get("__META__", {})
        # simulate partial metadata: drop N and avgdl (doc_lengths kept)
        if "__META__" in pkg:
            meta.pop("N", None)
            meta.pop("avgdl", None)
            pkg["__META__"] = meta
        dump_pkg(pkg, IDX_MISS_META)

    @classmethod
    def tearDownClass(cls):
        for p in (IDX, IDX_MISS_META):
            if os.path.exists(p):
                os.remove(p)

    def test_repeated_query_terms(self):
        # Repeating the same term in the query should not crash and typically
        # favors the doc with higher tf ('x','x' beats 'x')
        ranked, scores = rank_documents(["x","x","x"], self.docs, self.ids, IDX, method="bm25")
        self.assertEqual(ranked[0], 1)

    def test_unsorted_candidates_and_permutation(self):
        ranked, scores = rank_documents(["x"], self.docs, self.ids, IDX, method="bm25")
        self.assertEqual(set(ranked), set(self.ids))
        self.assertEqual(len(ranked), len(self.ids))
        self.assertEqual(len(scores), len(self.ids))

    def test_missing_meta_fallback(self):
        # Ensure ranking still runs if N/avgdl missing in __META__
        ranked, scores = rank_documents(["x"], self.docs, self.ids, IDX_MISS_META, method="bm25")
        self.assertEqual(set(ranked), set(self.ids))
        self.assertEqual(len(ranked), len(scores))

if __name__ == "__main__":
    unittest.main()