import os, unittest
from typing import List
from index.builders import create_all_indexes
from query_processing.wildcard import process_wildcard_query

IDX = "testcases/tmp_task2_wildcard.pkl"

class TestTask2Wildcard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        docs: List[List[str]] = [
            ["climate","change","effects"],          # 10
            ["machine","learning","algorithms"],     # 20
            ["climate","science","research"],        # 30
            ["renewable","energy","transition"],     # 40
        ]
        dids = [10, 20, 30, 40]
        create_all_indexes(docs, IDX, dids)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(IDX):
            os.remove(IDX)

    def test_prefix(self):
        # climat* -> matches 'climate' -> docs 10,30
        self.assertEqual(process_wildcard_query("climat*", IDX), {10, 30})

    def test_suffix(self):
        # *tion -> matches 'transition' -> doc 40
        self.assertEqual(process_wildcard_query("*tion", IDX), {40})

    def test_infix(self):
        # learn*ing -> 'learning' -> doc 20
        self.assertEqual(process_wildcard_query("learn*ing", IDX), {20})

if __name__ == "__main__":
    unittest.main()
