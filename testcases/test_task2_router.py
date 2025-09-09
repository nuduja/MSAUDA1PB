import os, unittest
from typing import List
from index.builders import create_all_indexes
from query_processing.query_process import process_query

IDX = "testcases/tmp_task2_router.pkl"

class TestTask2Router(unittest.TestCase):
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

    def test_router_boolean(self):
        self.assertEqual(process_query("climate AND change", IDX), {10})

    def test_router_wildcard(self):
        self.assertEqual(process_query("climat*", IDX), {10, 30})

    def test_router_proximity(self):
        self.assertEqual(process_query('climate NEAR/1 change', IDX), {10})

    def test_router_natural_language(self):
        # "climate change" -> "climate OR change" -> docs containing either: {10,30}
        self.assertEqual(process_query('climate change', IDX), {10, 30})

if __name__ == "__main__":
    unittest.main()
