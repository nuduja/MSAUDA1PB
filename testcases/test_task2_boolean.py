import os, unittest
from typing import List
from index.builders import create_all_indexes
from query_processing.boolean import process_boolean_query

IDX = "testcases/tmp_task2_boolean.pkl"

class TestTask2Boolean(unittest.TestCase):
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

    def test_and_or(self):
        # climate AND change -> only doc 10
        self.assertEqual(process_boolean_query("climate AND change", IDX), {10})
        # climate OR science -> docs 10 and 30
        self.assertEqual(process_boolean_query("climate OR science", IDX), {10, 30})

    def test_not(self):
        # climate AND NOT science -> doc 10 only
        self.assertEqual(process_boolean_query("climate AND NOT science", IDX), {10})

    def test_parentheses(self):
        # (climate AND change) OR science -> {10,30}
        self.assertEqual(process_boolean_query("( climate AND change ) OR science", IDX), {10, 30})

    def test_phrase(self):
        # exact bigram "machine learning" -> doc 20
        self.assertEqual(process_boolean_query('"machine learning"', IDX), {20})

if __name__ == "__main__":
    unittest.main()
