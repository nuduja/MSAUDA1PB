import os, unittest
from typing import List
from index.builders import create_all_indexes
from query_processing.proximity import process_proximity_query

IDX = "testcases/tmp_task2_proximity.pkl"

class TestTask2Proximity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        docs: List[List[str]] = [
            ["climate","change","effects"],          # 10: positions 0,1,2
            ["machine","learning","algorithms"],     # 20: 0,1,2
            ["climate","science","research"],        # 30
            ["renewable","energy","transition"],     # 40
        ]
        dids = [10, 20, 30, 40]
        create_all_indexes(docs, IDX, dids)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(IDX):
            os.remove(IDX)

    def test_adjacent_unigrams(self):
        # climate (0) NEAR/1 change (1) -> True in doc 10
        self.assertEqual(process_proximity_query("climate NEAR/1 change", IDX), {10})

    def test_phrase_to_term(self):
        # "machine learning" (0..1) NEAR/2 algorithms (2) -> distance min(|2-1|,|0-2|)=1 <= 2
        self.assertEqual(process_proximity_query('"machine learning" NEAR/2 algorithms', IDX), {20})

    def test_term_to_phrase_overlap_hit(self):
        # change (1..1) NEAR/0 "climate change" (0..1) -> spans overlap => distance 0 => hit
        self.assertEqual(process_proximity_query('change NEAR/0 "climate change"', IDX), {10})

if __name__ == "__main__":
    unittest.main()
