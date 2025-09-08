import os, unittest
from index.builders import create_all_indexes
from index.access import get_posting_list, get_term_positions, find_wildcard_matches

IDX = "testcases/tmp_index_sort.pkl"

class TestSortingAndDedup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Duplicate tokens + shuffled doc ids to stress dedup/sort
        docs = [
            ["alpha", "beta", "alpha", "alpha"],           # doc 101
            ["beta", "gamma", "beta"],                     # doc 7   (no 'alpha' here)
            ["alpha", "gamma", "beta", "beta", "alpha"],   # doc 999
        ]
        doc_ids = [101, 7, 999]  # non-contiguous + unsorted
        create_all_indexes(docs, IDX, doc_ids)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(IDX):
            os.remove(IDX)

    def test_postings_sorted_and_dedup(self):
        # 'alpha' appears only in docs 101 and 999 (NOT in 7)
        self.assertEqual(get_posting_list("alpha", IDX), [101, 999])
        self.assertEqual(get_posting_list("beta", IDX), [7, 101, 999])
        self.assertEqual(get_posting_list("gamma", IDX), [7, 999])

    def test_positions_sorted_and_dedup(self):
        # alpha positions in doc 101: [0, 2, 3]
        self.assertEqual(get_term_positions("alpha", 101, IDX), [0, 2, 3])
        # beta positions in doc 999: [2, 3]
        self.assertEqual(get_term_positions("beta", 999, IDX), [2, 3])

    def test_oov_returns_empty(self):
        self.assertEqual(get_posting_list("nope", IDX), [])
        self.assertEqual(get_term_positions("alpha", 123456, IDX), [])
        self.assertEqual(find_wildcard_matches("$", IDX), [])  # per spec

if __name__ == "__main__":
    unittest.main()
