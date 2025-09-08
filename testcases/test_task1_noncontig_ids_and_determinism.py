import os, unittest, copy
from index.builders import create_all_indexes
from index.access import get_posting_list, get_term_positions

IDX1 = "testcases/tmp_index_det1.pkl"
IDX2 = "testcases/tmp_index_det2.pkl"

class TestDeterminism(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        docs = [
            ["x", "y", "z", "x"],
            ["y", "z", "y"],
            ["z", "x", "y"],
        ]
        dids = [101, 5, 3001]  # non-contiguous, unsorted
        create_all_indexes(copy.deepcopy(docs), IDX1, dids)
        create_all_indexes(copy.deepcopy(docs), IDX2, dids)

    @classmethod
    def tearDownClass(cls):
        for p in (IDX1, IDX2):
            if os.path.exists(p):
                os.remove(p)

    def test_same_results_across_builds(self):
        self.assertEqual(get_posting_list("x", IDX1), get_posting_list("x", IDX2))
        self.assertEqual(get_posting_list(("y","z"), IDX1), get_posting_list(("y","z"), IDX2))
        self.assertEqual(get_term_positions("z", 3001, IDX1), get_term_positions("z", 3001, IDX2))

if __name__ == "__main__":
    unittest.main()
