import os, unittest
from index.builders import create_all_indexes
from index.access import get_posting_list, get_term_positions

IDX = "testcases/tmp_index_trigram.pkl"

class TestTrigramAndPositions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        docs = [
            ["laminar", "boundary", "layer", "theory"],     # doc 2
            ["turbulent", "boundary", "layer"],             # doc 4
            ["boundary", "layer", "transition"],            # doc 8
        ]
        dids = [2, 4, 8]
        create_all_indexes(docs, IDX, dids)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(IDX):
            os.remove(IDX)

    def test_bigram_postings(self):
        self.assertEqual(get_posting_list(("boundary","layer"), IDX), [2, 4, 8])

    def test_trigram_postings(self):
        self.assertEqual(get_posting_list(("laminar","boundary","layer"), IDX), [2])
        self.assertEqual(get_posting_list(("boundary","layer","transition"), IDX), [8])

    def test_trigram_positions_if_present(self):
        # Builder stores n-gram positions up to 3; check expected offsets
        self.assertEqual(get_term_positions(("laminar","boundary","layer"), 2, IDX), [0])
        self.assertEqual(get_term_positions(("boundary","layer","transition"), 8, IDX), [0])
        # Missing tuple in a doc -> []
        self.assertEqual(get_term_positions(("laminar","boundary","layer"), 8, IDX), [])

if __name__ == "__main__":
    unittest.main()
