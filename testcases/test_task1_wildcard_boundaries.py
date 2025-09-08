import os, unittest
from index.builders import create_all_indexes
from index.access import find_wildcard_matches

IDX = "testcases/tmp_index_wc.pkl"

class TestWildcardBoundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        docs = [
            ["climate", "class", "clear"],
            ["format", "automatic"],
            ["update", "create"],
        ]
        dids = [1, 3, 5]
        create_all_indexes(docs, IDX, dids)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(IDX):
            os.remove(IDX)

    def test_prefix_boundary(self):
        # Lexicographic order is required by the spec
        out = find_wildcard_matches("$cl", IDX)
        self.assertEqual(out, ["class", "clear", "climate"])

    def test_infix_ngram(self):
        # 'mat' appears inside multiple words; lexicographic order
        out = find_wildcard_matches("mat", IDX)
        self.assertEqual(out, ["automatic", "climate", "format"])

    def test_suffix_boundary(self):
        # Terms that END with 'te' (true suffix matches), lexicographic order
        out = find_wildcard_matches("te$", IDX)
        self.assertEqual(out, ["climate", "create", "update"])

    def test_reject_bare_dollar(self):
        self.assertEqual(find_wildcard_matches("$", IDX), [])

if __name__ == "__main__":
    unittest.main()
