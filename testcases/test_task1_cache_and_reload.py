import os, unittest
from index.builders import create_all_indexes
from index.access import get_posting_list  # uses package cache internally

IDX = "testcases/tmp_index_cache.pkl"

class TestCacheReload(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        docs = [["a","b","c"], ["a","a","d"]]
        dids = [2, 4]
        create_all_indexes(docs, IDX, dids)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(IDX):
            os.remove(IDX)

    def test_multiple_calls_same_path(self):
        # First call loads & caches; subsequent calls must behave identically
        p1 = get_posting_list("a", IDX)
        p2 = get_posting_list("a", IDX)
        self.assertEqual(p1, [2, 4])
        self.assertEqual(p1, p2)

if __name__ == "__main__":
    unittest.main()
