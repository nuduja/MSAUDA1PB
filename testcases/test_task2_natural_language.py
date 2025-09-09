import unittest
from query_processing.query_process import convert_natural_language

class TestTask2NaturalLanguage(unittest.TestCase):
    def test_or_join(self):
        self.assertEqual(convert_natural_language('climate change'), 'climate OR change')
        self.assertEqual(convert_natural_language('a b c'), 'a OR b OR c')

    def test_empty(self):
        self.assertEqual(convert_natural_language(''), '')
        self.assertEqual(convert_natural_language('   '), '')

if __name__ == "__main__":
    unittest.main()
