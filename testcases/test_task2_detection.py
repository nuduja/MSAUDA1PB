import unittest
from query_processing.detection import detect_query_type

class TestTask2Detection(unittest.TestCase):
    def test_boolean_ops(self):
        self.assertEqual(detect_query_type('climate AND change'), 'boolean')
        self.assertEqual(detect_query_type('"machine learning"'), 'boolean')
        self.assertEqual(detect_query_type('(climate OR policy) AND NOT change'), 'boolean')

    def test_wildcard(self):
        self.assertEqual(detect_query_type('climat*'), 'wildcard')
        self.assertEqual(detect_query_type('*tion'), 'wildcard')
        self.assertEqual(detect_query_type('learn*ing'), 'wildcard')

    def test_proximity(self):
        self.assertEqual(detect_query_type('climate NEAR/1 change'), 'proximity')
        self.assertEqual(detect_query_type('"machine learning" NEAR/2 algorithms'), 'proximity')

    def test_natural_language(self):
        self.assertEqual(detect_query_type('climate change effects'), 'natural_language')

    def test_malformed(self):
        with self.assertRaises(ValueError):
            detect_query_type('"unmatched')
        with self.assertRaises(ValueError):
            detect_query_type('climat* AND change')  # mixed wildcard + boolean
        with self.assertRaises(ValueError):
            detect_query_type('A XOR B')              # unknown uppercase op-like
        with self.assertRaises(ValueError):
            detect_query_type('climate NEAR / 2 change')  # non-strict NEAR form
        with self.assertRaises(ValueError):
            detect_query_type('NEAR/2 change')  # missing left operand
