import unittest
from retrieval import similarity_reminder
class ReminderTests(unittest.TestCase):
    def test_below_threshold(self):
        self.assertIn('适当降低',similarity_reminder(.3,.5))
    def test_equal_or_above(self):
        self.assertIsNone(similarity_reminder(.5,.5))
        self.assertIsNone(similarity_reminder(.6,.5))
    def test_empty(self):
        self.assertIsNone(similarity_reminder(None,.5))
