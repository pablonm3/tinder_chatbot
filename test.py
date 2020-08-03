import unittest
from app import chat

class MyTestCase(unittest.TestCase):
    def test_opener(self):
        result = chat()
        print("result: ", result)
        self.assertEqual("Robemos un banco!", result)


if __name__ == '__main__':
    unittest.main()
