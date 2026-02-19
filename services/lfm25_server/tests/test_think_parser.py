import unittest

from lfm25_server.think_parser import parse_think


class ThinkParserTests(unittest.TestCase):
    def test_no_thinking_block(self) -> None:
        result = parse_think("Hello world")
        self.assertIsNone(result.thinking)
        self.assertEqual(result.answer, "Hello world")

    def test_multiple_leading_blocks(self) -> None:
        text = "<think>a</think>\n<think>b</think>\nFinal answer"
        result = parse_think(text)
        self.assertEqual(result.thinking, "a\n\nb")
        self.assertEqual(result.answer, "Final answer")

    def test_malformed_block_falls_back_to_raw(self) -> None:
        text = "<think>broken"
        result = parse_think(text)
        self.assertIsNone(result.thinking)
        self.assertEqual(result.answer, "<think>broken")


if __name__ == "__main__":
    unittest.main()
