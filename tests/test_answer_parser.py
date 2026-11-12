"""Pure CPU checks; import benchmarks without the model package initializer."""
import importlib.util
import itertools
import os
from pathlib import Path
import unittest

SOURCE = Path(os.environ.get("CVRR_BENCHMARKS", Path(__file__).resolve().parents[1] / "cvrr/benchmarks.py"))
spec = importlib.util.spec_from_file_location("benchmarks", SOURCE)
benchmarks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmarks)


class AnswerParserTests(unittest.TestCase):
    def test_choice_word(self):
        self.assertEqual(benchmarks.extract_final_letter("The final answer is choice B."), "B")

    def test_definitely_word(self):
        self.assertEqual(benchmarks.extract_final_letter("The answer is definitely B."), "B")

    def test_based_word(self):
        self.assertEqual(benchmarks.extract_final_letter("Answer: Based on the image, option C."), "C")

    def test_word_initials_rejected(self):
        for text in ("Answer: Blue", "Option Excellent", "<answer>correct</answer>", "ordinary prose", "", None):
            with self.subTest(text=text):
                self.assertEqual(benchmarks.extract_final_letter(text), "")

    def test_valid_formats(self):
        for letter, template, prefix in itertools.product("ABCDE", ["{}", "({})", "<answer>{}</answer>", r"\boxed{{{}}}", "The answer is {}.", "Final answer: {}", "Option {}", "choice ({})"], ["", "Reasoning completed. ", "2026 examples. "]):
            with self.subTest(letter=letter, template=template, prefix=prefix):
                self.assertEqual(benchmarks.extract_final_letter(prefix + template.format(letter)), letter)

    def test_case_punctuation_and_end_tokens(self):
        for letter, suffix in itertools.product("abcde", ["", ".", "!", " <|im_end|>", " <|eot_id|>"]):
            self.assertEqual(benchmarks.extract_final_letter("Answer: " + letter + suffix), letter.upper())

    def test_precedence_and_last_match(self):
        for text, expected in (("<answer>B</answer> Answer: C", "B"), ("Answer: A. Final answer: D.", "D"), ("Option A. Choice E.", "E"), ("<answer>A</answer> <answer>C</answer>", "C")):
            self.assertEqual(benchmarks.extract_final_letter(text), expected)


if __name__ == "__main__":
    unittest.main()
