"""Exercise chunking independently of optional cloud and model dependencies."""
import ast
from pathlib import Path
import unittest


source = Path(__file__).resolve().parents[1] / "src" / "data_ingestion.py"
tree = ast.parse(source.read_text())
function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "chunk_text")
namespace = {"List": list}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
chunk_text = namespace["chunk_text"]


class ChunkBoundaryTests(unittest.TestCase):
    def test_short_and_empty_text(self):
        self.assertEqual(chunk_text("abc"), ["abc"])
        self.assertEqual(chunk_text(""), [])

    def test_overlap_and_final_chunk(self):
        self.assertEqual(chunk_text("abcdefghijk", 5, 2), ["abcde", "defgh", "ghijk"])
        self.assertEqual(len(chunk_text("a" * 25, 10, 2)), 3)

    def test_invalid_settings(self):
        for size, overlap in ((0, 0), (5, 5), (5, 6), (5, -1)):
            with self.subTest(size=size, overlap=overlap), self.assertRaises(ValueError):
                chunk_text("abc", size, overlap)


if __name__ == "__main__":
    unittest.main()
