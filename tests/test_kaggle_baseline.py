import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "kaggle_commfor_baseline.py"
SPEC = importlib.util.spec_from_file_location("kaggle_commfor_baseline", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BaselineHelpersTests(unittest.TestCase):
    def test_stratified_sample_rotates_categories(self) -> None:
        paths = [Path(category) / f"{index}.jpg" for category in ("a", "b") for index in range(3)]
        selected = MODULE.stratified_sample(paths, count=4, seed=323)
        self.assertEqual({path.parent.name for path in selected}, {"a", "b"})
        self.assertEqual(len(selected), 4)

    def test_find_class_directory_requires_one_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "real_dataset").mkdir()
            self.assertEqual(MODULE.find_class_directory(root, "real_dataset"), root / "real_dataset")
            (root / "nested" / "real_dataset").mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError, "found 2"):
                MODULE.find_class_directory(root, "real_dataset")

    def test_find_class_directory_supports_kaggle_mount_alias(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = root / "unexpected-mount-name" / "real_dataset"
            expected.mkdir(parents=True)

            self.assertEqual(MODULE.find_class_directory(root, "real_dataset"), expected)


if __name__ == "__main__":
    unittest.main()
