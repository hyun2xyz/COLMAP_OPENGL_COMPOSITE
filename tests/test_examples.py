"""The setup check and the downloader must agree about the examples outside the pipeline."""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/"scripts"/f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Examples(unittest.TestCase):
    def setUp(self):
        self.setup, self.fetch = load("setup_environment"), load("fetch_assets")

    def test_every_example_input_can_be_downloaded(self):
        known = {clip[0] for clip in self.fetch.CLIPS.values()} | {self.fetch.POSE[0]}
        for name, need in self.setup.EXAMPLES.items():
            self.assertTrue((ROOT/"examples"/name/"README.md").is_file(), name)
            for path in need.get("files", ()):
                self.assertIn(path, known, f"{name}: no download for {path}")

    def test_optional_groups_have_requirement_files(self):
        for name, need in self.setup.EXAMPLES.items():
            for extra in need.get("extras", ()):
                self.assertIn(extra, self.setup.EXTRAS, name)
        for filename, packages in self.setup.EXTRAS.values():
            text = (ROOT/filename).read_text(encoding="utf-8")
            for package in packages:
                self.assertIn(package, text)

    def test_downloads_stay_out_of_git(self):
        ignore = (ROOT/".gitignore").read_text(encoding="utf-8")
        for path, *_ in self.fetch.CLIPS.values():
            self.assertTrue(path.endswith(".mp4") and "*.mp4" in ignore, path)
        self.assertIn("*.task", ignore)

    def test_examples_report_does_not_change_ready(self):
        extras, examples = self.setup.examples_state(core_ready=False)
        self.assertEqual(set(examples), set(self.setup.EXAMPLES))
        self.assertTrue(all(not state["ready"] for state in examples.values()))


if __name__ == "__main__":
    unittest.main()
