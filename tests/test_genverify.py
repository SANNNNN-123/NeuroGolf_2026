from pathlib import Path
import unittest

from src import genverify


class GenverifyPathTest(unittest.TestCase):
    def test_load_gen_uses_repo_local_arc_gen_when_tmp_arc_gen_is_absent(self):
        self.assertFalse(Path("/tmp/arc-gen/tasks/task_7b6016b9.py").exists())

        gen = genverify.load_gen(187)

        self.assertEqual(
            Path(gen.__file__).resolve(),
            (Path.cwd() / "arc-gen" / "tasks" / "task_7b6016b9.py").resolve(),
        )
        self.assertTrue(hasattr(gen, "generate"))


if __name__ == "__main__":
    unittest.main()
