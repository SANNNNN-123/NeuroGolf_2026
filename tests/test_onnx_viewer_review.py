import unittest

import numpy as np

from tools import onnx_viewer as viewer


class ReviewSummaryTest(unittest.TestCase):
    def test_summarizes_fixed_overlay_and_bbox(self):
        inp = np.full((30, 30), -1, dtype=np.int16)
        out = np.full((30, 30), -1, dtype=np.int16)
        inp[:5, :5] = 0
        out[:5, :5] = 0
        out[1:4, 2:5] = 3

        summary = viewer.summarize_data_diff([{"input_grid": inp, "expected_grid": out}])

        self.assertEqual(summary["examples"], 1)
        self.assertEqual(summary["changed_examples"], 1)
        self.assertEqual(summary["changed_colors"], {3: 9})
        self.assertEqual(summary["bbox_min"], (1, 2))
        self.assertEqual(summary["bbox_max"], (3, 4))
        self.assertTrue(summary["fixed_delta"])

    def test_filters_oversized_examples_instead_of_crashing(self):
        good = {"source": "arc-gen", "index": 0,
                "input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}
        # 15x34 input exceeds the 30-wide ONNX grid; official scoring skips it.
        oversized = {"source": "arc-gen", "index": 1,
                     "input": [[0] * 34 for _ in range(15)],
                     "output": [[0] * 17 for _ in range(15)]}

        valid, skipped = viewer.convertible_examples([good, oversized])

        self.assertEqual(skipped, 1)
        self.assertEqual(len(valid), 1)
        self.assertEqual(valid[0]["index"], 0)
        # Every surviving example must be renderable without raising.
        for example in valid:
            viewer.example_to_arrays(example)

    def test_suggests_questions_from_cost_and_data_patterns(self):
        data_summary = {
            "fixed_delta": True,
            "mostly_copy": True,
            "avg_changed_cells": 12.0,
            "bbox_regular": True,
        }
        cost_summary = {
            "expensive_ops": [{"op": "Where", "bytes": 5400}, {"op": "MaxPool", "bytes": 3600}],
            "has_full_label_plane": True,
            "has_repeated_planes": True,
        }

        questions = viewer.review_questions(data_summary, cost_summary)

        joined = "\n".join(questions)
        self.assertIn("fixed colour", joined)
        self.assertIn("full label plane", joined)
        self.assertIn("walk-einsum", joined)


if __name__ == "__main__":
    unittest.main()
