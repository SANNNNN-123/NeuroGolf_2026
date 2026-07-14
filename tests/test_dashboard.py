import unittest

from tools import dashboard


MANIFEST = {
    "1": {"points": 25.0, "memory": 0, "params": 10, "method": "custom:task001"},
    "2": {"points": 22.0, "memory": 5, "params": 5, "method": "custom:task002"},
    "3": {"points": 20.0, "memory": 100, "params": 0, "method": "ext:task003"},
    "4": {"points": 18.0, "memory": 200, "params": 0, "method": "custom:task004"},
    "5": {"points": 17.999, "memory": 900, "params": 0, "method": "custom:task005"},
}

BY_OP = {
    "Einsum": [1, 2, 3],
    "Conv": [4, 5],
    "MaxPool": [5],
}

ARC = {
    "1": {"arc_id": "aaa", "generator": "g1"},
    "2": {"arc_id": "bbb", "generator": "g2"},
}


class BandTest(unittest.TestCase):
    def test_boundaries_land_in_expected_bands(self):
        self.assertEqual(dashboard.band_of(25.0), "25 (max)")
        self.assertEqual(dashboard.band_of(24.999), "25 (max)")
        self.assertEqual(dashboard.band_of(24.9), "22-25")
        self.assertEqual(dashboard.band_of(22.0), "22-25")
        self.assertEqual(dashboard.band_of(21.999), "20-22")
        self.assertEqual(dashboard.band_of(20.0), "20-22")
        self.assertEqual(dashboard.band_of(18.0), "18-20")
        self.assertEqual(dashboard.band_of(17.999), "<18")


class BuildRowsTest(unittest.TestCase):
    def test_one_row_per_task_with_merged_fields(self):
        rows = dashboard.build_rows(MANIFEST, BY_OP, ARC)
        self.assertEqual(len(rows), 5)
        self.assertEqual([r["task"] for r in rows], [1, 2, 3, 4, 5])

        r1 = rows[0]
        self.assertEqual(r1["arc_id"], "aaa")
        self.assertEqual(r1["mem_params"], 10)
        self.assertEqual(r1["method_prefix"], "custom")
        self.assertEqual(r1["headroom"], 0.0)
        self.assertEqual(r1["ops"], "Einsum")
        self.assertEqual(r1["n_ops"], 1)
        self.assertEqual(r1["band"], "25 (max)")

    def test_task_with_multiple_ops_and_missing_arc(self):
        rows = dashboard.build_rows(MANIFEST, BY_OP, ARC)
        r5 = rows[4]
        self.assertEqual(r5["arc_id"], "")  # no arc mapping entry
        self.assertEqual(sorted(r5["ops"].split(", ")), ["Conv", "MaxPool"])
        self.assertEqual(r5["n_ops"], 2)


class AggregateTest(unittest.TestCase):
    def setUp(self):
        self.rows = dashboard.build_rows(MANIFEST, BY_OP, ARC)

    def test_score_bands_counts(self):
        bands = {b["band"]: b for b in dashboard.score_bands(self.rows)}
        self.assertEqual(bands["25 (max)"]["count"], 1)
        self.assertEqual(bands["22-25"]["count"], 1)
        self.assertEqual(bands["20-22"]["count"], 1)
        self.assertEqual(bands["18-20"]["count"], 1)
        self.assertEqual(bands["<18"]["count"], 1)
        self.assertAlmostEqual(bands["25 (max)"]["pct"], 20.0)

    def test_kpis(self):
        k = dashboard.kpis(self.rows)
        self.assertEqual(k["task_count"], 5)
        self.assertEqual(k["theoretical_max"], 125.0)
        self.assertEqual(k["max_count"], 1)
        self.assertEqual(k["low_count"], 1)
        self.assertAlmostEqual(k["total_points"], 103.0, places=2)  # 102.999 -> round(2)
        self.assertAlmostEqual(k["gap_to_max"], 22.0, places=2)

    def test_op_breakdown_inverts_by_op(self):
        ob = {o["op"]: o["count"] for o in dashboard.op_breakdown(BY_OP)}
        self.assertEqual(ob["Einsum"], 3)
        self.assertEqual(ob["Conv"], 2)
        self.assertEqual(ob["MaxPool"], 1)
        # sorted most-used first
        self.assertEqual(dashboard.op_breakdown(BY_OP)[0]["op"], "Einsum")

    def test_method_breakdown(self):
        mb = {m["method"]: m for m in dashboard.method_breakdown(self.rows)}
        self.assertEqual(mb["custom"]["count"], 4)
        self.assertEqual(mb["ext"]["count"], 1)


class TileTest(unittest.TestCase):
    def setUp(self):
        self.rows = dashboard.build_rows(MANIFEST, BY_OP, ARC)

    def test_points_to_color_low_is_red_high_is_green(self):
        low = dashboard.points_to_color(14.0)
        high = dashboard.points_to_color(25.0)
        self.assertTrue(low.startswith("hsl(0,"))       # hue 0 = red
        self.assertTrue(high.startswith("hsl(130,"))     # hue 130 = green
        # clamps below/above range
        self.assertEqual(dashboard.points_to_color(5.0), dashboard.points_to_color(14.0))
        self.assertEqual(dashboard.points_to_color(99.0), dashboard.points_to_color(25.0))

    def test_order_rows(self):
        by_task = [r["task"] for r in dashboard.order_rows(self.rows, "task number")]
        self.assertEqual(by_task, [1, 2, 3, 4, 5])
        low_first = [r["task"] for r in dashboard.order_rows(self.rows, "score (low first)")]
        self.assertEqual(low_first[0], 5)   # 17.999 is lowest
        self.assertEqual(low_first[-1], 1)  # 25.0 is highest
        high_first = [r["task"] for r in dashboard.order_rows(self.rows, "score (high first)")]
        self.assertEqual(high_first[0], 1)

    def test_tiles_html_has_one_link_per_task(self):
        markup = dashboard._tiles_html(self.rows)
        self.assertEqual(markup.count('<a href="/viewer?task='), len(self.rows))
        self.assertIn("/viewer?task=3", markup)
        self.assertIn("neurogolf_viewer", markup)

    def test_tiles_height_scales_with_rows(self):
        # 5 tasks in 20 columns -> 1 row
        self.assertEqual(
            dashboard._tiles_height(5, columns=20),
            1 * (dashboard.TILE_HEIGHT + dashboard.TILE_GAP) + dashboard.TILE_GAP + 6,
        )
        # 400 tasks in 20 columns -> 20 rows
        self.assertEqual(
            dashboard._tiles_height(400, columns=20),
            20 * (dashboard.TILE_HEIGHT + dashboard.TILE_GAP) + dashboard.TILE_GAP + 6,
        )


if __name__ == "__main__":
    unittest.main()
