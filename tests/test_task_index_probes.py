import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

def _load():
    spec = importlib.util.spec_from_file_location(
        "task_index_probes", ROOT / "reports/scripts/task_index_probes.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

def _identity_samples():
    g = np.array([[1, 2, 0], [0, 3, 4]])
    return [(g.copy(), g.copy()) for _ in range(20)]

def _fill_samples():
    # output = input with all-0 cells set to colour 5 (fixed-colour delta)
    out = []
    for _ in range(20):
        g = np.random.randint(0, 4, size=(5, 5))
        o = g.copy(); o[o == 0] = 5
        out.append((g, o))
    return out

def test_shape_relation_equal_on_identity():
    m = _load()
    val, conf = m.probe_shape_relation(_identity_samples())
    assert val == "equal" and conf == 1.0

def test_delta_copy_frac_high_on_fill():
    m = _load()
    val, conf = m.probe_delta(_fill_samples())
    assert val["copy_frac"] < 1.0 and val["changed_cells"] > 0

def test_color_source_fixed_delta_on_fill():
    m = _load()
    val, conf = m.probe_color_source(_fill_samples())
    assert val == "FIXED_DELTA"

def test_run_probes_returns_all_keys_with_confidence():
    m = _load()
    res = m.run_probes(_identity_samples())
    assert "shape_relation" in res and "confidence" in res["shape_relation"]
    assert res["shape_relation"]["value"] == "equal"


def test_d4_confidence_direction():
    m = _load()
    g = np.array([[1, 2, 0], [0, 3, 4]])
    rotated = np.rot90(g, 1)
    samples = []
    # 2/10 samples are a rot90 of the input; the rest are unrelated (not any d4 transform).
    other = np.array([[9, 9, 9], [9, 9, 9]])
    for k in range(10):
        if k < 2:
            samples.append((g.copy(), rotated.copy()))
        else:
            samples.append((g.copy(), other.copy()))
    val, conf = m.probe_d4_transform_of_input(samples)
    assert val is False
    assert conf == pytest.approx(0.8, abs=1e-6)


def test_color_source_identity_not_small_k():
    m = _load()
    val, conf = m.probe_color_source(_identity_samples())
    assert val == "FIXED_DELTA"
    assert val != "SMALL_K"


def test_n_objects_counts_components_not_colors():
    m = _load()
    g = np.zeros((5, 5), dtype=int)
    # three disjoint 1-cell blobs, all the SAME colour (5) -> 3 components, not 1 colour.
    g[0, 0] = 5
    g[2, 2] = 5
    g[4, 4] = 5
    samples = [(g.copy(), g.copy())]
    val, conf = m.probe_n_objects_est(samples)
    assert val == 3


def test_flood_ccl_true_on_contiguous_recolor():
    m = _load()
    # background (colour 0) forms ONE large contiguous blob that gets filled with a single new
    # colour -- the classic background-fill flood, which the PROBE_VERSION-2 "bg_flood" branch
    # flags True unconditionally (untouched by the v3 connectivity tightening on the dominant-
    # source-colour branch).
    samples = []
    for _ in range(10):
        g = np.ones((8, 8), dtype=int) * 3
        g[1:7, 1:7] = 0  # one large contiguous background blob
        o = g.copy()
        o[g == 0] = 5
        samples.append((g, o))
    val, conf = m.probe_flood_ccl(samples)
    assert val is True


def test_flood_ccl_false_on_scattered_recolor():
    m = _load()
    # a handful of mutually-isolated single cells (same source colour 3, non-background) each
    # flipped to colour 5: dominant-source-colour concentration is high (all changed cells came
    # from colour 3), but there is no multi-region connectivity structure -- every diff-component
    # is a lone pixel (mean component size == 1, well below the count task077 needs). This is the
    # "common single-object/scattered recolor" false-positive class the v3 tightening excludes.
    samples = []
    for _ in range(10):
        g = np.zeros((8, 8), dtype=int)
        for r, c in [(0, 0), (0, 4), (4, 0), (7, 7)]:
            g[r, c] = 3
        o = g.copy()
        o[g == 3] = 5
        samples.append((g, o))
    val, conf = m.probe_flood_ccl(samples)
    assert val is False


def test_separable_rect_boundary():
    m = _load()
    i = np.zeros((10, 10), dtype=int)
    o5 = i.copy()
    # 5 disjoint solid rectangles, well separated so none merge into one run.
    rects = [(0, 0, 1, 1), (0, 3, 1, 1), (0, 6, 1, 1), (3, 0, 1, 1), (3, 3, 1, 1)]
    for r, c, h, w in rects:
        o5[r:r + h, c:c + w] = 1
    val5, conf5 = m.probe_separable_rect_output([(i.copy(), o5.copy())])
    assert val5 == {"is": True, "n_rects": 5}

    o6 = o5.copy()
    o6[6, 6] = 1  # a 6th disjoint rectangle
    val6, conf6 = m.probe_separable_rect_output([(i.copy(), o6.copy())])
    assert val6["is"] is False
    assert val6["n_rects"] == 6
