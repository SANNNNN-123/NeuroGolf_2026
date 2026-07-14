import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def _load():
    spec = importlib.util.spec_from_file_location(
        "build_task_index", ROOT / "reports/scripts/build_task_index.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

def test_economic_row_computes_cost_and_bloat():
    m = _load()
    row = m.economic_row({"memory": 21526, "params": 100, "points": 15.0, "method": "urad:x"})
    assert row["cost"] == 21626
    assert row["bloat"] == 21626  # class_floor_est None -> floor 0
    assert row["status"] == "unexamined"

def test_structural_row_flags_on_task150():
    # task150 = single-axis Gather col-mirror (memory ~136); must parse without error
    import onnx
    p = ROOT / "networks/task150.onnx"
    if not p.exists():
        import pytest; pytest.skip("networks/task150.onnx absent")
    m = _load()
    row = m.structural_row(onnx.load(str(p)))
    assert "Gather" in row["ops"]
    assert row["node_count"] >= 1
    assert isinstance(row["flags"]["has_topk"], bool)

def test_build_structural_economic_shape():
    m = _load()
    idx = m.build_structural_economic()
    assert isinstance(idx, dict) and len(idx) > 0
    missing_net_num = None
    for num, row in idx.items():
        assert set(row.keys()) >= {"arc_id", "structural", "economic"}
        econ = row["economic"]
        assert "cost" in econ and "bloat" in econ
        p = ROOT / f"networks/task{int(num):03d}.onnx"
        if not p.exists() and missing_net_num is None:
            missing_net_num = num
    if missing_net_num is not None:
        assert idx[missing_net_num]["structural"] == {}
    else:
        # all 400 networks present locally -> can't exercise the missing-net branch here
        pass

def test_sample_pairs_and_semantic_row_on_a_real_task():
    m = _load()
    import json
    mapping = json.loads((ROOT / "reports/arc_mapping.json").read_text())
    arc = mapping["1"]["arc_id"]
    pairs = m.sample_pairs(arc, 5)
    if not pairs:
        import pytest; pytest.skip("generator unavailable")
    assert len(pairs) == 5
    sem = m.semantic_row(arc, 5)
    assert "shape_relation" in sem and sem["probe_version"] == m.__dict__["_PV"]()
