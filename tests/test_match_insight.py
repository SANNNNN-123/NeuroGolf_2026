import importlib.util, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def _load(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "match_insight", ROOT / "reports/scripts/match_insight.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    idx = {
        "1": {"economic": {"bloat": 100, "cost": 100, "status": "unexamined", "method": "c"},
              "structural": {"ops": ["Gather"], "flags": {"has_gridsample": False, "has_topk": False}},
              "semantic": {"d4_transform_of_input": {"value": True, "confidence": 1.0},
                           "separable_rect_output": {"value": {"is": False, "n_rects": 9}, "confidence": 1.0}}},
        "2": {"economic": {"bloat": 5000, "cost": 5000, "status": "unexamined", "method": "c"},
              "structural": {"ops": ["GridSample"], "flags": {"has_gridsample": True, "has_topk": False}},
              "semantic": {"d4_transform_of_input": {"value": True, "confidence": 1.0},
                           "separable_rect_output": {"value": {"is": True, "n_rects": 2}, "confidence": 1.0}}},
    }
    p = tmp_path / "task_index.json"; p.write_text(json.dumps(idx))
    monkeypatch.setattr(mod, "INDEX_PATH", p)
    return mod

def test_where_and_ranking(tmp_path, monkeypatch):
    m = _load(tmp_path, monkeypatch)
    res = m.match("d4_transform_of_input and bloat > 50")
    assert [t for t, _ in res] == ["2", "1"]  # ranked by bloat desc

def test_flag_query(tmp_path, monkeypatch):
    m = _load(tmp_path, monkeypatch)
    res = m.match("has_gridsample")
    assert [t for t, _ in res] == ["2"]
