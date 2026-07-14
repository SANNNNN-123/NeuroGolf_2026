import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "coverage_lib", ROOT / "reports/scripts/coverage_lib.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "PATH", tmp_path / "cov.json")
    return mod


def test_record_and_resolved(tmp_path, monkeypatch):
    m = _load(tmp_path, monkeypatch)
    m.record("gridsample_warp", "209", "applied", delta=1.05, session="S15")
    m.record("gridsample_warp", "133", "pending")
    assert m.resolved_tasks("gridsample_warp") == {"209"}


def test_seed_writes_known_applications(tmp_path, monkeypatch):
    m = _load(tmp_path, monkeypatch)
    m.seed()
    assert "187" in m.resolved_tasks("walk_einsum")
    assert "234" in m.resolved_tasks("signed_rect")
