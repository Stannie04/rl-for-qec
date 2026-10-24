"""Contract tests for ``scripts/bp_debug/step4_config_sweep.py`` (card BD5, issue #13).

Run from the repository root::

    python -m pytest tests/bp_debug/test_step4.py
"""

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("galois")  # src.environment.code imports galois
pytest.importorskip("ldpc")
pytest.importorskip("torch_geometric")  # src.environment.code imports torch_geometric

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.bp_debug import step4_config_sweep as step4  # noqa: E402

CODE = "d5_surface"
_PAYLOAD = {}


def payload():
    """One shared quick sweep; decoding is not free, so reuse it across tests."""
    if "payload" not in _PAYLOAD:
        _PAYLOAD["payload"] = step4.run([CODE], 40, 0)
    return _PAYLOAD["payload"]


def test_sweep_settings_covers_contract_grid():
    settings = step4.sweep_settings()
    by_factor = {}
    for factor, setting, overrides in settings:
        by_factor.setdefault(factor, []).append((setting, overrides))

    assert {o["max_iter"] for _, o in by_factor["max_iter"]} == {10, 30, 100, 500}
    assert {o["bp_method"] for _, o in by_factor["bp_method"]} == {"product_sum", "minimum_sum"}
    assert {o["ms_scaling_factor"] for _, o in by_factor["ms_scaling_factor"]} == {0.5, 0.625, 0.75, 0.9, 1.0}
    # ms_scaling_factor only applies to minimum_sum.
    assert all(o["bp_method"] == "minimum_sum" for _, o in by_factor["ms_scaling_factor"])
    assert {o["schedule"] for _, o in by_factor["schedule"]} == {"parallel", "serial"}

    # The default appears as the max_iter=30 row.
    assert step4.BASE == {"max_iter": 30, "bp_method": "product_sum", "ms_scaling_factor": 1.0, "schedule": "parallel"}
    assert next(o for s, o in by_factor["max_iter"] if s == "max_iter=30") == {"max_iter": 30}


def test_contract_values_valid_and_bogus_rejected():
    for _, _, overrides in step4.sweep_settings():
        assert step4.valid_settings({**step4.BASE, **overrides})

    assert not step4.valid_settings({**step4.BASE, "bp_method": "bogus"})
    assert not step4.valid_settings({**step4.BASE, "schedule": "bogus"})


def test_run_payload_shape_and_ranges():
    result = payload()
    meta = result["meta"]
    assert meta["n_shots"] == 40 and meta["seed"] == 0
    assert meta["prior"] == pytest.approx(2 * 0.003 / 3)
    assert meta["codes"] == [CODE]

    rows = result["results"][CODE]
    assert len(rows) == len(step4.sweep_settings())
    for row in rows:
        assert not row["skipped"]
        assert 0.0 <= row["failure_fraction"] <= 1.0
        assert 0.0 <= row["non_converged_fraction"] <= 1.0
        assert row["setting"].startswith(row["factor"])

    # The default row is present and matches BASE.
    default = next(row for row in rows if row["setting"] == "max_iter=30")
    assert default["bp_method"] == "product_sum" and default["schedule"] == "parallel"


def test_observations_are_short_facts():
    bullets = step4.observations(payload())
    assert 1 <= len(bullets) <= 5
    assert all(bullet.startswith("- ") for bullet in bullets)


def test_write_outputs_writes_json_and_markdown(tmp_path):
    json_path, md_path = step4.write_outputs(payload(), tmp_path)

    assert json_path.name == f"{step4.MODULE_NAME}.json"
    assert md_path.name == f"{step4.MODULE_NAME}.md"
    assert json.loads(json_path.read_text()) == payload()

    markdown = md_path.read_text()
    assert "## Validity of ldpc 2.4.1 options" in markdown
    assert "## Observation" in markdown
    assert "iid depolarizing shots at p=0.003" in markdown


def test_main_quick_writes_to_out_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(step4, "OUT_DIR", tmp_path)

    result = step4.main(["--quick", "--codes", CODE])

    assert result["meta"]["n_shots"] == 200
    assert (tmp_path / f"{step4.MODULE_NAME}.json").exists()
    assert (tmp_path / f"{step4.MODULE_NAME}.md").exists()
