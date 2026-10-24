"""Smoke test for ``scripts/bp_debug/step1_rerun.py`` (card BD2, issue #13).

Run from the repository root::

    python -m pytest tests/bp_debug/test_step1.py
"""

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("galois")  # src.environment.code imports galois
pytest.importorskip("ldpc")
pytest.importorskip("pymatching")  # MWPMAgent

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.bp_debug import step1_rerun  # noqa: E402


@pytest.mark.slow  # ~1 min: the agent-stack import plus the repo's per-shot evaluation loop
def test_quick_run_d5_surface_writes_contract_outputs():
    payload = step1_rerun.main(["--quick", "--codes", "d5_surface"])

    assert payload["quick"] is True
    assert payload["codes"] == ["d5_surface"]
    assert payload["n_per_weight"] == step1_rerun.QUICK_N_PER_WEIGHT

    code = payload["results"]["d5_surface"]
    n = code["n_per_weight"]
    assert code["weights"] == [1, 2, 3, 4, 5]
    assert code["n_data"] == 25
    # repo prior convention for depolarizing noise: 2 * (0.001 * 144 / n_data) / 3
    assert code["p_prior"] == pytest.approx(2 * (0.001 * 144 / 25) / 3)

    assert set(code["repo"]) == set(step1_rerun.REPO_DECODERS)
    assert set(code["common"]) == set(step1_rerun.COMMON_DECODERS)
    # pymatching accepts the surface-code H, so the repo MWPM path runs for d5_surface
    assert code["repo_errors"] == {}
    for entry in list(code["repo"].values()) + list(code["common"].values()):
        assert entry["n"] == [n] * 5
        assert all(0.0 <= fraction <= 1.0 for fraction in entry["failure_fraction"])
        assert all(f <= size for f, size in zip(entry["failures"], entry["n"]))

    # repo BP and common.decode_bp are the same decoder on the same shots: no disagreement
    assert code["disagreements"]["bp"]["total"] == 0
    assert code["disagreements"]["bp"]["per_weight"] == [0] * 5

    assert json.loads(step1_rerun.JSON_PATH.read_text()) == payload
    markdown = step1_rerun.MD_PATH.read_text()
    assert markdown.startswith("# Step 1: rerun baseline per weight")
    assert "| d5_surface | 1 | bp |" in markdown
    assert "## Observation" in markdown
