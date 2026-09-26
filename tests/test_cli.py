from __future__ import annotations

import json
import subprocess
import sys

import pandas as pd
import pytest

from northstar.cli import main
from northstar.io import load_tables, read_manifest
from northstar.profile import BEGIN_MARKER, END_MARKER, extract_generated_block
from northstar.schema import TABLE_NAMES
from northstar.synthetic import generate
from northstar.synthetic import params as p


@pytest.fixture
def readme(tmp_path):
    path = tmp_path / "README.md"
    path.write_text(f"# Test\n\nintro\n\n{BEGIN_MARKER}\nstale\n{END_MARKER}\n\noutro\n")
    return path


def test_generate_command_writes_valid_data_profile_and_readme(tmp_path, readme):
    out, prof = tmp_path / "raw", tmp_path / "outputs"
    code = main(["generate-data", "--seed", "3", "--n-prospects", "800", "--out", str(out),
                 "--profile-dir", str(prof), "--readme", str(readme)])
    assert code == 0
    manifest = read_manifest(out)
    assert manifest["seed"] == 3 and manifest["n_prospects"] == 800
    assert set(manifest["tables"]) == set(TABLE_NAMES)
    for name in TABLE_NAMES:
        assert (out / f"{name}.parquet").exists()

    loaded = load_tables(out)
    expected = generate(seed=3, n_prospects=800)
    for name in TABLE_NAMES:
        pd.testing.assert_frame_equal(loaded[name], expected[name])
        assert manifest["tables"][name]["rows"] == len(expected[name])

    profile = json.loads((prof / "data_profile.json").read_text())
    assert profile["row_counts"]["orders"] == len(expected["orders"])
    assert profile["headline"]["net_revenue"] == pytest.approx(
        expected["orders"]["net_amount"].sum(), abs=0.01)
    assert (prof / "monthly_kpis.csv").exists()
    assert (prof / "figures" / "monthly_net_revenue.png").stat().st_size > 0

    text = readme.read_text()
    assert "stale" not in text and text.startswith("# Test") and text.rstrip().endswith("outro")
    assert f"| Orders | {len(expected['orders']):,} |" in extract_generated_block(readme)

    assert main(["validate-data", "--data-dir", str(out)]) == 0


def test_validate_command_fails_on_corrupted_files(tmp_path):
    out = tmp_path / "raw"
    assert main(["generate-data", "--n-prospects", "400", "--out", str(out),
                 "--skip-profile"]) == 0
    orders = pd.read_parquet(out / "orders.parquet")
    orders.loc[0, "net_amount"] = -1.0
    orders.to_parquet(out / "orders.parquet", index=False)
    assert main(["validate-data", "--data-dir", str(out)]) == 1


def test_load_tables_explains_how_to_generate_missing_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="northstar generate-data"):
        load_tables(tmp_path / "nothing-here")


def test_load_tables_can_generate_on_demand(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "DEFAULT_N_PROSPECTS", 300)
    tables = load_tables(tmp_path / "raw", names=["prospects", "orders"],
                         generate_if_missing=True)
    assert set(tables) == {"prospects", "orders"}
    assert len(tables["prospects"]) == 300
    assert read_manifest(tmp_path / "raw")["n_prospects"] == 300


def test_module_entry_point_runs(tmp_path):
    out = tmp_path / "dict.md"
    proc = subprocess.run([sys.executable, "-m", "northstar", "data-dictionary", "--out",
                           str(out)], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert out.read_text().startswith("# Northstar Consumer data dictionary")
