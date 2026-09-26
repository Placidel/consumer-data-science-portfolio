"""Section 08 end to end: CLI flow, committed-result traceability, Docker and CI configuration."""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess

import pandas as pd
import pytest
import yaml

from northstar.cli import main
from northstar.paths import PRODUCTION_DIR, PROJECTS_DIR, REPO_ROOT
from northstar.profile import extract_generated_block
from northstar.serving import report
from northstar.serving.registry import load_artifact
from northstar.serving.schemas import AcquisitionScoreRequest, ChurnScoreRequest
from northstar.serving.specs import ACQUISITION, CHURN, SPECS
from northstar.serving.training import train_and_register
from northstar.synthetic import generate
from northstar.synthetic import params as p

OUTPUTS = PRODUCTION_DIR / "outputs"
METRICS = OUTPUTS / "metrics.json"
README = PRODUCTION_DIR / "README.md"
TABLES = ("drift_features", "drift_by_run", "performance_by_run")
FIGURES = ("feature_psi", "performance_by_run")


@pytest.fixture(scope="module")
def small_data_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("prod") / "raw"
    assert main(["generate-data", "--seed", "11", "--n-prospects", "4000", "--out", str(out),
                 "--skip-profile"]) == 0
    return out


def _readme(path):
    path.write_text(f"# Test\n\n{report.BEGIN_MARKER}\nstale\n{report.END_MARKER}\n\nend\n")
    return path


def test_train_score_monitor_commands_end_to_end(small_data_dir, tmp_path, capsys):
    models, out = tmp_path / "models", tmp_path / "outputs"
    readme = _readme(tmp_path / "README.md")
    assert main(["train-models", "--data-dir", str(small_data_dir), "--model-dir", str(models),
                 "--no-generate"]) == 0
    for spec in SPECS.values():
        md = load_artifact(spec, models).metadata
        assert md["data"] == {"seed": 11, "n_prospects": 4000}

    # Batch scoring from a scoring date, then from a feature file written upstream.
    scores = tmp_path / "scores.csv"
    assert main(["score-batch", "--model", "churn", "--cutoff", "2025-11-01", "--data-dir",
                 str(small_data_dir), "--model-dir", str(models), "--out", str(scores),
                 "--no-generate"]) == 0
    scored = pd.read_csv(scores)
    assert {"customer_id", "churn_probability", "reference_percentile", "rank",
            "model_version"} <= set(scored.columns)
    assert scored["rank"].min() == 1 and scored["customer_id"].is_unique

    from northstar.io import load_tables

    tables = load_tables(small_data_dir)
    features = ACQUISITION.build_features(tables, "2025-08-01")
    features.loc[features.index[0], "sessions_7d"] = 999  # one broken upstream record
    feature_file = tmp_path / "leads.csv"
    features.to_csv(feature_file, index=False)
    lead_scores = tmp_path / "lead_scores.csv"
    base = ["score-batch", "--model", "acquisition_lead_score", "--features", str(feature_file),
            "--model-dir", str(models), "--out", str(lead_scores)]
    assert main(base) == 1  # strict by default
    assert "sessions_7d" in capsys.readouterr().err and not lead_scores.exists()
    assert main([*base, "--max-invalid-share", "0.01"]) == 0
    assert len(pd.read_csv(lead_scores)) == len(features) - 1
    rejected = pd.read_csv(tmp_path / "lead_scores.rejected.csv")
    assert rejected["row"].tolist() == [0] and "sessions_7d" in rejected["errors"][0]

    # A file with an outcome column, or without a scoring date, is refused before scoring.
    labelled = tmp_path / "labelled.csv"
    ACQUISITION.build_run(tables, pd.Timestamp("2025-08-01")).to_csv(labelled, index=False)
    undated = tmp_path / "undated.csv"
    ACQUISITION.build_features(tables, "2025-08-01").drop(columns="run_cutoff").to_csv(
        undated, index=False)
    capsys.readouterr()
    for path, extra, message in ((labelled, [], "'converted' is the outcome"),
                                 (undated, [], "no scoring date"),
                                 (undated, ["--cutoff", "2025-06-01"], "reuse training labels")):
        target = tmp_path / f"{path.stem}_scores.csv"
        assert main(["score-batch", "--model", "acquisition", "--features", str(path),
                     "--model-dir", str(models), "--out", str(target), *extra]) == 1
        assert message in capsys.readouterr().err and not target.exists()
    dated = tmp_path / "undated_scores.csv"
    assert main(["score-batch", "--model", "acquisition", "--features", str(undated),
                 "--cutoff", "2025-08-01", "--model-dir", str(models), "--out", str(dated)]) == 0
    assert (pd.read_csv(dated)["run_cutoff"] == "2025-08-01").all()
    assert main(["score-batch", "--model", "acquisition", "--model-dir", str(models)]) == 2

    # Monitoring writes machine- and human-readable outputs and the README block.
    assert main(["monitor", "--data-dir", str(small_data_dir), "--model-dir", str(models),
                 "--out-dir", str(out), "--readme", str(readme), "--projects-dir",
                 str(PROJECTS_DIR), "--no-generate"]) == 0
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["data"] == {"seed": 11, "n_prospects": 4000}
    assert set(metrics["monitoring"]) == set(SPECS)
    # Committed section outputs come from other data: the cross-check says so, not "no".
    assert all(r["consistent_with_section"] is None for r in metrics["registry"])
    for name in TABLES:
        assert len(pd.read_csv(out / f"{name}.csv")) > 0
    for name in FIGURES:
        assert (out / "figures" / f"{name}.png").stat().st_size > 0
    assert (out / "monitoring_report.md").read_text() == report.render_report(metrics)
    block = extract_generated_block(readme, report.BEGIN_MARKER, report.END_MARKER)
    assert block == report.render_markdown(metrics)
    assert "stale" not in readme.read_text() and readme.read_text().rstrip().endswith("end")
    AcquisitionScoreRequest.model_validate_json(
        (out / "example_requests" / "acquisition.json").read_text())
    ChurnScoreRequest.model_validate_json((out / "example_requests" / "churn.json").read_text())


def test_commands_fail_clearly_without_models(small_data_dir, tmp_path, capsys):
    empty = str(tmp_path / "none")
    assert main(["monitor", "--data-dir", str(small_data_dir), "--model-dir", empty,
                 "--out-dir", str(tmp_path / "out"), "--no-generate"]) == 1
    assert main(["score-batch", "--model", "churn", "--cutoff", "2025-08-01", "--data-dir",
                 str(small_data_dir), "--model-dir", empty, "--no-generate"]) == 1
    assert capsys.readouterr().err.count("northstar train-models") == 2
    assert not (tmp_path / "out").exists()


def test_training_on_unchanged_data_reproduces_the_version(small_data_dir, tmp_path):
    from northstar.io import load_tables

    tables = load_tables(small_data_dir, names=CHURN.tables_used)
    _, first = train_and_register(CHURN, tables, tmp_path / "a")
    _, second = train_and_register(CHURN, tables, tmp_path / "b")
    assert first["version"] == second["version"]
    assert first["holdout_metrics"] == second["holdout_metrics"]


# ---------------------------------------------------------------- committed results
@pytest.fixture(scope="module")
def committed():
    return json.loads(METRICS.read_text())


def test_readme_and_report_match_committed_metrics(committed):
    block = extract_generated_block(README, report.BEGIN_MARKER, report.END_MARKER)
    assert block == report.render_markdown(committed)
    assert (OUTPUTS / "monitoring_report.md").read_text() == report.render_report(committed)


def test_committed_run_uses_default_data_and_matches_the_sections(committed):
    assert committed["data"] == {"seed": p.DEFAULT_SEED, "n_prospects": p.DEFAULT_N_PROSPECTS}
    for entry in committed["registry"]:
        assert entry["leakage_audit_passed"]
        # The served champion is the model sections 01 and 02 selected and evaluated.
        assert entry["consistent_with_section"] is True
        section = json.loads((PROJECTS_DIR / entry["section"] / "outputs" / "metrics.json")
                             .read_text())
        assert entry["algorithm"] == section["champion"]
    for spec in SPECS.values():
        body = json.loads((OUTPUTS / "example_requests" / f"{spec.route}.json").read_text())
        assert len(body["records"]) == report.N_EXAMPLE_RECORDS


def test_prose_claims_in_the_readme_hold_for_the_committed_run(committed):
    """The narrative sections of the README make these claims; keep them true."""
    acq, churn = (committed["monitoring"][s.name] for s in (ACQUISITION, CHURN))
    # Lead score: no drift finding, every monthly run matured, performance holds.
    assert acq["status"] == "ok" and not acq["performance"]["pending_runs"]
    assert acq["feature_drift"][0]["psi"] < committed["config"]["psi_moderate"]
    assert acq["score_drift"]["psi"] < committed["config"]["psi_moderate"]
    # Churn: the ageing customer base shows as input drift led by tenure ...
    drift = {r["feature"]: r for r in churn["feature_drift"]}
    assert churn["feature_drift"][0]["feature"] == "tenure_days"
    assert drift["tenure_days"]["status"] == "investigate"
    assert drift["tenure_days"]["current_mean"] > drift["tenure_days"]["reference_mean"]
    assert drift["tenure_days"]["outside_reference_range"] > 0.1  # "more than one in ten"
    assert {"orders_total", "category_count"} <= {
        f for f, r in drift.items() if r["status"] != "ok"}
    assert churn["status"] == "investigate"
    # ... while scores barely move and matured-run discrimination and calibration hold.
    assert churn["score_drift"]["psi"] < committed["config"]["psi_moderate"]
    assert not [f for f in churn["findings"] if f["check"] in ("discrimination", "calibration")]
    assert churn["performance"]["pending_runs"] == ["2025-11-01", "2025-12-01"]
    for rep in (acq, churn):
        assert rep["performance"]["pooled"]["roc_auc"] >= (
            rep["performance"]["reference"]["roc_auc"] - committed["config"]["auc_drop"])


def _assert_close(fresh, committed, path="metrics"):
    if isinstance(committed, dict):
        assert set(fresh) == set(committed), path
        for k in committed:
            _assert_close(fresh[k], committed[k], f"{path}.{k}")
    elif isinstance(committed, list):
        assert len(fresh) == len(committed), path
        for i, (x, y) in enumerate(zip(fresh, committed, strict=True)):
            _assert_close(x, y, f"{path}[{i}]")
    elif isinstance(committed, float) and not isinstance(committed, bool):
        assert math.isclose(fresh, committed, rel_tol=1e-3, abs_tol=2e-3), path
    else:
        assert fresh == committed, path


@pytest.mark.slow
def test_committed_metrics_are_reproduced_from_default_generation(committed, tmp_path):
    tables = generate(seed=p.DEFAULT_SEED, n_prospects=p.DEFAULT_N_PROSPECTS)
    info = {"seed": p.DEFAULT_SEED, "n_prospects": p.DEFAULT_N_PROSPECTS}
    for spec in SPECS.values():
        train_and_register(spec, tables, tmp_path, data_info=info)
    fresh, _, _ = report.run_report(tables, tmp_path, PROJECTS_DIR)
    # Versions embed the library versions by design; everything else must reproduce.
    for entry in (*fresh["registry"], *committed["registry"]):
        entry.pop("version"), entry.pop("scikit_learn")
    for rep in (*fresh["monitoring"].values(), *committed["monitoring"].values()):
        rep["model"].pop("version")
    expected = {k: v for k, v in committed.items() if k != "data"}
    _assert_close(json.loads(json.dumps(fresh)), expected)


# ---------------------------------------------------------------- deployment configuration
SECRET_PATTERN = re.compile(r"(password|secret|token|api[_-]?key|credential)", re.IGNORECASE)


def test_dockerfile_is_pinned_starts_the_api_and_embeds_no_secrets():
    text = (REPO_ROOT / "Dockerfile").read_text()
    base = re.search(r"^FROM (\S+)", text, re.MULTILINE).group(1)
    assert "@sha256:" in base and ":3.12" in base  # tag and digest
    assert "--constraint constraints.txt" in text
    assert "northstar train-models" in text  # models baked at build time, not at start-up
    assert re.search(r'^CMD \["northstar", "serve"', text, re.MULTILINE)
    assert re.search(r"^USER (?!root)\w+", text, re.MULTILINE)
    for line in text.splitlines():
        if not line.lstrip().startswith("#"):
            assert not SECRET_PATTERN.search(line), line
    ignored = (REPO_ROOT / ".dockerignore").read_text().split()
    assert {".venv", ".env", "models", "data/raw", ".git"} <= set(ignored)


def test_compose_runs_api_and_dashboard_with_health_checks():
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text())
    services = compose["services"]
    assert set(services) == {"api", "dashboard"}
    assert services["api"]["command"][:2] == ["northstar", "serve"]
    assert "8000:8000" in services["api"]["ports"] and "8501:8501" in services["dashboard"]["ports"]
    for svc in services.values():
        assert "healthcheck" in svc and svc["build"] == "."
        assert not SECRET_PATTERN.search(json.dumps(svc.get("environment", {})))
        assert "env_file" not in svc and "secrets" not in svc


def test_ci_lints_and_tests_from_a_clean_environment():
    workflow = yaml.safe_load((REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text())
    job = workflow["jobs"]["lint-and-test"]
    script = "\n".join(step.get("run", "") for step in job["steps"])
    assert "python -m venv .venv" in script
    assert 'pip install --constraint constraints.txt -e ".[dev]"' in script
    assert "ruff check ." in script and "pytest" in script
    docker = "\n".join(step.get("run", "") for step in workflow["jobs"]["docker"]["steps"])
    assert "docker compose up" in docker and "/health" in docker


def test_constraints_pin_every_declared_dependency():
    pins = {}
    for line in (REPO_ROOT / "constraints.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            name, _, version = line.partition("==")
            pins[name.lower().replace("_", "-")] = version
    import tomllib

    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]
    declared = [*project["dependencies"], *project["optional-dependencies"]["dev"]]
    for requirement in declared:
        name = re.split(r"[<>=\[ ]", requirement, maxsplit=1)[0].lower()
        assert pins.get(name), f"{name} is not pinned in constraints.txt"


def test_generated_models_and_scores_stay_out_of_git_via_nested_ignore_files():
    # The root .gitignore is owned by the build harness, so the regenerated registry and batch
    # scores are ignored by self-contained .gitignore files (the data/ convention) instead.
    from northstar.cli import DEFAULT_SCORES_DIR

    for directory in (REPO_ROOT / "models", DEFAULT_SCORES_DIR):
        rules = (directory / ".gitignore").read_text().splitlines()
        assert "*" in rules and "!.gitignore" in rules
    assert "/models/" not in (REPO_ROOT / ".gitignore").read_text()

    if shutil.which("git") is None or not (REPO_ROOT / ".git").exists():
        return  # the rule files above are the whole contract outside a checkout

    def ignored(path: str) -> bool:
        return subprocess.run(["git", "-C", str(REPO_ROOT), "check-ignore", "-q", path],
                              check=False).returncode == 0

    assert ignored("models/churn_risk/v20250701-0000000000/model.joblib")
    assert ignored("models/churn_risk/LATEST")
    assert ignored(f"{DEFAULT_SCORES_DIR.name}/churn_risk_2025-07-01.csv")
    assert not ignored("models/.gitignore") and not ignored("scores/.gitignore")
