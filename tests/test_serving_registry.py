"""Section 08: training, versioned artifacts and verified model loading."""

from __future__ import annotations

import dataclasses
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from northstar.serving import registry
from northstar.serving.registry import ArtifactError, list_versions, load_artifact, save_artifact
from northstar.serving.specs import ACQUISITION, CATEGORY_LEVELS, CHURN, SPECS
from northstar.serving.training import LeakageError, artifact_version, predict, train_and_register

ALL_SPECS = pytest.mark.parametrize("spec", list(SPECS.values()), ids=list(SPECS))


@pytest.fixture
def registry_copy(model_registry, tmp_path):
    root = tmp_path / "models"
    shutil.copytree(model_registry, root)
    return root


def _run(spec, tables, split: str) -> pd.DataFrame:
    runs = getattr(spec.split, split)
    return pd.concat([spec.build_run(tables, c) for c in runs], ignore_index=True)


@ALL_SPECS
def test_artifact_round_trip_reproduces_the_trained_model(spec, tables, model_registry):
    artifact = load_artifact(spec, model_registry)
    assert (model_registry / spec.name / "LATEST").read_text().strip() == artifact.version
    assert artifact.version.startswith("v20250701-")
    md = artifact.metadata
    assert md["features"] == list(spec.features)
    assert md["training"]["deployable_from"] == str(min(spec.split.holdout).date())
    assert md["leakage_audit"]["passed"]
    # The champion is refit on fit + validation runs only: refitting the section's estimator on
    # exactly those rows gives identical holdout predictions, so no holdout row influenced the
    # served model.
    train_rows = pd.concat([_run(spec, tables, "fit"), _run(spec, tables, "validation")],
                           ignore_index=True)
    holdout = _run(spec, tables, "holdout")
    served = predict(spec, artifact.model, holdout)
    fresh = spec.fit_model(md["algorithm"], train_rows)
    np.testing.assert_allclose(served, predict(spec, fresh, holdout), rtol=0, atol=1e-12)
    assert md["training"]["train_rows"] == len(train_rows)
    # Negative controls: the comparison is sensitive to the training rows, so the match above
    # pins them down. A refit that also saw the holdout, or that skipped the validation runs,
    # would predict differently.
    leaky = spec.fit_model(md["algorithm"], pd.concat([train_rows, holdout], ignore_index=True))
    fit_only = spec.fit_model(md["algorithm"], _run(spec, tables, "fit"))
    for other in (leaky, fit_only):
        assert np.abs(served - predict(spec, other, holdout)).max() > 1e-6


@ALL_SPECS
def test_champion_follows_the_section_selection_rule(spec, model_registry):
    selection = load_artifact(spec, model_registry).metadata["selection"]
    assert set(selection["candidates"]) == set(spec.candidates)
    values = {n: m[spec.selection_metric] for n, m in selection["candidates"].items()}
    best = (max if spec.maximize else min)(values, key=values.get)
    assert selection["champion"] == best == spec.select_champion(selection["candidates"])


@ALL_SPECS
def test_reference_profile_is_a_valid_distribution(spec, model_registry):
    profile = load_artifact(spec, model_registry).profile
    for block in (*profile["numeric"].values(), *profile["categorical"].values()):
        assert sum(block["shares"]) == pytest.approx(1.0)
    for f in spec.categorical:
        assert profile["categorical"][f]["levels"] == list(CATEGORY_LEVELS[f])
    q = profile["score"]["quantiles"]
    assert len(q) == 1001 and np.all(np.diff(q) >= 0) and q[0] >= 0 and q[-1] <= 1


@ALL_SPECS
def test_encoder_levels_are_inside_the_request_schema(spec, model_registry):
    """Every level the model knows is accepted by the API, and no accepted level is unseen."""
    model = load_artifact(spec, model_registry).model
    if not hasattr(model, "named_steps"):
        pytest.skip("baseline champion has no encoder")
    encoder = model.named_steps["pre"].named_transformers_["cat"]
    for feature, levels in zip(spec.categorical, encoder.categories_, strict=True):
        assert set(levels) == set(CATEGORY_LEVELS[feature]), feature


def test_version_is_content_addressed(tables, mutable_tables, model_registry):
    for spec in SPECS.values():
        champion = load_artifact(spec, model_registry).metadata["algorithm"]
        version, _ = artifact_version(spec, tables, champion)
        assert version == load_artifact(spec, model_registry).version
    # One changed row in an input table, or another algorithm, is a new version.
    base, _ = artifact_version(ACQUISITION, tables, "logistic_regression")
    mutable_tables["sessions"].loc[0, "pages_viewed"] += 1
    assert artifact_version(ACQUISITION, mutable_tables, "logistic_regression")[0] != base
    assert artifact_version(ACQUISITION, tables, "gradient_boosting")[0] != base
    # A table the model does not read does not change its version.
    unused = {**tables, "support_contacts": tables["support_contacts"].iloc[1:]}
    assert artifact_version(ACQUISITION, unused, "logistic_regression")[0] == base


def test_versions_can_be_pinned_and_latest_is_only_moved_on_promotion(registry_copy):
    current = load_artifact(CHURN, registry_copy)
    metadata = {**current.metadata, "version": "v20250701-candidate"}
    save_artifact(current.model, metadata, current.profile, registry_copy, set_latest=False)
    assert list_versions(CHURN, registry_copy) == sorted([current.version,
                                                          "v20250701-candidate"])
    assert load_artifact(CHURN, registry_copy).version == current.version
    pinned = load_artifact(CHURN, registry_copy, "v20250701-candidate")
    assert pinned.version == "v20250701-candidate"
    save_artifact(current.model, metadata, current.profile, registry_copy)
    assert load_artifact(CHURN, registry_copy).version == "v20250701-candidate"
    assert not list((registry_copy / CHURN.name).glob(".*tmp"))


def test_promote_command_lists_promotes_and_rolls_back(registry_copy, capsys):
    from northstar.cli import main

    current = load_artifact(ACQUISITION, registry_copy)
    save_artifact(current.model, {**current.metadata, "version": "v20250701-next"},
                  current.profile, registry_copy, set_latest=False)
    args = ["promote-model", "--model", "acquisition", "--model-dir", str(registry_copy)]
    assert main(args) == 0
    assert f"{current.version}  <- LATEST" in capsys.readouterr().out
    assert main([*args, "--version", "v20250701-next"]) == 0
    assert load_artifact(ACQUISITION, registry_copy).version == "v20250701-next"
    assert main([*args, "--version", "v20250701-missing"]) == 1  # unknown: LATEST unchanged
    assert load_artifact(ACQUISITION, registry_copy).version == "v20250701-next"
    assert main([*args, "--version", current.version]) == 0  # rollback
    assert load_artifact(ACQUISITION, registry_copy).version == current.version


def test_missing_registry_explains_how_to_train(tmp_path):
    with pytest.raises(ArtifactError, match="northstar train-models"):
        load_artifact(ACQUISITION, tmp_path / "empty")


def test_unknown_version_lists_what_is_available(model_registry):
    version = load_artifact(ACQUISITION, model_registry).version
    with pytest.raises(ArtifactError, match=version):
        load_artifact(ACQUISITION, model_registry, "v19990101-nope")


def _edit_metadata(root, spec, **changes):
    path = root / spec.name / (root / spec.name / "LATEST").read_text().strip()
    md = json.loads((path / registry.METADATA_FILE).read_text())
    for key, value in changes.items():
        section, _, field = key.partition("__")
        if field:
            md[section][field] = value
        else:
            md[section] = value
    (path / registry.METADATA_FILE).write_text(json.dumps(md))


@pytest.mark.parametrize(("changes", "message"), [
    ({"environment__scikit_learn": "0.0.1"}, "scikit-learn 0.0.1"),
    ({"features": ["acquisition_channel"]}, "feature list differs"),
    ({"schema_version": 99}, "metadata schema"),
    ({"artifact__sha256": "0" * 64}, "does not match the hash"),
])
def test_incompatible_or_tampered_artifacts_are_refused(registry_copy, changes, message):
    _edit_metadata(registry_copy, ACQUISITION, **changes)
    with pytest.raises(ArtifactError, match=message):
        load_artifact(ACQUISITION, registry_copy)


def test_corrupted_model_file_is_refused_before_unpickling(registry_copy, monkeypatch):
    path = load_artifact(CHURN, registry_copy).path
    (path / registry.MODEL_FILE).write_bytes(b"not a model")
    monkeypatch.setattr(registry.joblib, "load", lambda *_: pytest.fail("unpickled bad file"))
    with pytest.raises(ArtifactError, match="does not match the hash"):
        load_artifact(CHURN, registry_copy)


def test_unpickling_errors_become_artifact_errors(registry_copy, monkeypatch):
    def broken(*_):
        raise ModuleNotFoundError("No module named 'northstar.old_models'")

    monkeypatch.setattr(registry.joblib, "load", broken)
    with pytest.raises(ArtifactError, match="could not be unpickled"):
        load_artifact(CHURN, registry_copy)


def test_unreadable_or_incomplete_versions_fail_clearly(registry_copy):
    path = load_artifact(ACQUISITION, registry_copy).path
    (path / registry.METADATA_FILE).write_text("{ not json")
    with pytest.raises(ArtifactError, match="Unreadable metadata"):
        load_artifact(ACQUISITION, registry_copy)
    (path / registry.PROFILE_FILE).unlink()
    with pytest.raises(ArtifactError, match=r"reference_profile\.json"):
        load_artifact(ACQUISITION, registry_copy)


def test_failed_leakage_audit_registers_nothing(tables, tmp_path):
    def failing_audit(*_):
        return {"passed": False, "checks": {"features_only_from_pre_cutoff_rows": False},
                "details": {}}

    spec = dataclasses.replace(ACQUISITION, leakage_audit=failing_audit)
    with pytest.raises(LeakageError, match="features_only_from_pre_cutoff_rows"):
        train_and_register(spec, tables, tmp_path / "models")
    assert not (tmp_path / "models").exists()
