"""Section 08: the scoring API - representative scoring, validation errors, failure behavior."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from northstar.cli import main
from northstar.serving.api import create_app
from northstar.serving.registry import load_artifact
from northstar.serving.schemas import (
    MAX_RECORDS,
    RECORD_MODELS,
    AcquisitionScoreRequest,
    ChurnScoreRequest,
)
from northstar.serving.specs import ACQUISITION, CATEGORY_LEVELS, CHURN, SPECS
from northstar.serving.training import predict

SCORING_DATE = pd.Timestamp("2025-09-01")
URL = {ACQUISITION.name: "/v1/acquisition/score", CHURN.name: "/v1/churn/score"}


@pytest.fixture(scope="module")
def client(model_registry):
    with TestClient(create_app(model_registry, versions={})) as c:
        yield c


@pytest.fixture(scope="module")
def run_features(tables):
    """Feature frames of one scoring run, exactly as the section pipelines build them."""
    return {name: spec.build_features(tables, SCORING_DATE) for name, spec in SPECS.items()}


def _records(spec, frame: pd.DataFrame, n: int | None = None) -> list[dict]:
    """JSON-ready records (what an upstream caller would send)."""
    rows = frame[[spec.entity, *spec.features]].head(n)
    return json.loads(rows.to_json(orient="records"))


def _lead(run_features) -> dict:
    return _records(ACQUISITION, run_features[ACQUISITION.name], 1)[0]


def test_health_and_model_cards(client, model_registry):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    for spec in SPECS.values():
        assert body["models"][spec.name] == {
            "loaded": True, "version": load_artifact(spec, model_registry).version}
    cards = client.get("/v1/models").json()
    assert set(cards) == set(SPECS)
    churn = cards[CHURN.name]
    assert churn["horizon_days"] == 90 and churn["features"] == list(CHURN.features)
    assert churn["holdout_metrics"]["roc_auc"] > 0.7
    assert "win-back" in churn["not_for"]


@pytest.mark.parametrize("spec", list(SPECS.values()), ids=list(SPECS))
def test_representative_records_score_like_the_offline_model(spec, client, run_features,
                                                             model_registry):
    frame = run_features[spec.name].sample(200, random_state=1)
    resp = client.post(URL[spec.name], json={"records": _records(spec, frame)})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    artifact = load_artifact(spec, model_registry)
    assert body["model"]["name"] == spec.name and body["model"]["version"] == artifact.version
    assert body["model"]["deployable_from"] == "2025-07-01"
    preds = pd.DataFrame(body["predictions"])
    # Same order and ids as the request; same probabilities as the offline pipeline.
    assert preds[spec.entity].tolist() == frame[spec.entity].tolist()
    np.testing.assert_allclose(preds[spec.score_field], predict(spec, artifact.model, frame),
                               rtol=0, atol=1e-9)
    assert preds[spec.score_field].between(0, 1).all()
    assert preds["reference_percentile"].between(0, 100).all()
    # Percentiles are monotone in the score.
    ranked = preds.sort_values(spec.score_field)
    assert ranked["reference_percentile"].is_monotonic_increasing


def test_committed_example_requests_are_valid_and_score(client):
    from northstar.paths import PRODUCTION_DIR

    for spec in SPECS.values():
        body = json.loads((PRODUCTION_DIR / "outputs" / "example_requests"
                           / f"{spec.route}.json").read_text())
        resp = client.post(URL[spec.name], json=body)
        assert resp.status_code == 200, resp.text
        assert len(resp.json()["predictions"]) == len(body["records"]) > 0


def test_every_record_the_feature_pipelines_build_is_accepted(tables):
    """Validation must not be stricter than the features it guards (all runs, both models)."""
    for spec in SPECS.values():
        model = RECORD_MODELS[spec.name]
        runs = (*spec.split.fit, *spec.split.validation, *spec.split.holdout,
                pd.Timestamp("2025-12-01"))
        for cutoff in runs:
            frame = spec.build_features(tables, cutoff)
            for record in frame[[spec.entity, *spec.features]].to_dict(orient="records"):
                model.model_validate(record)


def test_request_schema_matches_the_model_features():
    for spec in SPECS.values():
        fields = RECORD_MODELS[spec.name].model_fields
        assert set(fields) == {spec.entity, *spec.features}
        for f in spec.categorical:
            assert set(fields[f].annotation.__args__) == set(CATEGORY_LEVELS[f])


# ---------------------------------------------------------------- invalid requests
def _set(**changes):
    def apply(record: dict) -> dict:
        return {**record, **changes}
    return apply


def _drop(field):
    def apply(record: dict) -> dict:
        return {k: v for k, v in record.items() if k != field}
    return apply


INVALID_LEADS = {
    "missing field": (_drop("sessions_total"), "sessions_total"),
    "outcome column smuggled in": (_set(converted=1), "converted"),
    "unknown channel": (_set(acquisition_channel="tiktok"), "acquisition_channel"),
    "negative count": (_set(retarget_clicks=-1), "retarget_clicks"),
    "fractional count": (_set(emails_received=2.5), "emails_received"),
    "text in a number": (_set(pages_viewed="many"), "pages_viewed"),
    "share above one": (_set(mobile_session_share=1.2), "mobile_session_share"),
    "lead older than the pipeline window": (_set(lead_age_days=120.0), "lead_age_days"),
    "purchase stage means already converted": (_set(max_stage_reached=5), "max_stage_reached"),
    "7-day sessions above 30-day": (_set(sessions_7d=50, sessions_30d=1, sessions_total=60),
                                    "sessions_7d <= sessions_30d"),
    "open rate inconsistent with opens": (_set(emails_received=4, emails_opened=2,
                                               email_open_rate=0.9), "email_open_rate"),
    "bad id": (_set(prospect_id="drop table;"), "prospect_id"),
}


@pytest.mark.parametrize("case", list(INVALID_LEADS), ids=list(INVALID_LEADS))
def test_invalid_lead_records_get_422_with_the_offending_field(case, client, run_features):
    mutate, expected = INVALID_LEADS[case]
    resp = client.post(URL[ACQUISITION.name], json={"records": [mutate(_lead(run_features))]})
    assert resp.status_code == 422, resp.text
    detail = resp.json()["detail"]
    assert any(expected in json.dumps(err) for err in detail), detail


def test_invalid_churn_records_get_422(client, run_features):
    base = _records(CHURN, run_features[CHURN.name], 1)[0]
    cases = [
        {**base, "days_since_last_order": 200.0},  # lapsed: win-back, not this model
        {**base, "orders_total": 0},
        {**base, "store_order_share": 0.7, "app_order_share": 0.6},
        {**base, "low_csat_contacts_180d": base["support_contacts_180d"] + 1},
        {**base, "region": "Northeast"},  # levels are case-sensitive
    ]
    for record in cases:
        resp = client.post(URL[CHURN.name], json={"records": [record]})
        assert resp.status_code == 422, record


def test_request_level_validation(client, run_features):
    lead = _lead(run_features)
    assert client.post(URL[ACQUISITION.name], json={"records": []}).status_code == 422
    assert client.post(URL[ACQUISITION.name], json={}).status_code == 422
    assert client.post(URL[ACQUISITION.name], json=[lead]).status_code == 422
    too_many = [{**lead, "prospect_id": f"P{i}"} for i in range(MAX_RECORDS + 1)]
    assert client.post(URL[ACQUISITION.name], json={"records": too_many}).status_code == 422
    dupes = client.post(URL[ACQUISITION.name], json={"records": [lead, lead]})
    assert dupes.status_code == 422 and "duplicate prospect_id" in dupes.text
    nan = json.dumps({"records": [lead]}).replace(
        f'"lead_age_days": {json.dumps(lead["lead_age_days"])}', '"lead_age_days": NaN')
    assert "NaN" in nan
    resp = client.post(URL[ACQUISITION.name], content=nan,
                       headers={"Content-Type": "application/json"})
    assert resp.status_code == 422
    # A lead record sent to the churn endpoint is rejected, not scored with the wrong model.
    assert client.post(URL[CHURN.name], json={"records": [lead]}).status_code == 422


def test_request_models_are_usable_directly(run_features):
    lead = _lead(run_features)
    assert len(AcquisitionScoreRequest(records=[lead]).records) == 1
    with pytest.raises(ValueError, match="records"):
        ChurnScoreRequest(records=[lead])


# ---------------------------------------------------------------- no training at request time
def test_models_are_loaded_once_and_never_refit(model_registry, run_features, monkeypatch):
    calls = []

    def counting_loader(spec, root, version):
        calls.append(spec.name)
        return load_artifact(spec, root, version)

    def no_fit(*_, **__):
        raise AssertionError("a model was fit while serving a request")

    for cls in (Pipeline, LogisticRegression, HistGradientBoostingClassifier):
        monkeypatch.setattr(cls, "fit", no_fit)

    with TestClient(create_app(model_registry, versions={}, loader=counting_loader)) as c:
        models_before = {n: id(s.artifact.model) for n, s in c.app.state.models.scorers.items()}
        for _ in range(5):
            for spec in SPECS.values():
                frame = run_features[spec.name].head(20)
                resp = c.post(URL[spec.name], json={"records": _records(spec, frame)})
                assert resp.status_code == 200, resp.text
        models_after = {n: id(s.artifact.model) for n, s in c.app.state.models.scorers.items()}
    assert sorted(calls) == sorted(SPECS)  # one load per model for the app's lifetime
    assert models_before == models_after


# ---------------------------------------------------------------- failure behavior
def test_missing_artifacts_make_health_and_scoring_unavailable(tmp_path, run_features):
    with TestClient(create_app(tmp_path / "no-models", versions={})) as c:
        health = c.get("/health")
        assert health.status_code == 503
        body = health.json()
        assert body["status"] == "unavailable"
        assert "northstar train-models" in body["models"][ACQUISITION.name]["error"]
        resp = c.post(URL[ACQUISITION.name], json={"records": [_lead(run_features)]})
        assert resp.status_code == 503 and "not loaded" in resp.json()["detail"]
        assert c.get("/v1/models").json() == {}


def test_one_missing_model_does_not_block_the_other(model_registry, tmp_path, run_features):
    root = tmp_path / "models"
    shutil.copytree(model_registry / ACQUISITION.name, root / ACQUISITION.name)
    with TestClient(create_app(root, versions={})) as c:
        health = c.get("/health").json()
        assert health["models"][ACQUISITION.name]["loaded"]
        assert not health["models"][CHURN.name]["loaded"]
        assert c.post(URL[ACQUISITION.name],
                      json={"records": [_lead(run_features)]}).status_code == 200
        churn = _records(CHURN, run_features[CHURN.name], 1)
        assert c.post(URL[CHURN.name], json={"records": churn}).status_code == 503


def test_pinned_versions_come_from_the_environment(model_registry, monkeypatch):
    monkeypatch.setenv("NORTHSTAR_CHURN_MODEL_VERSION", "v20000101-missing")
    with TestClient(create_app(model_registry)) as c:
        health = c.get("/health")
        assert health.status_code == 503
        assert "v20000101-missing" in health.json()["models"][CHURN.name]["error"]
        assert health.json()["models"][ACQUISITION.name]["loaded"]


def test_serve_command_refuses_to_start_without_models(tmp_path, capsys):
    assert main(["serve", "--model-dir", str(tmp_path / "none")]) == 1
    err = capsys.readouterr().err
    assert "Refusing to start" in err and "northstar train-models" in err
