# Architecture: from analysis to a served, monitored model

This document describes how the Northstar models get from the section analyses to a running
service, and what happens when something fails. For stakeholder-facing context (intended use,
owners and cadence), see [section 08](../projects/08_productionization/README.md). For day-to-day
procedures, see [operations.md](operations.md).

## Components

```text
                           ┌───────────────────────────── build time (image or CI) ─────────┐
 northstar generate-data   │  data/raw/*.parquet (fixed seed, content-hashed manifest)       │
          │                │            │                                                   │
          ▼                │            ▼                                                   │
 section feature builders ─┼─> northstar train-models                                       │
 (01 acquisition,          │     1. build section dataset (point-in-time features + labels)  │
  02 retention)            │     2. section leakage audit  ── fail → nothing registered      │
                           │     3. fit candidates on fit runs, select on validation runs    │
                           │     4. refit champion on fit + validation runs                  │
                           │     5. score holdout once → model card                          │
                           │     6. write models/<name>/<version>/ atomically, move LATEST   │
                           └─────────────────────────────────────────────────────────────────┘
                                         │
             ┌───────────────────────────┼────────────────────────────┐
             ▼                           ▼                            ▼
   northstar serve (FastAPI)   northstar score-batch         northstar monitor
   load once at start-up       same validation + scorer      batch-score current runs,
   /health  /v1/models         ranked call list per run      PSI vs stored reference,
   /v1/acquisition/score       quarantine file for rejects   matured-run AUC/calibration,
   /v1/churn/score                                           JSON + CSV + Markdown report
             │                                                            │
             ▼                                                            ▼
   CRM / sales tools                                          model owner review
                                                              (projects/08_productionization)
```

The executive dashboard (section 07) runs from the same image. It reads the committed section
outputs and does not call the API.

## Code map

| Module | Responsibility |
|---|---|
| `serving/specs.py` | What is served: population, entity, horizon, features, split, candidates, selection rule, and the section functions that build features and fit models |
| `serving/training.py` | Offline training with the leakage gate, champion selection, content-addressed version, model card |
| `serving/registry.py` | Artifact layout, atomic writes, `LATEST` pointer, promotion and rollback, verified loading |
| `serving/schemas.py` | Pydantic request and response contracts: domains, population scope, feature identities |
| `serving/scoring.py` | One scoring path (`Scorer`) and row-level validation for frames |
| `serving/api.py` | FastAPI app: lifespan loading, health, model cards, scoring endpoints, 422/503 behavior |
| `serving/batch.py` | Batch scoring from a scoring date or a feature file; refuses unknown/outcome columns and undated or in-sample files; strict or quarantine mode |
| `serving/profiles.py` | Reference profiles, PSI, percentiles |
| `serving/monitoring.py` | Drift and delayed-label performance checks, severities, recommendations |
| `serving/report.py` | Section 08 outputs, figures and README block |

## Artifact layout and versioning

```text
models/
  acquisition_lead_score/
    LATEST                               -> "v20250701-<fingerprint>"
    v20250701-<fingerprint>/
      model.joblib                       fitted scikit-learn pipeline
      metadata.json                      model card (see below)
      reference_profile.json             training-population bins and shares, score quantiles
  churn_risk/ ...
```

- **Version** = `v<first usable date>-<first 10 hex of SHA-256>` over the model name, algorithm,
  feature list, split, content hashes of the input tables the model reads, and the scikit-learn
  and package versions. The same inputs give the same version; any change gives a new one. There
  are no timestamps, so a rebuild is byte-for-byte comparable where the libraries allow.
- **First usable date** (`deployable_from`) is the first holdout run. All training labels are
  observed by then, so scoring an earlier date would reuse training outcomes. The batch scorer
  refuses such dates.
- **Model card** (`metadata.json`) holds the population, intended use and prohibited uses, the
  features, the split, the selection table for every candidate, champion validation and holdout
  metrics, the leakage-audit checks, the data seed and size, the Python, scikit-learn and pandas
  versions, and the model file's SHA-256.
- **Writes are atomic.** A version is written to a hidden temporary folder and renamed into place;
  `LATEST` is replaced with an atomic rename. Readers never see a partial version.

## Request lifecycle

1. The request body is parsed and validated by Pydantic (`schemas.py`). Any problem returns
   **422** with `loc`, `type` and `msg` per error. Submitted values are not echoed back.
2. The endpoint looks up the scorer loaded at start-up. If that model failed to load, it returns
   **503** with the loader's message.
3. The validated records become a frame with the training dtypes, then go through
   `predict_proba` and the reference-percentile lookup.
4. The response carries the model name, version, algorithm, horizon and `deployable_from`, and
   one prediction per record in request order.
5. A log line records the model, version, record count and latency. The `X-Process-Time-Ms`
   header carries the request time.

The API is stateless apart from the loaded models, so it scales horizontally. Each worker loads
its own copy of the models, which take a few MB.

## Failure modes

| Failure | Detected by | Behavior |
|---|---|---|
| No registry / no `LATEST` | loader | `serve` refuses to start; app `/health` 503 naming `northstar train-models` |
| Pinned version missing | loader | 503 for that model, with the list of available versions |
| Model file corrupted or replaced | SHA-256 vs metadata, before unpickling | refused |
| Artifact from another scikit-learn | metadata vs running version | refused ("retrain with `northstar train-models`") |
| Feature list differs from the code | metadata vs spec | refused |
| One model bad, other fine | per-model loading | healthy model keeps serving; `/health` 503 |
| Malformed or out-of-scope record | Pydantic schema | 422 per request; batch aborts (or quarantines within a tolerance) |
| Leakage audit fails at training | section audit | nothing registered, command exits 1 |
| Input or score drift, degraded outcomes | `northstar monitor` | severity plus recommended action in the report |

## Deployment

- **Image** ([`Dockerfile`](../Dockerfile)). The Python 3.12 slim base is pinned by digest, and
  every dependency version comes from [`constraints.txt`](../constraints.txt). The package is
  installed so that `/app` mirrors the repository. The build generates the data with the default
  seed and trains the models, so starting a container only loads artifacts. The image runs as a
  non-root user. The default command is `northstar serve --host 0.0.0.0 --port 8000`. A
  `HEALTHCHECK` calls `/health`.
- **Compose** ([`docker-compose.yml`](../docker-compose.yml)). `api` (port 8000, read-only root
  filesystem, `/tmp` as tmpfs) and `dashboard` (Streamlit on port 8501) run from the same image,
  each with a health check. `docker compose up --build --wait` returns when both are healthy.
  No environment variables, `.env` files or secrets are used.
- **CI** ([`.github/workflows/ci.yml`](../.github/workflows/ci.yml)). Lint and the full test suite
  run in a fresh virtual environment on Python 3.12 with locked dependencies and on Python 3.11
  with the newest allowed ones. After that, the image is built, both services start, and the
  job smoke-tests health, both scoring endpoints (with the committed example requests), a 422
  case and the dashboard.

## Security and privacy notes

- All data is synthetic, and IDs are generated keys. Error responses and logs never include
  feature values.
- Artifacts are pickles. The hash check catches corruption, not a malicious writer, so load only
  from a registry with controlled write access.
- The service has no authentication and is meant to sit behind an internal gateway that provides
  identity, TLS and rate limits. It needs no credentials of its own, so there are none to leak.
