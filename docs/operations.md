# Operations runbook: scoring service and model lifecycle

How to run, refresh, monitor, promote and roll back the Northstar models. The design is
described in [architecture.md](architecture.md), and the business context in
[section 08](../projects/08_productionization/README.md). The commands assume the repository
virtual environment is active; `make` targets wrap most of them.

## Ownership (RACI)

R = responsible, A = accountable, C = consulted, I = informed. Roles belong to the fictional
Northstar organization.

| Activity | Data Science (model owner) | Data Engineering | ML Platform | Marketing Ops / Sales | CRM / Lifecycle |
|---|---|---|---|---|---|
| Monthly batch scoring | A | R (feature job) | C | I | I |
| Monthly monitoring review | R/A | C | I | I | I |
| Investigating input drift | A | R | I | C | C |
| Retraining and promotion | R/A | C | C | I | I |
| API deployment and uptime | C | I | R/A | I | I |
| Rollback | A (decision) | I | R (execution) | I | I |
| Use of lead lists | C | | | R/A | |
| Retention offer rules and budget | C | | | | R/A |

## Routine calendar

| When | What | Command |
|---|---|---|
| 1st of each month | Build the scoring run's call lists | `northstar score-batch --model acquisition --cutoff <YYYY-MM-01>` and `--model churn` |
| Right after scoring | Drift and performance report | `northstar monitor`, then review `projects/08_productionization/outputs/monitoring_report.md` |
| Continuous | API serving CRM and sales tools | `northstar serve` or `docker compose up -d` |
| Quarterly, or on a `recalibrate` / `retrain` finding | Retrain, compare, promote | see below |

Performance on a run can only be judged when its outcome window closes: 30 days for lead
scores and 90 days for churn. Until then the report lists the run as pending and checks drift
only.

## Reading the monitoring report

| Severity | Meaning | Response (owner) | Target time |
|---|---|---|---|
| `ok` | inputs, scores and matured outcomes in line with the reference | none | - |
| `watch` | some input PSI 0.10-0.25 | note it in the monthly review; check whether it persists (DS) | next review |
| `investigate` | input or score PSI ≥ 0.25 | confirm with Data Engineering whether it is a real population change or a pipeline change (DE); if a pipeline bug, stop using the list until fixed | before the next run |
| `recalibrate` | observed/predicted outcome ratio outside 0.80-1.25, beyond sampling noise | rankings still usable; do not use probabilities in value calculations until retrained or recalibrated (DS) | within 2 weeks |
| `retrain` | ROC AUC more than 0.05 below the validation estimate, beyond sampling noise | retrain on recent runs; consider pausing the list if lift is lost (DS, with the business owner) | within 1 week |

The thresholds live in `MonitoringConfig` (`src/northstar/serving/monitoring.py`), and every
report records them.

## Retraining and promotion

1. Regenerate or refresh the data, then train **without** moving `LATEST`:
   ```bash
   northstar train-models --no-promote
   northstar promote-model --model churn          # lists versions; LATEST is marked
   ```
2. Compare the candidate's model card (`models/<name>/<version>/metadata.json`) with the current
   one: validation and holdout metrics, selected algorithm, leakage-audit checks, data seed. The
   candidate must not be worse on the holdout runs than the served version, and its audit must
   pass (training refuses to register otherwise).
3. Promote, then restart the API so it loads the new version:
   ```bash
   northstar promote-model --model churn --version <new version>
   docker compose restart api      # or restart `northstar serve`
   curl -s localhost:8000/health   # confirm the served version
   ```
4. Re-run `northstar monitor` so the report reflects the served version.

In the container image, models are trained at build time. The equivalent promotion is a new
image build (new data or code gives a new content-addressed version) followed by a redeploy.

## Rollback

```bash
northstar promote-model --model churn --version <previous version>   # verifies it loads first
docker compose restart api
```

Alternatively, pin a version without touching `LATEST`, for example for a canary:
`NORTHSTAR_CHURN_MODEL_VERSION=<version> northstar serve` (or `--churn-version`). Previous
versions stay in the registry until deleted.

## Incident checklist

| Symptom | Likely cause | Action |
|---|---|---|
| `northstar serve` prints "Refusing to start" | registry missing or artifact incompatible | read the per-model reason; `northstar train-models` for a missing registry or library upgrade; roll back if a new version is bad |
| `/health` 503 with a model error | same, in a running app | as above; the other model keeps serving |
| Many 422 responses from a caller | upstream feature job changed or broke | each error names the field and rule; Data Engineering fixes the job, and the API rules are the contract |
| `score-batch` aborts with "records failed validation" | broken feature rows | fix upstream; only if the business accepts a partial list, rerun with `--max-invalid-share` and send the quarantine file to Data Engineering |
| `score-batch` refuses a date | date before the model's `deployable_from` or after the data | use a valid date or a model trained on an earlier split |
| `score-batch` reports "unexpected column(s)" | the feature file carries an outcome (`converted`, `churned`), a misspelt feature or another extra column | export only the ID, the features and `run_cutoff` (plus `margin_180d` for churn); never score a file that contains outcomes |
| `score-batch` reports "no scoring date" | the feature file has no `run_cutoff` column | add it, or pass the date the features were built for with `--cutoff` |
| Report `retrain` on a single small run | possible noise | the rule already requires significance; check the next run before acting if lift is still positive |

## Configuration reference

| Setting | Default | Used by |
|---|---|---|
| `NORTHSTAR_DATA_DIR` | `data/raw` | all data-reading commands |
| `NORTHSTAR_MODEL_DIR` | `models` | `train-models`, `score-batch`, `monitor`, `promote-model`, `serve` |
| `NORTHSTAR_ACQUISITION_MODEL_VERSION` / `NORTHSTAR_CHURN_MODEL_VERSION` | unset (serve `LATEST`) | `serve` / API |
| `NORTHSTAR_PROJECTS_DIR` | `projects` | dashboard |
| `--max-invalid-share` | 0 | `score-batch` |

No setting is secret. The service requires no credentials.
