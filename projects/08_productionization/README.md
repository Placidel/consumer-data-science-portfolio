# 08 · End-to-End Productionization and Stakeholder Adoption

## Business problem

Sections 01 and 02 show that a lead score and a churn model beat the rules marketing and CRM use
today, on out-of-time holdouts. As analysis code, though, they produce value only when an analyst
reruns a pipeline and hands over a CSV. Before the business depends on them, the models must be
**operable**. Another team has to be able to score leads and customers on demand or in bulk, know
which model version produced a score, and notice when the population drifts or accuracy falls. It
also needs to know what the scores are for and what they are not for.

**Business question:** can the analytical work be operated, scored, monitored and handed to
another team, rather than existing only as analysis code?

## Decision supported

- **Engineering and data platform:** can the scoring service be deployed, and is it safe to
  depend on? The service needs a versioned contract, a clear failure mode, a health check and a
  reproducible image.
- **Marketing operations and CRM:** which list do I work this month, which model version built it,
  and is it still trustworthy? The answer is the batch call list plus the monthly monitoring
  report.
- **Model owner (data science):** do we keep the current model, recalibrate it or retrain it? The
  monitoring checks give a severity and a recommended action for each model.

## Data used

- **Training:** the shared synthetic tables (section 00), assembled by the section 01 and 02
  point-in-time feature builders on those sections' out-of-time splits. No new features and no
  new labels are introduced here, so the served models are the ones the sections validated.
- **Serving:** one feature record per open lead or active customer, exactly as those builders
  produce it for a scoring date. The API receives features. It does not look up raw events.
- **Monitoring:** a *reference* profile of each model's training runs, stored with the artifact,
  compared with the *current* monthly scoring runs from the model's first usable date
  (2025-07-01) to the last run in the data (2025-12-01). Outcomes are joined only when their
  window has closed.

All data is synthetic. IDs are generated keys with no personal attributes, and API error
responses never echo submitted values.

## Method

Code: [`src/northstar/serving/`](../../src/northstar/serving). Architecture and operations
detail: [`docs/architecture.md`](../../docs/architecture.md) and
[`docs/operations.md`](../../docs/operations.md).

```text
data/raw ──> section feature builders ──> train-models ──> models/<name>/<version>/  (registry)
                                          │ leakage audit gate                │ LATEST pointer
                                          │ champion selection (section rule) ▼
                                          └───────────────────────  API / batch scorer / monitor
```

1. **Served models** (`specs.py`). One spec per model binds it to the section that designed it:
   population, entity, horizon, feature builder, split, candidates and selection rule. Training,
   serving, batch scoring and monitoring all read the same spec, so they cannot disagree on
   feature order or horizon.

   | Model | Scores | Horizon | Candidates | Selection (validation runs) |
   |---|---|---|---|---|
   | `acquisition_lead_score` | open leads (created in the last 90 days, no order yet) | P(first order within 30 days) | channel rate, logistic regression, gradient boosting | highest average precision (section 01's rule) |
   | `churn_risk` | active customers (ordered in the last 180 days) | P(no order within 90 days) | RFM cell rate, logistic regression, gradient boosting | lowest log loss (section 02's rule) |

   The API returns probabilities, so section 01's rank-only recency heuristic is not a serving
   candidate.
2. **Training and registry** (`training.py`, `registry.py`). `northstar train-models` rebuilds
   the section dataset and runs the **section's leakage audit as a hard gate**. It fits the
   candidates on the fit runs, selects on the validation runs, refits the champion on all
   training runs and scores the holdout once for the model card. Each artifact is a folder with
   the estimator (`model.joblib`), a model card (`metadata.json`: population, intended and
   prohibited uses, split, selection table, validation and holdout metrics, data seed, library
   versions, file hash) and the reference profile. Versions are **content-addressed**
   (`v<first usable date>-<fingerprint of data, features, split, algorithm, library versions>`).
   Retraining on unchanged inputs therefore reproduces the same version. `LATEST` names the
   served version. `northstar promote-model` moves it (promotion or rollback) only after the
   target version loads and verifies.
3. **Verified loading with clear failure behavior.** The loader refuses, with an actionable
   message, when a registry or version is missing, when the model file's SHA-256 differs from its
   metadata (checked *before* unpickling), when the artifact was trained with another
   scikit-learn, or when its feature list differs from the code that builds request frames.
4. **Request/response contracts** (`schemas.py`, Pydantic). Validation is strict because a bad
   record does not crash a model, it just scores wrongly:
   - unknown fields are rejected, including outcome columns such as `converted`;
   - categorical values must be levels of the shared schema. The one-hot encoder would otherwise
     score an unknown channel silently as "none of the above";
   - numbers must be finite and in their domain: whole non-negative counts, shares in [0, 1];
   - records outside the model's population are refused: a lead older than 90 days, a customer
     with no order in 180 days, a lead that has already reached the purchase stage;
   - identities that every correctly built record satisfies are enforced, for example 7-day
     sessions ≤ 30-day sessions ≤ total, open rate = opens / emails, and low-CSAT contacts ≤
     contacts.

   Each request holds 1-1,000 records with unique IDs. The response names the model version and
   gives each record's probability and **reference percentile** (the share of the training
   population scoring at or below it).
5. **Scoring API** (`api.py`, FastAPI):

   | Endpoint | Purpose |
   |---|---|
   | `GET /health` | 200 with loaded versions, or 503 naming the model that failed to load and why |
   | `GET /v1/models` | model cards of the loaded models |
   | `POST /v1/acquisition/score` | lead records → `conversion_probability`, `reference_percentile` |
   | `POST /v1/churn/score` | customer records → `churn_probability`, `reference_percentile` |
   | `GET /docs` | interactive OpenAPI documentation |

   Models are loaded **once** at start-up. Requests only call `predict_proba`. If an artifact is
   bad, the process still starts, `/health` returns 503 and only the affected endpoint returns
   503, so a load balancer takes the instance out of rotation. `northstar serve` also runs a
   preflight and refuses to start unless every model loads. Versions can be pinned per model
   (`--churn-version`, or `NORTHSTAR_CHURN_MODEL_VERSION` and
   `NORTHSTAR_ACQUISITION_MODEL_VERSION`). Logs record model, version, record count and latency,
   never feature values.
6. **Batch scoring** (`batch.py`, `northstar score-batch`). Scores a whole run, either from a
   scoring date (features built from the tables) or from a CSV/Parquet feature file. A file is
   checked as a whole first. Columns other than the ID, the features, `run_cutoff` and the
   builder's context columns (churn's `margin_180d`) are refused, so an outcome column such as
   `converted` or a misspelt feature fails the batch just as it fails the API. Every row needs a
   scoring date, either its `run_cutoff` or a `--cutoff` given for the whole file. A file with
   neither is refused, and so are dates before a model's first usable date, because those
   scores would reuse training labels. Rows then go through the same validation and scorer as
   the API. By default one invalid row aborts the batch, because a partial list silently drops
   people. `--max-invalid-share` accepts a small share and writes the rejected rows and their
   reasons to a quarantine file. The output is ranked within each run, so it is directly a call
   list.
7. **Monitoring** (`profiles.py`, `monitoring.py`, `northstar monitor`). Every current run is
   batch-scored, then three kinds of check run:
   - **Input drift:** Population Stability Index per feature against the stored reference
     (quantile bins at observed values; categorical level shares). The pooled current
     population is judged; the worst single run is reported for context.
   - **Score drift:** PSI of the score distribution. If it moves, capacity plans that assume
     "top 10% of scores" contain a given population need re-deriving.
   - **Outcome performance** on runs whose label window has closed (30 days for leads, 90 days
     for churn). Per-run ROC AUC with a Hanley-McNeil 95% interval is compared with the
     champion's validation estimate. The observed/predicted outcome ratio is tested with a
     binomial z-test. An alert needs the change to be **material and beyond sampling noise**,
     so a small month does not trigger a retrain by chance.

   Severities run `ok` < `watch` < `investigate` < `recalibrate` < `retrain`, each with a
   recommended action. Output is machine-readable (`metrics.json`, `drift_features.csv`,
   `drift_by_run.csv`, `performance_by_run.csv`) and human-readable (`monitoring_report.md`,
   figures and the block below).
8. **Deployment** ([`Dockerfile`](../../Dockerfile),
   [`docker-compose.yml`](../../docker-compose.yml),
   [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml)). One image serves both the API
   and the dashboard. The base image is pinned by digest, dependencies by
   [`constraints.txt`](../../constraints.txt), and the data and models are built inside the image
   from the fixed seed. The container starts by loading artifacts. It runs as a non-root user,
   the API container's filesystem is read-only, and no secrets or environment variables are
   required. CI installs into a fresh virtual environment, runs Ruff and the full test suite on
   locked (3.12) and latest (3.11) dependencies, then builds the image, starts both services, and
   smoke-tests `/health`, both scoring endpoints, a 422 case and the dashboard health endpoint.

## Validation design

Tests: `tests/test_serving_api.py`, `tests/test_serving_registry.py`,
`tests/test_serving_monitoring.py`, `tests/test_productionization_pipeline.py`.

- **Representative scoring and training-serving parity.** Records built by the section feature
  pipelines for a real scoring run are posted to both endpoints. The API's probabilities must
  equal the offline pipeline's to 1e-9, in request order, with monotone percentiles. The
  committed example requests must score.
- **Validation is neither too loose nor too strict.** A parametrized suite of invalid records
  must return 422 naming the offending field. The cases include missing and extra fields, an
  outcome column, an unknown channel, negative or fractional counts, text in a number, NaN, a
  share above one, an out-of-population lead, broken identities, a bad ID, duplicate IDs, empty
  and oversized batches, and a lead sent to the churn endpoint. Batch files with an outcome
  column, a misspelt feature, no scoring date, a garbled date or a date inside the training
  period are refused both by `score_frame` and by the `score-batch` command. Conversely,
  **every** record the feature builders produce for every run of both models must pass.
- **No training at request time.** With every `fit` method patched to raise, repeated requests
  must succeed. The loader must run exactly once per model, and the model objects must be the
  same before and after.
- **Failure behavior.** With no registry, `/health` returns 503 and names
  `northstar train-models`, scoring returns 503, and `northstar serve` refuses to start. With one
  model missing, the other keeps serving. A pinned missing version is reported. Tampered hashes,
  a corrupted model file (refused before unpickling), a different scikit-learn, a changed feature
  list, unreadable metadata and a missing profile are each refused with a specific message.
- **Serialization and training integrity.** The loaded model predicts identically to a refit of
  the section's estimator on the fit and validation runs only. Negative controls show that a
  refit which also saw the holdout, or skipped the validation runs, predicts differently. So the
  match shows that no holdout row reached the served model. The
  champion follows the section's selection rule. Encoder levels equal the schema's levels.
  Versions are reproducible, change with the data or algorithm, and ignore tables the model does
  not read. `LATEST` moves only on promotion, and rollback works. A failed leakage audit
  registers nothing.
- **Monitoring statistics.** PSI and binning are checked against hand computations. Scoring the
  training population itself gives PSI 0, and an injected shift is flagged on exactly the
  shifted features. The Hanley-McNeil standard error is checked against a bootstrap. A shuffled
  model triggers `retrain` without a calibration alert, and an under-predicting model triggers
  `recalibrate` without a discrimination alert. The pooled holdout performance measured by the
  monitor reproduces the model card's holdout metrics exactly.
- **Reproducibility and traceability.** The commands run end to end on freshly generated data.
  The results block and `monitoring_report.md` equal a fresh rendering of `metrics.json`. The
  served champions and holdout ROC AUC must match sections 01 and 02. A slow test regenerates the
  default data, retrains, and reproduces every committed number; version strings are excluded,
  because they include library versions by design.
- **Deployment configuration.** Tests check that the Dockerfile's base image is pinned by
  digest, that it uses the constraints, trains at build time and starts `northstar serve` as a
  non-root user, and that it contains no secret-like values. Compose must run exactly the API and
  the dashboard, each with a health check and no environment secrets. The CI workflow must lint
  and test from a new virtual environment and smoke-test the containers. Every declared
  dependency must be pinned in `constraints.txt`.

## Results generated from the current run

Figures: [`outputs/figures/feature_psi.png`](outputs/figures/feature_psi.png) (pooled input drift
per model) and [`outputs/figures/performance_by_run.png`](outputs/figures/performance_by_run.png)
(matured-run ROC AUC against the validation estimate). Full report:
[`outputs/monitoring_report.md`](outputs/monitoring_report.md). Example request bodies:
[`outputs/example_requests/`](outputs/example_requests).

<!-- BEGIN GENERATED: production-results -->
_data seed `20240101`, 40,000 prospects. Rendered from `outputs/metrics.json` by `northstar monitor`. Reference = each model's training runs; current = every monthly scoring run from the model's deployable date to 2025-12-01, scored through the batch scorer. PSI thresholds 0.1 (watch) and 0.25 (investigate); performance alerts need a ROC AUC drop over 0.05 or an observed/predicted ratio outside 0.8-1.25, beyond 95% sampling noise._

**Registered models** (served by the API; `LATEST` version of each)

| Model | Version | Algorithm | Training runs | Deployable from | Validation ROC AUC | Holdout ROC AUC | Holdout AP | Holdout log loss | Matches section champion |
|---|---|---|---|---|---:|---:|---:|---:|---|
| Lead conversion score (`acquisition_lead_score`) | `v20250701-ac88f76ea6` | Logistic regression | 2024-04-01 to 2025-06-01 (50,798 rows) | 2025-07-01 | 0.814 | 0.810 | 0.205 | 0.1807 | yes |
| 90-day churn risk (`churn_risk`) | `v20250701-089e978487` | Gradient boosting | 2024-07-01 to 2025-04-01 (28,703 rows) | 2025-07-01 | 0.802 | 0.811 | 0.782 | 0.5307 | yes |

**Monitoring summary**

| Model | Current runs | Records scored | Score PSI | Features at watch / investigate | Matured runs | Labels pending | Status |
|---|---|---:|---:|---:|---:|---|---|
| Lead conversion score | 2025-07-01 to 2025-12-01 (6) | 23,850 | 0.000 | 0 / 0 | 6 | none | **ok** |
| 90-day churn risk | 2025-07-01 to 2025-12-01 (6) | 37,595 | 0.033 | 3 / 1 | 4 | 2025-11-01, 2025-12-01 | **investigate** |

**Lead conversion score** - status **ok**. No action: inputs, scores and matured outcomes are in line with the reference.

- No checks triggered.

Largest input shifts (pooled PSI over current runs):

| Feature | PSI | Worst run (PSI) | Reference | Current | Status |
|---|---:|---|---:|---:|---|
| `acquisition_channel` | 0.009 | 2025-12-01 (0.023) | 23.0% paid_social | 26.1% | ok |
| `age_band` | 0.001 | 2025-10-01 (0.003) | 24.1% 25-34 | 25.3% | ok |
| `email_open_rate` | 0.001 | 2025-12-01 (0.006) | 0.17 | 0.17 | ok |
| `pages_viewed` | 0.001 | 2025-12-01 (0.005) | 12.25 | 12.19 | ok |
| `lead_age_days` | 0.001 | 2025-12-01 (0.019) | 42.64 | 42.32 | ok |

Matured runs vs validation estimate (ROC AUC 0.814, outcome rate 6.0%):

| Run | Records | Observed rate | Mean predicted | Observed / predicted | ROC AUC (95% CI) | AP |
|---|---:|---:|---:|---:|---|---:|
| 2025-07-01 | 3,695 | 5.6% | 6.0% | 0.94 | 0.821 (0.785 to 0.856) | 0.240 |
| 2025-08-01 | 3,972 | 5.3% | 5.8% | 0.92 | 0.800 (0.763 to 0.836) | 0.191 |
| 2025-09-01 | 4,035 | 5.4% | 5.5% | 0.98 | 0.804 (0.769 to 0.840) | 0.207 |
| 2025-10-01 | 4,010 | 5.2% | 5.6% | 0.93 | 0.823 (0.788 to 0.859) | 0.205 |
| 2025-11-01 | 3,848 | 6.0% | 5.8% | 1.04 | 0.796 (0.761 to 0.831) | 0.202 |
| 2025-12-01 | 4,290 | 6.2% | 6.5% | 0.95 | 0.814 (0.782 to 0.846) | 0.211 |
| **All matured** | 23,850 | 5.6% | 5.9% | 0.96 | 0.810 (0.796 to 0.824) | 0.205 |

**90-day churn risk** - status **investigate**. Keep scoring, but confirm with the data owner whether the shift is a real change in the population or an upstream pipeline change before the next run.

- `investigate` feature drift: `tenure_days` - PSI 0.423 vs reference (worst run 2025-12-01: 0.543)
- `watch` feature drift: `category_count` - PSI 0.123 vs reference (worst run 2025-12-01: 0.171)
- `watch` feature drift: `orders_total` - PSI 0.206 vs reference (worst run 2025-12-01: 0.280)
- `watch` feature drift: `store_order_share` - PSI 0.107 vs reference (worst run 2025-12-01: 0.146)

Largest input shifts (pooled PSI over current runs):

| Feature | PSI | Worst run (PSI) | Reference | Current | Status |
|---|---:|---|---:|---:|---|
| `tenure_days` | 0.423 | 2025-12-01 (0.543) | 126.40 | 219.02 | investigate |
| `orders_total` | 0.206 | 2025-12-01 (0.280) | 3.45 | 6.06 | watch |
| `category_count` | 0.123 | 2025-12-01 (0.171) | 2.32 | 2.78 | watch |
| `store_order_share` | 0.107 | 2025-12-01 (0.146) | 0.10 | 0.13 | watch |
| `discount_share` | 0.084 | 2025-12-01 (0.106) | 0.04 | 0.03 | ok |

Matured runs vs validation estimate (ROC AUC 0.802, outcome rate 52.9%):

| Run | Records | Observed rate | Mean predicted | Observed / predicted | ROC AUC (95% CI) | AP |
|---|---:|---:|---:|---:|---|---:|
| 2025-07-01 | 5,676 | 48.9% | 49.7% | 0.98 | 0.813 (0.802 to 0.824) | 0.793 |
| 2025-08-01 | 5,960 | 49.4% | 48.8% | 1.01 | 0.817 (0.806 to 0.827) | 0.799 |
| 2025-09-01 | 6,158 | 47.7% | 48.9% | 0.98 | 0.800 (0.789 to 0.811) | 0.765 |
| 2025-10-01 | 6,336 | 45.4% | 48.4% | 0.94 | 0.814 (0.803 to 0.825) | 0.771 |
| **All matured** | 24,130 | 47.8% | 48.9% | 0.98 | 0.811 (0.805 to 0.816) | 0.782 |
| 2025-11-01 | 6,561 | pending until 2026-01-30 | | | | |
| 2025-12-01 | 6,904 | pending until 2026-03-01 | | | | |
<!-- END GENERATED: production-results -->

## Business interpretation

- **The served models are the validated ones.** Each registered champion is the model its
  section selected, with the same holdout ROC AUC, and every artifact passed the section's
  leakage audit before registration. The API returns the same probabilities as the analysis, so
  the section 01 and 02 findings (lift over current rules, calibration, retention ROI) carry over
  to what the business consumes.
- **The lead score is stable.** Across the six post-deployment runs no input moved beyond the
  watch threshold, the score distribution did not move, and every matured run's ROC AUC and
  calibration stayed within tolerance of the validation estimate. The outreach capacity plans
  from section 01 remain valid.
- **The churn model shows why drift is not the same as degradation.** Its inputs moved, led by
  tenure: Northstar only started trading in January 2024, so the training base was young, and
  every month the active base gets older. More than one in ten current customers now has a longer
  tenure than anyone in the training runs, and order counts and category breadth rise with
  tenure. Yet the score distribution barely moved, and on every run whose outcomes have matured,
  ranking and calibration held. The right action is `investigate`, not an emergency retrain.
  Confirm that the shift is organic ageing rather than a pipeline change, and make sure the
  next scheduled retrain covers recent runs, so that longer tenures are inside the training
  range. The November and December runs cannot be judged until their 90-day windows close. The
  report states this instead of guessing.
- **Operational risk is contained by design.** A bad artifact or a malformed upstream feed
  produces a clear, early failure: a refused start or a 503 at deploy time, a 422 per request,
  an aborted or quarantined batch. It never produces silently wrong scores.

## Stakeholder adoption

**How the scores should be used**

| | Lead conversion score | 90-day churn risk |
|---|---|---|
| Use it to | rank the open pipeline and give outreach capacity (SDR calls, sales follow-up) to the top of the list | rank the active base for retention offers, combined with customer margin as in section 02's expected-value rule |
| Scope | leads created in the last 90 days that have not yet ordered | customers with an order in the last 180 days |
| Read the number as | the probability of a first order in the next 30 days *without a change in treatment*; `reference_percentile` places it within the training population | the probability of no order in the next 90 days *without an offer*; use it with margin, not alone |
| Typical cut | the top 10-30% of a run (section 01's capacity simulation) | the depth where expected net value turns negative (section 02's simulation) |

**How they should not be used**

- **Not as causal effects.** A high churn score does not mean an offer will save the customer,
  and a high lead score does not mean a call caused the order. Measure program impact with a
  holdout group (section 03's framework), never by comparing contacted high scorers with
  uncontacted low scorers.
- **Not outside their population:** win-back of lapsed customers, prospects older than the
  pipeline window, or other businesses and markets. The API refuses these records on purpose.
- **Not for individual pricing, credit, eligibility or service-level decisions,** and not as a
  customer-facing number. The models use behavioral and coarse demographic bands (age band,
  income band, region) that are appropriate for prioritizing marketing effort but not for
  decisions with individual consequences.
- **Not as a fixed threshold forever.** A "probability above 0.5" rule breaks silently when the
  base rate moves. Plan with ranks within each run, and re-derive cut-offs after score drift.

**Refresh cadence**

| Activity | Cadence | Trigger or dependency |
|---|---|---|
| Batch scoring (call lists) | monthly, on the 1st (the runs the models were designed for) | `northstar score-batch --cutoff <date>` |
| On-demand scoring | as needed, for example CRM or sales tools calling the API | feature record as of the scoring date |
| Drift monitoring | monthly, right after batch scoring | `northstar monitor` |
| Performance monitoring | when labels mature: 30 days (leads), 90 days (churn) | monthly report, listed as pending until then |
| Retraining | quarterly, or when monitoring returns `recalibrate` or `retrain` | `train-models --no-promote`, review, then `promote-model` |

**Owners** (roles in the fictional organization; a full RACI is in
[`docs/operations.md`](../../docs/operations.md))

| Responsibility | Owner |
|---|---|
| Model logic, retraining, promotion decision, monitoring review | Data Science (model owner) |
| Feature pipeline and source data quality | Data Engineering |
| API uptime, deployment, rollback execution | ML Platform / Engineering |
| Lead list use and outreach capacity | Marketing Operations and Sales leadership |
| Retention offer rules and budget | CRM / Lifecycle Marketing |

**Monitoring and feedback loop**

1. After each monthly run, the model owner reviews `monitoring_report.md`. `ok` and `watch` need
   no action. `investigate` goes to Data Engineering to check the upstream data.
   `recalibrate` and `retrain` start a retrain.
2. Outcomes flow back automatically: first orders and churn are observed from the same order
   tables, so every matured run becomes a new labeled training run. No manual labeling is
   needed.
3. Business users report surprises, such as a high-scoring lead that turned out to be a
   duplicate or an existing customer, to the model owner. Recurring patterns become validation
   rules or features.
4. Retention and outreach programs keep a **random holdout** (for example 10% of each list not
   contacted). The holdout measures the programs' real lift, and it keeps untreated outcomes
   available for retraining, so the models do not learn only from treated customers.
5. Every change is a new version, trained with `--no-promote`, compared on the same holdout
   runs, and then promoted. Rollback is `northstar promote-model --version <previous>` plus an
   API restart.

## Limitations and next steps

- **The feature store is the pipeline itself.** Callers must send point-in-time feature records.
  In production the section builders would run as a scheduled job writing to a feature table
  that the CRM and API callers read. The validation rules here would become that job's data
  contract.
- **A local file registry.** Versioned folders and a `LATEST` pointer show the mechanics.
  A company deployment would use a registry service (for example MLflow) with object storage,
  approvals and audit history. Pickled artifacts should only be loaded from a trusted registry;
  the hash check detects corruption, not a malicious writer.
- **Monitoring runs on the synthetic history.** "Current" runs come from the same generator as
  training, so the drift found is organic (ageing), not a pipeline failure. The injected-drift
  and degraded-model tests show the checks fire when they should. PSI thresholds are industry
  conventions, not tuned to Northstar's cost of false alarms.
- **Reference performance is a point estimate.** Alerts account for the current run's sampling
  noise but not for the uncertainty in the validation AUC. The significance rule is applied per
  run without a multiplicity correction, which is acceptable for a monthly review but would
  need one for daily checks.
- **No online A/B assignment or explanations in the API.** Section 02's SHAP reason codes are
  computed in batch. Serving them per request, and assigning program holdouts inside the
  service, are natural extensions.
- **No authentication or rate limiting.** The service is meant to run behind an internal gateway
  that provides single sign-on, TLS and quotas. It deliberately holds no secrets.

## Reproduction commands

From the repository root, with the virtual environment activated (see the root README):

```bash
northstar generate-data                  # shared data (section 00), if not already present
northstar train-models                   # train, audit and register both models (~40 s)
northstar monitor                        # drift + performance report, outputs/ and this README (~20 s)
# or: make production

northstar score-batch --model churn --cutoff 2025-12-01          # -> scores/churn_risk_2025-12-01.csv
northstar score-batch --model acquisition --features leads.csv    # score an upstream feature file
northstar promote-model --model churn                             # list versions (LATEST marked)
northstar promote-model --model churn --version <version>         # promote or roll back

northstar serve                          # API at http://127.0.0.1:8000 (docs at /docs)
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/v1/churn/score -H "Content-Type: application/json" \
  -d @projects/08_productionization/outputs/example_requests/churn.json

docker compose up --build --wait         # API on :8000 and dashboard on :8501 in containers
docker compose down

python -m pytest tests/test_serving_api.py tests/test_serving_registry.py \
  tests/test_serving_monitoring.py tests/test_productionization_pipeline.py
```

`models/` (the registry) and `scores/` (batch output) are build artifacts and are git-ignored:
`northstar train-models` rebuilds the registry deterministically. `NORTHSTAR_MODEL_DIR` points
all commands at another registry.
