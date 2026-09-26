# Northstar Consumer: an end-to-end consumer data science portfolio

This repository takes one fictional business from business question to deployed model:
acquisition, churn, conversion optimization, customer value, forecasting, lifecycle analytics,
an executive dashboard and production ML. Every section shares one reproducible synthetic
dataset, so the work reads as a single analytics program rather than disconnected notebooks.

## The story in one page

| Stage | What happens | Where |
|---|---|---|
| **Business requirements** | Each section starts from a decision someone at Northstar has to make: who sales should call, who gets a retention offer, whether to ship a checkout redesign, where growth money goes, what revenue to plan for | section READMEs, "Decision supported" |
| **Data** | One synthetic, seeded, validated data model of the whole customer journey, with a point-in-time `snapshot` so every feature is built only from what was known at the scoring date | [section 00](projects/00_foundation/README.md), [`docs/data_dictionary.md`](docs/data_dictionary.md) |
| **Models and analysis** | Baselines first, time-aware splits, leakage audits that block results, calibrated probabilities, intervals, and decision simulations under explicit assumptions | sections [01](projects/01_acquisition/README.md)-[06](projects/06_lifecycle/README.md) |
| **Consumption** | Executives read one dashboard whose every number traces to a pipeline output field | [section 07](projects/07_dashboard/README.md) |
| **Deployment** | The lead and churn champions become versioned, verified artifacts behind a FastAPI service and a batch scorer, in a pinned Docker image with CI | [section 08](projects/08_productionization/README.md), [`docs/architecture.md`](docs/architecture.md) |
| **Adoption** | Monitoring for drift and matured-outcome performance, a runbook, owners, refresh cadence, and explicit guidance on how scores should and should not be used | [section 08](projects/08_productionization/README.md#stakeholder-adoption), [`docs/operations.md`](docs/operations.md) |

## The business and its data

**Northstar Consumer** is a fictional omnichannel retailer. It sells apparel, footwear, home,
beauty, outdoor and accessories through its website, app and physical stores, runs quarterly
paid acquisition, lifecycle email and seasonal promotions, and sells a paid membership called
**Northstar Plus**.

All data is synthetic and generated with a fixed seed. No real or scraped personal data is used,
and there are no names, emails or addresses. The shared model (details in
[`docs/data_dictionary.md`](docs/data_dictionary.md)):

```text
campaigns ─┬─< prospects ──1:0..1── customers ─┬─< orders ──< order_lines >── products
           │        │                          ├─< subscription_events
           │        ├─< sessions ──< funnel_events
           │        ├─< marketing_touches      └─< support_contacts
           │        └─< experiment_assignments >── experiments
           └─< marketing_touches / sessions / orders (campaign_id)
```

- 24 months of history (2024-01-01 to 2025-12-31); about 1.5M rows and 13 MB of Parquet.
- `prospect_id` follows a person from lead to customer. Entity tables hold only attributes known
  at creation. Outcomes such as conversion, churn and value must be derived from event logs.
- **Time-aware by construction:** the default modelling cutoff is 2025-07-01, and
  `northstar.timeline.snapshot(tables, cutoff)` gives the point-in-time view that later sections
  use for leak-free features.
- An A/B test (one-page checkout) is embedded with deterministic hash-based assignment.

## Sections

| # | Section | Business question | Status |
|---|---|---|---|
| 00 | [Foundation and synthetic data](projects/00_foundation/README.md) | Can every analysis share one trustworthy, reproducible, leak-safe view of the customer journey? | Complete |
| 01 | [Customer acquisition optimization](projects/01_acquisition/README.md) | Which prospects should sales/marketing prioritize to maximize acquisition efficiency? | Complete |
| 02 | [Retention and churn prediction](projects/02_retention/README.md) | Which active customers are most at risk of churning, and how should a limited retention budget be prioritized? | Complete |
| 03 | [Conversion funnel and experimentation](projects/03_conversion/README.md) | Where do prospects drop out of the funnel, and does the checkout redesign improve conversion? | Complete |
| 04 | [Revenue growth and customer value](projects/04_revenue_growth/README.md) | Which customers represent the highest future value, and where should growth efforts focus? | Complete |
| 05 | [Predictive analytics and revenue forecasting](projects/05_predictive_analytics/README.md) | What revenue should leadership expect in upcoming periods, and how uncertain is that forecast? | Complete |
| 06 | [Customer lifecycle analytics](projects/06_lifecycle/README.md) | How do customers move from prospect to new, active, loyal, at-risk and churned states, and where are the largest lifecycle opportunities? | Complete |
| 07 | [Executive sales and marketing dashboard](projects/07_dashboard/README.md) | How can sales and marketing stakeholders consume these insights without reading code? | Complete |
| 08 | [End-to-end productionization and adoption](projects/08_productionization/README.md) | Can the analytical work be operated, scored, monitored and handed to another team rather than existing only as analysis code? | Complete |

Each section README covers the business problem, the decision it supports, data, method,
validation design, results generated by code, interpretation, limitations and reproduction
commands.

## Setup

Requires Python 3.11+ and uses the standard `venv` and pip:

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

To install exactly the versions the Docker image and the locked CI job use, add the constraints
file: `python -m pip install -c constraints.txt -e ".[dev]"`.

## Reproduce the data and results

One command regenerates all shared data from a clean checkout (about 1 minute on a laptop):

```bash
northstar generate-data
```

This generates the data with the default seed (`20240101`, 40,000 prospects) and validates
schema, keys and business rules. It writes `data/raw/*.parquet` plus `manifest.json`, which is
git-ignored and byte-identical on every run. It also refreshes section 00's generated tables,
figures and README results. Related commands:

```bash
northstar validate-data              # re-validate data/raw
northstar profile-data               # rebuild section 00 outputs from data/raw
northstar data-dictionary            # re-render docs/data_dictionary.md from the schema
python -m northstar --help           # same CLI without the console script
```

In Python:

```python
from northstar.io import load_tables
from northstar.timeline import DEFAULT_CUTOFF, PredictionWindow, snapshot

tables = load_tables()                          # dict of DataFrames with canonical dtypes
history = snapshot(tables, DEFAULT_CUTOFF)      # features may only use this view
window = PredictionWindow(DEFAULT_CUTOFF, horizon_days=90)
```

### Section pipelines

Each completed section has one deterministic command that rebuilds its tables, figures and the
generated results block in its README from `data/raw` (and generates the default data first if
it is missing):

```bash
northstar acquisition                # section 01: lead scoring, lift, budget simulation (~40 s)
northstar retention                  # section 02: churn models, targeting depth, retention ROI (~1 min)
northstar conversion                 # section 03: funnel diagnostics, A/B test analysis (~10 s)
northstar revenue                    # section 04: customer value models, segments, next best action (~30 s)
northstar forecast                   # section 05: 13-week revenue forecast, rolling backtest, intervals (~5 s)
northstar lifecycle                  # section 06: lifecycle states, transitions, cohorts, RFM segments (~10 s)
northstar dashboard-kpis             # section 07: KPI catalog with source fields from the outputs above (~1 s)
northstar train-models               # section 08: train, audit and register the served models (~40 s)
northstar monitor                    # section 08: batch-score current runs, drift + performance report (~20 s)
```

`make results` runs all of the above in order.

## Tests and linting

```bash
python -m pytest                     # full suite, including a full-scale reproducibility check
python -m pytest -m "not slow"       # skip the full-scale reproducibility checks
python -m ruff check .
```

`make setup`, `make data`, `make acquisition`, `make retention`, `make conversion`, `make revenue`,
`make forecast`, `make lifecycle`, `make dashboard-kpis`, `make production` (train models and
monitor), `make results` (data plus every section pipeline, the dashboard KPI snapshot and the
section 08 report), `make dashboard`, `make serve`, `make docker-up`, `make test` and `make lint`
wrap the same commands.

Continuous integration ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs Ruff and the
full test suite in a fresh virtual environment, on Python 3.12 with the locked versions in
[`constraints.txt`](constraints.txt) and on Python 3.11 with the newest allowed versions. It then
builds the Docker image and smoke-tests the running API and dashboard.

## Dashboard and API

The executive sales and marketing dashboard ([section 07](projects/07_dashboard/README.md)) is a
Streamlit app that reads the saved outputs of sections 00-06. The committed outputs are enough,
so it runs from a clean checkout after setup:

```bash
northstar dashboard                  # opens http://localhost:8501 (Ctrl+C to stop)
northstar dashboard --port 8600 --headless   # other port, no browser
streamlit run src/northstar/dashboard/app.py # equivalent
```

It has an executive overview (acquisition, conversion, retention, revenue, lifecycle and
forecast KPIs), one page per section with filters over the saved results, and a definitions
page that lists every KPI's definition and source file and field. No model is refit in the
app. After regenerating results, run `northstar dashboard-kpis` to refresh the KPI snapshot in
the section README.

The **scoring API** ([section 08](projects/08_productionization/README.md)) serves the section 01
lead score and the section 02 churn model from a local, versioned model registry. Train the
models once (they are git-ignored build artifacts), then start the service:

```bash
northstar train-models               # registers models/<name>/<version>/ and LATEST (~40 s)
northstar serve                      # http://127.0.0.1:8000, interactive docs at /docs
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/v1/acquisition/score -H "Content-Type: application/json" \
  -d @projects/08_productionization/outputs/example_requests/acquisition.json
northstar score-batch --model churn --cutoff 2025-12-01   # monthly call list -> scores/
```

Endpoints: `GET /health`, `GET /v1/models` (model cards), `POST /v1/acquisition/score` and
`POST /v1/churn/score` (1-1,000 validated feature records per request). Invalid records return
422 with the field and rule. If a model artifact is missing or fails verification, `serve` refuses
to start and `/health` returns 503.

**Both services in containers** (Docker with Compose v2; no secrets or environment variables
needed). The image generates the data and trains the models at build time (about 3-5 minutes on
first build), so start-up only loads artifacts:

```bash
docker compose up --build --wait     # API on http://localhost:8000, dashboard on http://localhost:8501
docker compose down
```

## Skills demonstrated

| Skill | Where |
|---|---|
| Data modelling and simulation design (entities, keys, realistic causal structure) | `src/northstar/synthetic/`, `src/northstar/schema.py` |
| Leakage-safe, time-aware analysis design (cutoffs, point-in-time snapshots) | `src/northstar/timeline.py` |
| Data quality engineering (schema, referential integrity, business-rule validation) | `src/northstar/validation.py`, `tests/test_validation.py` |
| Experimentation foundations (stable hash randomization, pre-registered metrics) | `experiments` / `experiment_assignments` tables, `synthetic.assignment_bucket` |
| Propensity modelling for imbalanced targeting (baselines, logistic regression, gradient boosting) | `src/northstar/acquisition/`, [section 01](projects/01_acquisition/README.md) |
| Out-of-time validation, runtime leakage audit, clustered bootstrap intervals | `acquisition/dataset.py` (`SplitPlan`, `leakage_audit`), `acquisition/evaluation.py` |
| Calibration, lift/gains and budget (top-N outreach) simulation | `acquisition/evaluation.py`, `projects/01_acquisition/outputs/` |
| Model explainability (coefficients, permutation importance, exact SHAP) | `acquisition/models.py` |
| Churn prediction with explicit observation/prediction windows, purged out-of-time validation | `src/northstar/retention/dataset.py`, [section 02](projects/02_retention/README.md) |
| Calibration monitoring by run and segment; RFM benchmark; per-customer reason codes | `retention/evaluation.py`, `retention/models.py` |
| Decision economics under explicit assumptions (expected-value targeting, break-even, sensitivity) | `retention/simulation.py`, `projects/02_retention/outputs/` |
| Funnel analytics from event logs (ordered stages, integrity checks, segment drop-off, gap sizing) | `src/northstar/conversion/funnel.py`, [section 03](projects/03_conversion/README.md) |
| A/B test analysis: pre-registered plan, SRM and balance audit, z-test with matching CI, guardrails, decision rule | `conversion/experiment.py` (`ExperimentPlan`, `assignment_audit`, `decide`) |
| Power, MDE and retrodesign (exaggeration of underpowered wins); peeking illustration | `conversion/stats.py`, `projects/03_conversion/outputs/` |
| Multiple comparisons (Holm, Benjamini-Hochberg), heterogeneity (Cochran's Q), A/A and permutation checks, CUPED-style regression adjustment | `conversion/stats.py`, `tests/test_conversion_stats.py` |
| Statistical vs business significance (revenue per unit, break-even lift) | [section 03](projects/03_conversion/README.md) |
| Forward-looking customer value with leak-free windows (spend history reconciled against the target period) | `src/northstar/revenue/dataset.py`, [section 04](projects/04_revenue_growth/README.md) |
| Probabilistic CLV from first principles (BG/NBD + Gamma-Gamma maximum likelihood, validated by simulation) | `revenue/clv.py`, `tests/test_revenue_clv.py` |
| Zero-inflated revenue regression (Poisson gradient boosting) vs RFM baselines; RMSE, calibration-in-the-large, normalized Gini, revenue capture | `revenue/models.py`, `revenue/evaluation.py` |
| Value segmentation, value migration and auditable next-best-action rules | `revenue/actions.py`, `projects/04_revenue_growth/outputs/` |
| Growth scenario analysis with explicit assumptions (break-even uplift, plan vs outcome, sensitivity) | `revenue/scenarios.py` |
| Time-series forecasting: daily revenue aggregation, damped-trend harmonic regression with calendar and planned-promotion effects vs. run-rate and last-year-plus-growth baselines | `src/northstar/forecasting/`, [section 05](projects/05_predictive_analytics/README.md) |
| Rolling-origin (expanding-window) backtesting with design/evaluation separation; MAE, WAPE, sMAPE, bias by lead time; Diebold-Mariano tests for overlapping horizons | `forecasting/evaluation.py`, `tests/test_forecasting_evaluation.py` |
| Forecast uncertainty: empirical prediction intervals from past errors, with coverage honestly measured; leakage audit that rebuilds every forecast from truncated data | `forecasting/evaluation.py`, `forecasting/report.py` (`leakage_audit`) |
| Customer lifecycle analytics: explicit, mutually exclusive state definitions with documented precedence; deterministic person-month panel; month-to-month transition matrices with rule-derived structural zeros | `src/northstar/lifecycle/`, [section 06](projects/06_lifecycle/README.md) |
| Cohort retention with fixed acquisition denominators, explicit right-censoring and fixed-cohort pooled curves; RFM segmentation that complements lifecycle states | `lifecycle/cohorts.py`, `lifecycle/segments.py`, `tests/test_lifecycle_cohorts.py` |
| Descriptive opportunity sizing kept separate from causal claims (decision-point rates with Wilson intervals, revenue gaps as upper bounds, integrity audit with conservation checks) | `lifecycle/transitions.py`, `lifecycle/report.py` (`integrity_audit`) |
| Executive dashboard over pipeline outputs (Streamlit, Altair): KPI registry with definitions and source-field traceability, filters that never refit models, cross-section provenance check | `src/northstar/dashboard/`, [section 07](projects/07_dashboard/README.md) |
| Headless UI testing (Streamlit AppTest, real server launch, interaction and tamper tests) | `tests/test_dashboard_app.py` |
| Model serving: FastAPI service with Pydantic contracts (domain, population-scope and feature-identity validation), 422/503 failure behavior, models loaded once | `src/northstar/serving/api.py`, `serving/schemas.py`, [section 08](projects/08_productionization/README.md) |
| Model registry: content-addressed versions, atomic writes, hash-verified loading, promotion and rollback, leakage audit as a registration gate | `serving/registry.py`, `serving/training.py`, `tests/test_serving_registry.py` |
| Batch scoring and training-serving parity (API and batch share one validation and scoring path) | `serving/batch.py`, `serving/scoring.py`, `tests/test_serving_api.py` |
| Model monitoring: PSI input and score drift against stored reference profiles, delayed-label performance with significance-aware alerts, severities and actions | `serving/monitoring.py`, `serving/profiles.py`, `tests/test_serving_monitoring.py` |
| Deployment and MLOps: digest-pinned Docker image, Compose with health checks, locked dependencies, GitHub Actions CI with container smoke tests | `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml` |
| Stakeholder adoption: intended and prohibited uses, owners (RACI), refresh cadence, runbook, feedback loop | [section 08](projects/08_productionization/README.md#stakeholder-adoption), `docs/operations.md` |
| Reproducibility (seeded generation, content-hashed manifests, generated docs) | `northstar generate-data`, `tests/test_generation.py` |
| Stakeholder communication (business framing, traceable results) | `projects/*/README.md` |
| Python engineering (packaging, CLI, tests, lint) | `pyproject.toml`, `src/northstar/cli.py`, `tests/` |

## Repository layout

```text
src/northstar/          shared package (data generation, schema, timeline, validation, CLI)
src/northstar/<section>/ section pipelines (e.g. `acquisition/` for section 01)
src/northstar/dashboard/ section 07 Streamlit app, KPI registry and filters
src/northstar/serving/  section 08 registry, training, API, batch scoring and monitoring
projects/<section>/     stakeholder-facing README and generated outputs per section
docs/                   data dictionary (generated), architecture and operations runbook
data/raw/               generated tables (not committed; `northstar generate-data`)
models/                 model registry (not committed; `northstar train-models`)
tests/                  automated tests
Dockerfile, docker-compose.yml, .github/workflows/ci.yml, constraints.txt   deployment and CI
```
