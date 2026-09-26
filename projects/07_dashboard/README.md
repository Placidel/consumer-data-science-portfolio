# 07 · Executive Sales and Marketing Dashboard

## Business problem

Sections 01-06 each answer one question well, but their results live in READMEs, CSV files and
`metrics.json` files. Sales, marketing and CRM leaders will not read code or JSON. If they get
numbers by asking an analyst, each copy of a number drifts a little from the pipeline that
produced it. They need one place to see the headline KPIs and to explore the trade-offs behind
each recommendation. Every number there must be the one the analysis produced, with its
definition.

**Business question:** how can sales and marketing stakeholders consume the portfolio insights
without reading code?

## Decision supported

The dashboard serves the recurring business reviews where budget and program decisions are
made:

- **Monthly marketing and sales review.** Is the business growing? Which channels bring
  customers that stay? How much better than random is lead prioritization at the capacity the
  sales team actually has?
- **Retention and growth program sizing.** How deep should a retention voucher or growth perk
  go, and which ranking should choose the customers? Each depth is shown with its net value,
  ROI and break-even point, under the assumptions stated on the page.
- **Product and experimentation review.** Should the checkout redesign ship, and why not yet?
  Where in the funnel are prospects lost, and for which segments?
- **Quarterly planning.** What revenue to plan for next quarter, with an interval, and how
  accurate the forecast has been.
- **Lifecycle program portfolio.** Which lifecycle moment loses the most value, and which
  acquisition channels bring customers who keep buying.

It supports those conversations. It does not replace the section analyses. Each page names the
section it comes from, and each section README holds the method, validation and caveats.

## Data used

Only **saved outputs of the section pipelines**. The dashboard never reads `data/raw`, never
refits a model and never recomputes a metric that a pipeline already produced:

| Section | Files read (under `projects/<section>/outputs/`) |
|---|---|
| 00 Foundation | `data_profile.json`, `monthly_kpis.csv`, `channel_summary.csv` |
| 01 Acquisition | `metrics.json`, `model_comparison.csv`, `budget_simulation.csv`, `decile_lift.csv` |
| 02 Retention | `metrics.json`, `model_comparison.csv`, `retention_value_curve.csv`, `segment_drivers.csv` |
| 03 Conversion | `metrics.json`, `funnel_prospect_cohort.csv`, `funnel_segments.csv`, `experiment_results.csv`, `cumulative_effect.csv` |
| 04 Revenue growth | `metrics.json`, `model_comparison.csv`, `value_tiers.csv`, `next_best_action.csv`, `growth_value_curve.csv` |
| 05 Forecast | `metrics.json`, `weekly_revenue.csv`, `forecast.csv`, `accuracy_by_horizon.csv` |
| 06 Lifecycle | `metrics.json`, `state_counts_by_month.csv`, `decision_points.csv`, `decision_points_by_channel.csv`, `cohort_retention_by_channel.csv` |

All data is synthetic (section 00); there is no customer-level personal data. Customer IDs appear
only in the section 02 and 04 samples, and the dashboard does not display them.

## Method

Code: [`src/northstar/dashboard/`](../../src/northstar/dashboard)
([`artifacts.py`](../../src/northstar/dashboard/artifacts.py),
[`kpis.py`](../../src/northstar/dashboard/kpis.py),
[`views.py`](../../src/northstar/dashboard/views.py),
[`app.py`](../../src/northstar/dashboard/app.py)).

1. **Artifact loading** (`artifacts.py`). Each section declares the files the dashboard needs.
   A missing file never crashes the app. The section is marked unavailable, its tiles show
   `n/a`, and its page names the command that regenerates it. Loads are cached on each file's
   size and modification time, so rerunning a pipeline refreshes an open dashboard.
2. **Provenance.** Each section's `metrics.json` records the data seed and population it was
   built from. The dashboard compares them with section 00 and warns in the sidebar if any
   section was built from a different data run.
3. **KPI registry** (`kpis.py`). Each KPI is declared once: label, theme, a **definition**, the
   **section file** it lives in and a **field path** into that file (for example
   `models[champion=True].holdout_lift_top10`), plus optional interval bounds and a comparison
   value. Values are resolved from the saved files at run time. Definitions quote windows and
   assumptions through placeholders (`{config.horizon_days}`), so a tooltip cannot contradict
   the run it describes. Nothing is typed in by hand. A path that does not resolve is an error,
   not a guess.
4. **Pages** (`app.py`), one per analytical section, plus an overview and a reference page:

   | Page | What it shows | Interactive controls (all slice saved outputs) |
   |---|---|---|
   | Executive overview | Two headline KPIs per theme (acquisition, conversion, retention, revenue, lifecycle, forecast); monthly revenue and new customers; one "where to act" line per section | Month range → window totals and charts |
   | Acquisition | Channel conversion and cost per customer; lead-score outreach simulation; lift by decile | Channel filter; outreach capacity (10/20/30% of open leads) |
   | Conversion | Lead funnel; session funnel by segment; checkout experiment results, guardrails, decision and cumulative lift | Segment dimension |
   | Retention | Net value per run by targeting depth and policy; churn by customer attribute | Targeting depth; policies; attribute |
   | Revenue growth | Value tiers; next-best-action groups; growth program net value by depth | Targeting depth; ranking models |
   | Forecast | Weekly actuals, 13-week forecast with interval band, backtest accuracy | Interval level (50/80%); history length; baseline overlays |
   | Lifecycle | State mix by month; decision points with Wilson CIs and revenue gaps; cohort retention by channel | Count/share toggle; channels (up to 4); measure |
   | Definitions & sources | Every KPI with value, interval, comparison, definition, source file and field; section provenance | Theme filter |

5. **Filters without recomputation** (`views.py`). Every control selects rows of a table a
   pipeline already saved. Targeting depths snap to the depth grid the pipeline evaluated, the
   forecast interval switches between the two saved levels, and the month window sums saved
   monthly KPIs. Interactions therefore take milliseconds and can only show results that were
   validated upstream. The filter functions are plain pandas and are tested without Streamlit.
6. **Design choices.** The layout is wide, with two theme cards per row on the overview so
   labels and intervals are not truncated on a 1280-1440 px laptop screen. Charts are one
   series or a few series, with no dual axes. Colours come from the same validated palette as
   the section figures and are assigned per entity, so a filter never repaints a surviving
   series. Baselines and "random" are drawn in neutral grey. Every chart has hover tooltips
   and a legend when it shows two or more series. Each chart or table cites its source file
   underneath. KPI tooltips (the "?" icon) give the definition, interval and source field.
7. **Traceability snapshot.** `northstar dashboard-kpis` writes the resolved catalog to
   [`outputs/kpi_catalog.csv`](outputs/kpi_catalog.csv) and section provenance to
   [`outputs/provenance.csv`](outputs/provenance.csv), and renders the results block below.
   The dashboard and this README therefore show the same values from the same fields.

## Validation design

Tests in `tests/test_dashboard_app.py`, `tests/test_dashboard_kpis.py` and
`tests/test_dashboard_views.py`:

- **Launches headlessly.** Streamlit's `AppTest` runs the app script with no browser, and
  `northstar dashboard --headless` is started as a real server whose health endpoint and page
  must answer.
- **Every page renders without exceptions or error elements**, with complete outputs and with
  one section's outputs deleted. In the second case the overview shows `n/a`, the sidebar
  warns, and the page names `northstar retention`.
- **Interactions are covered.** Changing the month range must reproduce window totals summed
  directly from `monthly_kpis.csv`. The retention depth slider must show the net value saved in
  `retention_value_curve.csv` at that depth. The channel filter, outreach capacity, forecast
  interval level and every other control must re-render cleanly.
- **Nothing is hard-coded.** A test edits a private copy of the saved outputs (net revenue and
  churn rate set to sentinel values) and requires the dashboard to display the sentinels.
- **Traceability.** Every registered KPI must resolve against the committed outputs with no
  unfilled placeholder. A sample from every section is checked against independent lookups of
  the JSON/CSV files. Intervals and comparisons must equal the saved bounds and reference
  values. Sections built from a different seed are flagged.
- **Documentation stays in sync.** The results block below and `outputs/kpi_catalog.csv` must
  equal a fresh rendering from the committed section outputs, as for sections 00-06.

## Results generated from the current run

The dashboard's numbers are the section outputs themselves. The block below is the snapshot
written by `northstar dashboard-kpis`: which outputs were read, whether they come from the same
data run, and every KPI the dashboard displays with its source field.

<!-- BEGIN GENERATED: dashboard-kpis -->
_Rendered by `northstar dashboard-kpis` from saved section outputs (data seed `20240101`, 40,000 prospects). The dashboard reads the same files and fields; the full catalog with definitions is [`outputs/kpi_catalog.csv`](outputs/kpi_catalog.csv)._

**Section outputs read by the dashboard:**

| Section | Available | Seed | Prospects | Consistent with section 00 | Regenerate with |
|---|---|---:|---:|---|---|
| 00 Foundation and synthetic data | yes | 20240101 | 40,000 | yes | `northstar generate-data` |
| 01 Customer acquisition | yes | 20240101 | 40,000 | yes | `northstar acquisition` |
| 02 Retention and churn | yes | 20240101 | 40,000 | yes | `northstar retention` |
| 03 Conversion and experimentation | yes | 20240101 | 40,000 | yes | `northstar conversion` |
| 04 Revenue growth and customer value | yes | 20240101 | 40,000 | yes | `northstar revenue` |
| 05 Revenue forecast | yes | 20240101 | 40,000 | yes | `northstar forecast` |
| 06 Customer lifecycle | yes | 20240101 | 40,000 | yes | `northstar lifecycle` |

**KPIs shown on the dashboard** (headline tiles on the executive overview are marked ★):

| Theme | KPI | Value | Interval / comparison | Source field |
|---|---|---:|---|---|
| Acquisition | Leads generated | 40,000 |  | `00_foundation/data_profile.json` `headline.leads` |
| Acquisition | Lead → customer (60 days) ★ | 30.0% |  | `00_foundation/data_profile.json` `headline.lead_conversion_rate_60d` |
| Acquisition | Lead score lift, top 10% ★ | 3.86× |  | `01_acquisition/metrics.json` `models[champion=True].holdout_lift_top10` |
| Acquisition | Lead score ROC AUC | 0.810 | 95% CI 0.801 to 0.820 | `01_acquisition/metrics.json` `models[champion=True].holdout_roc_auc` |
| Conversion | Lead → purchase (30 days) ★ | 24.9% |  | `03_conversion/metrics.json` `funnel.prospect_funnel[stage=purchase].share_of_start` |
| Conversion | Largest funnel loss | product view → add to cart |  | `03_conversion/metrics.json` `funnel.largest_loss_step` |
| Conversion | Checkout test lift ★ | +2.9 pp | 95% CI +0.6 to +5.1 pp | `03_conversion/metrics.json` `experiment.primary.diff` |
| Conversion | Checkout test: order value | -14.5% | 95% CI -20.2% to -8.3% | `03_conversion/metrics.json` `experiment.guardrails.first_order_value.relative` |
| Retention | 90-day churn rate ★ | 47.8% |  | `02_retention/metrics.json` `models[champion=True].holdout_base_rate` |
| Retention | Churn model ROC AUC | 0.811 | 95% CI 0.803 to 0.817 | `02_retention/metrics.json` `models[champion=True].holdout_roc_auc` |
| Retention | Retention net value / run ★ | $14,764 |  | `02_retention/metrics.json` `simulation.expected_net_positive.net_value_per_run` |
| Revenue | Net revenue, 24 months ★ | $5.40M |  | `00_foundation/data_profile.json` `headline.net_revenue` |
| Revenue | Average order value | $84.13 |  | `00_foundation/data_profile.json` `headline.average_order_value` |
| Revenue | Repeat-order revenue | 81.2% |  | `00_foundation/data_profile.json` `headline.repeat_order_revenue_share` |
| Revenue | Top-10% value capture ★ | 40.6% |  | `04_revenue_growth/metrics.json` `models[champion=True].holdout_capture_top10` |
| Revenue | Growth program value | $8,976 |  | `04_revenue_growth/metrics.json` `scenario.expected_net_positive.net_value` |
| Lifecycle | Second-purchase rate ★ | 46.6% | 95% CI 45.4% to 47.8% | `06_lifecycle/metrics.json` `decision_points[decision_point=second_purchase].favourable_rate` |
| Lifecycle | Loyal customers ★ | 15.4% | +4.6 pp vs 2024-12 | `06_lifecycle/metrics.json` `states.current.customer_shares.loyal` |
| Lifecycle | At-risk recovery rate | 32.0% | 95% CI 30.9% to 33.1% | `06_lifecycle/metrics.json` `decision_points[decision_point=at_risk_recovery].favourable_rate` |
| Lifecycle | Churned customers | 40.3% | +19.9 pp vs 2024-12 | `06_lifecycle/metrics.json` `states.current.customer_shares.churned` |
| Forecast | Revenue, next 13 weeks ★ | $1.10M | 80% interval $1.07M to $1.22M; +62.6% vs same weeks last year | `05_predictive_analytics/metrics.json` `forecast.total.forecast` |
| Forecast | Forecast error (WAPE) ★ | 10.0% |  | `05_predictive_analytics/metrics.json` `evaluation.overall[model=@config.champion].wape` |
<!-- END GENERATED: dashboard-kpis -->

## Business interpretation

- **One number, one definition.** Executives see the same values the analysts validated, and
  the "?" tooltip gives the definition and source field. A disputed number can be traced to the
  file, field and pipeline command in seconds, which settles "whose number is right" before
  it starts.
- **Trade-offs rather than point answers.** The retention and growth pages show how net value
  moves with targeting depth and ranking, and the acquisition page shows outreach capacity. A
  leader can see how much is lost by going deeper, or by ranking with a simple rule instead of
  a model, and where a program stops paying back. Every figure is from the holdout period and
  sits next to its assumptions.
- **Decisions are shown with their caveats.** The checkout experiment is shown as the
  pre-registered decision (hold for the guardrail), not as a "+x% conversion" headline.
  Lifecycle revenue gaps are labelled upper bounds, not program effects. The forecast is
  always shown with its interval.
- **Adoption path.** Section owners keep their pipelines. The dashboard adds no new modelling
  that could drift, so adding a KPI means one registry entry pointing at a field a pipeline
  already writes.

## Limitations and next steps

- **Snapshot, not live.** Outputs come from a batch run over the synthetic history. In
  production the section pipelines would run on a schedule (section 08) and the dashboard
  would read their latest versioned outputs, from object storage or a warehouse table rather
  than the repository.
- **Filters are limited to what pipelines saved.** Channel filters cannot cut the lead-scoring
  or churn results by channel, because those outputs are not saved by channel. Adding such a
  cut means saving it in the section pipeline, where it is validated, not computing it in the
  app.
- **Assumption scenarios are fixed.** Retention and growth economics are shown at the
  assumptions and sensitivity grid the sections evaluated. A free-form "what if the save rate
  were 22%?" input would need the scenario functions (cheap, no model refit) exposed to the
  app with their own tests.
- **No access control or audit log.** Streamlit is suitable for an internal tool or a demo. A
  company rollout would add single sign-on, row-level permissions for any customer-level
  views, and usage analytics to measure adoption.
- **Visual regression is checked by hand.** Tests verify content and interactions, not pixels.
  Layout was reviewed in a headless browser at 1440 px.

## Reproduction commands

From the repository root, with the virtual environment activated (see the root README):

```bash
make results                            # data plus every section pipeline and the KPI snapshot
# or step by step:
northstar generate-data                 # shared data (section 00)
northstar acquisition && northstar retention && northstar conversion \
  && northstar revenue && northstar forecast && northstar lifecycle
northstar dashboard-kpis                # rebuild outputs/ and the results block above (~1 s)

northstar dashboard                     # launch at http://localhost:8501 (Ctrl+C to stop)
northstar dashboard --port 8600 --headless   # other port, no browser (servers, CI)
streamlit run src/northstar/dashboard/app.py # equivalent, using .streamlit/config.toml

python -m pytest tests/test_dashboard_app.py tests/test_dashboard_kpis.py \
  tests/test_dashboard_views.py
```

The committed outputs of sections 00-06 are enough to launch the dashboard from a clean
checkout; regenerating them is only needed after changing data or code. `--projects-dir`
(both commands) or the `NORTHSTAR_PROJECTS_DIR` environment variable points the dashboard at
another folder of section outputs. `northstar dashboard-kpis --strict` exits with an error if
any KPI cannot be read, which is useful as a pipeline check.
