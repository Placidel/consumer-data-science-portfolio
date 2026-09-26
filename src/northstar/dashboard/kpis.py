"""KPI registry for the executive dashboard: definitions, sources and values.

Every KPI names the section whose saved metrics file holds it and a *field path* into that file,
so each number on the dashboard can be traced to ``projects/<section>/outputs/<file>`` and a
key. Nothing is computed from raw data here and no value is typed in by hand.

Field paths are dot-separated keys with two extras:

* ``models[champion=True]`` selects the first list item whose ``champion`` field equals ``True``
  (values are compared as strings), and ``[model=@config.champion]`` compares against another
  field of the same file;
* integer tokens index lists (``ci.0``; ``-1`` is the last item).

Definitions may embed fields as ``{path}`` or ``{path:fmt}`` so windows and assumptions quoted in
a tooltip always match the run being shown.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from northstar.dashboard.artifacts import SectionArtifacts, load_all, provenance

BEGIN_MARKER = "<!-- BEGIN GENERATED: dashboard-kpis -->"
END_MARKER = "<!-- END GENERATED: dashboard-kpis -->"

_TOKEN = re.compile(r"(-?[A-Za-z0-9_]+)(?:\[([A-Za-z0-9_]+)=([^\]]+)\])?")
_PLACEHOLDER = re.compile(r"\{([^{}:]+)(?::(\w+))?\}")


class FieldError(KeyError):
    """A field path does not exist in a metrics file."""


def _split(path: str) -> list[str]:
    """Split on dots that are not inside ``[...]``."""
    parts, depth, cur = [], 0, ""
    for ch in path:
        if ch == "." and depth == 0:
            parts.append(cur)
            cur = ""
            continue
        depth += {"[": 1, "]": -1}.get(ch, 0)
        cur += ch
    parts.append(cur)
    return parts


def resolve(obj: Any, path: str, root: Any = None) -> Any:
    """Return the value at ``path`` in a parsed metrics file (see module docstring)."""
    root = obj if root is None else root
    cur = obj
    for token in _split(path):
        m = _TOKEN.fullmatch(token)
        if m is None:
            raise FieldError(f"Malformed field path {path!r} at {token!r}")
        name, sel_key, sel_value = m.groups()
        if isinstance(cur, list) and re.fullmatch(r"-?\d+", name):
            try:
                cur = cur[int(name)]
            except IndexError as exc:
                raise FieldError(f"{path!r}: index {name} out of range") from exc
        elif isinstance(cur, Mapping) and name in cur:
            cur = cur[name]
        else:
            raise FieldError(f"{path!r}: no field {name!r}")
        if sel_key is not None:
            want = resolve(root, sel_value[1:]) if sel_value.startswith("@") else sel_value
            matches = [item for item in cur
                       if isinstance(item, Mapping) and str(item.get(sel_key)) == str(want)]
            if not matches:
                raise FieldError(f"{path!r}: no item with {sel_key} == {want!r}")
            cur = matches[0]
    return cur


# --- formatting ------------------------------------------------------------------------------

def fmt_value(value: Any, fmt: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if fmt == "text":
        return str(value).replace("->", " → ").replace("_", " ")
    v = float(value)
    if fmt == "usd":
        sign, v = ("-" if v < 0 else ""), abs(v)
        if v >= 1e6:
            return f"{sign}${v / 1e6:,.2f}M"
        return f"{sign}${v:,.0f}" if v >= 1e3 else f"{sign}${v:,.2f}"
    if fmt == "pct":
        return f"{v:.1%}"
    if fmt == "pp":
        return f"{v * 100:+.1f} pp"
    if fmt == "x":
        return f"{v:.2f}×"
    if fmt == "int":
        return f"{v:,.0f}"
    if fmt == "auc":
        return f"{v:.3f}"
    if fmt == "num":
        return f"{v:,.4g}"
    raise ValueError(f"Unknown format {fmt!r}")


def render_template(text: str, metrics: Mapping) -> str:
    """Fill ``{path}`` / ``{path:fmt}`` placeholders from a metrics file."""
    def sub(m: re.Match) -> str:
        value = resolve(metrics, m.group(1))
        return fmt_value(value, m.group(2)) if m.group(2) else str(value)
    return _PLACEHOLDER.sub(sub, text)


# --- registry --------------------------------------------------------------------------------

@dataclass(frozen=True)
class KPI:
    key: str
    theme: str
    label: str
    section: str
    field: str
    fmt: str
    definition: str
    higher_is_better: bool | None = True
    lower: str | None = None          # interval bounds (field paths in the same file)
    upper: str | None = None
    interval_label: str = "95% CI"
    reference: str | None = None      # comparison value (field path in the same file)
    reference_label: str = ""         # template, e.g. "vs {states.year_ago.period}"
    reference_kind: str = "relative"  # "relative" (% change) or "pp" (difference of shares)


THEMES = ("Acquisition", "Conversion", "Retention", "Revenue", "Lifecycle", "Forecast")

KPIS: tuple[KPI, ...] = (
    # Acquisition --------------------------------------------------------------------------
    KPI("leads", "Acquisition", "Leads generated", "foundation", "headline.leads", "int",
        "Prospects (identified leads) created from {data_start} until the data end "
        "({data_end_exclusive}, exclusive).",
        higher_is_better=None),
    KPI("lead_conversion_60d", "Acquisition", "Lead → customer (60 days)", "foundation",
        "headline.lead_conversion_rate_60d", "pct",
        "Share of leads that place a first order within 60 days of lead creation, among leads "
        "with a complete 60-day window (all channels)."),
    KPI("lead_score_lift_top10", "Acquisition", "Lead score lift, top 10%", "acquisition",
        "models[champion=True].holdout_lift_top10", "x",
        "Conversion rate of the top 10% of open leads ranked by the champion lead score "
        "({champion}) divided by the average rate. Conversion = first order within "
        "{config.horizon_days} days of a monthly scoring run; out-of-time holdout runs "
        "{config.holdout_runs.0} to {config.holdout_runs.-1}."),
    KPI("lead_score_auc", "Acquisition", "Lead score ROC AUC", "acquisition",
        "models[champion=True].holdout_roc_auc", "auc",
        "Holdout ROC AUC of the champion lead score ({champion}); 0.5 is random ranking. "
        "Interval: clustered bootstrap.",
        lower="models[champion=True].holdout_roc_auc_ci_low",
        upper="models[champion=True].holdout_roc_auc_ci_high"),
    # Conversion ---------------------------------------------------------------------------
    KPI("lead_purchase_rate_30d", "Conversion", "Lead → purchase (30 days)", "conversion",
        "funnel.prospect_funnel[stage=purchase].share_of_start", "pct",
        "Share of the {funnel.prospect_cohort.leads:int} leads created {funnel.prospect_cohort."
        "created_from} to {funnel.prospect_cohort.created_before} (exclusive) who reach "
        "purchase within {funnel.prospect_cohort.window_days} days of lead creation."),
    KPI("largest_loss_step", "Conversion", "Largest funnel loss", "conversion",
        "funnel.largest_loss_step", "text",
        "Session funnel step (prospect sessions, {funnel.period_start} to "
        "{funnel.period_end_exclusive}) where the most sessions are lost.",
        higher_is_better=None),
    KPI("checkout_lift", "Conversion", "Checkout test lift", "conversion",
        "experiment.primary.diff", "pp",
        "One-page checkout minus control in first-purchase conversion "
        "({experiment.registry.start_date} to {experiment.registry.end_date}; "
        "{experiment.primary.n_treatment:int} treatment vs {experiment.primary.n_control:int} "
        "control prospects). Decision: {experiment.decision.recommendation_text}",
        lower="experiment.primary.ci.0", upper="experiment.primary.ci.1"),
    KPI("checkout_order_value", "Conversion", "Checkout test: order value", "conversion",
        "experiment.guardrails.first_order_value.relative", "pct",
        "Relative change in average first-order net value, one-page checkout vs control "
        "(guardrail; fails if the decline is worse than "
        "{experiment.guardrails.first_order_value.margin:pct}). Status: "
        "{experiment.guardrails.first_order_value.status}.",
        lower="experiment.guardrails.first_order_value.relative_ci.0",
        upper="experiment.guardrails.first_order_value.relative_ci.1"),
    # Retention ----------------------------------------------------------------------------
    KPI("churn_rate_90d", "Retention", "90-day churn rate", "retention",
        "models[champion=True].holdout_base_rate", "pct",
        "Share of active customers (ordered in the {config.active_days} days before a monthly "
        "run) who place no order in the next {config.horizon_days} days. Holdout runs "
        "{config.holdout_runs.0} to {config.holdout_runs.-1}.",
        higher_is_better=False),
    KPI("churn_auc", "Retention", "Churn model ROC AUC", "retention",
        "models[champion=True].holdout_roc_auc", "auc",
        "Holdout ROC AUC of the champion churn model ({champion}). Interval: clustered "
        "bootstrap.",
        lower="models[champion=True].holdout_roc_auc_ci_low",
        upper="models[champion=True].holdout_roc_auc_ci_high"),
    KPI("retention_net_value", "Retention", "Retention net value / run", "retention",
        "simulation.expected_net_positive.net_value_per_run", "usd",
        "Holdout net margin per monthly run from offering a retention voucher only to customers "
        "with a positive expected value ({simulation.expected_net_positive.depth:pct} of the "
        "active base). Assumes a {simulation.assumptions.save_rate:pct} save rate and "
        "{simulation.assumptions.incentive_cost:usd} incentive; ROI "
        "{simulation.expected_net_positive.roi:x}."),
    # Revenue ------------------------------------------------------------------------------
    KPI("net_revenue", "Revenue", "Net revenue, 24 months", "foundation",
        "headline.net_revenue", "usd",
        "Sum of order net amounts (after discounts) from {data_start} until the data end "
        "({data_end_exclusive}, exclusive)."),
    KPI("average_order_value", "Revenue", "Average order value", "foundation",
        "headline.average_order_value", "usd",
        "Mean net amount per order over the same period."),
    KPI("repeat_revenue_share", "Revenue", "Repeat-order revenue", "foundation",
        "headline.repeat_order_revenue_share", "pct",
        "Share of net revenue from orders other than a customer's first."),
    KPI("revenue_capture_top10", "Revenue", "Top-10% value capture", "revenue",
        "models[champion=True].holdout_capture_top10", "pct",
        "Share of next-{config.horizon_days}-day revenue generated by the 10% of customers "
        "with the highest predicted value ({champion}); holdout run {config.holdout_runs.0}. "
        "A random 10% captures 10%."),
    KPI("growth_net_value", "Revenue", "Growth program value", "revenue",
        "scenario.expected_net_positive.net_value", "usd",
        "Holdout net value (incremental margin minus cost) of offering the growth perk to "
        "customers whose predicted incremental margin covers its cost "
        "({scenario.expected_net_positive.customers_targeted:int} customers). Assumes a "
        "{scenario.assumptions.uplift:pct} revenue uplift; break-even uplift "
        "{scenario.expected_net_positive.break_even_uplift:pct}."),
    # Lifecycle ----------------------------------------------------------------------------
    KPI("second_purchase_rate", "Lifecycle", "Second-purchase rate", "lifecycle",
        "decision_points[decision_point=second_purchase].favourable_rate", "pct",
        "{decision_points[decision_point=second_purchase].label}: share of new customers who "
        "order again (become active or loyal) rather than slipping to at risk, "
        "{transitions.first_origin} to {transitions.last_destination}. Interval: Wilson 95%.",
        lower="decision_points[decision_point=second_purchase].rate_ci_low",
        upper="decision_points[decision_point=second_purchase].rate_ci_high"),
    KPI("loyal_share", "Lifecycle", "Loyal customers", "lifecycle",
        "states.current.customer_shares.loyal", "pct",
        "Share of customers in the Loyal state at the end of {states.current.period}: "
        "{config.definitions.loyal}",
        reference="states.year_ago.customer_shares.loyal",
        reference_label="vs {states.year_ago.period}", reference_kind="pp"),
    KPI("at_risk_recovery", "Lifecycle", "At-risk recovery rate", "lifecycle",
        "decision_points[decision_point=at_risk_recovery].favourable_rate", "pct",
        "{decision_points[decision_point=at_risk_recovery].label}: share of at-risk customers "
        "who order again (return to active or loyal) rather than churning, "
        "{transitions.first_origin} to {transitions.last_destination}. Interval: Wilson 95%.",
        lower="decision_points[decision_point=at_risk_recovery].rate_ci_low",
        upper="decision_points[decision_point=at_risk_recovery].rate_ci_high"),
    KPI("churned_share", "Lifecycle", "Churned customers", "lifecycle",
        "states.current.customer_shares.churned", "pct",
        "Share of all customers ever acquired who are churned at the end of "
        "{states.current.period}: {config.definitions.churned}",
        higher_is_better=False, reference="states.year_ago.customer_shares.churned",
        reference_label="vs {states.year_ago.period}", reference_kind="pp"),
    # Forecast -----------------------------------------------------------------------------
    KPI("forecast_13w", "Forecast", "Revenue, next 13 weeks", "forecast",
        "forecast.total.forecast", "usd",
        "Champion model ({config.champion}) forecast of net revenue for the "
        "{config.plan.horizon_weeks} weeks from {forecast.origin} to {forecast.horizon_end}, "
        "with its 80% empirical prediction interval.",
        lower="forecast.total.lower_80", upper="forecast.total.upper_80",
        interval_label="80% interval", reference="forecast.total.same_weeks_last_year",
        reference_label="vs same weeks last year"),
    KPI("forecast_wape", "Forecast", "Forecast error (WAPE)", "forecast",
        "evaluation.overall[model=@config.champion].wape", "pct",
        "Weighted absolute percentage error of the champion over the rolling-origin "
        "evaluation backtest (all lead weeks). Naive 4-week run rate: "
        "{evaluation.overall[model=naive_4wk].wape:pct}.",
        higher_is_better=False),
)

KPI_BY_KEY = {k.key: k for k in KPIS}

# Two headline tiles per theme on the executive overview.
OVERVIEW: dict[str, tuple[str, str]] = {
    "Acquisition": ("lead_conversion_60d", "lead_score_lift_top10"),
    "Conversion": ("lead_purchase_rate_30d", "checkout_lift"),
    "Retention": ("churn_rate_90d", "retention_net_value"),
    "Revenue": ("net_revenue", "revenue_capture_top10"),
    "Lifecycle": ("second_purchase_rate", "loyal_share"),
    "Forecast": ("forecast_13w", "forecast_wape"),
}


@dataclass(frozen=True)
class KPIValue:
    kpi: KPI
    value: Any
    lower: float | None
    upper: float | None
    reference: float | None
    definition: str
    reference_label: str
    source: str
    error: str | None = None

    @property
    def available(self) -> bool:
        return self.error is None

    @property
    def display(self) -> str:
        return fmt_value(self.value, self.kpi.fmt) if self.available else "n/a"

    @property
    def interval(self) -> str:
        if self.lower is None or self.upper is None:
            return ""
        lo, hi = fmt_value(self.lower, self.kpi.fmt), fmt_value(self.upper, self.kpi.fmt)
        if self.kpi.fmt == "pp":
            lo, hi = lo.removesuffix(" pp"), hi
        return f"{self.kpi.interval_label} {lo} to {hi}"

    @property
    def change(self) -> float | None:
        if self.reference is None or not self.available:
            return None
        if self.kpi.reference_kind == "pp":
            return float(self.value) - float(self.reference)
        return float(self.value) / float(self.reference) - 1 if self.reference else None

    @property
    def comparison(self) -> str:
        change = self.change
        if change is None:
            return ""
        text = (f"{change * 100:+.1f} pp" if self.kpi.reference_kind == "pp"
                else f"{change:+.1%}")
        return f"{text} {self.reference_label}".strip()

    @property
    def tooltip(self) -> str:
        parts = [self.definition]
        if self.interval:
            parts.append(self.interval + ".")
        parts.append(f"Source: `{self.source}` → `{self.kpi.field}`")
        return "\n\n".join(parts)


def evaluate(kpi: KPI, artifacts: Mapping[str, SectionArtifacts]) -> KPIValue:
    art = artifacts[kpi.section]
    source = art.source()
    if not art.metrics:
        return KPIValue(kpi, None, None, None, None, kpi.definition, "", source,
                        error=f"{art.section.metrics_file} not found; run "
                              f"`{art.section.command}`")
    m = art.metrics
    try:
        value = resolve(m, kpi.field)
        lower = float(resolve(m, kpi.lower)) if kpi.lower else None
        upper = float(resolve(m, kpi.upper)) if kpi.upper else None
        reference = float(resolve(m, kpi.reference)) if kpi.reference else None
        definition = render_template(kpi.definition, m)
        ref_label = render_template(kpi.reference_label, m)
    except (FieldError, TypeError, ValueError) as exc:
        return KPIValue(kpi, None, None, None, None, kpi.definition, "", source,
                        error=f"{type(exc).__name__}: {exc}")
    return KPIValue(kpi, value, lower, upper, reference, definition, ref_label, source)


def evaluate_all(artifacts: Mapping[str, SectionArtifacts]) -> dict[str, KPIValue]:
    return {k.key: evaluate(k, artifacts) for k in KPIS}


def kpi_catalog(artifacts: Mapping[str, SectionArtifacts]) -> pd.DataFrame:
    """Every KPI with its value, interval, comparison, definition and source field."""
    rows = []
    for key, v in evaluate_all(artifacts).items():
        rows.append({
            "key": key, "theme": v.kpi.theme, "kpi": v.kpi.label,
            "value": v.value if v.available else None, "display": v.display,
            "interval": v.interval, "comparison": v.comparison,
            "definition": v.definition if v.available else v.error,
            "source": v.source, "field": v.kpi.field,
        })
    return pd.DataFrame(rows)


# --- README results block --------------------------------------------------------------------

def _cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def render_markdown(catalog: pd.DataFrame, prov: pd.DataFrame) -> str:
    seeds = prov.loc[prov["available"], ["seed", "n_prospects"]].drop_duplicates()
    if len(seeds) == 1:
        seed, n = seeds.iloc[0]
        data_line = f"data seed `{int(seed)}`, {int(n):,} prospects"
    else:
        data_line = "sections were produced from **different data runs**"
    lines = [
        f"_Rendered by `northstar dashboard-kpis` from saved section outputs ({data_line}). "
        f"The dashboard reads the same files and fields; the full catalog with definitions "
        f"is [`outputs/kpi_catalog.csv`](outputs/kpi_catalog.csv)._",
        "",
        "**Section outputs read by the dashboard:**",
        "",
        "| Section | Available | Seed | Prospects | Consistent with section 00 | Regenerate with |",
        "|---|---|---:|---:|---|---|",
    ]
    for r in prov.itertuples():
        seed = "" if pd.isna(r.seed) else f"{int(r.seed)}"
        n = "" if pd.isna(r.n_prospects) else f"{int(r.n_prospects):,}"
        lines.append(f"| {r.section} | {'yes' if r.available else 'no'} | {seed} | {n} | "
                     f"{'yes' if r.consistent else 'no'} | `{r.regenerate_with}` |")
    lines += [
        "",
        "**KPIs shown on the dashboard** (headline tiles on the executive overview are marked "
        "★):",
        "",
        "| Theme | KPI | Value | Interval / comparison | Source field |",
        "|---|---|---:|---|---|",
    ]
    headline = {k for pair in OVERVIEW.values() for k in pair}
    for r in catalog.itertuples():
        extra = "; ".join(x for x in (r.interval, r.comparison) if x)
        star = " ★" if r.key in headline else ""
        src = Path(r.source)
        lines.append(f"| {r.theme} | {_cell(r.kpi)}{star} | {_cell(r.display)} | "
                     f"{_cell(extra)} | `{src.parts[-3]}/{src.name}` `{_cell(r.field)}` |")
    return "\n".join(lines)


def write_outputs(out_dir: Path, readme: Path | None = None, root: Path | None = None
                  ) -> pd.DataFrame:
    """Write the KPI catalog and provenance tables, and refresh the README results block."""
    from northstar.profile import update_generated_block

    artifacts = load_all(root)
    catalog, prov = kpi_catalog(artifacts), provenance(artifacts)
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(out_dir / "kpi_catalog.csv", index=False)
    prov.to_csv(out_dir / "provenance.csv", index=False)
    if readme is not None and readme.exists():
        update_generated_block(readme, render_markdown(catalog, prov), BEGIN_MARKER, END_MARKER)
    return catalog
