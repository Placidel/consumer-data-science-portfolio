"""Section 02 pipeline: build the churn dataset, audit leakage, select and evaluate models, explain
the champion, simulate a retention budget, and write tables, figures and the README results block.

Everything reported in ``projects/02_retention/README.md`` between the generated-block markers is
rendered from ``outputs/metrics.json`` by :func:`render_markdown`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from northstar.acquisition.evaluation import cluster_bootstrap, gains_curve, per_run_auc
from northstar.acquisition.report import LeakageError
from northstar.profile import GRID, TEXT_PRIMARY, TEXT_SECONDARY, _style, update_generated_block
from northstar.retention import evaluation as ev
from northstar.retention import simulation as sim
from northstar.retention.dataset import (
    ACTIVE_DAYS,
    DEFAULT_SPLIT,
    FEATURES,
    TARGET,
    VALUE,
    SplitPlan,
    build_dataset,
    leakage_audit,
)
from northstar.retention.models import (
    BASELINES,
    LEARNED,
    MODEL_LABELS,
    MODEL_NAMES,
    PROBABILISTIC,
    SEED,
    fit_model,
    logistic_coefficients,
    predict,
    reason_codes,
    shap_importance,
)

__all__ = ["LeakageError", "RetentionConfig", "render_markdown", "run_analysis", "write_outputs"]

BEGIN_MARKER = "<!-- BEGIN GENERATED: retention-results -->"
END_MARKER = "<!-- END GENERATED: retention-results -->"
SELECTION_METRIC = "log_loss"  # proper scoring rule: the simulation needs calibrated risk

# Reference categorical palette (fixed slot order) plus a muted ink for the random reference.
COLORS = {"champion": "#2a78d6", "recency_rule": "#eb6834", "rfm_cell_rate": "#1baf7a",
          "other_model": "#eda100", "random": "#898781"}
POLICY_LABELS = {
    "random": "Random (no targeting)",
    "recency_rule": MODEL_LABELS["recency_rule"],
    "rfm_cell_rate": MODEL_LABELS["rfm_cell_rate"],
    "risk_ranked": "Champion, ranked by churn risk",
    "value_ranked": "Champion, ranked by expected net value",
    "expected_net_positive": "Champion, every customer with expected net value > 0",
}


@dataclass(frozen=True)
class RetentionConfig:
    split: SplitPlan = DEFAULT_SPLIT
    depths: tuple[float, ...] = ev.DEPTHS
    n_bootstrap: int = 200
    shap_sample: int = 2000
    permutation_repeats: int = 5
    assumptions: sim.RetentionAssumptions = field(default_factory=sim.RetentionAssumptions)
    sensitivity_depth: float = 0.1
    sensitivity_save_rates: tuple[float, ...] = (0.05, 0.10, 0.15, 0.25)
    sensitivity_incentives: tuple[float, ...] = (5.0, 10.0, 20.0)
    n_reason_codes: int = 15
    curve_depths: tuple[float, ...] = field(
        default_factory=lambda: tuple(np.round(np.linspace(0, 0.6, 31), 2)))
    gains_shares: tuple[float, ...] = field(
        default_factory=lambda: tuple(np.round(np.linspace(0, 1, 51), 2)))


def _clean(obj):
    """JSON-safe copy with floats rounded (stable diffs; NaN -> None, inf -> "inf")."""
    if isinstance(obj, Mapping):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.integer | bool | np.bool_):
        return obj.item() if isinstance(obj, np.generic) else obj
    if isinstance(obj, float | np.floating):
        if np.isnan(obj):
            return None
        if np.isinf(obj):
            return "inf"
        return round(float(obj), 4)
    if isinstance(obj, pd.Timestamp):
        return str(obj.date())
    return obj


def _split_summary(data: pd.DataFrame, plan: SplitPlan) -> list[dict]:
    rows = []
    for split, runs in (("fit", plan.fit), ("validation", plan.validation),
                        ("holdout", plan.holdout)):
        d = data.loc[data["split"] == split]
        rows.append({"split": split, "runs": len(runs), "first_run": str(min(runs).date()),
                     "last_run": str(max(runs).date()), "customer_runs": len(d),
                     "unique_customers": int(d["customer_id"].nunique()),
                     "churners": int(d[TARGET].sum()), "churn_rate": d[TARGET].mean()})
    return rows


def _per_run(hold: pd.DataFrame, champion: str) -> list[dict]:
    auc = per_run_auc(hold, champion, TARGET)
    return [{"run": str(cutoff.date()), "active_customers": len(run),
             "observed_churn_rate": run[TARGET].mean(), "mean_predicted": run[champion].mean(),
             "roc_auc": auc[str(cutoff.date())]}
            for cutoff, run in hold.groupby("run_cutoff")]


def _reason_code_table(model, hold: pd.DataFrame, train: pd.DataFrame, champion: str,
                       n: int) -> pd.DataFrame:
    """Top of the latest run's value-ranked list with the features pushing each risk up."""
    latest = hold.loc[hold["run_cutoff"] == hold["run_cutoff"].max()]
    top = latest.sort_values(["value_ranked", "customer_id"], ascending=[False, True]).head(n)
    _, per_feature = shap_importance(model, top, train, sample_size=len(top))
    return pd.DataFrame({
        "run_cutoff": top["run_cutoff"].dt.date.astype(str),
        "customer_id": top["customer_id"],
        "churn_probability": top[champion],
        "margin_180d": top[VALUE],
        "expected_net_value": top["value_ranked"],
        "days_since_last_order": top["days_since_last_order"],
        "orders_total": top["orders_total"],
        "top_risk_drivers": reason_codes(per_feature.loc[top.index]),
        "churned_in_next_90d": top[TARGET],
    }).reset_index(drop=True)


def run_analysis(tables: Mapping[str, pd.DataFrame], config: RetentionConfig | None = None
                 ) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Run the full section 02 analysis; returns (metrics dict, output tables)."""
    config = config or RetentionConfig()
    plan = config.split
    assumptions = config.assumptions
    data = build_dataset(tables, plan.fit + plan.validation + plan.holdout,
                         horizon_days=plan.horizon_days)
    data["split"] = data["run_cutoff"].map(plan.role)

    audit = leakage_audit(tables, data, plan)
    if not audit["passed"]:
        failed = [k for k, ok in audit["checks"].items() if not ok]
        raise LeakageError(f"Leakage audit failed: {failed}; details: {audit['details']}")

    fit = data.loc[data["split"] == "fit"]
    val = data.loc[data["split"] == "validation"].copy()
    train = data.loc[data["split"].isin(["fit", "validation"])]
    hold = data.loc[data["split"] == "holdout"].copy()

    # Stage 1 - model selection on later validation runs (lowest log loss among probabilistic).
    validation = {}
    for name in MODEL_NAMES:
        val[name] = predict(fit_model(name, fit), val)
        validation[name] = ev.score_metrics(val, name, name in PROBABILISTIC, config.depths)
    champion = min(PROBABILISTIC, key=lambda n: validation[n][SELECTION_METRIC])

    # Stage 2 - refit every model on all training runs, evaluate once on the holdout runs.
    models = {name: fit_model(name, train) for name in MODEL_NAMES}
    for name, model in models.items():
        hold[name] = predict(model, hold)
    holdout = {name: ev.score_metrics(hold, name, name in PROBABILISTIC, config.depths)
               for name in MODEL_NAMES}
    intervals = cluster_bootstrap(hold, MODEL_NAMES, references=BASELINES,
                                  n_boot=config.n_bootstrap, seed=SEED, cluster="customer_id",
                                  target=TARGET)

    comparison = pd.DataFrame([
        {"model": name, "role": "baseline" if name in BASELINES else "model",
         "champion": name == champion,
         **{f"validation_{k}": validation[name].get(k) for k in ("roc_auc", "log_loss")},
         **{f"holdout_{k}": v for k, v in holdout[name].items()},
         "holdout_roc_auc_ci_low": intervals[f"{name}.roc_auc"][0],
         "holdout_roc_auc_ci_high": intervals[f"{name}.roc_auc"][1],
         "holdout_ap_ci_low": intervals[f"{name}.average_precision"][0],
         "holdout_ap_ci_high": intervals[f"{name}.average_precision"][1]}
        for name in MODEL_NAMES])
    deltas = {
        f"{champion}_minus_{ref}": {
            "roc_auc": holdout[champion]["roc_auc"] - holdout[ref]["roc_auc"],
            "roc_auc_ci": intervals.get(f"{champion}.roc_auc_minus_{ref}"),
            "average_precision": (holdout[champion]["average_precision"]
                                  - holdout[ref]["average_precision"]),
            "average_precision_ci": intervals.get(f"{champion}.average_precision_minus_{ref}"),
        }
        for ref in BASELINES if ref != champion
    }

    calibration = ev.reliability(hold, PROBABILISTIC)
    depth = ev.depth_table(hold, {name: name for name in MODEL_NAMES}, config.depths)
    deciles = ev.decile_table(hold, champion)
    drivers = ev.segment_drivers(hold, champion)
    shares = np.array(config.gains_shares)
    gain_policies = dict.fromkeys([*BASELINES, champion])
    gains = pd.DataFrame({"share_targeted": shares, "random": shares,
                          **{n: gains_curve(hold, n, shares, TARGET) for n in gain_policies}})

    # Explainability: SHAP for both learned models (agreement is a robustness check), centered
    # logistic coefficients and permutation importance (holdout ROC AUC drop) for the champion.
    coefficients = logistic_coefficients(models["logistic_regression"])
    importance = pd.DataFrame({"feature": list(FEATURES)})
    for name in LEARNED:
        summary, _ = shap_importance(models[name], hold, train, config.shap_sample)
        summary = summary.rename(columns={"mean_abs_shap": f"shap_{name}",
                                          "direction": f"direction_{name}"})
        importance = importance.merge(summary, on="feature", how="left")
    lead = champion if champion in LEARNED else LEARNED[0]
    if champion in LEARNED:
        perm = permutation_importance(
            models[champion], hold[list(FEATURES)], hold[TARGET], scoring="roc_auc",
            n_repeats=config.permutation_repeats, random_state=SEED)
        importance["permutation_auc_drop"] = perm.importances_mean
        importance["permutation_auc_drop_std"] = perm.importances_std
    importance = importance.sort_values(f"shap_{lead}", ascending=False).reset_index(drop=True)

    # Retention budget simulation under explicit assumptions (see simulation.py).
    hold["risk_ranked"] = hold[champion]
    hold["value_ranked"] = sim.value_score(hold, champion, assumptions)
    policies = {"random": None, "recency_rule": "recency_rule", "rfm_cell_rate": "rfm_cell_rate",
                "risk_ranked": "risk_ranked", "value_ranked": "value_ranked"}
    budget = sim.simulate_policies(hold, policies, config.depths, assumptions)
    curve = sim.simulate_policies(hold, policies, config.curve_depths, assumptions)
    threshold = sim.threshold_policy(hold, champion, assumptions)
    sensitivity = sim.sensitivity(hold, champion, config.sensitivity_depth, assumptions,
                                  config.sensitivity_save_rates, config.sensitivity_incentives)
    reasons = (_reason_code_table(models[lead], hold, train, lead, config.n_reason_codes)
               if lead == champion else pd.DataFrame())

    metrics = {
        "config": {**plan.as_dict(), "active_days": ACTIVE_DAYS,
                   "depths": list(config.depths), "n_bootstrap": config.n_bootstrap,
                   "shap_sample": config.shap_sample, "seed": SEED, "features": list(FEATURES)},
        "splits": _split_summary(data, plan),
        "leakage_audit": audit,
        "champion": champion,
        "selection_metric": f"validation {SELECTION_METRIC.replace('_', ' ')}",
        "models": comparison.to_dict(orient="records"),
        "champion_vs_baselines": deltas,
        "champion_per_run": _per_run(hold, champion),
        "targeting_depths": depth.to_dict(orient="records"),
        "deciles": deciles.to_dict(orient="records"),
        "segment_drivers": drivers.to_dict(orient="records"),
        "feature_importance": importance.head(12).to_dict(orient="records"),
        "logistic_top_coefficients": coefficients.head(8).to_dict(orient="records"),
        "simulation": {
            "assumptions": assumptions.as_dict(),
            "value_definition": f"{assumptions.value_multiplier:g} x trailing 180-day gross margin"
                                " (net line revenue minus unit cost), floored at 0",
            "holdout_mean_value_churners": hold.loc[hold[TARGET] == 1, VALUE].mean(),
            "holdout_mean_value_non_churners": hold.loc[hold[TARGET] == 0, VALUE].mean(),
            "policies": budget.to_dict(orient="records"),
            "expected_net_positive": threshold,
            "sensitivity": sensitivity.to_dict(orient="records"),
            "sensitivity_depth": config.sensitivity_depth,
        },
    }
    tables_out = {"model_comparison": comparison, "calibration": calibration,
                  "targeting_depths": depth, "decile_table": deciles, "gains_curve": gains,
                  "segment_drivers": drivers, "feature_importance": importance,
                  "logistic_coefficients": coefficients, "retention_simulation": budget,
                  "retention_value_curve": curve, "roi_sensitivity": sensitivity,
                  "reason_codes_sample": reasons}
    return _clean(metrics), tables_out


# ---------------------------------------------------------------- figures
def _color(name: str, champion: str) -> str:
    if name in (champion, "risk_ranked", "value_ranked"):
        return COLORS["champion"]
    return COLORS.get(name, COLORS["other_model"])


def _finish(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, metadata={"Software": None})
    plt.close(fig)
    return path


def _axes_labels(ax: plt.Axes, x: str, y: str) -> None:
    ax.set_xlabel(x, color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel(y, color=TEXT_SECONDARY, fontsize=9)
    ax.set_axisbelow(True)


def save_figures(metrics: Mapping, out: Mapping[str, pd.DataFrame], fig_dir: Path) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    champion = metrics["champion"]
    runs = metrics["config"]["holdout_runs"]
    period = f"{len(runs)} monthly runs, {runs[0]} to {runs[-1]}"
    paths = []

    # Gains: share of churners reached vs share of each run's active base targeted.
    gains = out["gains_curve"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=120)
    ax.plot(gains["share_targeted"] * 100, gains["random"] * 100, color=COLORS["random"],
            linewidth=1.5, linestyle="--", label="Random")
    for name in [c for c in gains.columns if c not in ("share_targeted", "random")]:
        ax.plot(gains["share_targeted"] * 100, gains[name] * 100, color=_color(name, champion),
                linewidth=2, label=MODEL_LABELS[name] + (" (champion)" if name == champion else ""))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True, color=GRID, linewidth=0.8)
    _axes_labels(ax, "Share of active base targeted (%)", "Share of 90-day churners reached (%)")
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=TEXT_PRIMARY)
    _style(ax, "Cumulative gains on the out-of-time holdout",
           f"{period}; customers ranked within each run")
    paths.append(_finish(fig, fig_dir / "gains_curve.png"))

    # Churn rate by champion risk decile.
    deciles = out["decile_table"]
    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=120)
    rate = deciles["churn_rate"] * 100
    ax.bar(deciles["decile"], rate, color=COLORS["champion"], width=0.7)
    base = next(s["churn_rate"] for s in metrics["splits"] if s["split"] == "holdout") * 100
    ax.axhline(base, color=COLORS["random"], linewidth=1.2, linestyle="--")
    ax.text(10.4, base + 1, f"average {base:.0f}%", color=TEXT_SECONDARY, fontsize=9,
            va="bottom", ha="right")
    for x, v in zip(deciles["decile"], rate, strict=True):
        ax.text(x, v + 1, f"{v:.0f}%", ha="center", fontsize=8, color=TEXT_PRIMARY)
    ax.set_xticks(deciles["decile"])
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    _axes_labels(ax, "Churn-risk decile within run (1 = highest risk)", "90-day churn rate (%)")
    _style(ax, f"Observed churn by risk decile - {MODEL_LABELS[champion].lower()}",
           f"Out-of-time holdout, {period}")
    paths.append(_finish(fig, fig_dir / "decile_churn.png"))

    # Reliability diagram.
    cal = out["calibration"]
    fig, ax = plt.subplots(figsize=(6.2, 4.6), dpi=120)
    ax.plot([0, 100], [0, 100], color=COLORS["random"], linestyle="--", linewidth=1.2,
            label="Perfect calibration")
    for name, group in cal.groupby("model", sort=False):
        ax.plot(group["mean_predicted"] * 100, group["observed_rate"] * 100, marker="o",
                markersize=5, linewidth=2, color=_color(name, champion),
                label=MODEL_LABELS[name])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True, color=GRID, linewidth=0.8)
    _axes_labels(ax, "Mean predicted churn probability (%)", "Observed churn rate (%)")
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=TEXT_PRIMARY)
    _style(ax, "Calibration on the out-of-time holdout", "Ten equal-size bins of predicted risk")
    paths.append(_finish(fig, fig_dir / "calibration.png"))

    # Net value vs targeting depth, per policy (assumption-driven).
    curve = out["retention_value_curve"]
    a = metrics["simulation"]["assumptions"]
    fig, ax = plt.subplots(figsize=(9.5, 4.4), dpi=120)
    styles = {"random": ("--", COLORS["random"]), "recency_rule": ("-", COLORS["recency_rule"]),
              "rfm_cell_rate": ("-", COLORS["rfm_cell_rate"]),
              "risk_ranked": (":", COLORS["champion"]), "value_ranked": ("-", COLORS["champion"])}
    for policy, (ls, color) in styles.items():
        sub = curve.loc[curve["policy"] == policy]
        ax.plot(sub["depth"] * 100, sub["net_value_per_run"] / 1000, linestyle=ls, color=color,
                linewidth=2, label=POLICY_LABELS[policy])
    thr = metrics["simulation"]["expected_net_positive"]
    ax.plot([thr["depth"] * 100], [thr["net_value_per_run"] / 1000], marker="o", markersize=8,
            color=COLORS["champion"], markeredgecolor="white", markeredgewidth=2, linestyle="none",
            label="Stop where expected net value turns negative")
    ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
    ax.grid(True, color=GRID, linewidth=0.8)
    _axes_labels(ax, "Share of active base targeted per run (%)",
                 "Net value per monthly run ($ thousands)")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              labelcolor=TEXT_PRIMARY)
    # "\$" keeps matplotlib from reading the dollar signs as mathtext delimiters.
    _style(ax, "Retention budget simulation (ASSUMED economics)",
           f"Save rate {a['save_rate']:.0%}, contact \\${a['contact_cost']:.2f}, voucher "
           f"\\${a['incentive_cost']:.0f}, {a['nonchurner_redemption']:.0%} redemption by "
           "non-churners; out-of-time holdout")
    paths.append(_finish(fig, fig_dir / "retention_value_curve.png"))

    # SHAP importance: champion (or the logistic model if a baseline won) next to the challenger.
    imp = out["feature_importance"]
    lead = champion if champion in LEARNED else LEARNED[0]
    other = next(n for n in LEARNED if n != lead)
    top = imp.dropna(subset=[f"shap_{lead}"]).head(12)[::-1]
    ys = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=120)
    ax.barh(ys + 0.19, top[f"shap_{lead}"], height=0.36, color=COLORS["champion"],
            label=MODEL_LABELS[lead] + (" (champion)" if lead == champion else ""))
    ax.barh(ys - 0.19, top[f"shap_{other}"], height=0.36, color=COLORS["other_model"],
            label=MODEL_LABELS[other])
    ax.set_yticks(ys)
    ax.set_yticklabels(top["feature"])
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    _axes_labels(ax, "Mean |SHAP value| (log-odds of churn)", "")
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=TEXT_PRIMARY)
    _style(ax, "What drives the churn score",
           "Exact SHAP on a holdout sample; one-hot columns summed per feature")
    paths.append(_finish(fig, fig_dir / "shap_importance.png"))
    return paths


# ---------------------------------------------------------------- README rendering
def _pct(v: float | None, digits: int = 1) -> str:
    return "-" if v is None else f"{v * 100:.{digits}f}%"


def _num(v: float | None, digits: int = 3) -> str:
    return "-" if v is None else f"{v:.{digits}f}"


def _usd(v: float | None) -> str:
    if v is None:
        return "-"
    return f"-${-v:,.0f}" if v < 0 else f"${v:,.0f}"


def _ci(ci: list[float] | None, digits: int = 3) -> str:
    return "" if ci is None else f" [{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"


def _rate(v: float | str | None) -> str:
    return "never" if v == "inf" else _pct(v)


def render_markdown(metrics: Mapping) -> str:
    champion = metrics["champion"]
    cfg = metrics["config"]
    models = {m["model"]: m for m in metrics["models"]}
    data = metrics.get("data")
    source = (f"Data seed `{data['seed']}`, {data['n_prospects']:,} prospects. " if data else "")
    lines = [
        f"_{source}Rendered from `outputs/metrics.json`. Monthly scoring runs; active base = "
        f"customers with an order in the {cfg['active_days']} days before the run; churned = no "
        f"order in the {cfg['horizon_days']} days from the run; {cfg['n_bootstrap']} "
        "customer-clustered bootstrap resamples for 95% intervals._",
        "",
        "**Time-aware split**",
        "",
        "| Split | Runs | First run | Last run | Customer-runs | Unique customers | Churners | "
        "Churn rate |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for s in metrics["splits"]:
        lines.append(f"| {s['split']} | {s['runs']} | {s['first_run']} | {s['last_run']} | "
                     f"{s['customer_runs']:,} | {s['unique_customers']:,} | {s['churners']:,} | "
                     f"{_pct(s['churn_rate'])} |")
    lines += [
        "",
        "**Leakage audit** (the pipeline refuses to write results if any check fails)",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    audit = metrics["leakage_audit"]
    for name, ok in audit["checks"].items():
        lines.append(f"| {name.replace('_', ' ')} | {'pass' if ok else 'FAIL'} |")
    d = audit["details"]
    lines += [
        "",
        f"Most predictive single feature on the holdout: `{d['most_predictive_single_feature']}`"
        f" (AUC {d['most_predictive_single_feature_auc']:.3f}, limit "
        f"{d['single_feature_auc_limit']}). Last training label window ends "
        f"{d['last_train_label_end']}; first holdout run {d['first_holdout_run']}.",
        "",
        f"**Model comparison** (champion selected on {metrics['selection_metric']}: "
        f"**{MODEL_LABELS[champion]}**; all models refit on fit + validation runs and scored on "
        "the same holdout)",
        "",
        "| Model | Validation log loss | Holdout ROC AUC [95% CI] | Holdout AP [95% CI] | "
        "Log loss | Brier | ECE | Calibration slope | Mean predicted / observed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in models.items():
        label = MODEL_LABELS[name] + (" **(champion)**" if name == champion else "")
        mean_pred = m.get("holdout_mean_predicted")
        lines.append(
            f"| {label} | {_num(m.get('validation_log_loss'), 4)} | "
            f"{_num(m['holdout_roc_auc'])}{_ci([m['holdout_roc_auc_ci_low'], m['holdout_roc_auc_ci_high']])} | "  # noqa: E501
            f"{_num(m['holdout_average_precision'])}{_ci([m['holdout_ap_ci_low'], m['holdout_ap_ci_high']])} | "  # noqa: E501
            f"{_num(m.get('holdout_log_loss'), 4)} | {_num(m.get('holdout_brier'), 4)} | "
            f"{_num(m.get('holdout_ece'), 4)} | {_num(m.get('holdout_calibration_slope'), 2)} | "
            + ("-" if mean_pred is None else
               f"{_pct(mean_pred)} / {_pct(m['holdout_base_rate'])}") + " |")
    lines.append("")
    for key, delta in metrics["champion_vs_baselines"].items():
        ref = key.split("_minus_")[1]
        lines.append(
            f"- {MODEL_LABELS[champion]} minus {MODEL_LABELS[ref].lower()}: ROC AUC "
            f"{delta['roc_auc']:+.3f}{_ci(delta['roc_auc_ci'])}, average precision "
            f"{delta['average_precision']:+.3f}{_ci(delta['average_precision_ci'])}.")
    lines += [
        "",
        "Champion by holdout run (does calibration hold as the base matures?):",
        "",
        "| Run | Active customers | Observed churn | Mean predicted | ROC AUC |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in metrics["champion_per_run"]:
        lines.append(f"| {r['run']} | {r['active_customers']:,} | "
                     f"{_pct(r['observed_churn_rate'])} | {_pct(r['mean_predicted'])} | "
                     f"{r['roc_auc']:.3f} |")
    depth_rows = metrics["targeting_depths"]
    order = ["random", *MODEL_NAMES]
    lines += [
        "",
        "**Precision and recall at operational targeting depths** (share of each run's active "
        "base contacted; holdout)",
        "",
        "| Depth | Customers / run | " + " | ".join(
            ("Random" if n == "random" else MODEL_LABELS[n]) + " precision / recall"
            for n in order) + " |",
        "|---:|---:|" + "---:|" * len(order),
    ]
    for dep in cfg["depths"]:
        by_model = {r["model"]: r for r in depth_rows if r["depth"] == dep}
        cells = " | ".join(f"{_pct(by_model[n]['precision'])} / {_pct(by_model[n]['recall'])}"
                           for n in order)
        lines.append(f"| {dep:.0%} | {by_model['random']['customers_per_run']:,.0f} | {cells} |")
    lines += [
        "",
        f"**Churn by risk decile - {MODEL_LABELS[champion].lower()}** (deciles within each "
        "holdout run)",
        "",
        "| Decile | Customer-runs | Churners | Churn rate | Mean predicted | Lift | Cumulative "
        "share of churners |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in metrics["deciles"]:
        lines.append(f"| {r['decile']} | {r['customers']:,} | {r['churners']:,} | "
                     f"{_pct(r['churn_rate'])} | {_pct(r['mean_predicted'])} | "
                     f"{r['lift']:.2f}x | {_pct(r['cumulative_capture'])} |")
    lead = champion if champion in LEARNED else LEARNED[0]
    lines += [
        "",
        "**Drivers: model explanations** (mean |SHAP value| in log-odds on a holdout sample for "
        "both learned models; permutation importance = drop in the champion's holdout ROC AUC "
        "when the feature is shuffled)",
        "",
        "| Feature | " + " | ".join(f"SHAP: {MODEL_LABELS[n].lower()}" for n in LEARNED)
        + " | Direction (champion) | Permutation AUC drop |",
        "|---|" + "---:|" * len(LEARNED) + "---|---:|",
    ]
    for r in metrics["feature_importance"]:
        shap_cols = " | ".join(_num(r[f"shap_{n}"]) for n in LEARNED)
        lines.append(f"| `{r['feature']}` | {shap_cols} | {r[f'direction_{lead}']} | "
                     f"{_num(r.get('permutation_auc_drop'))} |")
    lines += [
        "",
        "Largest logistic-regression coefficients (numeric features are log1p-scaled and "
        "standardized, so odds ratios are per standard deviation):",
        "",
        "| Term | Coefficient | Odds ratio |",
        "|---|---:|---:|",
    ]
    for r in metrics["logistic_top_coefficients"]:
        lines.append(f"| `{r['term']}` | {r['coefficient']:+.3f} | {r['odds_ratio']:.2f} |")
    lines += [
        "",
        "**Drivers: observed churn by segment** (holdout; descriptive associations, not causal "
        "effects; mean predicted doubles as a per-segment calibration check)",
        "",
        "| Segment | Level | Share of base | Observed churn | Mean predicted | Relative risk |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in metrics["segment_drivers"]:
        lines.append(f"| {r['segment'].replace('_', ' ')} | {r['level']} | "
                     f"{_pct(r['share_of_base'])} | {_pct(r['churn_rate'])} | "
                     f"{_pct(r['mean_predicted'])} | {r['relative_risk']:.2f}x |")

    s = metrics["simulation"]
    a = s["assumptions"]
    lines += [
        "",
        "**Retention budget simulation.** The churn outcomes and customer margins below are "
        "observed on the holdout. The program economics are **assumptions, not observed facts**:"
        " no retention offer has been randomized, so the save rate in particular is unknown.",
        "",
        "| Assumption | Value | Meaning |",
        "|---|---:|---|",
    ]
    for key, desc in sim.DESCRIPTIONS.items():
        v = a[key]
        shown = f"${v:,.2f}" if key.endswith("_cost") else (
            f"{v:g}x" if key == "value_multiplier" else f"{v:.0%}")
        lines.append(f"| `{key}` (assumed) | {shown} | {desc} |")
    lines += [
        "",
        f"Value of a save = {s['value_definition']}. Observed on the holdout: mean trailing "
        f"margin {_usd(s['holdout_mean_value_churners'])} for customers who went on to churn vs "
        f"{_usd(s['holdout_mean_value_non_churners'])} for those who kept buying.",
        "",
        "| Depth | Policy | Churners targeted / run | Expected saves / run | Incremental margin"
        " / run | Program cost / run | Net value / run | ROI | Break-even save rate |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in s["policies"]:
        lines.append(
            f"| {r['depth']:.0%} | {POLICY_LABELS[r['policy']]} | "
            f"{r['churners_targeted_per_run']:,.0f} | {r['expected_saves_per_run']:,.1f} | "
            f"{_usd(r['incremental_margin_per_run'])} | {_usd(r['program_cost_per_run'])} | "
            f"{_usd(r['net_value_per_run'])} | {_num(r['roi'], 2)} | "
            f"{_rate(r['break_even_save_rate'])} |")
    t = s["expected_net_positive"]
    lines += [
        "",
        f"- Without a budget cap, targeting every customer whose *ex-ante* expected net value is "
        f"positive would contact {_pct(t['depth'])} of the base "
        f"({t['customers_targeted_per_run']:,.0f} customers per run) for a planned "
        f"{_usd(t['planned_net_value_per_run'])} and a "
        f"realized-under-assumptions {_usd(t['net_value_per_run'])} net per run (ROI "
        f"{t['roi']:.2f}, break-even save rate {_rate(t['break_even_save_rate'])}).",
        "",
        f"Sensitivity of net value per run at {s['sensitivity_depth']:.0%} depth to the two "
        "least certain assumptions (the value-ranked list is re-planned under each scenario):",
        "",
    ]
    sens = s["sensitivity"]
    incentives = sorted({r["incentive_cost"] for r in sens})
    lines += [
        "| Save rate | Policy | " + " | ".join(f"Voucher ${i:,.0f}" for i in incentives) + " |",
        "|---:|---|" + "---:|" * len(incentives),
    ]
    for rate in sorted({r["save_rate"] for r in sens}):
        for policy in ("risk_ranked", "value_ranked"):
            cells = {r["incentive_cost"]: r["net_value_per_run"] for r in sens
                     if r["save_rate"] == rate and r["policy"] == policy}
            lines.append(f"| {rate:.0%} | {POLICY_LABELS[policy]} | "
                         + " | ".join(_usd(cells[i]) for i in incentives) + " |")
    return "\n".join(lines)


def write_outputs(tables: Mapping[str, pd.DataFrame], out_dir: Path, readme: Path | None = None,
                  config: RetentionConfig | None = None, data_manifest: Mapping | None = None
                  ) -> dict:
    """Run the analysis and write metrics JSON, CSV tables, figures and the README block."""
    metrics, out = run_analysis(tables, config)
    if data_manifest is not None:
        metrics = {"data": {"seed": data_manifest["seed"],
                            "n_prospects": data_manifest["n_prospects"]}, **metrics}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    for name, df in out.items():
        df.round(6).to_csv(out_dir / f"{name}.csv", index=False)
    save_figures(metrics, out, out_dir / "figures")
    if readme is not None and readme.exists():
        update_generated_block(readme, render_markdown(metrics), BEGIN_MARKER, END_MARKER)
    return metrics
