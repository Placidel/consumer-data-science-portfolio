"""Section 04 pipeline: build the customer value dataset, audit leakage, select and evaluate value
models, segment the base, assign next best actions, run growth scenarios, and write tables,
figures and the README results block.

Everything reported in ``projects/04_revenue_growth/README.md`` between the generated-block
markers is rendered from ``outputs/metrics.json`` by :func:`render_markdown`.
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

from northstar.acquisition.dataset import SplitPlan
from northstar.acquisition.report import LeakageError
from northstar.profile import GRID, TEXT_PRIMARY, TEXT_SECONDARY, _style, update_generated_block
from northstar.revenue import actions as act
from northstar.revenue import evaluation as ev
from northstar.revenue import scenarios as sc
from northstar.revenue.dataset import (
    DEFAULT_SPLIT,
    FEATURES,
    TARGET,
    build_dataset,
    leakage_audit,
)
from northstar.revenue.models import (
    BASELINES,
    MODEL_LABELS,
    MODEL_NAMES,
    SEED,
    fit_model,
    predict,
)
from northstar.timeline import PredictionWindow

__all__ = ["LeakageError", "RevenueConfig", "render_markdown", "run_analysis", "write_outputs"]

BEGIN_MARKER = "<!-- BEGIN GENERATED: revenue-results -->"
END_MARKER = "<!-- END GENERATED: revenue-results -->"
SELECTION_METRIC = "rmse"  # minimized by the conditional mean, which is what revenue plans sum

# Reference categorical palette in fixed slot order; muted ink for random, primary ink for oracle.
COLORS = {"gradient_boosting": "#2a78d6", "run_rate": "#eb6834", "rfm_cell_mean": "#1baf7a",
          "bgnbd_gamma_gamma": "#eda100", "random": "#898781", "oracle": TEXT_PRIMARY}
POLICY_LABELS = {"random": "Random (no targeting)", "oracle": "Perfect foresight (unattainable)",
                 **MODEL_LABELS}


@dataclass(frozen=True)
class RevenueConfig:
    split: SplitPlan = DEFAULT_SPLIT
    shares: tuple[float, ...] = ev.TOP_SHARES
    n_bootstrap: int = 200
    permutation_repeats: int = 5
    rules: act.ActionRules = field(default_factory=act.ActionRules)
    per_action: int = 10
    assumptions: sc.GrowthAssumptions = field(default_factory=sc.GrowthAssumptions)
    scenario_depths: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.5)
    sensitivity_depth: float = 0.1
    sensitivity_uplifts: tuple[float, ...] = (0.02, 0.05, 0.10)
    sensitivity_perk_costs: tuple[float, ...] = (5.0, 15.0, 30.0)
    curve_depths: tuple[float, ...] = field(
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
                     "buyer_rate": float((d[TARGET] > 0).mean()),
                     "mean_future_revenue": float(d[TARGET].mean()),
                     "median_tenure_days": float(d["tenure_days"].median())})
    return rows


def _clv_level_by_run(data: pd.DataFrame, horizon_days: int) -> list[dict]:
    """Predicted vs realized total revenue of the BG/NBD model at every run.

    The probabilistic model never sees a label, so every run (fit runs included) is an honest
    out-of-sample check of its calibration-in-the-large and shows how the level moves with the
    season of the outcome window.
    """
    rows = []
    for cutoff, run in data.groupby("run_cutoff"):
        actual, pred = run[TARGET].sum(), run["clv_expected_revenue"].sum()
        rows.append({"run": str(cutoff.date()), "split": run["split"].iloc[0],
                     "label_window_end": str((cutoff + pd.Timedelta(days=horizon_days)).date()),
                     "customers": len(run), "realized_revenue": actual,
                     "predicted_revenue": pred, "bias": pred / actual - 1})
    return rows


def _importance(model, hold: pd.DataFrame, repeats: int) -> pd.DataFrame:
    """Permutation importance of the learned model: holdout RMSE increase when shuffled."""
    perm = permutation_importance(model, hold[list(FEATURES)], hold[TARGET],
                                  scoring="neg_root_mean_squared_error", n_repeats=repeats,
                                  random_state=SEED)
    return (pd.DataFrame({"feature": list(FEATURES), "rmse_increase": perm.importances_mean,
                          "rmse_increase_std": perm.importances_std})
            .sort_values("rmse_increase", ascending=False).reset_index(drop=True))


def run_analysis(tables: Mapping[str, pd.DataFrame], config: RevenueConfig | None = None
                 ) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Run the full section 04 analysis; returns (metrics dict, output tables)."""
    config = config or RevenueConfig()
    plan = config.split
    data, clv_params = build_dataset(tables, plan.fit + plan.validation + plan.holdout,
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

    # Stage 1 - model selection on the later validation run (lowest RMSE).
    validation = {}
    for name in MODEL_NAMES:
        val[name] = predict(fit_model(name, fit), val)
        validation[name] = ev.score_metrics(val[TARGET], val[name], config.shares)
    champion = min(MODEL_NAMES, key=lambda n: validation[n][SELECTION_METRIC])

    # Stage 2 - refit every model on all training runs, evaluate once on the holdout.
    models = {name: fit_model(name, train) for name in MODEL_NAMES}
    for name, model in models.items():
        hold[name] = predict(model, hold)
    y = hold[TARGET].to_numpy(float)
    holdout = {name: ev.score_metrics(y, hold[name], config.shares) for name in MODEL_NAMES}
    intervals = ev.cluster_bootstrap(hold, TARGET, MODEL_NAMES, references=MODEL_NAMES,
                                     n_boot=config.n_bootstrap, seed=SEED)
    comparison = pd.DataFrame([
        {"model": name, "role": "baseline" if name in BASELINES else "model",
         "champion": name == champion,
         **{f"validation_{k}": validation[name][k]
            for k in ("rmse", "mae", "bias", "normalized_gini")},
         **{f"holdout_{k}": v for k, v in holdout[name].items()},
         **{f"holdout_{m}_ci_{side}": intervals[f"{name}.{m}"][i]
            for m in ev.BOOTSTRAP_METRICS for i, side in enumerate(("low", "high"))}}
        for name in MODEL_NAMES])
    deltas = {
        f"{champion}_minus_{ref}": {
            **{m: holdout[champion][k] - holdout[ref][k]
               for m, k in (("rmse", "rmse"), ("mae", "mae"),
                            ("normalized_gini", "normalized_gini"),
                            ("capture_top10", "capture_top10"))},
            **{f"{m}_ci": intervals[f"{champion}.{m}_minus_{ref}"] for m in ev.BOOTSTRAP_METRICS},
        }
        for ref in MODEL_NAMES if ref != champion
    }
    deciles = ev.decile_table(y, hold[champion])
    scores = {name: hold[name].to_numpy(float) for name in MODEL_NAMES}
    gains = ev.gains_table(y, scores, config.curve_depths)
    captured = ev.capture_table(y, scores, config.shares)
    active = ev.active_subset_metrics(hold, TARGET, MODEL_NAMES)
    importance = _importance(models["gradient_boosting"], hold, config.permutation_repeats)

    # Segmentation, next best action and scenarios act on the latest holdout run: the list the
    # business would work from.
    latest_cutoff = hold["run_cutoff"].max()
    latest = hold.loc[hold["run_cutoff"] == latest_cutoff].reset_index(drop=True)
    ids = pd.Index(latest["customer_id"], name="customer_id")
    tiers = act.tier_table(latest, champion, TARGET)
    migration = act.migration_table(latest, champion, TARGET)
    actions = act.assign_actions(latest, champion, config.rules)
    profile = act.category_profile(tables, latest_cutoff, ids)
    bought = act.future_categories(
        tables, PredictionWindow(latest_cutoff, horizon_days=plan.horizon_days), ids)
    category_rules = act.category_rule_evaluation(
        profile, bought, pd.Series((actions == "cross_sell").to_numpy(), index=ids))
    action_summary = act.action_table(latest, actions, champion, TARGET)
    priorities = act.priority_list(latest, actions, profile.reset_index(drop=True), champion,
                                   TARGET, config.per_action)

    a = config.assumptions
    margin_rate = sc.observed_margin_rate(tables, latest_cutoff)
    y_latest = latest[TARGET].to_numpy(float)
    policies = {"random": None, **{n: latest[n].to_numpy(float) for n in MODEL_NAMES}}
    scenario = sc.simulate_policies(y_latest, policies, config.scenario_depths, margin_rate, a)
    curve = sc.simulate_policies(y_latest, policies, config.curve_depths, margin_rate, a)
    plan_check = sc.planned_vs_realized(y_latest, latest[champion].to_numpy(float),
                                        config.scenario_depths, margin_rate, a)
    threshold = sc.threshold_policy(y_latest, latest[champion].to_numpy(float), margin_rate, a)
    sens_policies = {"random": None, "run_rate": policies["run_rate"],
                     champion: policies[champion]}
    sensitivity = sc.sensitivity(y_latest, sens_policies, config.sensitivity_depth, margin_rate,
                                 a, config.sensitivity_uplifts, config.sensitivity_perk_costs)

    metrics = {
        "config": {**plan.as_dict(), "shares": list(config.shares),
                   "n_bootstrap": config.n_bootstrap, "seed": SEED, "features": list(FEATURES)},
        "splits": _split_summary(data, plan),
        "clv_parameters": clv_params,
        "clv_level_by_run": _clv_level_by_run(data, plan.horizon_days),
        "leakage_audit": audit,
        "champion": champion,
        "selection_metric": f"validation {SELECTION_METRIC.upper()}",
        "models": comparison.to_dict(orient="records"),
        "champion_vs_models": deltas,
        "active_base": active,
        "deciles": deciles.to_dict(orient="records"),
        "revenue_capture": captured.to_dict(orient="records"),
        "feature_importance": importance.head(12).to_dict(orient="records"),
        "segmentation": {
            "run": str(latest_cutoff.date()),
            "tiers": tiers.to_dict(orient="records"),
            "migration": migration.to_dict(orient="records"),
        },
        "next_best_action": {
            "rules": config.rules.as_dict(),
            "actions": action_summary.to_dict(orient="records"),
            "category_rules": category_rules,
        },
        "scenario": {
            "assumptions": a.as_dict(),
            "cost_per_customer": a.cost_per_customer,
            "observed_margin_rate": margin_rate,
            "customers_in_run": len(latest),
            "policies": scenario.to_dict(orient="records"),
            "planned_vs_realized": plan_check.to_dict(orient="records"),
            "expected_net_positive": threshold,
            "sensitivity": sensitivity.to_dict(orient="records"),
            "sensitivity_depth": config.sensitivity_depth,
        },
    }
    tables_out = {"model_comparison": comparison, "decile_table": deciles, "gains_curve": gains,
                  "revenue_capture": captured, "feature_importance": importance,
                  "value_tiers": tiers, "value_migration": migration,
                  "next_best_action": action_summary, "nba_priority_list": priorities,
                  "growth_scenarios": scenario, "growth_value_curve": curve,
                  "scenario_sensitivity": sensitivity, "planned_vs_realized": plan_check}
    return _clean(metrics), tables_out


# ---------------------------------------------------------------- figures
def _finish(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, metadata={"Software": None})
    plt.close(fig)
    return path


def _axes_labels(ax: plt.Axes, x: str, y: str) -> None:
    ax.set_xlabel(x, color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel(y, color=TEXT_SECONDARY, fontsize=9)
    ax.set_axisbelow(True)


def _label(name: str, champion: str) -> str:
    return POLICY_LABELS[name] + (" (champion)" if name == champion else "")


def save_figures(metrics: Mapping, out: Mapping[str, pd.DataFrame], fig_dir: Path) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    champion = metrics["champion"]
    run = metrics["segmentation"]["run"]
    horizon = metrics["config"]["horizon_days"]
    paths = []

    # Revenue capture (Lorenz / gains) curves.
    gains = out["gains_curve"]
    fig, ax = plt.subplots(figsize=(8.8, 4.4), dpi=120)
    styles = {"random": "--", "oracle": ":"}
    for name in ["oracle", *MODEL_NAMES, "random"]:
        ax.plot(gains["share_targeted"] * 100, gains[name] * 100, color=COLORS[name],
                linestyle=styles.get(name, "-"), linewidth=2.6 if name == champion else 1.8,
                label=_label(name, champion))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True, color=GRID, linewidth=0.8)
    _axes_labels(ax, "Share of customer base targeted, highest predicted value first (%)",
                 f"Share of next-{horizon}-day revenue captured (%)")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              labelcolor=TEXT_PRIMARY)
    _style(ax, "Revenue capture on the out-of-time holdout",
           f"Customer base at {run}; realized net revenue in the following {horizon} days")
    paths.append(_finish(fig, fig_dir / "revenue_gains.png"))

    # Predicted vs realized mean revenue by champion decile.
    deciles = out["decile_table"]
    fig, ax = plt.subplots(figsize=(7.5, 3.9), dpi=120)
    x = deciles["decile"].to_numpy()
    ax.bar(x - 0.2, deciles["mean_predicted"], width=0.38, color=COLORS[champion],
           label="Mean predicted")
    ax.bar(x + 0.2, deciles["mean_actual"], width=0.38, color=COLORS["random"],
           label="Mean realized")
    ax.set_xticks(x)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    _axes_labels(ax, "Predicted-value decile (1 = highest)",
                 f"Net revenue per customer, next {horizon} days (USD)")
    ax.legend(frameon=False, fontsize=8, loc="upper right", labelcolor=TEXT_PRIMARY)
    _style(ax, f"Calibration by decile - {MODEL_LABELS[champion]}",
           f"Out-of-time holdout, customer base at {run}")
    paths.append(_finish(fig, fig_dir / "decile_calibration.png"))

    # Value tiers: share of customers vs share of realized future revenue.
    tiers = out["value_tiers"]
    fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=120)
    ys = np.arange(len(tiers))[::-1]
    ax.barh(ys + 0.19, tiers["share_of_customers"] * 100, height=0.36, color=COLORS["random"],
            label="Share of customers")
    ax.barh(ys - 0.19, tiers["share_of_actual"] * 100, height=0.36, color=COLORS[champion],
            label=f"Share of realized {horizon}-day revenue")
    for yv, v in zip(ys - 0.19, tiers["share_of_actual"] * 100, strict=True):
        ax.text(v + 1, yv, f"{v:.0f}%", va="center", fontsize=8, color=TEXT_PRIMARY)
    ax.set_yticks(ys)
    ax.set_yticklabels(tiers["group"])
    ax.set_xlim(0, 100)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    _axes_labels(ax, "Share (%)", "Predicted-value tier")
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=TEXT_PRIMARY)
    _style(ax, "Where future revenue sits", f"Tiers by champion prediction at {run}")
    paths.append(_finish(fig, fig_dir / "value_tiers.png"))

    # Growth scenario: net value vs depth by policy (assumption-driven).
    curve = out["growth_value_curve"]
    s = metrics["scenario"]
    a = s["assumptions"]
    fig, ax = plt.subplots(figsize=(9.5, 4.4), dpi=120)
    for name in [*MODEL_NAMES, "random"]:
        sub = curve.loc[curve["policy"] == name]
        ax.plot(sub["depth"] * 100, sub["net_value"] / 1000, color=COLORS[name],
                linestyle=styles.get(name, "-"), linewidth=2.6 if name == champion else 1.8,
                label=_label(name, champion))
    thr = s["expected_net_positive"]
    ax.plot([thr["depth"] * 100], [thr["net_value"] / 1000], marker="o", markersize=8,
            color=COLORS[champion], markeredgecolor="white", markeredgewidth=2, linestyle="none",
            label="Stop where predicted net value turns negative")
    ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
    ax.grid(True, color=GRID, linewidth=0.8)
    _axes_labels(ax, "Share of customer base targeted (%)",
                 "Net value of the program ($ thousands)")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              labelcolor=TEXT_PRIMARY)
    # "\$" keeps matplotlib from reading the dollar signs as mathtext delimiters.
    _style(ax, "Growth program scenario (ASSUMED economics)",
           f"Uplift {a['uplift']:.0%} of 180-day revenue, contact \\${a['contact_cost']:.0f}, perk "
           f"\\${a['perk_cost']:.0f} at {a['redemption_rate']:.0%} redemption; observed margin "
           f"{s['observed_margin_rate']:.0%}")
    paths.append(_finish(fig, fig_dir / "growth_scenario.png"))

    # Permutation importance of the learned model.
    imp = out["feature_importance"].head(12)[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=120)
    ax.barh(np.arange(len(imp)), imp["rmse_increase"], color=COLORS["gradient_boosting"],
            xerr=imp["rmse_increase_std"], error_kw={"ecolor": TEXT_SECONDARY, "lw": 1})
    ax.set_yticks(np.arange(len(imp)))
    ax.set_yticklabels(imp["feature"])
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    _axes_labels(ax, "Holdout RMSE increase when the feature is shuffled (USD)", "")
    _style(ax, "What drives the gradient boosting value score",
           "Permutation importance on the out-of-time holdout (mean ± sd over repeats)")
    paths.append(_finish(fig, fig_dir / "feature_importance.png"))
    return paths


# ---------------------------------------------------------------- README rendering
def _pct(v: float | None, digits: int = 1) -> str:
    return "-" if v is None else f"{v * 100:.{digits}f}%"


def _num(v: float | None, digits: int = 3) -> str:
    return "-" if v is None else f"{v:.{digits}f}"


def _usd(v: float | None, digits: int = 0) -> str:
    if v is None:
        return "-"
    if v == "inf":
        return "never"
    return f"-${-v:,.{digits}f}" if v < 0 else f"${v:,.{digits}f}"


def _ci(ci: list[float] | None, fmt=_num) -> str:
    return "" if ci is None else f" [{fmt(ci[0])}, {fmt(ci[1])}]"


def _uplift(v: float | str) -> str:
    return "never" if v == "inf" else _pct(v)


def render_markdown(metrics: Mapping) -> str:
    champion = metrics["champion"]
    cfg = metrics["config"]
    h = cfg["horizon_days"]
    models = {m["model"]: m for m in metrics["models"]}
    data = metrics.get("data")
    source = (f"Data seed `{data['seed']}`, {data['n_prospects']:,} prospects. " if data else "")
    lines = [
        f"_{source}Rendered from `outputs/metrics.json`. Customer base = every customer whose "
        f"first order was before the run; target = net revenue in the {h} days from the run; "
        f"{cfg['n_bootstrap']} customer-level bootstrap resamples for 95% intervals._",
        "",
        "**Time-aware split**",
        "",
        "| Split | Runs | First run | Last run | Customers scored | Unique customers | Median "
        "tenure (days) | Buyer rate | Mean future revenue |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for s in metrics["splits"]:
        lines.append(f"| {s['split']} | {s['runs']} | {s['first_run']} | {s['last_run']} | "
                     f"{s['customer_runs']:,} | {s['unique_customers']:,} | "
                     f"{s['median_tenure_days']:.0f} | {_pct(s['buyer_rate'])} | "
                     f"{_usd(s['mean_future_revenue'], 2)} |")
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
        f" (|Spearman| {d['most_predictive_single_feature_rank_corr']:.3f}, limit "
        f"{d['single_feature_rank_corr_limit']}). Last training label window ends "
        f"{d['last_train_label_end']}; holdout run {d['first_holdout_run']}. Spend "
        f"reconciliation mismatches: {d['spend_reconciliation_mismatches']}.",
        "",
        "Probabilistic CLV parameters, refit on pre-cutoff history at each run (time unit: "
        "weeks):",
        "",
        "| Run | Customers | Repeat customers | BG/NBD r | alpha | a | b | Gamma-Gamma p | q | v |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run, prm in metrics["clv_parameters"].items():
        bg, gg = prm["bgnbd"], prm["gamma_gamma"]
        lines.append(f"| {run} | {prm['customers']:,} | {prm['repeat_customers']:,} | "
                     f"{bg['r']:.3f} | {bg['alpha']:.2f} | {bg['a']:.3f} | {bg['b']:.3f} | "
                     f"{gg['p']:.2f} | {gg['q']:.2f} | {gg['v']:.1f} |")
    lines += [
        "",
        "BG/NBD + Gamma-Gamma total predicted vs. realized revenue at every run (the model never "
        "sees labels, so each run is an out-of-sample check of the *level*):",
        "",
        "| Run | Split | Label window ends | Customers | Predicted total | Realized total | Bias |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in metrics["clv_level_by_run"]:
        lines.append(f"| {r['run']} | {r['split']} | {r['label_window_end']} | "
                     f"{r['customers']:,} | {_usd(r['predicted_revenue'])} | "
                     f"{_usd(r['realized_revenue'])} | {r['bias']:+.1%} |")
    lines += [
        "",
        f"**Model comparison** (champion selected on {metrics['selection_metric']}: "
        f"**{MODEL_LABELS[champion]}**; learned models refit on all training runs and every model "
        "scored once on the same holdout)",
        "",
        "| Model | Validation RMSE | Holdout RMSE [95% CI] | MAE | R² | Bias (total) | Spearman | "
        "Normalized Gini [95% CI] | Top-10% revenue capture [95% CI] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in models.items():
        label = MODEL_LABELS[name] + (" **(champion)**" if name == champion else "")
        lines.append(
            f"| {label} | {_num(m['validation_rmse'], 1)} | {_num(m['holdout_rmse'], 1)}"
            f"{_ci([m['holdout_rmse_ci_low'], m['holdout_rmse_ci_high']], lambda v: _num(v, 1))} | "
            f"{_num(m['holdout_mae'], 1)} | {_num(m['holdout_r2'])} | {m['holdout_bias']:+.1%} | "
            f"{_num(m['holdout_spearman'])} | {_num(m['holdout_normalized_gini'])}"
            f"{_ci([m['holdout_normalized_gini_ci_low'], m['holdout_normalized_gini_ci_high']])} "
            f"| {_pct(m['holdout_capture_top10'])}"
            f"{_ci([m['holdout_capture_top10_ci_low'], m['holdout_capture_top10_ci_high']], _pct)}"
            " |")
    lines.append("")
    for key, delta in metrics["champion_vs_models"].items():
        ref = key.split("_minus_")[1]
        lines.append(
            f"- {MODEL_LABELS[champion]} minus {MODEL_LABELS[ref]}: RMSE "
            f"{delta['rmse']:+.1f}{_ci(delta['rmse_ci'], lambda v: f'{v:+.1f}')}, normalized "
            f"Gini {delta['normalized_gini']:+.3f}"
            f"{_ci(delta['normalized_gini_ci'], lambda v: f'{v:+.3f}')}, top-10% capture "
            f"{delta['capture_top10'] * 100:+.1f} pp"
            f"{_ci(delta['capture_top10_ci'], lambda v: f'{v * 100:+.1f}')}.")
    act_base = metrics["active_base"]
    lines += [
        "",
        f"Within the active base only ({act_base['customers']:,} customers with an order in the "
        f"last 180 days, {_pct(act_base['share_of_base'])} of the base holding "
        f"{_pct(act_base['share_of_future_revenue'])} of future revenue):",
        "",
        "| Model | RMSE | Normalized Gini | Top-10% capture | Bias (total) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in MODEL_NAMES:
        r = act_base[name]
        lines.append(f"| {MODEL_LABELS[name]} | {_num(r['rmse'], 1)} | "
                     f"{_num(r['normalized_gini'])} | {_pct(r['capture_top10'])} | "
                     f"{r['bias']:+.1%} |")
    lines += [
        "",
        f"**Revenue capture of top-ranked customers** (share of the holdout's realized {h}-day "
        "revenue held by the top of each ranked list; lift = capture / share targeted)",
        "",
        "| Share targeted | " + " | ".join(POLICY_LABELS[n] for n in
                                           ["random", *MODEL_NAMES, "oracle"]) + " |",
        "|---:|" + "---:|" * (len(MODEL_NAMES) + 2),
    ]
    for share in cfg["shares"]:
        row = {r["policy"]: r for r in metrics["revenue_capture"] if r["share_targeted"] == share}
        cells = " | ".join(f"{_pct(row[n]['share_of_revenue'])} ({row[n]['lift']:.1f}x)"
                           for n in ["random", *MODEL_NAMES, "oracle"])
        lines.append(f"| {share:.0%} | {cells} |")
    lines += [
        "",
        f"**Calibration by decile - {MODEL_LABELS[champion]}** (holdout)",
        "",
        "| Decile | Customers | Mean predicted | Mean realized | Predicted / realized | Buyer "
        "rate | Share of revenue | Cumulative |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in metrics["deciles"]:
        lines.append(f"| {r['decile']} | {r['customers']:,} | {_usd(r['mean_predicted'])} | "
                     f"{_usd(r['mean_actual'])} | {r['predicted_to_actual']:.2f} | "
                     f"{_pct(r['buyer_rate'])} | {_pct(r['share_of_revenue'])} | "
                     f"{_pct(r['cumulative_capture'])} |")
    lines += [
        "",
        "**What drives the learned value score** (permutation importance: increase in holdout "
        "RMSE, USD, when the feature is shuffled)",
        "",
        "| Feature | RMSE increase | SD over repeats |",
        "|---|---:|---:|",
    ]
    for r in metrics["feature_importance"]:
        lines.append(f"| `{r['feature']}` | {r['rmse_increase']:.1f} | "
                     f"{r['rmse_increase_std']:.1f} |")

    seg = metrics["segmentation"]
    group_header = ("| {} | Customers | Share of base | Predicted revenue share | Realized "
                    "revenue share | Mean predicted | Mean realized | Buyer rate | Active share "
                    "| Plus members | Mean categories |")
    group_rule = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"

    def group_row(label: str, r: Mapping) -> str:
        return (f"| {label} | {r['customers']:,} | {_pct(r['share_of_customers'])} | "
                f"{_pct(r['share_of_predicted'])} | {_pct(r['share_of_actual'])} | "
                f"{_usd(r['mean_predicted'])} | {_usd(r['mean_actual'])} | "
                f"{_pct(r['buyer_rate'])} | {_pct(r['active_share'])} | "
                f"{_pct(r['plus_member_rate'])} | {r['mean_category_count']:.1f} |")

    lines += [
        "",
        f"**Value segmentation** (customer base at {seg['run']}, tiers by the champion's "
        "prediction; realized columns are the check, not an input)",
        "",
        group_header.format("Tier"),
        group_rule,
        *[group_row(r["group"], r) for r in seg["tiers"]],
        "",
        "Value migration (top 20% by trailing 180-day revenue vs. top 20% by predicted value):",
        "",
        group_header.format("Group"),
        group_rule,
        *[group_row(act.MIGRATION_LABELS[r["group"]], r) for r in seg["migration"]],
    ]
    nba = metrics["next_best_action"]
    rules = nba["rules"]
    cr = nba["category_rules"]
    lines += [
        "",
        "**Next best action** (ordered rules, first match wins: "
        f"top-{rules['at_risk_past_share']:.0%} past spender with BG/NBD P(alive) < "
        f"{rules['at_risk_p_alive']:.2f} → retention; top "
        f"{rules['vip_share']:.0%} predicted → VIP; top {rules['growth_share']:.0%} non-member → "
        f"Plus invite; top {rules['engaged_share']:.0%} with ≤ "
        f"{rules['cross_sell_max_categories']} categories → cross-sell; rest of top "
        f"{rules['engaged_share']:.0%} → personalized; others → low touch)",
        "",
        group_header.format("Action"),
        group_rule,
        *[group_row(act.ACTION_LABELS[r["group"]], r) for r in nba["actions"]],
        "",
        f"- Featured category (the customer's most-bought category) was among the categories "
        f"bought by {_pct(cr['featured_category_hit_rate'])} of the {cr['window_buyers']:,} "
        f"customers who bought in the window, vs. {_pct(cr['best_seller_hit_rate'])} for "
        f"featuring the best seller (`{cr['best_seller_category']}`) to everyone.",
        f"- Suggested new category (most widely bought category not yet owned) was among the new "
        f"categories of {_pct(cr['suggested_new_category_hit_rate'])} of the "
        f"{cr['cross_sell_new_category_buyers']:,} cross-sell customers who bought a new "
        f"category, vs. {_pct(cr['random_new_category_hit_rate'])} expected from a random "
        "unowned category.",
    ]

    s = metrics["scenario"]
    a = s["assumptions"]
    lines += [
        "",
        "**Growth program scenarios.** Realized revenue and the margin rate are observed. The "
        "program economics are **assumptions, not observed facts**: no growth treatment has "
        "been randomized, so the uplift in particular is unknown.",
        "",
        "| Input | Value | Meaning |",
        "|---|---:|---|",
    ]
    for key, desc in sc.DESCRIPTIONS.items():
        v = a[key]
        shown = f"${v:,.2f}" if key.endswith("_cost") else f"{v:.0%}"
        lines.append(f"| `{key}` (assumed) | {shown} | {desc} |")
    lines += [
        f"| gross margin rate (observed) | {_pct(s['observed_margin_rate'])} | Gross margin / "
        "net revenue on all order lines before the run |",
        "",
        f"Cost per targeted customer {_usd(s['cost_per_customer'], 2)}; base of "
        f"{s['customers_in_run']:,} customers.",
        "",
        "| Depth | Policy | Customers | Their realized revenue | Incremental margin | Program "
        "cost | Net value | ROI | Break-even uplift |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in s["policies"]:
        lines.append(
            f"| {r['depth']:.0%} | {POLICY_LABELS[r['policy']]} | "
            f"{r['customers_targeted']:,.0f} | {_usd(r['baseline_revenue'])} | "
            f"{_usd(r['incremental_margin'])} | {_usd(r['program_cost'])} | "
            f"{_usd(r['net_value'])} | {_num(r['roi'], 2)} | "
            f"{_uplift(r['break_even_uplift'])} |")
    t = s["expected_net_positive"]
    lines += [
        "",
        f"- Without a fixed budget, targeting every customer whose *predicted* incremental margin "
        f"covers their cost (predicted {h}-day revenue ≥ {_usd(t['min_predicted_revenue'])}) "
        f"would reach {_pct(t['depth'])} of the base ({t['customers_targeted']:,.0f} customers) "
        f"for a planned {_usd(t['planned_net_value'])} and a realized-under-assumptions "
        f"{_usd(t['net_value'])} (break-even uplift {_uplift(t['break_even_uplift'])}).",
        "",
        "Plan vs. outcome for the champion's list (planned = predicted revenue of the list; "
        "realized = what those customers actually spent):",
        "",
        "| Depth | Planned revenue of list | Realized revenue of list | Planned net value | "
        "Realized net value |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in s["planned_vs_realized"]:
        lines.append(f"| {r['depth']:.0%} | {_usd(r['planned_baseline_revenue'])} | "
                     f"{_usd(r['realized_baseline_revenue'])} | {_usd(r['planned_net_value'])} | "
                     f"{_usd(r['realized_net_value'])} |")
    sens = s["sensitivity"]
    perks = sorted({r["perk_cost"] for r in sens})
    policies = list(dict.fromkeys(r["policy"] for r in sens))
    lines += [
        "",
        f"Sensitivity of net value at {s['sensitivity_depth']:.0%} depth to the two least "
        "certain assumptions:",
        "",
        "| Uplift | Policy | " + " | ".join(f"Perk ${c:,.0f}" for c in perks) + " |",
        "|---:|---|" + "---:|" * len(perks),
    ]
    for uplift in sorted({r["uplift"] for r in sens}):
        for policy in policies:
            cells = {r["perk_cost"]: r["net_value"] for r in sens
                     if r["uplift"] == uplift and r["policy"] == policy}
            lines.append(f"| {uplift:.0%} | {POLICY_LABELS[policy]} | "
                         + " | ".join(_usd(cells[c]) for c in perks) + " |")
    return "\n".join(lines)


def write_outputs(tables: Mapping[str, pd.DataFrame], out_dir: Path, readme: Path | None = None,
                  config: RevenueConfig | None = None, data_manifest: Mapping | None = None
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
