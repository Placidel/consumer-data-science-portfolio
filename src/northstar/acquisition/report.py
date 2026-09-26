"""Section 01 pipeline: build the dataset, audit leakage, select and evaluate models, simulate
outreach budgets, explain the champion, and write tables, figures and the README results block.

Everything reported in ``projects/01_acquisition/README.md`` between the generated-block markers
is rendered from ``outputs/metrics.json`` by :func:`render_markdown`.
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

from northstar.acquisition import evaluation as ev
from northstar.acquisition.dataset import (
    DEFAULT_SPLIT,
    FEATURES,
    PIPELINE_DAYS,
    TARGET,
    SplitPlan,
    build_dataset,
    leakage_audit,
)
from northstar.acquisition.models import (
    BASELINES,
    LEARNED,
    MODEL_LABELS,
    MODEL_NAMES,
    PROBABILISTIC,
    SEED,
    fit_model,
    logistic_coefficients,
    predict,
    shap_importance,
)
from northstar.profile import (
    GRID,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    _style,
    update_generated_block,
)

BEGIN_MARKER = "<!-- BEGIN GENERATED: acquisition-results -->"
END_MARKER = "<!-- END GENERATED: acquisition-results -->"
REFERENCE_BASELINE = "channel_rate"  # current practice that the champion must beat

# Reference categorical palette (fixed slot order) plus a muted ink for the random reference.
COLORS = {"champion": "#2a78d6", "channel_rate": "#eb6834", "recent_activity": "#1baf7a",
          "other_model": "#eda100", "random": "#898781"}
POLICY_LABELS = {"random": "Random (no targeting)", **MODEL_LABELS}


class LeakageError(RuntimeError):
    """Raised when the runtime leakage audit fails; no results are written."""


@dataclass(frozen=True)
class AcquisitionConfig:
    split: SplitPlan = DEFAULT_SPLIT
    capacities: tuple[float, ...] = (0.1, 0.2, 0.3)
    n_bootstrap: int = 200
    shap_sample: int = 2000
    permutation_repeats: int = 5
    gains_shares: tuple[float, ...] = field(
        default_factory=lambda: tuple(np.round(np.linspace(0, 1, 51), 2)))


def _clean(obj):
    """JSON-safe copy with floats rounded (stable diffs; NaN -> None)."""
    if isinstance(obj, Mapping):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.integer | bool | np.bool_):
        return obj.item() if isinstance(obj, np.generic) else obj
    if isinstance(obj, float | np.floating):
        return None if np.isnan(obj) else round(float(obj), 4)
    return obj


def _records(df: pd.DataFrame) -> list[dict]:
    return _clean(df.to_dict(orient="records"))


def _split_summary(data: pd.DataFrame, plan: SplitPlan) -> list[dict]:
    rows = []
    for split, runs in (("fit", plan.fit), ("validation", plan.validation),
                        ("holdout", plan.holdout)):
        d = data.loc[data["split"] == split]
        rows.append({"split": split, "runs": len(runs), "first_run": str(min(runs).date()),
                     "last_run": str(max(runs).date()), "lead_runs": len(d),
                     "unique_leads": int(d["prospect_id"].nunique()),
                     "conversions": int(d[TARGET].sum()), "conversion_rate": d[TARGET].mean()})
    return rows


def _pipeline_coverage(tables: Mapping[str, pd.DataFrame], plan: SplitPlan) -> dict:
    """Where first orders in the holdout outcome windows come from (scorable or not)."""
    customers = tables["customers"]
    created = customers["prospect_id"].map(
        tables["prospects"].set_index("prospect_id")["created_at"])
    counts = {"open_pipeline": 0, "older_leads": 0, "leads_created_in_window": 0}
    for cutoff in plan.holdout:
        end = cutoff + pd.Timedelta(days=plan.horizon_days)
        in_window = (customers["customer_since"] >= cutoff) & (customers["customer_since"] < end)
        c = created[in_window]
        counts["leads_created_in_window"] += int((c >= cutoff).sum())
        counts["open_pipeline"] += int(((c < cutoff)
                                        & (c >= cutoff - pd.Timedelta(days=PIPELINE_DAYS))).sum())
        counts["older_leads"] += int((c < cutoff - pd.Timedelta(days=PIPELINE_DAYS)).sum())
    total = sum(counts.values())
    return {"first_orders_in_holdout_windows": total,
            **{f"share_{k}": v / total for k, v in counts.items()}}


def run_analysis(tables: Mapping[str, pd.DataFrame],
                 config: AcquisitionConfig | None = None) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Run the full section 01 analysis; returns (metrics dict, output tables)."""
    config = config or AcquisitionConfig()
    plan = config.split
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

    # Stage 1 - model selection: fit on the fit runs, compare on later validation runs.
    validation = {}
    for name in MODEL_NAMES:
        val[name] = predict(fit_model(name, fit), val)
        validation[name] = ev.score_metrics(val, name, name in PROBABILISTIC)
    champion = max(MODEL_NAMES, key=lambda n: validation[n]["average_precision"])

    # Stage 2 - refit every model on all training runs, evaluate once on the holdout runs.
    models = {name: fit_model(name, train) for name in MODEL_NAMES}
    for name, model in models.items():
        hold[name] = predict(model, hold)
    holdout = {name: ev.score_metrics(hold, name, name in PROBABILISTIC) for name in MODEL_NAMES}
    intervals = ev.cluster_bootstrap(hold, MODEL_NAMES, references=BASELINES,
                                     n_boot=config.n_bootstrap, seed=SEED)

    comparison = pd.DataFrame([
        {"model": name, "role": "baseline" if name in BASELINES else "model",
         "champion": name == champion,
         **{f"validation_{k}": v for k, v in validation[name].items()
            if k in ("roc_auc", "average_precision")},
         **{f"holdout_{k}": v for k, v in holdout[name].items()},
         "holdout_roc_auc_ci_low": intervals[f"{name}.roc_auc"][0],
         "holdout_roc_auc_ci_high": intervals[f"{name}.roc_auc"][1],
         "holdout_ap_ci_low": intervals[f"{name}.average_precision"][0],
         "holdout_ap_ci_high": intervals[f"{name}.average_precision"][1]}
        for name in MODEL_NAMES])

    calibration = pd.concat([
        ev.calibration_table(hold[TARGET], hold[name]).assign(model=name)
        for name in PROBABILISTIC], ignore_index=True)
    deciles = ev.decile_table(hold, champion)
    policies = {name: name for name in dict.fromkeys([*BASELINES, champion])}
    budget = ev.budget_simulation(hold, policies, config.capacities)
    shares = np.array(config.gains_shares)
    gains = pd.DataFrame({"share_contacted": shares, "random": shares,
                          **{name: ev.gains_curve(hold, name, shares) for name in policies}})

    # Explainability: logistic coefficients, SHAP for both learned models (agreement is a
    # robustness check) and permutation importance for the champion.
    coefficients = logistic_coefficients(models["logistic_regression"])
    importance = pd.DataFrame({"feature": list(FEATURES)})
    for name in LEARNED:
        summary, _ = shap_importance(models[name], hold, train, config.shap_sample)
        summary = summary.rename(columns={"mean_abs_shap": f"shap_{name}",
                                          "direction": f"direction_{name}"})
        importance = importance.merge(summary, on="feature", how="left")
    if champion in LEARNED:
        perm = permutation_importance(
            models[champion], hold[list(FEATURES)], hold[TARGET], scoring="average_precision",
            n_repeats=config.permutation_repeats, random_state=SEED)
        importance["permutation_ap_drop"] = perm.importances_mean
        importance["permutation_ap_drop_std"] = perm.importances_std
    rank_by = f"shap_{champion if champion in LEARNED else LEARNED[0]}"
    importance = importance.sort_values(rank_by, ascending=False).reset_index(drop=True)

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
    metrics = {
        "config": {**plan.as_dict(), "pipeline_days": PIPELINE_DAYS,
                   "capacities": list(config.capacities), "n_bootstrap": config.n_bootstrap,
                   "shap_sample": config.shap_sample, "seed": SEED, "features": list(FEATURES)},
        "splits": _split_summary(data, plan),
        "pipeline_coverage": _pipeline_coverage(tables, plan),
        "leakage_audit": audit,
        "champion": champion,
        "selection_metric": "validation average precision",
        "models": comparison.to_dict(orient="records"),
        "champion_vs_baselines": deltas,
        "champion_per_run_auc": ev.per_run_auc(hold, champion),
        "deciles": deciles.to_dict(orient="records"),
        "budget_simulation": budget.to_dict(orient="records"),
        "feature_importance": importance.head(12).to_dict(orient="records"),
        "logistic_top_coefficients": coefficients.head(8).to_dict(orient="records"),
    }
    tables_out = {"model_comparison": comparison, "calibration": calibration,
                  "decile_lift": deciles, "budget_simulation": budget, "gains_curve": gains,
                  "feature_importance": importance, "logistic_coefficients": coefficients}
    return _clean(metrics), tables_out


# ---------------------------------------------------------------- figures
def _policy_color(name: str, champion: str) -> str:
    if name == champion:
        return COLORS["champion"]
    return COLORS.get(name, COLORS["other_model"])


def _finish(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, metadata={"Software": None})
    plt.close(fig)
    return path


def save_figures(metrics: Mapping, out: Mapping[str, pd.DataFrame], fig_dir: Path) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    champion = metrics["champion"]
    paths = []

    # Gains: share of conversions reached vs share of each monthly pipeline contacted.
    gains = out["gains_curve"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=120)
    ax.plot(gains["share_contacted"] * 100, gains["random"] * 100, color=COLORS["random"],
            linewidth=1.5, linestyle="--")
    ax.text(62, 55, "Random", color=TEXT_SECONDARY, fontsize=9)
    for name in [c for c in gains.columns if c not in ("share_contacted", "random")]:
        color = _policy_color(name, champion)
        ax.plot(gains["share_contacted"] * 100, gains[name] * 100, color=color, linewidth=2,
                label=POLICY_LABELS[name])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Share of monthly open pipeline contacted (%)", color=TEXT_SECONDARY,
                  fontsize=9)
    ax.set_ylabel("Share of 30-day conversions reached (%)", color=TEXT_SECONDARY, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=TEXT_PRIMARY)
    runs = metrics["config"]["holdout_runs"]
    _style(ax, "Cumulative gains on the out-of-time holdout",
           f"{len(runs)} monthly scoring runs, {runs[0]} to {runs[-1]}; leads ranked within "
           "each run")
    paths.append(_finish(fig, fig_dir / "gains_curve.png"))

    # Decile lift for the champion.
    deciles = out["decile_lift"]
    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=120)
    rate = deciles["conversion_rate"] * 100
    ax.bar(deciles["decile"], rate, color=COLORS["champion"], width=0.7)
    base = next(s["conversion_rate"] for s in metrics["splits"] if s["split"] == "holdout") * 100
    ax.axhline(base, color=COLORS["random"], linewidth=1.2, linestyle="--")
    ax.text(10.4, base, f"average {base:.1f}%", color=TEXT_SECONDARY, fontsize=9, va="bottom",
            ha="right")
    for x, v in zip(deciles["decile"], rate, strict=True):
        ax.text(x, v + 0.5, f"{v:.1f}%", ha="center", fontsize=8, color=TEXT_PRIMARY)
    ax.set_xticks(deciles["decile"])
    ax.set_xlabel("Score decile within run (1 = highest)", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("30-day conversion rate (%)", color=TEXT_SECONDARY, fontsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _style(ax, f"Conversion rate by score decile - {MODEL_LABELS[champion].lower()}",
           "Out-of-time holdout; deciles formed within each monthly run")
    paths.append(_finish(fig, fig_dir / "decile_lift.png"))

    # Reliability diagram for the probabilistic models.
    cal = out["calibration"]
    fig, ax = plt.subplots(figsize=(6.2, 4.6), dpi=120)
    top = float(max(cal["mean_predicted"].max(), cal["observed_rate"].max())) * 100 * 1.05
    ax.plot([0, top], [0, top], color=COLORS["random"], linestyle="--", linewidth=1.2)
    ax.text(top * 0.72, top * 0.64, "perfect calibration", color=TEXT_SECONDARY, fontsize=8,
            rotation=33)
    for name, group in cal.groupby("model", sort=False):
        ax.plot(group["mean_predicted"] * 100, group["observed_rate"] * 100, marker="o",
                markersize=5, linewidth=2, color=_policy_color(name, champion),
                label=MODEL_LABELS[name])
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel("Mean predicted probability (%)", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("Observed conversion rate (%)", color=TEXT_SECONDARY, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=TEXT_PRIMARY)
    _style(ax, "Calibration on the out-of-time holdout", "Ten equal-size bins of predicted risk")
    paths.append(_finish(fig, fig_dir / "calibration.png"))

    # Budget policies: conversions reached per monthly run at each capacity.
    budget = out["budget_simulation"]
    policies = list(dict.fromkeys(budget["policy"]))
    caps = list(dict.fromkeys(budget["capacity_share"]))
    width = 0.8 / len(policies)
    fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=120)
    for i, policy in enumerate(policies):
        sub = budget.loc[budget["policy"] == policy]
        xs = np.arange(len(caps)) + (i - (len(policies) - 1) / 2) * width
        color = COLORS["random"] if policy == "random" else _policy_color(policy, champion)
        ax.bar(xs, sub["conversions_reached_per_run"], width=width * 0.92, color=color,
               label=POLICY_LABELS[policy])
        for x, v in zip(xs, sub["conversions_reached_per_run"], strict=True):
            ax.text(x, v + 1, f"{v:.0f}", ha="center", fontsize=7, color=TEXT_PRIMARY)
    ax.set_xticks(np.arange(len(caps)))
    contacts = budget.groupby("capacity_share", sort=False)["contacts_per_run"].first()
    ax.set_xticklabels([f"Top {c:.0%}\n(~{contacts[c]:,.0f} leads/run)" for c in caps])
    ax.set_ylabel("Conversions reached per monthly run", color=TEXT_SECONDARY, fontsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=TEXT_PRIMARY)
    _style(ax, "Outreach budget simulation", "Same contact capacity, different targeting policy;"
           " holdout average per run")
    paths.append(_finish(fig, fig_dir / "budget_policies.png"))

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
    ax.set_axisbelow(True)
    ax.set_xlabel("Mean |SHAP value| (log-odds)", color=TEXT_SECONDARY, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=TEXT_PRIMARY)
    _style(ax, "What drives the lead score",
           "Exact SHAP on a holdout sample; one-hot columns summed per feature")
    paths.append(_finish(fig, fig_dir / "shap_importance.png"))
    return paths


# ---------------------------------------------------------------- README rendering
def _pct(v: float | None, digits: int = 1) -> str:
    return "-" if v is None else f"{v * 100:.{digits}f}%"


def _num(v: float | None, digits: int = 3) -> str:
    return "-" if v is None else f"{v:.{digits}f}"


def _ci(ci: list[float] | None, digits: int = 3) -> str:
    return "" if ci is None else f" [{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"


def render_markdown(metrics: Mapping) -> str:
    champion = metrics["champion"]
    cfg = metrics["config"]
    models = {m["model"]: m for m in metrics["models"]}
    data = metrics.get("data")
    source = (f"Data seed `{data['seed']}`, {data['n_prospects']:,} prospects. " if data else "")
    lines = [
        f"_{source}Rendered from `outputs/metrics.json`. Monthly scoring runs; open pipeline = "
        f"leads created in the previous {cfg['pipeline_days']} days without a first order; label = "
        f"first order within {cfg['horizon_days']} days of the run; {cfg['n_bootstrap']} "
        "lead-clustered bootstrap resamples for 95% intervals._",
        "",
        "**Time-aware split**",
        "",
        "| Split | Runs | First run | Last run | Lead-runs | Unique leads | Conversions | "
        "Conversion rate |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for s in metrics["splits"]:
        lines.append(f"| {s['split']} | {s['runs']} | {s['first_run']} | {s['last_run']} | "
                     f"{s['lead_runs']:,} | {s['unique_leads']:,} | {s['conversions']:,} | "
                     f"{_pct(s['conversion_rate'])} |")
    cov = metrics["pipeline_coverage"]
    lines += [
        "",
        f"Of {cov['first_orders_in_holdout_windows']:,} first orders placed in the holdout "
        f"outcome windows, {_pct(cov['share_open_pipeline'])} came from scorable open-pipeline "
        f"leads, {_pct(cov['share_leads_created_in_window'])} from leads created during the "
        f"window and {_pct(cov['share_older_leads'])} from leads older than "
        f"{cfg['pipeline_days']} days.",
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
        "| Model | Validation AP | Holdout ROC AUC [95% CI] | Holdout AP [95% CI] | "
        "Precision @10% | Lift @10% | Recall @20% | Brier | ECE | Calibration slope | "
        "Mean predicted / observed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in models.items():
        label = MODEL_LABELS[name] + (" **(champion)**" if name == champion else "")
        mean_pred = m.get("holdout_mean_predicted")
        lines.append(
            f"| {label} | {_num(m['validation_average_precision'])} | "
            f"{_num(m['holdout_roc_auc'])}{_ci([m['holdout_roc_auc_ci_low'], m['holdout_roc_auc_ci_high']])} | "  # noqa: E501
            f"{_num(m['holdout_average_precision'])}{_ci([m['holdout_ap_ci_low'], m['holdout_ap_ci_high']])} | "  # noqa: E501
            f"{_pct(m['holdout_precision_top10'])} | {_num(m['holdout_lift_top10'], 2)}x | "
            f"{_pct(m['holdout_recall_top20'])} | {_num(m.get('holdout_brier'), 4)} | "
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
    runs = metrics["champion_per_run_auc"]
    lines += [
        f"- Champion ROC AUC by holdout run ranges from {min(runs.values()):.3f} to "
        f"{max(runs.values()):.3f} across {len(runs)} monthly runs.",
        "",
        f"**Decile lift - {MODEL_LABELS[champion].lower()}** (deciles within each holdout run)",
        "",
        "| Decile | Lead-runs | Conversions | Conversion rate | Lift | Cumulative share of "
        "conversions |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in metrics["deciles"]:
        lines.append(f"| {r['decile']} | {r['leads']:,} | {r['conversions']:,} | "
                     f"{_pct(r['conversion_rate'])} | {r['lift']:.2f}x | "
                     f"{_pct(r['cumulative_capture'])} |")
    lines += [
        "",
        "**Outreach budget simulation** (contact a fixed share of each monthly open pipeline; "
        "holdout averages per run)",
        "",
        "| Capacity | Policy | Contacts / run | Conversions reached / run | Precision | Share of "
        "conversions reached | Lift vs random | Contacts per conversion |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in metrics["budget_simulation"]:
        lines.append(
            f"| {r['capacity_share']:.0%} | {POLICY_LABELS[r['policy']]} | "
            f"{r['contacts_per_run']:,.0f} | {r['conversions_reached_per_run']:,.1f} | "
            f"{_pct(r['precision'])} | {_pct(r['share_of_conversions_captured'])} | "
            f"{r['lift_vs_random']:.2f}x | {r['contacts_per_conversion']:.1f} |")
    budget = {(r["capacity_share"], r["policy"]): r for r in metrics["budget_simulation"]}
    lines.append("")
    for cap in cfg["capacities"]:
        champ, ref = budget[(cap, champion)], budget[(cap, REFERENCE_BASELINE)]
        gain = champ["conversions_reached_per_run"] - ref["conversions_reached_per_run"]
        lines.append(
            f"- At {cap:.0%} capacity the champion reaches {gain:+.1f} conversions per run "
            f"({gain / ref['conversions_reached_per_run']:+.0%}) versus the channel rule, with "
            f"{ref['contacts_per_conversion'] - champ['contacts_per_conversion']:.1f} fewer "
            "contacts per conversion.")
    lines += [
        "",
        "**Drivers** (mean |SHAP value| in log-odds on a holdout sample for both learned models; "
        "permutation importance = drop in the champion's holdout AP when the feature is "
        "shuffled)",
        "",
        "| Feature | " + " | ".join(f"SHAP: {MODEL_LABELS[n].lower()}" for n in LEARNED)
        + " | Direction (champion) | Permutation AP drop |",
        "|---|" + "---:|" * len(LEARNED) + "---|---:|",
    ]
    lead = champion if champion in LEARNED else LEARNED[0]
    for r in metrics["feature_importance"]:
        shap_cols = " | ".join(_num(r[f"shap_{n}"]) for n in LEARNED)
        lines.append(f"| `{r['feature']}` | {shap_cols} | {r[f'direction_{lead}']} | "
                     f"{_num(r.get('permutation_ap_drop'))} |")
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
    return "\n".join(lines)


def write_outputs(tables: Mapping[str, pd.DataFrame], out_dir: Path, readme: Path | None = None,
                  config: AcquisitionConfig | None = None, data_manifest: Mapping | None = None
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
