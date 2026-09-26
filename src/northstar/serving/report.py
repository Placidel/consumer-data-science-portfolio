"""Section 08 pipeline: registry summary, batch-scored monitoring runs, drift and performance
report, example API requests, figures and the README results block.

Everything reported in ``projects/08_productionization/README.md`` between the generated-block
markers is rendered from ``outputs/metrics.json`` by :func:`render_markdown`; the same function
with ``full=True`` writes the stand-alone ``outputs/monitoring_report.md``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from northstar.paths import PROJECTS_DIR
from northstar.profile import GRID, TEXT_PRIMARY, TEXT_SECONDARY, _style, update_generated_block
from northstar.serving.monitoring import MonitoringConfig, monitor_model
from northstar.serving.registry import load_artifact
from northstar.serving.scoring import Scorer, validate_frame
from northstar.serving.specs import SPECS

BEGIN_MARKER = "<!-- BEGIN GENERATED: production-results -->"
END_MARKER = "<!-- END GENERATED: production-results -->"
N_EXAMPLE_RECORDS = 3
# Reference categorical palette, fixed slot per model; neutral ink for reference lines.
COLORS = {"acquisition_lead_score": "#2a78d6", "churn_risk": "#eb6834"}
REFERENCE_INK = "#898781"


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


def section_champion(section: str, projects_dir: Path) -> dict | None:
    """Champion and holdout ROC AUC recorded by the section pipeline, if its outputs exist."""
    path = projects_dir / section / "outputs" / "metrics.json"
    if not path.exists():
        return None
    metrics = json.loads(path.read_text())
    row = next(m for m in metrics["models"] if m["champion"])
    return {"champion": metrics["champion"], "holdout_roc_auc": row["holdout_roc_auc"],
            "data": metrics.get("data")}


def registry_entry(scorer: Scorer, projects_dir: Path) -> dict:
    md = scorer.artifact.metadata
    spec = scorer.spec
    section = section_champion(spec.section, projects_dir)
    same_data = section is not None and section["data"] == md["data"]
    consistent = (None if not same_data else
                  section["champion"] == md["algorithm"]
                  and abs(section["holdout_roc_auc"] - md["holdout_metrics"]["roc_auc"]) < 5e-4)
    return {
        "name": md["name"], "title": md["title"], "version": md["version"],
        "section": md["section"], "algorithm": md["algorithm"],
        "algorithm_label": md["algorithm_label"], "horizon_days": md["horizon_days"],
        "population": md["population"], "deployable_from": md["training"]["deployable_from"],
        "first_train_run": md["training"]["fit_runs"][0],
        "last_train_run": md["training"]["validation_runs"][-1],
        "train_rows": md["training"]["train_rows"],
        "selection_metric": md["selection"]["metric"],
        "candidates": sorted(md["selection"]["candidates"]),
        "validation": md["validation_metrics"], "holdout": md["holdout_metrics"],
        "leakage_audit_passed": md["leakage_audit"]["passed"],
        "section_champion": section["champion"] if section else None,
        "section_holdout_roc_auc": section["holdout_roc_auc"] if section else None,
        "consistent_with_section": consistent,
        "scikit_learn": md["environment"]["scikit_learn"],
    }


def example_request(scorer: Scorer, tables: Mapping[str, pd.DataFrame], cutoff: pd.Timestamp
                    ) -> dict:
    """A real request body: the first few records (by id) of a scoring run."""
    spec = scorer.spec
    frame = spec.build_features(tables, cutoff).sort_values(spec.entity).head(N_EXAMPLE_RECORDS)
    valid, rejected = validate_frame(spec, frame)
    if rejected:
        raise ValueError(f"example records for {spec.name} fail validation: {rejected}")
    records = valid[[spec.entity, *spec.features]].to_dict(orient="records")
    numeric = set(spec.numeric)
    for rec in records:
        for k, v in rec.items():
            if k in numeric:
                rec[k] = int(v) if float(v).is_integer() else round(float(v), 4)
    return {"records": records}


def run_report(tables: Mapping[str, pd.DataFrame], model_dir: Path | None = None,
               projects_dir: Path = PROJECTS_DIR, config: MonitoringConfig | None = None
               ) -> tuple[dict, dict[str, pd.DataFrame], dict[str, dict]]:
    """Returns (metrics, output tables, example requests keyed by API route)."""
    config = config or MonitoringConfig()
    registry, monitoring, frames, examples = [], {}, {}, {}
    for name, spec in SPECS.items():
        scorer = Scorer(load_artifact(spec, model_dir))
        registry.append(registry_entry(scorer, projects_dir))
        report, out = monitor_model(scorer, tables, config)
        monitoring[name] = report
        for key, df in out.items():
            frames[key] = pd.concat([frames.get(key), df], ignore_index=True)
        examples[spec.route] = example_request(scorer, tables, config.last_run)
    metrics = {"config": config.as_dict(), "registry": registry, "monitoring": monitoring}
    return _clean(metrics), frames, _clean(examples)


# ---------------------------------------------------------------- figures
def _finish(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, metadata={"Software": None})
    plt.close(fig)
    return path


def save_figures(metrics: Mapping, fig_dir: Path, top: int = 10) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    cfg = metrics["config"]
    models = metrics["monitoring"]
    paths = []

    # Pooled feature PSI per model (top features), with the watch / investigate thresholds.
    fig, axes = plt.subplots(1, len(models), figsize=(11, 4.6))
    xmax = max(cfg["psi_major"] * 1.3,
               max(r["psi"] for m in models.values() for r in m["feature_drift"]) * 1.15)
    for ax, (name, rep) in zip(np.atleast_1d(axes), models.items(), strict=True):
        rows = rep["feature_drift"][:top][::-1]
        ys = np.arange(len(rows))
        ax.barh(ys, [r["psi"] for r in rows], height=0.6, color=COLORS[name])
        ax.set_yticks(ys, [r["feature"] for r in rows], fontsize=8.5, color=TEXT_PRIMARY)
        for x, label in ((cfg["psi_moderate"], "watch"), (cfg["psi_major"], "investigate")):
            ax.axvline(x, color=REFERENCE_INK, linewidth=1, linestyle="--")
            ax.text(x, -0.95, f" {label} {x:g}", fontsize=8, color=TEXT_SECONDARY, va="center")
        ax.set_xlim(0, xmax)
        ax.set_ylim(-1.3, len(rows) - 0.4)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("PSI, current runs vs training reference", color=TEXT_SECONDARY)
        _style(ax, SPECS[name].title,
               f"{rep['current']['first_run']} to {rep['current']['last_run']}, "
               f"{rep['current']['records']:,} scored records")
    paths.append(_finish(fig, fig_dir / "feature_psi.png"))

    # Matured-run ROC AUC with 95% intervals against the validation estimate.
    fig, axes = plt.subplots(1, len(models), figsize=(11, 4.6))
    for ax, (name, rep) in zip(np.atleast_1d(axes), models.items(), strict=True):
        perf = rep["performance"]
        runs = [r["run"] for r in rep["current"]["runs"]]
        by_run = {r["run"]: r for r in perf["by_run"]}
        xs = [i for i, run in enumerate(runs) if run in by_run]
        auc = np.array([by_run[runs[i]]["roc_auc"] for i in xs])
        lo = np.array([by_run[runs[i]]["roc_auc_ci_low"] for i in xs])
        hi = np.array([by_run[runs[i]]["roc_auc_ci_high"] for i in xs])
        ref = perf["reference"]["roc_auc"]
        ax.axhline(ref, color=REFERENCE_INK, linewidth=1, linestyle="--",
                   label="validation estimate")
        ax.axhline(ref - cfg["auc_drop"], color=REFERENCE_INK, linewidth=1, linestyle=":",
                   label=f"alert tolerance (-{cfg['auc_drop']:g})")
        ax.errorbar(xs, auc, yerr=[auc - lo, hi - auc], color=COLORS[name], linewidth=2,
                    marker="o", markersize=6, capsize=3, label="run ROC AUC")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False,
                  fontsize=8, labelcolor=TEXT_SECONDARY)
        low = min(lo.min(), ref - cfg["auc_drop"]) - 0.02
        for i, run in enumerate(runs):
            if run not in by_run:
                ax.text(i, low + 0.01, "labels\npending", ha="center", fontsize=8,
                        color=TEXT_SECONDARY)
        ax.set_xticks(range(len(runs)), [r[:7] for r in runs], fontsize=8.5)
        ax.set_xlim(-0.5, len(runs) - 0.5)
        ax.set_ylim(low, max(hi.max(), ref) + 0.02)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_ylabel("ROC AUC (95% CI)", color=TEXT_SECONDARY)
        _style(ax, SPECS[name].title, f"Scoring runs with matured {SPECS[name].horizon_days}-day "
               "outcomes")
    paths.append(_finish(fig, fig_dir / "performance_by_run.png"))
    return paths


# ---------------------------------------------------------------- markdown
def _num(v, digits: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{digits}f}"


def _pct(v, digits: int = 1) -> str:
    return "n/a" if v is None else f"{v * 100:.{digits}f}%"


def _yes(v) -> str:
    return {True: "yes", False: "**no**", None: "n/a (different data)"}[v]


def render_markdown(metrics: Mapping, full: bool = False) -> str:
    cfg = metrics["config"]
    data = metrics.get("data") or {}
    source = (f"data seed `{data['seed']}`, {data['n_prospects']:,} prospects. "
              if data else "")
    lines = [
        f"_{source}Rendered from `outputs/metrics.json` by `northstar monitor`. Reference = each "
        "model's training runs; current = every monthly scoring run from the model's deployable "
        f"date to {cfg['last_run']}, scored through the batch scorer. PSI thresholds "
        f"{cfg['psi_moderate']:g} (watch) and {cfg['psi_major']:g} (investigate); performance "
        f"alerts need a ROC AUC drop over {cfg['auc_drop']:g} or an observed/predicted ratio "
        f"outside {cfg['calibration_low']:g}-{cfg['calibration_high']:g}, beyond 95% sampling "
        "noise._",
        "",
        "**Registered models** (served by the API; `LATEST` version of each)",
        "",
        "| Model | Version | Algorithm | Training runs | Deployable from | Validation ROC AUC "
        "| Holdout ROC AUC | Holdout AP | Holdout log loss | Matches section champion |",
        "|---|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for r in metrics["registry"]:
        lines.append(
            f"| {r['title']} (`{r['name']}`) | `{r['version']}` | {r['algorithm_label']} | "
            f"{r['first_train_run']} to {r['last_train_run']} ({r['train_rows']:,} rows) | "
            f"{r['deployable_from']} | {_num(r['validation']['roc_auc'])} | "
            f"{_num(r['holdout']['roc_auc'])} | {_num(r['holdout']['average_precision'])} | "
            f"{_num(r['holdout']['log_loss'], 4)} | {_yes(r['consistent_with_section'])} |")
    lines += [
        "",
        "**Monitoring summary**",
        "",
        "| Model | Current runs | Records scored | Score PSI | Features at watch / investigate "
        "| Matured runs | Labels pending | Status |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for name, rep in metrics["monitoring"].items():
        drift = rep["feature_drift"]
        watch = sum(r["status"] == "watch" for r in drift)
        inv = sum(r["status"] == "investigate" for r in drift)
        pending = ", ".join(rep["performance"]["pending_runs"]) or "none"
        lines.append(
            f"| {SPECS[name].title} | {rep['current']['first_run']} to "
            f"{rep['current']['last_run']} ({len(rep['current']['runs'])}) | "
            f"{rep['current']['records']:,} | {_num(rep['score_drift']['psi'])} | "
            f"{watch} / {inv} | {len(rep['performance']['by_run'])} | {pending} | "
            f"**{rep['status']}** |")

    for name, rep in metrics["monitoring"].items():
        spec = SPECS[name]
        perf = rep["performance"]
        lines += ["", f"**{spec.title}** - status **{rep['status']}**. {rep['recommendation']}",
                  ""]
        if rep["findings"]:
            for f in rep["findings"]:
                lines.append(f"- `{f['severity']}` {f['check'].replace('_', ' ')}: "
                             f"`{f['subject']}` - {f['message']}")
        else:
            lines.append("- No checks triggered.")
        shown = rep["feature_drift"] if full else rep["feature_drift"][:5]
        lines += ["", f"{'All features' if full else 'Largest input shifts'} (pooled PSI over "
                  "current runs):", "",
                  "| Feature | PSI | Worst run (PSI) | Reference | Current | Status |",
                  "|---|---:|---|---:|---:|---|"]
        for r in shown:
            ref, cur = _num(r["reference_mean"], 2), _num(r["current_mean"], 2)
            if r["kind"] == "categorical":
                level = r["most_shifted_level"]
                ref, cur = f"{_pct(r['reference_mean'])} {level}", f"{_pct(r['current_mean'])}"
            lines.append(f"| `{r['feature']}` | {_num(r['psi'])} | {r['max_run']} "
                         f"({_num(r['max_run_psi'])}) | {ref} | {cur} | {r['status']} |")
        ref = perf["reference"]
        lines += ["", f"Matured runs vs validation estimate (ROC AUC {_num(ref['roc_auc'])}, "
                  f"outcome rate {_pct(ref['base_rate'])}):", "",
                  "| Run | Records | Observed rate | Mean predicted | Observed / predicted "
                  "| ROC AUC (95% CI) | AP |",
                  "|---|---:|---:|---:|---:|---|---:|"]
        for r in perf["by_run"]:
            lines.append(f"| {r['run']} | {r['rows']:,} | {_pct(r['observed_rate'])} | "
                         f"{_pct(r['mean_predicted'])} | {_num(r['observed_to_predicted'], 2)} | "
                         f"{_num(r['roc_auc'])} ({_num(r['roc_auc_ci_low'])} to "
                         f"{_num(r['roc_auc_ci_high'])}) | {_num(r['average_precision'])} |")
        if perf["pooled"]:
            p = perf["pooled"]
            lines.append(f"| **All matured** | {p['rows']:,} | {_pct(p['observed_rate'])} | "
                         f"{_pct(p['mean_predicted'])} | {_num(p['observed_to_predicted'], 2)} | "
                         f"{_num(p['roc_auc'])} ({_num(p['roc_auc_ci_low'])} to "
                         f"{_num(p['roc_auc_ci_high'])}) | {_num(p['average_precision'])} |")
        for run in rep["current"]["runs"]:
            if not run["labels_mature"]:
                lines.append(f"| {run['run']} | {run['records']:,} | pending until "
                             f"{run['labels_complete_on']} | | | | |")
    return "\n".join(lines)


def render_report(metrics: Mapping) -> str:
    return ("# Northstar model monitoring report\n\n"
            "Generated by `northstar monitor`; machine-readable version: `metrics.json`, "
            "`drift_features.csv`, `drift_by_run.csv`, `performance_by_run.csv`.\n\n"
            + render_markdown(metrics, full=True) + "\n")


def write_outputs(tables: Mapping[str, pd.DataFrame], out_dir: Path, readme: Path | None = None,
                  model_dir: Path | None = None, projects_dir: Path = PROJECTS_DIR,
                  config: MonitoringConfig | None = None, data_manifest: Mapping | None = None
                  ) -> dict:
    """Run the report and write metrics JSON, CSV tables, markdown, examples, figures, README."""
    metrics, frames, examples = run_report(tables, model_dir, projects_dir, config)
    if data_manifest is not None:
        metrics = {"data": {"seed": data_manifest["seed"],
                            "n_prospects": data_manifest["n_prospects"]}, **metrics}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    for name, df in frames.items():
        numeric = df.select_dtypes("number").columns
        df.assign(**df[numeric].round(4)).to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / "monitoring_report.md").write_text(render_report(metrics))
    examples_dir = out_dir / "example_requests"
    examples_dir.mkdir(exist_ok=True)
    for route, body in examples.items():
        (examples_dir / f"{route}.json").write_text(json.dumps(body, indent=2) + "\n")
    save_figures(metrics, out_dir / "figures")
    if readme is not None and readme.exists():
        update_generated_block(readme, render_markdown(metrics), BEGIN_MARKER, END_MARKER)
    return metrics
