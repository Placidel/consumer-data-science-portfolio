"""Command-line entry points: ``northstar <command>`` or ``python -m northstar <command>``."""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from northstar.io import load_tables, read_manifest, write_tables
from northstar.paths import (
    ACQUISITION_DIR,
    CONVERSION_DIR,
    DASHBOARD_DIR,
    DOCS_DIR,
    FORECAST_DIR,
    FOUNDATION_DIR,
    LIFECYCLE_DIR,
    PRODUCTION_DIR,
    PROJECTS_DIR,
    REPO_ROOT,
    RETENTION_DIR,
    REVENUE_DIR,
    default_data_dir,
    default_model_dir,
)
from northstar.profile import write_profile
from northstar.schema import render_data_dictionary
from northstar.synthetic import generate
from northstar.synthetic import params as p
from northstar.validation import validate_all

DEFAULT_PROFILE_DIR = FOUNDATION_DIR / "outputs"
DEFAULT_README = FOUNDATION_DIR / "README.md"
DEFAULT_DICTIONARY = DOCS_DIR / "data_dictionary.md"
DEFAULT_ACQUISITION_OUT = ACQUISITION_DIR / "outputs"
DEFAULT_ACQUISITION_README = ACQUISITION_DIR / "README.md"
DEFAULT_RETENTION_OUT = RETENTION_DIR / "outputs"
DEFAULT_RETENTION_README = RETENTION_DIR / "README.md"
DEFAULT_CONVERSION_OUT = CONVERSION_DIR / "outputs"
DEFAULT_CONVERSION_README = CONVERSION_DIR / "README.md"
DEFAULT_REVENUE_OUT = REVENUE_DIR / "outputs"
DEFAULT_REVENUE_README = REVENUE_DIR / "README.md"
DEFAULT_FORECAST_OUT = FORECAST_DIR / "outputs"
DEFAULT_FORECAST_README = FORECAST_DIR / "README.md"
DEFAULT_LIFECYCLE_OUT = LIFECYCLE_DIR / "outputs"
DEFAULT_LIFECYCLE_README = LIFECYCLE_DIR / "README.md"
DEFAULT_DASHBOARD_OUT = DASHBOARD_DIR / "outputs"
DEFAULT_DASHBOARD_README = DASHBOARD_DIR / "README.md"
DEFAULT_PRODUCTION_OUT = PRODUCTION_DIR / "outputs"
DEFAULT_PRODUCTION_README = PRODUCTION_DIR / "README.md"
DEFAULT_SCORES_DIR = REPO_ROOT / "scores"


def _report(issues: list[str]) -> int:
    if issues:
        print(f"Validation FAILED with {len(issues)} issue(s):", file=sys.stderr)
        for issue in issues:
            print(f"  - {issue}", file=sys.stderr)
        return 1
    print("Validation passed: schema, keys and business rules are consistent.")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    tables = generate(seed=args.seed, n_prospects=args.n_prospects)
    print(f"Generated {sum(len(t) for t in tables.values()):,} rows in "
          f"{time.perf_counter() - started:.1f}s (seed={args.seed}, "
          f"n_prospects={args.n_prospects:,}).")
    if _report(validate_all(tables)):
        return 1
    manifest = write_tables(tables, args.out, seed=args.seed, n_prospects=args.n_prospects)
    for name, entry in manifest["tables"].items():
        print(f"  {name:<24}{entry['rows']:>10,} rows")
    print(f"Wrote tables and manifest to {args.out}")
    if not args.skip_profile:
        write_profile(tables, manifest, args.profile_dir, readme=args.readme)
        print(f"Wrote data profile to {args.profile_dir}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    return _report(validate_all(load_tables(args.data_dir)))


def cmd_profile(args: argparse.Namespace) -> int:
    tables = load_tables(args.data_dir)
    write_profile(tables, read_manifest(args.data_dir), args.profile_dir, readme=args.readme)
    print(f"Wrote data profile to {args.profile_dir}")
    return 0


def cmd_dictionary(args: argparse.Namespace) -> int:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_data_dictionary())
    print(f"Wrote data dictionary to {args.out}")
    return 0


def cmd_acquisition(args: argparse.Namespace) -> int:
    # Imported lazily: scikit-learn and SHAP are only needed by the modelling sections.
    from northstar.acquisition.report import LeakageError, write_outputs

    started = time.perf_counter()
    tables = load_tables(args.data_dir, generate_if_missing=args.generate_if_missing)
    try:
        metrics = write_outputs(tables, args.out_dir, readme=args.readme,
                                data_manifest=read_manifest(args.data_dir))
    except LeakageError as exc:
        print(f"Section 01 aborted: {exc}", file=sys.stderr)
        return 1
    holdout = {m["model"]: m for m in metrics["models"]}[metrics["champion"]]
    print(f"Section 01 finished in {time.perf_counter() - started:.1f}s. Leakage audit passed; "
          f"champion = {metrics['champion']} (holdout ROC AUC {holdout['holdout_roc_auc']:.3f}, "
          f"AP {holdout['holdout_average_precision']:.3f}).")
    print(f"Wrote results to {args.out_dir}")
    return 0


def cmd_retention(args: argparse.Namespace) -> int:
    from northstar.retention.report import LeakageError, write_outputs

    started = time.perf_counter()
    tables = load_tables(args.data_dir, generate_if_missing=args.generate_if_missing)
    try:
        metrics = write_outputs(tables, args.out_dir, readme=args.readme,
                                data_manifest=read_manifest(args.data_dir))
    except LeakageError as exc:
        print(f"Section 02 aborted: {exc}", file=sys.stderr)
        return 1
    holdout = {m["model"]: m for m in metrics["models"]}[metrics["champion"]]
    print(f"Section 02 finished in {time.perf_counter() - started:.1f}s. Leakage audit passed; "
          f"champion = {metrics['champion']} (holdout ROC AUC {holdout['holdout_roc_auc']:.3f}, "
          f"log loss {holdout['holdout_log_loss']:.4f}).")
    print(f"Wrote results to {args.out_dir}")
    return 0


def cmd_conversion(args: argparse.Namespace) -> int:
    from northstar.conversion.report import AssignmentError, write_outputs

    started = time.perf_counter()
    tables = load_tables(args.data_dir, generate_if_missing=args.generate_if_missing)
    try:
        metrics = write_outputs(tables, args.out_dir, readme=args.readme,
                                data_manifest=read_manifest(args.data_dir))
    except AssignmentError as exc:
        print(f"Section 03 aborted: {exc}", file=sys.stderr)
        return 1
    e = metrics["experiment"]
    prim = e["primary"]
    print(f"Section 03 finished in {time.perf_counter() - started:.1f}s. Assignment audit "
          f"passed; conversion lift {prim['diff'] * 100:+.2f} pp "
          f"[{prim['ci'][0] * 100:+.2f}, {prim['ci'][1] * 100:+.2f}], p = {prim['p_value']:.4f};"
          f" decision: {e['decision']['recommendation']}.")
    print(f"Wrote results to {args.out_dir}")
    return 0


def cmd_revenue(args: argparse.Namespace) -> int:
    from northstar.revenue.report import LeakageError, write_outputs

    started = time.perf_counter()
    tables = load_tables(args.data_dir, generate_if_missing=args.generate_if_missing)
    try:
        metrics = write_outputs(tables, args.out_dir, readme=args.readme,
                                data_manifest=read_manifest(args.data_dir))
    except LeakageError as exc:
        print(f"Section 04 aborted: {exc}", file=sys.stderr)
        return 1
    holdout = {m["model"]: m for m in metrics["models"]}[metrics["champion"]]
    print(f"Section 04 finished in {time.perf_counter() - started:.1f}s. Leakage audit passed; "
          f"champion = {metrics['champion']} (holdout RMSE {holdout['holdout_rmse']:.1f}, "
          f"normalized Gini {holdout['holdout_normalized_gini']:.3f}, top-10% revenue capture "
          f"{holdout['holdout_capture_top10']:.1%}).")
    print(f"Wrote results to {args.out_dir}")
    return 0


def cmd_forecast(args: argparse.Namespace) -> int:
    from northstar.forecasting.report import LeakageError, write_outputs

    started = time.perf_counter()
    tables = load_tables(args.data_dir, generate_if_missing=args.generate_if_missing)
    try:
        metrics = write_outputs(tables, args.out_dir, readme=args.readme,
                                data_manifest=read_manifest(args.data_dir))
    except LeakageError as exc:
        print(f"Section 05 aborted: {exc}", file=sys.stderr)
        return 1
    overall = {m["model"]: m for m in metrics["evaluation"]["overall"]}
    champion = overall[metrics["config"]["champion"]]
    f = metrics["forecast"]
    print(f"Section 05 finished in {time.perf_counter() - started:.1f}s. Leakage audit passed; "
          f"backtest WAPE {champion['wape']:.1%} vs naive {overall['naive_4wk']['wape']:.1%}; "
          f"next {metrics['config']['plan']['horizon_weeks']} weeks from {f['origin']}: "
          f"${f['total']['forecast']:,.0f}.")
    print(f"Wrote results to {args.out_dir}")
    return 0


def cmd_lifecycle(args: argparse.Namespace) -> int:
    from northstar.lifecycle.report import IntegrityError, write_outputs

    started = time.perf_counter()
    tables = load_tables(args.data_dir, generate_if_missing=args.generate_if_missing)
    try:
        metrics = write_outputs(tables, args.out_dir, readme=args.readme,
                                data_manifest=read_manifest(args.data_dir))
    except IntegrityError as exc:
        print(f"Section 06 aborted: {exc}", file=sys.stderr)
        return 1
    cur = metrics["states"]["current"]
    points = {r["decision_point"]: r for r in metrics["decision_points"]}
    print(f"Section 06 finished in {time.perf_counter() - started:.1f}s. Integrity audit passed; "
          f"{cur['customers']:,} customers at {cur['period']} month end "
          f"({cur['customer_shares']['churned']:.0%} churned); second-purchase rate "
          f"{points['second_purchase']['favourable_rate']:.1%}, at-risk recovery "
          f"{points['at_risk_recovery']['favourable_rate']:.1%}.")
    print(f"Wrote results to {args.out_dir}")
    return 0


def cmd_dashboard_kpis(args: argparse.Namespace) -> int:
    from northstar.dashboard.artifacts import PROJECTS_ENV
    from northstar.dashboard.kpis import write_outputs

    os.environ[PROJECTS_ENV] = str(args.projects_dir)
    catalog = write_outputs(args.out_dir, readme=args.readme, root=args.projects_dir)
    missing = catalog.loc[catalog["value"].isna(), "key"].tolist()
    print(f"Wrote {len(catalog)} KPIs with sources to {args.out_dir}"
          + (f"; unavailable: {', '.join(missing)}" if missing else "."))
    return 1 if missing and args.strict else 0


def dashboard_command(port: int, headless: bool) -> list[str]:
    """The ``streamlit run`` command line used by ``northstar dashboard``."""
    from northstar.dashboard import APP_PATH

    return [sys.executable, "-m", "streamlit", "run", str(APP_PATH),
            "--server.port", str(port), "--server.headless", str(headless).lower(),
            "--browser.gatherUsageStats", "false", "--theme.base", "light",
            "--theme.primaryColor", "#2a78d6", "--client.toolbarMode", "minimal"]


def cmd_dashboard(args: argparse.Namespace) -> int:
    if importlib.util.find_spec("streamlit") is None:
        print("Streamlit is not installed. Run `python -m pip install -e \".[dev]\"`.",
              file=sys.stderr)
        return 1
    from northstar.dashboard.artifacts import PROJECTS_ENV

    env = {**os.environ, PROJECTS_ENV: str(args.projects_dir)}
    print(f"Starting the dashboard on http://localhost:{args.port} (Ctrl+C to stop)")
    try:
        return subprocess.call(dashboard_command(args.port, args.headless), env=env)
    except KeyboardInterrupt:
        return 0


def _data_info(data_dir: Path) -> dict:
    manifest = read_manifest(data_dir)
    return {"seed": manifest["seed"], "n_prospects": manifest["n_prospects"]}


def cmd_train_models(args: argparse.Namespace) -> int:
    from northstar.serving.specs import SPECS, get_spec
    from northstar.serving.training import LeakageError, train_and_register

    tables = load_tables(args.data_dir, generate_if_missing=args.generate_if_missing)
    for name in args.models or list(SPECS):
        started = time.perf_counter()
        spec = get_spec(name)
        try:
            path, md = train_and_register(spec, tables, args.model_dir, _data_info(args.data_dir),
                                          set_latest=not args.no_promote)
        except LeakageError as exc:
            print(f"Training aborted: {exc}", file=sys.stderr)
            return 1
        print(f"{name}: {md['algorithm']} {md['version']} in "
              f"{time.perf_counter() - started:.1f}s (validation {md['selection']['metric']} "
              f"{md['validation_metrics'][md['selection']['metric']]:.4f}; holdout ROC AUC "
              f"{md['holdout_metrics']['roc_auc']:.3f}). Wrote {path}"
              + ("" if args.no_promote else " and promoted it to LATEST."))
    return 0


def _resolve_model(name: str):
    from northstar.serving.specs import SPECS

    by_route = {s.route: s for s in SPECS.values()}
    return SPECS.get(name) or by_route[name]


def cmd_score_batch(args: argparse.Namespace) -> int:
    from northstar.serving.batch import (
        BatchValidationError,
        read_features,
        score_frame,
        score_run,
        write_result,
    )
    from northstar.serving.registry import ArtifactError, load_artifact
    from northstar.serving.scoring import Scorer

    if args.features is None and args.cutoff is None:
        print("Give a scoring date (--cutoff), a feature file (--features), or both.",
              file=sys.stderr)
        return 2
    spec = _resolve_model(args.model)
    try:
        scorer = Scorer(load_artifact(spec, args.model_dir, args.version))
    except ArtifactError as exc:
        print(f"Cannot score: {exc}", file=sys.stderr)
        return 1
    try:
        if args.features is not None:
            result = score_frame(scorer, read_features(args.features), args.max_invalid_share,
                                 cutoff=args.cutoff)
            label = args.features.stem
        else:
            tables = load_tables(args.data_dir, names=spec.tables_used,
                                 generate_if_missing=args.generate_if_missing)
            result = score_run(scorer, tables, args.cutoff)
            label = str(pd.Timestamp(args.cutoff).date())
    except BatchValidationError as exc:
        print(f"Batch rejected: {exc}", file=sys.stderr)
        return 1
    out = args.out or DEFAULT_SCORES_DIR / f"{spec.name}_{label}.csv"
    written = write_result(result, out)
    s = result.scores[spec.score_field]
    print(f"Scored {len(s):,} records with {spec.name} {scorer.artifact.version} "
          f"(mean {spec.score_field} {s.mean():.3f}; top-10% cut-off {s.quantile(0.9):.3f}; "
          f"{len(result.rejected)} rejected). Wrote {', '.join(map(str, written))}")
    return 0


def cmd_monitor(args: argparse.Namespace) -> int:
    from northstar.serving.registry import ArtifactError
    from northstar.serving.report import write_outputs

    started = time.perf_counter()
    tables = load_tables(args.data_dir, generate_if_missing=args.generate_if_missing)
    try:
        metrics = write_outputs(tables, args.out_dir, readme=args.readme,
                                model_dir=args.model_dir, projects_dir=args.projects_dir,
                                data_manifest=read_manifest(args.data_dir))
    except ArtifactError as exc:
        print(f"Section 08 aborted: {exc}", file=sys.stderr)
        return 1
    status = ", ".join(f"{name} {rep['status']}" for name, rep in metrics["monitoring"].items())
    print(f"Section 08 monitoring finished in {time.perf_counter() - started:.1f}s: {status}.")
    print(f"Wrote results to {args.out_dir}")
    return 0


def cmd_promote_model(args: argparse.Namespace) -> int:
    from northstar.serving.registry import ArtifactError, list_versions, promote, resolve_version

    spec = _resolve_model(args.model)
    if args.version is None:
        try:
            latest = resolve_version(spec, args.model_dir)
        except ArtifactError as exc:
            print(exc, file=sys.stderr)
            return 1
        for version in list_versions(spec, args.model_dir):
            print(f"{version}{'  <- LATEST' if version == latest else ''}")
        return 0
    try:
        artifact = promote(spec, args.version, args.model_dir)
    except ArtifactError as exc:
        print(f"Not promoted: {exc}", file=sys.stderr)
        return 1
    print(f"{spec.name}: LATEST -> {artifact.version} ({artifact.metadata['algorithm']}, "
          f"holdout ROC AUC {artifact.metadata['holdout_metrics']['roc_auc']:.3f}). Restart the "
          "API to serve it.")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import logging

    import uvicorn

    from northstar.serving.api import create_app, preflight

    versions = {"acquisition_lead_score": args.acquisition_version,
                "churn_risk": args.churn_version}
    problems = preflight(args.model_dir, versions)
    if problems:
        print("Refusing to start: not every model loads.", file=sys.stderr)
        for name, problem in problems.items():
            print(f"  - {name}: {problem}", file=sys.stderr)
        return 1
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    print(f"Serving on http://{args.host}:{args.port} (docs at /docs; Ctrl+C to stop)")
    uvicorn.run(create_app(args.model_dir, versions), host=args.host, port=args.port,
                log_level=args.log_level)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="northstar",
                                     description="Northstar Consumer portfolio tooling.")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate-data", help="Generate, validate and write all shared tables.")
    gen.add_argument("--seed", type=int, default=p.DEFAULT_SEED)
    gen.add_argument("--n-prospects", type=int, default=p.DEFAULT_N_PROSPECTS)
    gen.add_argument("--out", type=Path, default=default_data_dir())
    gen.add_argument("--profile-dir", type=Path, default=DEFAULT_PROFILE_DIR)
    gen.add_argument("--readme", type=Path, default=DEFAULT_README,
                     help="README whose generated results block is refreshed.")
    gen.add_argument("--skip-profile", action="store_true",
                     help="Only write the data (no profile outputs or README refresh).")
    gen.set_defaults(func=cmd_generate)

    val = sub.add_parser("validate-data", help="Validate previously generated tables.")
    val.add_argument("--data-dir", type=Path, default=default_data_dir())
    val.set_defaults(func=cmd_validate)

    prof = sub.add_parser("profile-data", help="Rebuild section 00 profile outputs.")
    prof.add_argument("--data-dir", type=Path, default=default_data_dir())
    prof.add_argument("--profile-dir", type=Path, default=DEFAULT_PROFILE_DIR)
    prof.add_argument("--readme", type=Path, default=DEFAULT_README)
    prof.set_defaults(func=cmd_profile)

    dd = sub.add_parser("data-dictionary", help="Render docs/data_dictionary.md from the schema.")
    dd.add_argument("--out", type=Path, default=DEFAULT_DICTIONARY)
    dd.set_defaults(func=cmd_dictionary)

    acq = sub.add_parser("acquisition",
                         help="Section 01: lead scoring models, lift and budget simulation.")
    acq.add_argument("--data-dir", type=Path, default=default_data_dir())
    acq.add_argument("--out-dir", type=Path, default=DEFAULT_ACQUISITION_OUT)
    acq.add_argument("--readme", type=Path, default=DEFAULT_ACQUISITION_README,
                     help="README whose generated results block is refreshed.")
    acq.add_argument("--no-generate", dest="generate_if_missing", action="store_false",
                     help="Fail instead of generating the default data when it is missing.")
    acq.set_defaults(func=cmd_acquisition)

    ret = sub.add_parser("retention",
                         help="Section 02: churn models, targeting depth and retention ROI.")
    ret.add_argument("--data-dir", type=Path, default=default_data_dir())
    ret.add_argument("--out-dir", type=Path, default=DEFAULT_RETENTION_OUT)
    ret.add_argument("--readme", type=Path, default=DEFAULT_RETENTION_README,
                     help="README whose generated results block is refreshed.")
    ret.add_argument("--no-generate", dest="generate_if_missing", action="store_false",
                     help="Fail instead of generating the default data when it is missing.")
    ret.set_defaults(func=cmd_retention)

    conv = sub.add_parser("conversion",
                          help="Section 03: funnel diagnostics and the checkout experiment.")
    conv.add_argument("--data-dir", type=Path, default=default_data_dir())
    conv.add_argument("--out-dir", type=Path, default=DEFAULT_CONVERSION_OUT)
    conv.add_argument("--readme", type=Path, default=DEFAULT_CONVERSION_README,
                      help="README whose generated results block is refreshed.")
    conv.add_argument("--no-generate", dest="generate_if_missing", action="store_false",
                      help="Fail instead of generating the default data when it is missing.")
    conv.set_defaults(func=cmd_conversion)

    rev = sub.add_parser("revenue",
                         help="Section 04: customer value models, segmentation, next best action.")
    rev.add_argument("--data-dir", type=Path, default=default_data_dir())
    rev.add_argument("--out-dir", type=Path, default=DEFAULT_REVENUE_OUT)
    rev.add_argument("--readme", type=Path, default=DEFAULT_REVENUE_README,
                     help="README whose generated results block is refreshed.")
    rev.add_argument("--no-generate", dest="generate_if_missing", action="store_false",
                     help="Fail instead of generating the default data when it is missing.")
    rev.set_defaults(func=cmd_revenue)

    fc = sub.add_parser("forecast",
                        help="Section 05: weekly revenue forecast, rolling backtest, intervals.")
    fc.add_argument("--data-dir", type=Path, default=default_data_dir())
    fc.add_argument("--out-dir", type=Path, default=DEFAULT_FORECAST_OUT)
    fc.add_argument("--readme", type=Path, default=DEFAULT_FORECAST_README,
                    help="README whose generated results block is refreshed.")
    fc.add_argument("--no-generate", dest="generate_if_missing", action="store_false",
                    help="Fail instead of generating the default data when it is missing.")
    fc.set_defaults(func=cmd_forecast)

    lc = sub.add_parser("lifecycle",
                        help="Section 06: lifecycle states, transitions, cohorts, RFM segments.")
    lc.add_argument("--data-dir", type=Path, default=default_data_dir())
    lc.add_argument("--out-dir", type=Path, default=DEFAULT_LIFECYCLE_OUT)
    lc.add_argument("--readme", type=Path, default=DEFAULT_LIFECYCLE_README,
                    help="README whose generated results block is refreshed.")
    lc.add_argument("--no-generate", dest="generate_if_missing", action="store_false",
                    help="Fail instead of generating the default data when it is missing.")
    lc.set_defaults(func=cmd_lifecycle)

    dash = sub.add_parser("dashboard",
                          help="Section 07: launch the executive dashboard (Streamlit).")
    dash.add_argument("--port", type=int, default=8501)
    dash.add_argument("--headless", action="store_true",
                      help="Do not open a browser (for servers and automated checks).")
    dash.add_argument("--projects-dir", type=Path, default=PROJECTS_DIR,
                      help="Folder holding the section outputs (projects/<section>/outputs).")
    dash.set_defaults(func=cmd_dashboard)

    dk = sub.add_parser("dashboard-kpis",
                        help="Section 07: write the dashboard KPI catalog and README block.")
    dk.add_argument("--projects-dir", type=Path, default=PROJECTS_DIR,
                    help="Folder holding the section outputs (projects/<section>/outputs).")
    dk.add_argument("--out-dir", type=Path, default=DEFAULT_DASHBOARD_OUT)
    dk.add_argument("--readme", type=Path, default=DEFAULT_DASHBOARD_README,
                    help="README whose generated results block is refreshed.")
    dk.add_argument("--strict", action="store_true",
                    help="Exit with an error if any KPI cannot be read from saved outputs.")
    dk.set_defaults(func=cmd_dashboard_kpis)

    model_names = ["acquisition_lead_score", "churn_risk"]
    tm = sub.add_parser("train-models",
                        help="Section 08: train the served models and register versioned "
                             "artifacts.")
    tm.add_argument("--data-dir", type=Path, default=default_data_dir())
    tm.add_argument("--model-dir", type=Path, default=default_model_dir())
    tm.add_argument("--models", nargs="+", choices=model_names,
                    help="Models to train (default: all).")
    tm.add_argument("--no-promote", action="store_true",
                    help="Register the new version without pointing LATEST at it.")
    tm.add_argument("--no-generate", dest="generate_if_missing", action="store_false",
                    help="Fail instead of generating the default data when it is missing.")
    tm.set_defaults(func=cmd_train_models)

    sb = sub.add_parser("score-batch",
                        help="Section 08: score a scoring run or a feature file with a "
                             "registered model.")
    sb.add_argument("--model", required=True,
                    choices=[*model_names, "acquisition", "churn"])
    sb.add_argument("--cutoff",
                    help="Scoring date: build point-in-time features from data or, with "
                         "--features, the date the file's features were built for (needed if "
                         "the file has no run_cutoff column).")
    sb.add_argument("--features", type=Path,
                    help="CSV or Parquet file of feature records: id, every feature and "
                         "run_cutoff; any other column (e.g. an outcome) is refused.")
    sb.add_argument("--data-dir", type=Path, default=default_data_dir())
    sb.add_argument("--model-dir", type=Path, default=default_model_dir())
    sb.add_argument("--version", help="Model version (default: LATEST).")
    sb.add_argument("--out", type=Path,
                    help="Output CSV (default: scores/<model>_<cutoff or file name>.csv).")
    sb.add_argument("--max-invalid-share", type=float, default=0.0,
                    help="Share of invalid records tolerated (quarantined); default 0.")
    sb.add_argument("--no-generate", dest="generate_if_missing", action="store_false",
                    help="Fail instead of generating the default data when it is missing.")
    sb.set_defaults(func=cmd_score_batch)

    mon = sub.add_parser("monitor",
                         help="Section 08: drift and performance report for the registered "
                              "models.")
    mon.add_argument("--data-dir", type=Path, default=default_data_dir())
    mon.add_argument("--model-dir", type=Path, default=default_model_dir())
    mon.add_argument("--projects-dir", type=Path, default=PROJECTS_DIR,
                     help="Folder with section outputs (to cross-check section champions).")
    mon.add_argument("--out-dir", type=Path, default=DEFAULT_PRODUCTION_OUT)
    mon.add_argument("--readme", type=Path, default=DEFAULT_PRODUCTION_README,
                     help="README whose generated results block is refreshed.")
    mon.add_argument("--no-generate", dest="generate_if_missing", action="store_false",
                     help="Fail instead of generating the default data when it is missing.")
    mon.set_defaults(func=cmd_monitor)

    pm = sub.add_parser("promote-model",
                        help="Section 08: list a model's versions, or point LATEST at one "
                             "(promotion/rollback).")
    pm.add_argument("--model", required=True, choices=[*model_names, "acquisition", "churn"])
    pm.add_argument("--version", help="Version to promote; omit to list versions.")
    pm.add_argument("--model-dir", type=Path, default=default_model_dir())
    pm.set_defaults(func=cmd_promote_model)

    srv = sub.add_parser("serve", help="Section 08: run the scoring API (FastAPI + uvicorn).")
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8000)
    srv.add_argument("--model-dir", type=Path, default=default_model_dir())
    srv.add_argument("--acquisition-version",
                     default=os.environ.get("NORTHSTAR_ACQUISITION_MODEL_VERSION"),
                     help="Pin a lead-score version (default: LATEST).")
    srv.add_argument("--churn-version", default=os.environ.get("NORTHSTAR_CHURN_MODEL_VERSION"),
                     help="Pin a churn-model version (default: LATEST).")
    srv.add_argument("--log-level", default="info",
                     choices=["critical", "error", "warning", "info", "debug"])
    srv.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
