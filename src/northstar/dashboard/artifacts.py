"""Load the saved outputs of each section pipeline for the dashboard.

Each section declares the files the dashboard reads. A missing file never raises: the section is
marked unavailable and the dashboard tells the reader which command regenerates it.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from northstar.paths import PROJECTS_DIR, REPO_ROOT

PROJECTS_ENV = "NORTHSTAR_PROJECTS_DIR"


@dataclass(frozen=True)
class Section:
    key: str
    slug: str
    title: str
    command: str
    metrics_file: str
    tables: tuple[str, ...]

    @property
    def number(self) -> str:
        return self.slug[:2]


SECTIONS: dict[str, Section] = {s.key: s for s in (
    Section("foundation", "00_foundation", "Foundation and synthetic data",
            "northstar generate-data", "data_profile.json",
            ("monthly_kpis", "channel_summary")),
    Section("acquisition", "01_acquisition", "Customer acquisition", "northstar acquisition",
            "metrics.json", ("model_comparison", "budget_simulation", "decile_lift")),
    Section("retention", "02_retention", "Retention and churn", "northstar retention",
            "metrics.json", ("model_comparison", "retention_value_curve", "segment_drivers")),
    Section("conversion", "03_conversion", "Conversion and experimentation",
            "northstar conversion", "metrics.json",
            ("funnel_prospect_cohort", "funnel_segments", "experiment_results",
             "cumulative_effect")),
    Section("revenue", "04_revenue_growth", "Revenue growth and customer value",
            "northstar revenue", "metrics.json",
            ("model_comparison", "value_tiers", "next_best_action", "growth_value_curve")),
    Section("forecast", "05_predictive_analytics", "Revenue forecast", "northstar forecast",
            "metrics.json", ("weekly_revenue", "forecast", "accuracy_by_horizon")),
    Section("lifecycle", "06_lifecycle", "Customer lifecycle", "northstar lifecycle",
            "metrics.json",
            ("state_counts_by_month", "decision_points", "decision_points_by_channel",
             "cohort_retention_by_channel")),
)}


def projects_dir() -> Path:
    """Where section outputs are read from; override with ``NORTHSTAR_PROJECTS_DIR``."""
    override = os.environ.get(PROJECTS_ENV)
    return Path(override) if override else PROJECTS_DIR


def display_path(path: Path) -> str:
    """Repository-relative path when possible (what the dashboard shows as a source)."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


@dataclass
class SectionArtifacts:
    section: Section
    directory: Path
    metrics: dict = field(default_factory=dict)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return not self.missing

    @property
    def metrics_path(self) -> Path:
        return self.directory / self.section.metrics_file

    def table_path(self, name: str) -> Path:
        return self.directory / f"{name}.csv"

    def source(self, name: str | None = None) -> str:
        """Display path of the metrics file (``name=None``) or of one table."""
        return display_path(self.metrics_path if name is None else self.table_path(name))

    @property
    def data(self) -> dict:
        """Seed and population the section was produced from (``{}`` if not recorded)."""
        if "data" in self.metrics:
            return dict(self.metrics["data"])
        return {k: self.metrics[k] for k in ("seed", "n_prospects") if k in self.metrics}


def source_files(section: Section, root: Path | None = None) -> list[Path]:
    directory = (root or projects_dir()) / section.slug / "outputs"
    return [directory / section.metrics_file, *(directory / f"{t}.csv" for t in section.tables)]


def load_section(key: str, root: Path | None = None) -> SectionArtifacts:
    section = SECTIONS[key]
    directory = (root or projects_dir()) / section.slug / "outputs"
    art = SectionArtifacts(section, directory)
    if art.metrics_path.exists():
        art.metrics = json.loads(art.metrics_path.read_text())
    else:
        art.missing.append(section.metrics_file)
    for name in section.tables:
        path = art.table_path(name)
        if path.exists():
            art.tables[name] = pd.read_csv(path)
        else:
            art.missing.append(path.name)
    return art


def load_all(root: Path | None = None, keys: Iterable[str] | None = None
             ) -> dict[str, SectionArtifacts]:
    return {key: load_section(key, root) for key in (keys or SECTIONS)}


def fingerprint(root: Path | None = None) -> tuple:
    """Size and modification time of every file the dashboard reads.

    Used as a cache key, so rerunning a section pipeline refreshes an open dashboard.
    """
    out = []
    for section in SECTIONS.values():
        for path in source_files(section, root):
            stat = path.stat() if path.exists() else None
            out.append((str(path), stat.st_mtime_ns if stat else None,
                        stat.st_size if stat else None))
    return tuple(out)


def provenance(artifacts: Mapping[str, SectionArtifacts]) -> pd.DataFrame:
    """One row per section: availability, the data it was produced from, and consistency.

    Sections built from a different seed or population than the foundation profile cannot be
    compared with each other; ``consistent`` flags them.
    """
    reference = artifacts["foundation"].data if "foundation" in artifacts else {}
    rows = []
    for key, art in artifacts.items():
        data = art.data
        rows.append({
            "section": f"{art.section.number} {art.section.title}",
            "key": key,
            "available": art.available,
            "seed": data.get("seed"),
            "n_prospects": data.get("n_prospects"),
            "consistent": bool(art.available and data and data == reference),
            "missing_files": ", ".join(art.missing),
            "regenerate_with": art.section.command,
            "outputs": display_path(art.directory),
        })
    return pd.DataFrame(rows)
