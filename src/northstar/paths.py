"""Repository-relative locations shared by all sections."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECTS_DIR = REPO_ROOT / "projects"
DOCS_DIR = REPO_ROOT / "docs"
FOUNDATION_DIR = PROJECTS_DIR / "00_foundation"
ACQUISITION_DIR = PROJECTS_DIR / "01_acquisition"
RETENTION_DIR = PROJECTS_DIR / "02_retention"
CONVERSION_DIR = PROJECTS_DIR / "03_conversion"
REVENUE_DIR = PROJECTS_DIR / "04_revenue_growth"
FORECAST_DIR = PROJECTS_DIR / "05_predictive_analytics"
LIFECYCLE_DIR = PROJECTS_DIR / "06_lifecycle"
DASHBOARD_DIR = PROJECTS_DIR / "07_dashboard"
PRODUCTION_DIR = PROJECTS_DIR / "08_productionization"


def default_data_dir() -> Path:
    """Where generated tables live; override with the ``NORTHSTAR_DATA_DIR`` env variable."""
    override = os.environ.get("NORTHSTAR_DATA_DIR")
    return Path(override) if override else REPO_ROOT / "data" / "raw"


def default_model_dir() -> Path:
    """Model registry root; override with the ``NORTHSTAR_MODEL_DIR`` env variable."""
    override = os.environ.get("NORTHSTAR_MODEL_DIR")
    return Path(override) if override else REPO_ROOT / "models"
