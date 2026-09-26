"""Section 01 - customer acquisition optimization: which open leads to prioritize for outreach."""

from northstar.acquisition.dataset import (
    DEFAULT_SPLIT,
    FEATURES,
    HORIZON_DAYS,
    PIPELINE_DAYS,
    SplitPlan,
    build_dataset,
    build_features,
    build_run,
    leakage_audit,
    open_leads,
)

__all__ = [
    "DEFAULT_SPLIT",
    "FEATURES",
    "HORIZON_DAYS",
    "PIPELINE_DAYS",
    "SplitPlan",
    "build_dataset",
    "build_features",
    "build_run",
    "leakage_audit",
    "open_leads",
]
