"""Section 03 - conversion funnel diagnostics and the one-page checkout experiment."""

from northstar.conversion.experiment import (
    AssignmentError,
    ExperimentPlan,
    Guardrail,
    assignment_audit,
    build_units,
    decide,
)
from northstar.conversion.funnel import (
    FunnelIntegrityError,
    assert_monotone,
    session_depth,
    stage_table,
)

__all__ = [
    "AssignmentError",
    "ExperimentPlan",
    "FunnelIntegrityError",
    "Guardrail",
    "assert_monotone",
    "assignment_audit",
    "build_units",
    "decide",
    "session_depth",
    "stage_table",
]
