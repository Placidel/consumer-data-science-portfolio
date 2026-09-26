"""Section 06 - customer lifecycle analytics: states, transitions, cohorts and RFM segments."""

from northstar.lifecycle.cohorts import cohort_retention, pooled_retention
from northstar.lifecycle.segments import customer_profile, rfm_scores, segment_rule
from northstar.lifecycle.states import (
    CUSTOMER_STATES,
    PRECEDENCE,
    STATES,
    LifecyclePanel,
    LifecycleRules,
    assign_states,
    build_panel,
    classify,
)
from northstar.lifecycle.transitions import (
    DECISION_POINTS,
    impossible_transitions,
    transition_counts,
    transition_matrix,
)

__all__ = [
    "CUSTOMER_STATES",
    "DECISION_POINTS",
    "PRECEDENCE",
    "STATES",
    "LifecyclePanel",
    "LifecycleRules",
    "assign_states",
    "build_panel",
    "classify",
    "cohort_retention",
    "customer_profile",
    "impossible_transitions",
    "pooled_retention",
    "rfm_scores",
    "segment_rule",
    "transition_counts",
    "transition_matrix",
]
