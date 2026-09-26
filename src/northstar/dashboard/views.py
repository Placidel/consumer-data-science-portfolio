"""Filters and reshaping behind the dashboard's interactive controls.

Every function slices or reshapes a table that a section pipeline already saved; none of them
fits a model or reads raw data, so interactions stay instant. Kept free of Streamlit so the
behaviour can be tested directly.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

CUSTOMER_STATES = ("new", "active", "loyal", "at_risk", "churned")

LABELS = {
    # policies and models
    "random": "Random",
    "channel_rate": "Channel rate (baseline)",
    "recent_activity": "Recent activity (baseline)",
    "logistic_regression": "Logistic regression",
    "gradient_boosting": "Gradient boosting",
    "recency_rule": "Recency rule (baseline)",
    "rfm_cell_rate": "RFM cell rate (baseline)",
    "risk_ranked": "Highest churn risk first",
    "value_ranked": "Highest expected value first",
    "run_rate": "Run rate (baseline)",
    "rfm_cell_mean": "RFM cell mean (baseline)",
    "bgnbd_gamma_gamma": "BG/NBD + Gamma-Gamma",
    "naive_4wk": "Naive 4-week run rate",
    "seasonal_naive_yoy": "Last year + growth",
    "harmonic_no_promo": "Harmonic, no promotions",
    "harmonic": "Harmonic regression",
    # lifecycle states
    "prospect": "Prospect",
    "new": "New",
    "active": "Active",
    "loyal": "Loyal",
    "at_risk": "At risk",
    "churned": "Churned",
    # next best actions
    "vip_care": "VIP care",
    "personalized_grow": "Personalized growth",
    "cross_sell": "Cross-sell",
    "plus_invite": "Plus invite",
    "retention_save": "Retention save",
    "low_touch": "Low touch",
    # funnel dimensions
    "device_type": "Device",
    "platform": "Platform",
    "traffic_source": "Traffic source",
    "visit_number": "Visit number",
}


def label(key: object) -> str:
    text = str(key)
    return LABELS.get(text, text.replace("_", " ").replace("->", " → ").capitalize())


# --- executive overview ------------------------------------------------------------------------

def month_window(monthly: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Rows of ``monthly_kpis`` with ``start <= month <= end`` (``YYYY-MM`` strings)."""
    if start > end:
        start, end = end, start
    months = monthly["month"].astype(str)
    return monthly.loc[(months >= start) & (months <= end)].reset_index(drop=True)


def window_totals(window: pd.DataFrame) -> dict[str, float]:
    """Sums over a month window, plus the window's average order value."""
    totals = {c: float(window[c].sum()) for c in
              ("net_revenue", "orders", "new_customers", "new_leads", "marketing_cost")}
    totals["months"] = len(window)
    totals["average_order_value"] = (totals["net_revenue"] / totals["orders"]
                                     if totals["orders"] else float("nan"))
    return totals


# --- acquisition -------------------------------------------------------------------------------

def filter_channels(df: pd.DataFrame, channels: Sequence[str],
                    column: str = "acquisition_channel") -> pd.DataFrame:
    return df.loc[df[column].isin(list(channels))].reset_index(drop=True)


def budget_at_capacity(budget: pd.DataFrame, capacity: float) -> pd.DataFrame:
    """Outreach policies compared at one capacity (share of open leads contacted per run)."""
    rows = budget.loc[(budget["capacity_share"] - capacity).abs() < 1e-9]
    if rows.empty:
        raise ValueError(f"No budget simulation at capacity {capacity}")
    return rows.sort_values("conversions_reached_per_run", ascending=False).reset_index(drop=True)


# --- targeting curves (retention and growth programs) ------------------------------------------

def depths(curve: pd.DataFrame) -> list[float]:
    """Targeting depths available in a saved value curve (excluding the empty program)."""
    return sorted(float(d) for d in curve["depth"].unique() if d > 0)


def curve_at_depth(curve: pd.DataFrame, depth: float, policies: Sequence[str] | None = None
                   ) -> pd.DataFrame:
    """Rows of a saved policy-by-depth curve at the saved depth closest to ``depth``."""
    available = curve["depth"].unique()
    nearest = min(available, key=lambda d: abs(d - depth))
    rows = curve.loc[curve["depth"] == nearest]
    if policies is not None:
        rows = rows.loc[rows["policy"].isin(list(policies))]
    return rows.reset_index(drop=True)


def best_depth(curve: pd.DataFrame, policy: str, value: str) -> pd.Series:
    """The saved row with the highest ``value`` for ``policy``."""
    rows = curve.loc[curve["policy"] == policy]
    return rows.loc[rows[value].idxmax()]


# --- conversion --------------------------------------------------------------------------------

STEP_COLUMNS = ("rate_session_start->product_view", "rate_product_view->add_to_cart",
                "rate_add_to_cart->checkout_start", "rate_checkout_start->purchase")


def segment_funnel(segments: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Session funnel rates for each level of one dimension, largest level first."""
    rows = segments.loc[segments["dimension"] == dimension]
    if rows.empty:
        raise ValueError(f"No funnel segments for dimension {dimension!r}")
    return rows.sort_values("sessions", ascending=False).reset_index(drop=True)


# --- forecast ----------------------------------------------------------------------------------

def forecast_view(weekly: pd.DataFrame, forecast: pd.DataFrame, champion: str, level: int,
                  history_weeks: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recent actual weeks, and the forecast with the champion's interval at ``level`` (50/80).

    The saved intervals belong to the champion; baseline forecasts are kept for comparison.
    """
    if level not in (50, 80):
        raise ValueError("level must be 50 or 80")
    history = weekly.sort_values("week_start").tail(history_weeks).reset_index(drop=True)
    ahead = forecast.rename(columns={f"lower_{level}": "lower", f"upper_{level}": "upper"})
    keep = ["week_start", "horizon_week", champion, "lower", "upper", "naive_4wk",
            "seasonal_naive_yoy", "same_week_last_year"]
    return history, ahead[list(dict.fromkeys(keep))].reset_index(drop=True)


# --- lifecycle ---------------------------------------------------------------------------------

def state_mix(state_counts: pd.DataFrame, as_share: bool) -> pd.DataFrame:
    """Long table of customers by lifecycle state and month (prospects excluded)."""
    long = state_counts.melt(id_vars=["period", "customers"], value_vars=list(CUSTOMER_STATES),
                             var_name="state", value_name="count")
    long["share"] = long["count"] / long["customers"].where(long["customers"] > 0)
    long["value"] = long["share"] if as_share else long["count"]
    long["order"] = long["state"].map({s: i for i, s in enumerate(CUSTOMER_STATES)})
    return long.sort_values(["period", "order"]).reset_index(drop=True)


def largest_channels(by_channel: pd.DataFrame, n: int) -> list[str]:
    """The ``n`` acquisition channels with the most acquired customers in the pooled cohorts."""
    first = by_channel.loc[by_channel["months_since_acquisition"] == 0]
    return first.nlargest(n, "customers")["acquisition_channel"].tolist()
