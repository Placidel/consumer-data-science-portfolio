"""Lifecycle states: explicit definitions and deterministic person-month assignment.

Every person (``prospect_id``) is assigned exactly one state at each **as-of date** ``d``, using
only orders with ``order_ts < d`` (the repository-wide cutoff convention). States are defined
from three purchase measures at ``d``:

* ``tenure`` = days since the first order,
* ``idle`` = days since the most recent order,
* ``orders_window`` = orders in ``[d - loyal_window_days, d)``.

Rules are evaluated in a fixed **precedence** order and the first match wins, so states are
mutually exclusive and exhaustive by construction:

1. ``prospect`` - a lead with no order yet.
2. ``churned`` - idle for more than ``churn_days`` (180). Same boundary as section 02's active
   base: a churned customer is exactly one section 02 would not score.
3. ``at_risk`` - idle for more than ``at_risk_days`` (90) but not churned. Matches the
   business's win-back eligibility and section 02's 90-day churn horizon.
4. ``new`` - first order within the last ``new_days`` (90).
5. ``loyal`` - at least ``loyal_min_orders`` (6) orders in the last ``loyal_window_days`` (365)
   and first order more than ``loyal_min_tenure_days`` (180) ago.
6. ``active`` - every other customer who ordered within ``at_risk_days``.

``LifecycleRules`` validates that the thresholds keep ``new`` and ``loyal`` disjoint and keep
``new`` customers out of ``at_risk`` regardless of precedence. Monthly snapshots are taken at
the first day of each month, so the state labelled ``2025-03`` is the state at the end of
March 2025 (as of 2025-04-01).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from northstar.timeline import DATA_END, DATA_START

STATES = ("prospect", "new", "active", "loyal", "at_risk", "churned")
CUSTOMER_STATES = STATES[1:]
ENGAGED_STATES = ("new", "active", "loyal")
PRECEDENCE = ("prospect", "churned", "at_risk", "new", "loyal", "active")
STATE_LABELS = {
    "prospect": "Prospect",
    "new": "New",
    "active": "Active",
    "loyal": "Loyal",
    "at_risk": "At risk",
    "churned": "Churned",
}
NOT_YET = -1
"""State code for person-months before the person's lead record existed."""
MAX_STEP_DAYS = 31
"""Longest gap between consecutive monthly as-of dates."""


@dataclass(frozen=True)
class LifecycleRules:
    """State thresholds. These are business policy choices, not estimates."""

    new_days: int = 90
    at_risk_days: int = 90
    churn_days: int = 180
    loyal_window_days: int = 365
    loyal_min_orders: int = 6
    loyal_min_tenure_days: int = 180

    def __post_init__(self) -> None:
        if not 0 < self.new_days <= self.at_risk_days < self.churn_days:
            raise ValueError("need 0 < new_days <= at_risk_days < churn_days")
        if self.loyal_min_orders < 2 or self.loyal_window_days <= 0:
            raise ValueError("loyalty needs at least two orders in a positive window")
        if self.loyal_min_tenure_days < self.new_days:
            raise ValueError("loyal_min_tenure_days must be at least new_days")

    def as_dict(self) -> dict:
        return asdict(self)

    def definitions(self) -> dict[str, str]:
        """Plain-language definition of every state under these thresholds."""
        return {
            "prospect": "Identified lead with no order yet.",
            "new": f"First order within the last {self.new_days} days.",
            "active": f"Ordered within the last {self.at_risk_days} days; neither new nor loyal.",
            "loyal": (f"Ordered within the last {self.at_risk_days} days, at least "
                      f"{self.loyal_min_orders} orders in the last {self.loyal_window_days} "
                      f"days and first order more than {self.loyal_min_tenure_days} days ago."),
            "at_risk": (f"Last order more than {self.at_risk_days} and at most "
                        f"{self.churn_days} days ago."),
            "churned": f"Last order more than {self.churn_days} days ago.",
        }


def classify(tenure_days, idle_days, orders_window, rules: LifecycleRules | None = None
             ) -> np.ndarray:
    """Vectorised state rule. ``tenure_days`` is NaN for people without an order."""
    rules = rules or LifecycleRules()
    tenure = np.asarray(tenure_days, dtype=float)
    idle = np.asarray(idle_days, dtype=float)
    n_window = np.asarray(orders_window, dtype=float)
    if np.any(idle[~np.isnan(tenure)] > tenure[~np.isnan(tenure)]):
        raise ValueError("idle days cannot exceed tenure")
    matches = {
        "prospect": np.isnan(tenure),
        "churned": idle > rules.churn_days,
        "at_risk": idle > rules.at_risk_days,
        "new": tenure <= rules.new_days,
        "loyal": (n_window >= rules.loyal_min_orders) & (tenure > rules.loyal_min_tenure_days),
    }
    conditions = [matches[s] for s in PRECEDENCE[:-1]]
    return np.select(conditions, list(PRECEDENCE[:-1]), PRECEDENCE[-1])


def _days(delta: pd.Series) -> pd.Series:
    return delta / pd.Timedelta(days=1)


def people_frame(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per person (prospect), sorted by ``prospect_id``, with their customer record."""
    customers = tables["customers"][["prospect_id", "customer_id", "customer_since"]]
    people = tables["prospects"][["prospect_id", "created_at", "acquisition_channel"]]
    out = people.merge(customers, on="prospect_id", how="left", validate="one_to_one")
    return out.sort_values("prospect_id", kind="stable").reset_index(drop=True)


def order_history(orders: pd.DataFrame, as_of: pd.Timestamp, window_days: int) -> pd.DataFrame:
    """First/last order time and orders in ``[as_of - window_days, as_of)`` per customer."""
    hist = orders.loc[orders["order_ts"] < as_of, ["customer_id", "order_ts"]]
    in_window = hist["order_ts"] >= as_of - pd.Timedelta(days=window_days)
    g = hist.assign(in_window=in_window).groupby("customer_id")
    return pd.DataFrame({"first_ts": g["order_ts"].min(), "last_ts": g["order_ts"].max(),
                         "orders_window": g["in_window"].sum()})


def assign_states(people: pd.DataFrame, orders: pd.DataFrame, as_of: str | pd.Timestamp,
                  rules: LifecycleRules | None = None) -> pd.DataFrame:
    """State of every person at ``as_of`` from orders strictly before it.

    People whose lead was created on or after ``as_of`` get ``state = None``: they do not exist
    yet. The result has the same index as ``people``.
    """
    rules = rules or LifecycleRules()
    as_of = pd.Timestamp(as_of)
    hist = order_history(orders, as_of, rules.loyal_window_days)
    cid = people["customer_id"]
    tenure = _days(as_of - cid.map(hist["first_ts"]))
    idle = _days(as_of - cid.map(hist["last_ts"]))
    n_window = cid.map(hist["orders_window"]).fillna(0).astype(int)
    exists = people["created_at"] < as_of
    state = np.where(exists, classify(tenure, idle, n_window, rules), None)
    return pd.DataFrame({
        "state": pd.Series(state, index=people.index, dtype=object),
        "tenure_days": tenure.where(exists),
        "idle_days": idle.where(exists),
        "orders_window": n_window.where(exists),
        "lead_age_days": _days(as_of - people["created_at"]).where(exists),
    })


def month_starts(start: pd.Timestamp = DATA_START, end: pd.Timestamp = DATA_END
                 ) -> pd.DatetimeIndex:
    """Month starts from ``start`` to ``end`` inclusive (the monthly as-of dates plus origin)."""
    return pd.date_range(start, end, freq="MS")


@dataclass
class LifecyclePanel:
    """Person x month lifecycle panel in wide (person rows, month columns) arrays.

    ``state[i, t]`` is the index into ``STATES`` of person ``i`` at the end of month
    ``periods[t]`` (as of ``as_of[t]``), or ``NOT_YET``. ``orders`` and ``revenue`` are what the
    person bought **during** month ``t``.
    """

    people: pd.DataFrame
    periods: pd.DatetimeIndex
    rules: LifecycleRules
    state: np.ndarray
    tenure_days: np.ndarray
    idle_days: np.ndarray
    orders_window: np.ndarray
    lead_age_days: np.ndarray
    orders: np.ndarray
    revenue: np.ndarray

    @property
    def as_of(self) -> pd.DatetimeIndex:
        return self.periods + pd.offsets.MonthBegin(1)

    @property
    def n_periods(self) -> int:
        return len(self.periods)

    def labels(self, t: int) -> np.ndarray:
        """State names at period ``t`` (``None`` before the lead existed)."""
        codes = self.state[:, t]
        names = np.array(STATES, dtype=object)[np.clip(codes, 0, None)]
        return np.where(codes == NOT_YET, None, names)

    def state_counts(self) -> pd.DataFrame:
        """People in each state at the end of every month (period x state)."""
        counts = np.stack([np.bincount(self.state[:, t][self.state[:, t] >= 0],
                                       minlength=len(STATES)) for t in range(self.n_periods)])
        return pd.DataFrame(counts, index=pd.Index(self.periods, name="period"),
                            columns=list(STATES))


def build_panel(tables: Mapping[str, pd.DataFrame], rules: LifecycleRules | None = None,
                periods: pd.DatetimeIndex | None = None) -> LifecyclePanel:
    """Assign a state to every person at the end of every month of the history."""
    rules = rules or LifecycleRules()
    periods = month_starts()[:-1] if periods is None else pd.DatetimeIndex(periods)
    people = people_frame(tables)
    orders = tables["orders"]
    n, T = len(people), len(periods)
    code = {s: i for i, s in enumerate(STATES)}
    arrays = {k: np.full((n, T), np.nan) for k in ("tenure_days", "idle_days", "orders_window",
                                                   "lead_age_days")}
    state = np.full((n, T), NOT_YET, dtype=np.int8)
    for t, as_of in enumerate(periods + pd.offsets.MonthBegin(1)):
        s = assign_states(people, orders, as_of, rules)
        exists = s["state"].notna().to_numpy()
        state[exists, t] = s.loc[exists, "state"].map(code).to_numpy()
        for k, arr in arrays.items():
            arr[:, t] = s[k].to_numpy(dtype=float)

    # Orders and revenue per person-month; orders outside the panel months are ignored.
    is_customer = people["customer_id"].notna()
    person = pd.Series(people.index[is_customer],
                       index=people.loc[is_customer, "customer_id"].to_numpy())
    month = orders["order_ts"].dt.to_period("M").dt.to_timestamp()
    col = pd.Series(np.arange(T), index=periods).reindex(month.to_numpy()).to_numpy()
    row = orders["customer_id"].map(person).to_numpy(dtype=float)
    keep = ~np.isnan(col) & ~np.isnan(row)
    n_orders = np.zeros((n, T), dtype=np.int64)
    revenue = np.zeros((n, T))
    np.add.at(n_orders, (row[keep].astype(int), col[keep].astype(int)), 1)
    np.add.at(revenue, (row[keep].astype(int), col[keep].astype(int)),
              orders["net_amount"].to_numpy()[keep])
    return LifecyclePanel(people=people, periods=periods, rules=rules, state=state,
                          orders=n_orders, revenue=revenue, **arrays)
