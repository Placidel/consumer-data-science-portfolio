"""Assemble the simulated event logs into the shared relational tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

from northstar.schema import FUNNEL_STAGES, TABLES
from northstar.synthetic import params as p
from northstar.synthetic.catalog import build_campaigns, build_experiments, build_products
from northstar.synthetic.simulate import (
    CustomerSimulator,
    EventLog,
    build_customer_context,
    build_prospects,
    simulate_prospect_journeys,
)
from northstar.timeline import DATA_END, DATA_START


def _to_ts(days: np.ndarray) -> pd.Series:
    """Float days since DATA_START -> timestamps at one-second resolution (NaN -> NaT)."""
    days = np.asarray(days, dtype=float)
    out = pd.Series(pd.NaT, index=range(len(days)), dtype="datetime64[ns]")
    ok = ~np.isnan(days)
    seconds = np.round(days[ok] * 86400).astype(np.int64)
    out[ok] = DATA_START + pd.to_timedelta(seconds, unit="s")
    return out


def _chrono_rank(times: np.ndarray, *tiebreak: np.ndarray) -> np.ndarray:
    """rank[i] = position of internal row i after a stable chronological sort."""
    keys = (*reversed(tiebreak), np.arange(len(times)), times)
    order = np.lexsort(keys) if len(times) else np.array([], dtype=np.int64)
    rank = np.empty(len(times), dtype=np.int64)
    rank[order] = np.arange(len(times))
    return rank


def _ids(prefix: str, width: int, rank: np.ndarray) -> np.ndarray:
    return np.array([f"{prefix}{r + 1:0{width}d}" for r in rank], dtype=object)


def _map_optional(values: list, id_array: np.ndarray) -> list:
    return [None if v is None or v < 0 else id_array[v] for v in values]


def _column(rows: list[tuple], k: int) -> list:
    return [r[k] for r in rows]


def _frame(name: str, data: dict[str, object], sort_col: str) -> pd.DataFrame:
    df = pd.DataFrame(data)
    df = df.sort_values(sort_col, kind="stable").reset_index(drop=True)
    return df[TABLES[name].column_names]


def generate(seed: int = p.DEFAULT_SEED, n_prospects: int = p.DEFAULT_N_PROSPECTS
             ) -> dict[str, pd.DataFrame]:
    """Generate every shared table. Identical ``(seed, n_prospects)`` gives identical output."""
    if n_prospects < 100:
        raise ValueError("n_prospects must be at least 100")
    start, end = DATA_START, DATA_END
    n_days = (end - start).days
    r_products, r_prospects, r_journeys, r_customers = (
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)
    )

    products = build_products(r_products)
    book = build_campaigns(start, end)
    experiments = build_experiments()
    prospects = build_prospects(r_prospects, n_prospects, start, end, book)
    log = EventLog()

    conversions = simulate_prospect_journeys(r_journeys, prospects, book, n_days, start, log)
    conversions.sort(key=lambda c: (c.t, c.person))
    ctx = build_customer_context(r_customers, products, book, start, end)
    records = prospects.table.to_dict("records")
    for cust, conv in enumerate(conversions):
        CustomerSimulator(ctx, log, cust, conv.person, records[conv.person],
                          float(prospects.intent[conv.person]), conv).run()

    prospect_ids = prospects.table["prospect_id"].to_numpy()
    customer_ids = np.array([f"C{k + 1:06d}" for k in range(len(conversions))], dtype=object)
    conv_people = np.array([c.person for c in conversions], dtype=np.int64)
    conv_t = np.array([c.t for c in conversions])
    attrs = prospects.table.set_index("prospect_id").loc[prospect_ids[conv_people]]
    customers = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "prospect_id": prospect_ids[conv_people],
            "customer_since": _to_ts(conv_t),
            **{c: attrs[c].to_numpy() for c in ("acquisition_channel", "region", "age_band",
                                                  "income_band", "device_type", "email_opt_in")},
        }
    )[TABLES["customers"].column_names]

    # Sessions and funnel events.
    s = log.sessions
    s_start = np.array(_column(s, 2))
    session_ids = _ids("S", 7, _chrono_rank(s_start))
    sessions = _frame(
        "sessions",
        {
            "session_id": session_ids,
            "prospect_id": prospect_ids[np.array(_column(s, 0), dtype=np.int64)],
            "customer_id": _map_optional(_column(s, 1), customer_ids),
            "session_start": _to_ts(s_start),
            "platform": _column(s, 3),
            "device_type": _column(s, 4),
            "traffic_source": _column(s, 5),
            "campaign_id": _column(s, 6),
            "pages_viewed": np.array(_column(s, 7), dtype=np.int64),
        },
        "session_id",
    )
    e = log.events
    e_session = np.array(_column(e, 0), dtype=np.int64)
    e_t = np.array(_column(e, 1))
    e_stage = np.array(_column(e, 2), dtype=np.int64)
    funnel_events = _frame(
        "funnel_events",
        {
            "event_id": _ids("E", 8, _chrono_rank(e_t, e_stage)),
            "session_id": session_ids[e_session],
            "event_ts": _to_ts(e_t),
            "event_type": np.array(FUNNEL_STAGES, dtype=object)[e_stage - 1],
            "stage_number": e_stage,
        },
        "event_id",
    )

    # Orders and lines.
    o = log.orders
    o_t = np.array(_column(o, 1))
    o_rank = _chrono_rank(o_t)
    order_ids = _ids("O", 7, o_rank)
    orders = _frame(
        "orders",
        {
            "order_id": order_ids,
            "customer_id": customer_ids[np.array(_column(o, 0), dtype=np.int64)],
            "order_ts": _to_ts(o_t),
            "order_channel": _column(o, 2),
            "session_id": _map_optional(_column(o, 3), session_ids),
            "campaign_id": _column(o, 4),
            "item_count": np.array(_column(o, 5), dtype=np.int64),
            "gross_amount": _column(o, 6),
            "discount_amount": _column(o, 7),
            "net_amount": _column(o, 8),
        },
        "order_id",
    )
    ln = log.lines
    l_order = np.array(_column(ln, 0), dtype=np.int64)
    l_rank = _chrono_rank(o_rank[l_order])  # lines follow their order, then insertion order
    order_lines = _frame(
        "order_lines",
        {
            "order_line_id": _ids("L", 7, l_rank),
            "order_id": order_ids[l_order],
            "product_id": _column(ln, 1),
            "quantity": np.array(_column(ln, 2), dtype=np.int64),
            "unit_price": _column(ln, 3),
            "discount_amount": _column(ln, 4),
            "net_amount": _column(ln, 5),
        },
        "order_line_id",
    )

    # Marketing touches.
    tc = log.touches
    t_t = np.array(_column(tc, 5))
    marketing_touches = _frame(
        "marketing_touches",
        {
            "touch_id": _ids("T", 7, _chrono_rank(t_t)),
            "prospect_id": prospect_ids[np.array(_column(tc, 0), dtype=np.int64)],
            "customer_id": _map_optional(_column(tc, 1), customer_ids),
            "campaign_id": _column(tc, 2),
            "channel": _column(tc, 3),
            "touch_type": _column(tc, 4),
            "touch_at": _to_ts(t_t),
            "opened": np.array(_column(tc, 6), dtype=bool),
            "clicked": np.array(_column(tc, 7), dtype=bool),
            "cost": _column(tc, 8),
        },
        "touch_id",
    )

    # Membership events.
    sb = log.subscriptions
    sb_t = np.array(_column(sb, 1))
    subscription_events = _frame(
        "subscription_events",
        {
            "subscription_event_id": _ids("M", 6, _chrono_rank(sb_t)),
            "customer_id": customer_ids[np.array(_column(sb, 0), dtype=np.int64)],
            "event_ts": _to_ts(sb_t),
            "event_type": _column(sb, 2),
            "plan": _column(sb, 3),
            "amount": np.array(_column(sb, 4), dtype=float),
        },
        "subscription_event_id",
    )

    # Support contacts.
    ct = log.contacts
    c_t = np.array(_column(ct, 2))
    resolved = np.array([np.nan if v is None else v for v in _column(ct, 5)], dtype=float)
    support_contacts = _frame(
        "support_contacts",
        {
            "contact_id": _ids("K", 6, _chrono_rank(c_t)),
            "customer_id": customer_ids[np.array(_column(ct, 0), dtype=np.int64)],
            "order_id": _map_optional(_column(ct, 1), order_ids),
            "contact_ts": _to_ts(c_t),
            "contact_channel": _column(ct, 3),
            "reason": _column(ct, 4),
            "resolved_at": _to_ts(resolved),
            "csat_score": pd.array(_column(ct, 6), dtype="Int64"),
        },
        "contact_id",
    )

    asg = log.assignments
    experiment_assignments = pd.DataFrame(
        {
            "experiment_id": p.EXPERIMENT_ID,
            "prospect_id": prospect_ids[np.array(_column(asg, 0), dtype=np.int64)],
            "variant": _column(asg, 1),
            "assigned_at": _to_ts(np.array(_column(asg, 2))),
        }
    ).sort_values("prospect_id", kind="stable").reset_index(drop=True)

    tables = {
        "products": products,
        "campaigns": book.table,
        "experiments": experiments,
        "prospects": prospects.table,
        "customers": customers,
        "marketing_touches": marketing_touches,
        "sessions": sessions,
        "funnel_events": funnel_events,
        "orders": orders,
        "order_lines": order_lines,
        "subscription_events": subscription_events,
        "support_contacts": support_contacts,
        "experiment_assignments": experiment_assignments,
    }
    return {name: normalize_dtypes(name, tables[name]) for name in TABLES}


def normalize_dtypes(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Coerce every column to the canonical dtype for its schema kind."""
    df = df[TABLES[name].column_names].copy()
    for col in TABLES[name].columns:
        s = df[col.name]
        if col.kind == "timestamp":
            df[col.name] = pd.to_datetime(s).astype("datetime64[ns]")
        elif col.kind == "int":
            df[col.name] = s.astype("Int64") if col.nullable else s.astype("int64")
        elif col.kind == "float":
            df[col.name] = s.astype("float64")
        elif col.kind == "bool":
            df[col.name] = s.astype("bool")
        else:
            df[col.name] = s.astype("string")
    return df
