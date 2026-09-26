"""Descriptive profile of the generated data: KPIs, tables and figures for section 00.

Everything reported in ``projects/00_foundation/README.md`` between the generated-block markers
is rendered from ``data_profile.json`` by this module, so documented numbers always trace back
to the current run.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from northstar.schema import CHANNELS, TABLE_NAMES
from northstar.timeline import DATA_END, DATA_START, DEFAULT_CUTOFF

CONVERSION_WINDOW_DAYS = 60
REPEAT_WINDOW_DAYS = 180
BEGIN_MARKER = "<!-- BEGIN GENERATED: data-profile -->"
END_MARKER = "<!-- END GENERATED: data-profile -->"

# Reference palette (light mode): one series per chart, so a single hue suffices.
SERIES = "#2a78d6"
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"


def _round(x: float, digits: int = 4) -> float:
    return float(round(float(x), digits))


def conversion_within(prospects: pd.DataFrame, customers: pd.DataFrame, days: int
                      ) -> pd.DataFrame:
    """Leads with a complete ``days`` outcome window, flagged if they converted within it."""
    window = pd.Timedelta(days=days)
    eligible = prospects.loc[prospects["created_at"] < DATA_END - window]
    since = eligible["prospect_id"].map(customers.set_index("prospect_id")["customer_since"])
    converted = since.notna() & (since < eligible["created_at"] + window)
    return eligible.assign(converted=converted.to_numpy())


def channel_summary(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    prospects, customers = tables["prospects"], tables["customers"]
    touches = tables["marketing_touches"]
    conv = conversion_within(prospects, customers, CONVERSION_WINDOW_DAYS)
    # Sourcing spend: the touch on the lead's own sourcing campaign at the moment of creation.
    sourcing = touches.merge(
        prospects[["prospect_id", "campaign_id", "created_at", "acquisition_channel"]],
        left_on=["prospect_id", "campaign_id", "touch_at"],
        right_on=["prospect_id", "campaign_id", "created_at"],
    )
    spend = sourcing.groupby("acquisition_channel")["cost"].sum()
    out = pd.DataFrame(
        {
            "leads": prospects.groupby("acquisition_channel").size(),
            "customers": customers.groupby("acquisition_channel").size(),
            f"conversion_rate_{CONVERSION_WINDOW_DAYS}d": conv.groupby("acquisition_channel")[
                "converted"].mean(),
            "sourcing_spend": spend,
        }
    ).reindex(list(CHANNELS)).fillna({"customers": 0, "sourcing_spend": 0.0})
    # Organic leads have no sourcing touch, so unit costs are undefined rather than zero.
    paid_spend = out["sourcing_spend"].where(out["sourcing_spend"] > 0)
    out["cost_per_lead"] = paid_spend / out["leads"]
    out["cost_per_customer"] = paid_spend / out["customers"].where(out["customers"] > 0)
    out = out.rename_axis("acquisition_channel").reset_index()
    out["leads"] = out["leads"].astype(int)
    out["customers"] = out["customers"].astype(int)
    return out.round(4)


def monthly_kpis(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    def by_month(df: pd.DataFrame, col: str) -> pd.core.groupby.DataFrameGroupBy:
        return df.groupby(df[col].dt.to_period("M"))

    orders = tables["orders"]
    subs = tables["subscription_events"]
    out = pd.DataFrame(
        {
            "new_leads": by_month(tables["prospects"], "created_at").size(),
            "new_customers": by_month(tables["customers"], "customer_since").size(),
            "orders": by_month(orders, "order_ts").size(),
            "ordering_customers": by_month(orders, "order_ts")["customer_id"].nunique(),
            "net_revenue": by_month(orders, "order_ts")["net_amount"].sum(),
            "membership_revenue": by_month(subs, "event_ts")["amount"].sum(),
            "marketing_cost": by_month(tables["marketing_touches"], "touch_at")["cost"].sum(),
        }
    ).fillna(0.0)
    out.index = out.index.astype(str)
    out = out.rename_axis("month").reset_index()
    for col in ("new_leads", "new_customers", "orders", "ordering_customers"):
        out[col] = out[col].astype(int)
    return out.round(2)


def build_profile(tables: Mapping[str, pd.DataFrame], manifest: Mapping) -> dict:
    prospects, customers, orders = tables["prospects"], tables["customers"], tables["orders"]
    conv = conversion_within(prospects, customers, CONVERSION_WINDOW_DAYS)

    # Repeat purchase: customers with a full observation window, second order within it.
    window = pd.Timedelta(days=REPEAT_WINDOW_DAYS)
    mature = customers.loc[customers["customer_since"] < DATA_END - window]
    o = orders.merge(mature[["customer_id", "customer_since"]], on="customer_id")
    repeaters = o.loc[(o["order_ts"] > o["customer_since"])
                      & (o["order_ts"] < o["customer_since"] + window), "customer_id"].nunique()

    subs = tables["subscription_events"]
    contacts = tables["support_contacts"]
    assignments = tables["experiment_assignments"]
    history = orders.loc[orders["order_ts"] < DEFAULT_CUTOFF]
    first_order = orders["order_ts"] == orders["customer_id"].map(
        customers.set_index("customer_id")["customer_since"])
    headline = {
        "leads": len(prospects),
        "customers": len(customers),
        f"lead_conversion_rate_{CONVERSION_WINDOW_DAYS}d": _round(conv["converted"].mean()),
        "orders": len(orders),
        "net_revenue": _round(orders["net_amount"].sum(), 2),
        "average_order_value": _round(orders["net_amount"].mean(), 2),
        "repeat_order_revenue_share": _round(
            orders.loc[~first_order, "net_amount"].sum() / orders["net_amount"].sum()),
        "discounted_order_share": _round(orders["campaign_id"].notna().mean()),
        "store_order_share": _round((orders["order_channel"] == "store").mean()),
        f"repeat_purchase_rate_{REPEAT_WINDOW_DAYS}d": _round(repeaters / len(mature)),
        "members_ever": int(subs.loc[subs["event_type"] == "subscribe", "customer_id"].nunique()),
        "membership_revenue": _round(subs["amount"].sum(), 2),
        "support_contacts_per_100_orders": _round(100 * len(contacts) / len(orders), 2),
        "mean_csat": _round(contacts["csat_score"].mean(), 3),
        "marketing_cost": _round(tables["marketing_touches"]["cost"].sum(), 2),
        "experiment_assigned_control": int((assignments["variant"] == "control").sum()),
        "experiment_assigned_treatment": int((assignments["variant"] == "treatment").sum()),
        "orders_before_default_cutoff": len(history),
        "orders_on_or_after_default_cutoff": len(orders) - len(history),
    }
    return {
        "seed": manifest["seed"],
        "n_prospects": manifest["n_prospects"],
        "data_start": str(DATA_START.date()),
        "data_end_exclusive": str(DATA_END.date()),
        "default_cutoff": str(DEFAULT_CUTOFF.date()),
        "row_counts": {name: len(tables[name]) for name in TABLE_NAMES},
        "headline": headline,
        "channels": [
            {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in row.items()}
            for row in channel_summary(tables).to_dict(orient="records")
        ],
    }


def _style(ax: plt.Axes, title: str, subtitle: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    ax.set_title(title, loc="left", fontsize=12, color=TEXT_PRIMARY, pad=22)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=TEXT_SECONDARY)


def save_figures(tables: Mapping[str, pd.DataFrame], monthly: pd.DataFrame, fig_dir: Path
                 ) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    meta = {"Software": None}

    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=120)
    x = pd.PeriodIndex(monthly["month"], freq="M").to_timestamp()
    ax.plot(x, monthly["net_revenue"] / 1000, color=SERIES, linewidth=2)
    ax.scatter(x, monthly["net_revenue"] / 1000, color=SERIES, s=14, zorder=3)
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel("Net revenue ($K)", color=TEXT_SECONDARY, fontsize=9)
    _style(ax, "Monthly net order revenue",
           "Growth from an expanding customer base; holiday peaks each Nov-Dec")
    fig.tight_layout()
    path = fig_dir / "monthly_net_revenue.png"
    fig.savefig(path, metadata=meta)
    plt.close(fig)
    paths.append(path)

    ch = channel_summary(tables).sort_values(f"conversion_rate_{CONVERSION_WINDOW_DAYS}d")
    rate = ch[f"conversion_rate_{CONVERSION_WINDOW_DAYS}d"] * 100
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=120)
    ax.barh(ch["acquisition_channel"], rate, color=SERIES, height=0.6)
    for y, v in enumerate(rate):
        ax.text(v + 0.4, y, f"{v:.1f}%", va="center", fontsize=9, color=TEXT_PRIMARY)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Leads converting within 60 days (%)", color=TEXT_SECONDARY, fontsize=9)
    _style(ax, "Lead-to-customer conversion by acquisition channel",
           "Leads with a complete 60-day window; descriptive, not causal")
    fig.tight_layout()
    path = fig_dir / "conversion_by_channel.png"
    fig.savefig(path, metadata=meta)
    plt.close(fig)
    paths.append(path)
    return paths


def _fmt_int(v: float) -> str:
    return f"{int(v):,}"


def render_profile_markdown(profile: Mapping) -> str:
    h = profile["headline"]
    conv_key = f"lead_conversion_rate_{CONVERSION_WINDOW_DAYS}d"
    rep_key = f"repeat_purchase_rate_{REPEAT_WINDOW_DAYS}d"
    lines = [
        f"_Seed `{profile['seed']}`, {profile['n_prospects']:,} prospects, history "
        f"{profile['data_start']} to {profile['data_end_exclusive']} (exclusive), default cutoff "
        f"{profile['default_cutoff']}. Rendered from `outputs/data_profile.json`._",
        "",
        "| KPI | Value |",
        "|---|---:|",
        f"| Leads | {_fmt_int(h['leads'])} |",
        f"| Customers | {_fmt_int(h['customers'])} |",
        f"| Lead conversion within {CONVERSION_WINDOW_DAYS} days | {h[conv_key]:.1%} |",
        f"| Orders | {_fmt_int(h['orders'])} |",
        f"| Net order revenue | ${h['net_revenue']:,.0f} |",
        f"| Average order value | ${h['average_order_value']:,.2f} |",
        f"| Revenue from repeat (non-first) orders | {h['repeat_order_revenue_share']:.1%} |",
        f"| Orders with a discount campaign | {h['discounted_order_share']:.1%} |",
        f"| Store share of orders | {h['store_order_share']:.1%} |",
        f"| Repeat purchase within {REPEAT_WINDOW_DAYS} days | {h[rep_key]:.1%} |",
        f"| Customers who ever joined Northstar Plus | {_fmt_int(h['members_ever'])} |",
        f"| Membership fee revenue | ${h['membership_revenue']:,.0f} |",
        f"| Support contacts per 100 orders | {h['support_contacts_per_100_orders']:.1f} |",
        f"| Mean CSAT (answered surveys) | {h['mean_csat']:.2f} |",
        f"| Variable marketing cost | ${h['marketing_cost']:,.0f} |",
        f"| Experiment assignments (control / treatment) | "
        f"{_fmt_int(h['experiment_assigned_control'])} / "
        f"{_fmt_int(h['experiment_assigned_treatment'])} |",
        f"| Orders before / on-or-after default cutoff | "
        f"{_fmt_int(h['orders_before_default_cutoff'])} / "
        f"{_fmt_int(h['orders_on_or_after_default_cutoff'])} |",
        "",
        "| Table | Rows |",
        "|---|---:|",
        *[f"| `{name}` | {_fmt_int(n)} |" for name, n in profile["row_counts"].items()],
        "",
        f"| Acquisition channel | Leads | Customers | {CONVERSION_WINDOW_DAYS}-day conversion | "
        "Cost per lead | Cost per customer |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in profile["channels"]:
        cpc = row["cost_per_customer"]
        paid = row["sourcing_spend"] > 0
        cpc_text = f"${cpc:,.2f}" if paid and cpc is not None else "-"
        cpl_text = f"${row['cost_per_lead']:,.2f}" if paid else "-"
        lines.append(
            f"| {row['acquisition_channel']} | {_fmt_int(row['leads'])} | "
            f"{_fmt_int(row['customers'])} | "
            f"{row[f'conversion_rate_{CONVERSION_WINDOW_DAYS}d']:.1%} | {cpl_text} | {cpc_text} |"
        )
    return "\n".join(lines)


def update_generated_block(readme: Path, markdown: str, begin: str = BEGIN_MARKER,
                           end: str = END_MARKER) -> None:
    """Replace the text between ``begin`` and ``end`` markers (other sections pass their own)."""
    text = readme.read_text()
    start, stop = text.index(begin), text.index(end)
    readme.write_text(text[: start + len(begin)] + "\n" + markdown + "\n" + text[stop:])


def extract_generated_block(readme: Path, begin: str = BEGIN_MARKER, end: str = END_MARKER
                            ) -> str:
    text = readme.read_text()
    start, stop = text.index(begin), text.index(end)
    return text[start + len(begin) : stop].strip("\n")


def write_profile(tables: Mapping[str, pd.DataFrame], manifest: Mapping, out_dir: Path,
                  readme: Path | None = None) -> dict:
    """Write profile JSON, KPI tables and figures; refresh the README block if given."""
    out_dir.mkdir(parents=True, exist_ok=True)
    profile = build_profile(tables, manifest)
    (out_dir / "data_profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    monthly = monthly_kpis(tables)
    monthly.to_csv(out_dir / "monthly_kpis.csv", index=False)
    channel_summary(tables).to_csv(out_dir / "channel_summary.csv", index=False)
    save_figures(tables, monthly, out_dir / "figures")
    if readme is not None and readme.exists():
        update_generated_block(readme, render_profile_markdown(profile))
    return profile
