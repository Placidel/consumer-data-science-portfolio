"""Static reference tables (products, campaigns, experiments) and the demand calendar."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from northstar.schema import CATEGORIES
from northstar.synthetic import params as p

# Category: (number of SKUs, median price, price log-sd, popularity weight).
CATEGORY_SPECS = {
    "apparel": (40, 42.0, 0.45, 0.30),
    "footwear": (15, 85.0, 0.35, 0.14),
    "home": (25, 34.0, 0.55, 0.18),
    "beauty": (20, 21.0, 0.40, 0.16),
    "outdoor": (15, 68.0, 0.50, 0.10),
    "accessories": (15, 26.0, 0.45, 0.12),
}
SOFT_GOODS = ("apparel", "footwear")

ACQUISITION_CAMPAIGN_CHANNELS = ("paid_search", "paid_social", "display", "affiliate", "email")
RETARGETING_CHANNELS = ("paid_social", "display")

# (key, name, month, first day, last day, discount) - Black Friday dates depend on the year.
PROMOTIONS = {
    2024: [
        ("spring_sale", "Spring Refresh Sale", (3, 14), (3, 24), 0.15),
        ("summer_sale", "Summer Sale", (7, 10), (7, 20), 0.20),
        ("black_friday", "Black Friday / Cyber Week", (11, 28), (12, 2), 0.25),
        ("holiday", "Holiday Gift Event", (12, 8), (12, 18), 0.15),
    ],
    2025: [
        ("spring_sale", "Spring Refresh Sale", (3, 13), (3, 23), 0.15),
        ("summer_sale", "Summer Sale", (7, 9), (7, 19), 0.20),
        ("black_friday", "Black Friday / Cyber Week", (11, 27), (12, 1), 0.25),
        ("holiday", "Holiday Gift Event", (12, 7), (12, 17), 0.15),
    ],
}


def build_products(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    sku = 1
    for category in CATEGORIES:
        n, median, sd, _ = CATEGORY_SPECS[category]
        prices = np.round(median * np.exp(rng.normal(0.0, sd, n)), 0) - 0.01
        prices = np.maximum(prices, 4.99)
        margins = rng.uniform(0.38, 0.62, n)
        for k in range(n):
            rows.append(
                (f"SKU{sku:04d}", f"{category.title()} item {k + 1:02d}", category,
                 float(prices[k]), float(round(prices[k] * (1 - margins[k]), 2)))
            )
            sku += 1
    return pd.DataFrame(
        rows, columns=["product_id", "product_name", "category", "list_price", "unit_cost"]
    )


@dataclass
class CampaignBook:
    """Campaign table plus lookups used while simulating behavior."""

    table: pd.DataFrame
    acquisition: dict[tuple[str, int], str]  # (channel, quarter index) -> campaign_id
    referral: str
    nurture: str
    retargeting: dict[str, str]
    welcome: str
    newsletter: str
    winback: str
    winback_discount: float
    welcome_discount: float
    promo_by_day: np.ndarray  # campaign index per day (-1 when no promotion)
    promo_ids: list[str]
    promo_discounts: np.ndarray


def quarter_index(ts: pd.Timestamp, start: pd.Timestamp) -> int:
    return (ts.year - start.year) * 4 + (ts.quarter - 1)


def build_campaigns(start: pd.Timestamp, end: pd.Timestamp) -> CampaignBook:
    last_day = end - pd.Timedelta(days=1)
    rows: list[tuple] = []

    def add(name: str, channel: str, objective: str, first: pd.Timestamp, last: pd.Timestamp,
            discount: float = 0.0) -> str:
        cid = f"CMP{len(rows) + 1:03d}"
        rows.append((cid, name, channel, objective, first, last, discount))
        return cid

    acquisition: dict[tuple[str, int], str] = {}
    for q_start in pd.date_range(start, last_day, freq="QS"):
        q_end = min(q_start + pd.offsets.QuarterEnd(0), last_day)
        label = f"{q_start.year}Q{q_start.quarter}"
        for channel in ACQUISITION_CAMPAIGN_CHANNELS:
            cid = add(f"{channel.replace('_', ' ').title()} Prospecting {label}", channel,
                      "acquisition", q_start, q_end)
            acquisition[(channel, quarter_index(q_start, start))] = cid

    referral = add("Refer-a-Friend Program", "referral", "acquisition", start, last_day)
    nurture = add("New Lead Nurture Series", "email", "nurture", start, last_day)
    retargeting = {
        ch: add(f"{ch.replace('_', ' ').title()} Cart Retargeting", ch, "retargeting", start,
                last_day)
        for ch in RETARGETING_CHANNELS
    }
    welcome_discount = 0.10
    welcome = add("Welcome Offer 10%", "email", "conversion", start, last_day, welcome_discount)
    newsletter = add("Customer Newsletter", "email", "retention", start, last_day)
    winback_discount = 0.20
    winback = add("We Miss You Win-back", "email", "winback", start, last_day, winback_discount)

    n_days = (end - start).days
    promo_by_day = np.full(n_days, -1, dtype=np.int64)
    promo_ids: list[str] = []
    promo_discounts: list[float] = []
    for year, promos in PROMOTIONS.items():
        for _, name, (m0, d0), (m1, d1), discount in promos:
            first = pd.Timestamp(year=year, month=m0, day=d0)
            last = pd.Timestamp(year=year, month=m1, day=d1)
            if first < start or last >= end:
                continue
            cid = add(f"{name} {year}", "email", "promotion", first, last, discount)
            promo_by_day[(first - start).days : (last - start).days + 1] = len(promo_ids)
            promo_ids.append(cid)
            promo_discounts.append(discount)

    table = pd.DataFrame(
        rows,
        columns=["campaign_id", "campaign_name", "channel", "objective", "start_date",
                 "end_date", "discount_pct"],
    )
    return CampaignBook(
        table=table,
        acquisition=acquisition,
        referral=referral,
        nurture=nurture,
        retargeting=retargeting,
        welcome=welcome,
        newsletter=newsletter,
        winback=winback,
        winback_discount=winback_discount,
        welcome_discount=welcome_discount,
        promo_by_day=promo_by_day,
        promo_ids=promo_ids,
        promo_discounts=np.array(promo_discounts),
    )


def build_experiments() -> pd.DataFrame:
    return pd.DataFrame(
        [
            (
                p.EXPERIMENT_ID,
                "one_page_checkout",
                "Replacing the three-step checkout with a one-page checkout increases the share of "
                "new visitors who complete a first purchase.",
                "prospect (person)",
                "Unconverted prospects with a web/app session during the experiment window",
                p.EXPERIMENT_START,
                p.EXPERIMENT_END,
                p.EXPERIMENT_TREATMENT_SHARE,
                "First-purchase conversion: share of assigned prospects whose first order is "
                "placed during the experiment window",
                "Average first-order net value; checkout-start rate",
            )
        ],
        columns=["experiment_id", "experiment_name", "hypothesis", "randomization_unit",
                 "eligible_population", "start_date", "end_date", "treatment_share",
                 "primary_metric", "guardrail_metrics"],
    )


def seasonal_day_multiplier(start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    """Annual demand shape: holiday peak, mid-summer bump and a post-holiday January lull."""
    days = pd.date_range(start, end - pd.Timedelta(days=1), freq="D")
    doy = days.dayofyear.to_numpy().astype(float)
    shape = (
        1.0
        + 0.55 * np.exp(-0.5 * ((doy - 338) / 13) ** 2)
        + 0.12 * np.exp(-0.5 * ((doy - 192) / 18) ** 2)
        - 0.12 * np.exp(-0.5 * ((doy - 30) / 16) ** 2)
    )
    weekday = np.array(p.WEEKDAY_MULT)[days.dayofweek.to_numpy()]
    return shape * weekday


@dataclass
class DemandCalendar:
    """Hourly multiplier of purchase demand used to thin a homogeneous Poisson process."""

    hourly: np.ndarray
    max_mult: float

    @classmethod
    def build(cls, start: pd.Timestamp, end: pd.Timestamp, book: CampaignBook) -> DemandCalendar:
        daily = seasonal_day_multiplier(start, end)
        daily = daily * np.where(book.promo_by_day >= 0, p.PROMO_DEMAND_MULT, 1.0)
        hour = np.array(p.HOURLY_PROFILE)
        hour = hour / hour.mean()
        hourly = np.repeat(daily, 24) * np.tile(hour, len(daily))
        return cls(hourly=hourly, max_mult=float(hourly.max()))
