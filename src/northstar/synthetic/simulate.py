"""Agent-level simulation of Northstar Consumer prospects and customers.

Time inside the simulation is a float number of days since ``DATA_START``. Each person is
simulated independently as a small discrete-event process:

1. **Prospect journey** - after lead creation the person visits the site/app a random number of
   times (more often with higher latent purchase intent), receives weekly nurture emails if
   opted in, may be retargeted after abandoning a cart, and may be randomized into the checkout
   experiment. Each session walks the ordered funnel; the first purchase converts the prospect.
2. **Customer lifecycle** - repeat purchases follow a Poisson process thinned by a seasonal,
   weekday, hourly and promotion demand calendar. A latent churn hazard (raised by bad service
   experiences, lowered by membership) competes with the next purchase; churned customers can be
   reactivated, usually by a win-back email. Membership, support contacts, browsing and email
   engagement are generated along the way.

Latent traits (intent, frequency, churn hazard) are never written out. Only the observable
event logs are, so later sections must *learn* the structure from behavior before a cutoff.
"""

from __future__ import annotations

import hashlib
import heapq
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from northstar.schema import AGE_BANDS, CATEGORIES, CHANNELS, DEVICE_TYPES, INCOME_BANDS, REGIONS
from northstar.synthetic import params as p
from northstar.synthetic.catalog import (
    CATEGORY_SPECS,
    RETARGETING_CHANNELS,
    SOFT_GOODS,
    CampaignBook,
    DemandCalendar,
    quarter_index,
    seasonal_day_multiplier,
)

MINUTE = 1.0 / 1440.0
HOUR = 1.0 / 24.0
STAGE_KEYS = ("product_view", "add_to_cart", "checkout_start", "purchase")
STAGE_GAP_MINUTES = (1.5, 3.0, 2.0, 2.5)  # mean minutes between consecutive funnel stages


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def assignment_bucket(experiment_id: str, prospect_id: str) -> float:
    """Uniform [0, 1) bucket from a hash of the experiment and unit ids (salted per experiment)."""
    digest = hashlib.sha256(f"{experiment_id}:{prospect_id}".encode()).hexdigest()
    return int(digest[:8], 16) / 0x1_0000_0000


def experiment_variant(experiment_id: str, prospect_id: str, treatment_share: float) -> str:
    """Stable hash-based assignment: the same person always lands in the same arm."""
    bucket = assignment_bucket(experiment_id, prospect_id)
    return "treatment" if bucket < treatment_share else "control"


@dataclass
class EventLog:
    """Row buffers. Ids are assigned later, after chronological sorting."""

    sessions: list[tuple] = field(default_factory=list)
    events: list[tuple] = field(default_factory=list)
    touches: list[tuple] = field(default_factory=list)
    orders: list[tuple] = field(default_factory=list)
    lines: list[tuple] = field(default_factory=list)
    subscriptions: list[tuple] = field(default_factory=list)
    contacts: list[tuple] = field(default_factory=list)
    assignments: list[tuple] = field(default_factory=list)

    def add_touch(self, person: int, customer: int, campaign: str, channel: str, kind: str,
                  t: float, opened: bool, clicked: bool, cost: float) -> None:
        self.touches.append((person, customer, campaign, channel, kind, t, opened, clicked,
                             round(cost, 2)))

    def add_session(self, rng: np.random.Generator, person: int, customer: int, start: float,
                    platform: str, device: str, source: str, campaign: str | None,
                    depth: int) -> tuple[int, float]:
        """Record a session reaching funnel ``depth`` (1-5); returns (index, last event time)."""
        idx = len(self.sessions)
        t = start
        self.events.append((idx, t, 1))
        for stage in range(2, depth + 1):
            t += (0.3 + rng.exponential(STAGE_GAP_MINUTES[stage - 2])) * MINUTE
            self.events.append((idx, t, stage))
        pages = depth + int(rng.poisson(1.0 + 1.2 * depth))
        self.sessions.append((person, customer, start, platform, device, source, campaign, pages))
        return idx, t

    def add_purchase_session(self, rng: np.random.Generator, person: int, customer: int,
                             order_t: float, platform: str, device: str, source: str,
                             campaign: str | None) -> int:
        """Full-funnel session whose purchase event lands exactly on ``order_t``."""
        gaps = [(0.3 + rng.exponential(g)) * MINUTE for g in STAGE_GAP_MINUTES]
        idx = len(self.sessions)
        t = order_t - sum(gaps)
        self.events.append((idx, t, 1))
        for stage, gap in enumerate(gaps, start=2):
            t += gap
            self.events.append((idx, order_t if stage == 5 else t, stage))
        pages = 5 + int(rng.poisson(7.0))
        self.sessions.append((person, customer, order_t - sum(gaps), platform, device, source,
                              campaign, pages))
        return idx


@dataclass
class Prospects:
    table: pd.DataFrame
    created: np.ndarray  # float days
    intent: np.ndarray


def _choice_from_weights(rng: np.random.Generator, weights: dict[str, float], n: int
                         ) -> np.ndarray:
    keys = list(weights)
    probs = np.array([weights[k] for k in keys], dtype=float)
    return np.array(keys, dtype=object)[rng.choice(len(keys), size=n, p=probs / probs.sum())]


def build_prospects(rng: np.random.Generator, n: int, start: pd.Timestamp, end: pd.Timestamp,
                    book: CampaignBook) -> Prospects:
    n_days = (end - start).days
    frac = np.arange(n_days) / n_days
    day_w = seasonal_day_multiplier(start, end) * (1 + (p.LEAD_GROWTH_END_MULT - 1) * frac)
    day_w = day_w * np.where(book.promo_by_day >= 0, 1.1, 1.0)
    days = np.sort(rng.choice(n_days, size=n, p=day_w / day_w.sum()))
    hour_w = np.array(p.HOURLY_PROFILE)
    hours = rng.choice(24, size=n, p=hour_w / hour_w.sum())
    created = days + (hours + rng.random(n)) / 24.0
    created = np.sort(created)

    # Channel mix drifts linearly over time.
    t_frac = created / n_days
    base = np.array([p.CHANNEL_WEIGHTS[c] for c in CHANNELS])
    drift = np.array([p.CHANNEL_DRIFT.get(c, 1.0) for c in CHANNELS])
    w = base[None, :] * (1 + (drift[None, :] - 1) * t_frac[:, None])
    cum = np.cumsum(w / w.sum(axis=1, keepdims=True), axis=1)
    channel_idx = np.minimum((rng.random(n)[:, None] > cum).sum(axis=1), len(CHANNELS) - 1)
    channel = np.array(CHANNELS, dtype=object)[channel_idx]

    region = _choice_from_weights(rng, p.REGION_WEIGHTS, n)
    age = _choice_from_weights(rng, p.AGE_WEIGHTS, n)
    income = _choice_from_weights(rng, p.INCOME_WEIGHTS, n)
    device = _choice_from_weights(rng, p.DEVICE_WEIGHTS, n)
    opt_in = (rng.random(n) < p.EMAIL_OPT_IN_RATE) | (channel == "email")

    intent = (
        np.array([p.CHANNEL_INTENT_LOGIT[c] for c in channel])
        + np.array([p.REGION_INTENT_LOGIT[r] for r in region])
        + np.array([p.AGE_INTENT_LOGIT[a] for a in age])
        + np.array([p.INCOME_INTENT_LOGIT[i] for i in income])
        + p.EMAIL_OPT_IN_LOGIT * opt_in
        + rng.normal(0.0, p.INTENT_NOISE_SD, n)
    )

    created_ts = start + pd.to_timedelta(np.round(created * 86400).astype(np.int64), unit="s")
    campaign = []
    for c, ts in zip(channel, created_ts, strict=True):
        if c == "referral":
            campaign.append(book.referral)
        elif c == "organic_search":
            campaign.append(None)
        else:
            campaign.append(book.acquisition[(c, quarter_index(ts, start))])

    table = pd.DataFrame(
        {
            "prospect_id": [f"P{i + 1:06d}" for i in range(n)],
            "created_at": created_ts,
            "acquisition_channel": channel,
            "campaign_id": campaign,
            "region": region,
            "age_band": age,
            "income_band": income,
            "device_type": device,
            "email_opt_in": opt_in,
        }
    )
    for col, values in (("region", REGIONS), ("age_band", AGE_BANDS),
                        ("income_band", INCOME_BANDS), ("device_type", DEVICE_TYPES)):
        assert set(table[col]) <= set(values), col
    return Prospects(table=table, created=created, intent=intent)


@dataclass
class Conversion:
    person: int
    t: float
    session: int
    platform: str
    treated: bool


def simulate_prospect_journeys(rng: np.random.Generator, prospects: Prospects,
                               book: CampaignBook, n_days: int, start: pd.Timestamp,
                               log: EventLog) -> list[Conversion]:
    table = prospects.table
    exp_start = (p.EXPERIMENT_START - start).days
    exp_end = (p.EXPERIMENT_END - start).days + 1.0
    conversions: list[Conversion] = []

    channels = table["acquisition_channel"].to_numpy()
    campaigns = table["campaign_id"].to_numpy()
    devices = table["device_type"].to_numpy()
    opt_ins = table["email_opt_in"].to_numpy()
    ids = table["prospect_id"].to_numpy()

    for i in range(len(table)):
        t0 = float(prospects.created[i])
        z = float(prospects.intent[i])
        channel = channels[i]
        device = devices[i]
        primary_platform = "app" if rng.random() < p.DEVICE_APP_SHARE[device] else "web"

        if channel != "organic_search":
            kind = {"email": "email", "referral": "referral_credit"}.get(channel, "ad_click")
            cost = p.CHANNEL_LEAD_COST[channel] * math.exp(rng.normal(0.0, 0.35))
            log.add_touch(i, -1, campaigns[i], channel, kind, t0, True, True, cost)

        seq = 0
        queue: list[tuple] = [(t0, seq, "session", channel, campaigns[i], 0.0)]
        research = rng.exponential(p.RESEARCH_DAYS_MEAN * (0.6 + 0.8 * sigmoid(z)))
        for _ in range(rng.poisson(math.exp(p.SESSIONS_LOG_RATE + p.SESSIONS_INTENT_SLOPE * z))):
            seq += 1
            source = "direct" if rng.random() < 0.55 else "organic_search"
            queue.append((t0 + rng.uniform(0.0, research), seq, "session", source, None, 0.0))
        if opt_ins[i]:
            for k in range(p.NURTURE_EMAILS):
                seq += 1
                queue.append((math.floor(t0) + 1 + 7 * k + 10 * HOUR, seq, "email", "email",
                              book.nurture, 0.0))
        heapq.heapify(queue)

        variant: str | None = None
        prior_sessions = 0
        saved_cart = False
        while queue:
            t, _, kind, source, campaign, boost = heapq.heappop(queue)
            if t >= n_days:
                break
            if kind == "email":
                opened = rng.random() < sigmoid(-0.8 + 0.4 * z)
                clicked = opened and rng.random() < sigmoid(-1.2 + 0.5 * z)
                log.add_touch(i, -1, campaign, "email", "email", t, opened, clicked, 0.01)
                if clicked:
                    seq += 1
                    heapq.heappush(queue, (t + rng.exponential(1.5 * HOUR), seq, "session",
                                           "email", campaign, p.CLICK_SESSION_CART_LOGIT))
                continue
            if kind == "retarget":
                log.add_touch(i, -1, campaign, source, "ad_click", t, True, True,
                              1.8 * math.exp(rng.normal(0.0, 0.3)))
                seq += 1
                heapq.heappush(queue, (t + MINUTE, seq, "session", source, campaign,
                                       p.CLICK_SESSION_CART_LOGIT))
                continue

            in_window = exp_start <= t < exp_end
            if in_window and variant is None:
                variant = experiment_variant(p.EXPERIMENT_ID, ids[i], p.EXPERIMENT_TREATMENT_SHARE)
                log.assignments.append((i, variant, t))
            treated = in_window and variant == "treatment"

            session_device = device if rng.random() < 0.85 else DEVICE_TYPES[rng.integers(3)]
            platform = primary_platform if rng.random() < 0.8 else (
                "web" if primary_platform == "app" else "app")
            if session_device == "desktop":
                platform = "web"
            mobile = session_device == "mobile"
            logits = {
                k: p.FUNNEL_BASE_LOGIT[k] + p.FUNNEL_INTENT_SLOPE[k] * z for k in STAGE_KEYS
            }
            logits["add_to_cart"] += boost + p.RETURN_VISIT_CART_LOGIT * min(prior_sessions, 4)
            logits["checkout_start"] += (p.MOBILE_CHECKOUT_LOGIT * mobile
                                         + p.SAVED_CART_CHECKOUT_LOGIT * saved_cart)
            logits["purchase"] += p.MOBILE_CHECKOUT_LOGIT * mobile
            logits["purchase"] += p.EXPERIMENT_PURCHASE_LOGIT * treated
            depth = 1
            for key in STAGE_KEYS:
                if rng.random() >= sigmoid(logits[key]):
                    break
                depth += 1
            idx, last_t = log.add_session(rng, i, -1, t, platform, session_device, source,
                                          campaign, depth)
            prior_sessions += 1
            saved_cart = saved_cart or depth >= 3
            if depth == 5:
                if last_t < n_days:
                    conversions.append(Conversion(i, last_t, idx, platform, treated))
                break
            if depth >= 3 and rng.random() < p.RETARGET_PROB:
                rt_channel = RETARGETING_CHANNELS[rng.integers(len(RETARGETING_CHANNELS))]
                seq += 1
                heapq.heappush(queue, (t + rng.uniform(0.5, 4.0), seq, "retarget", rt_channel,
                                       book.retargeting[rt_channel], 0.0))
    return conversions


@dataclass
class CustomerContext:
    """Per-simulation lookups shared by all customers."""

    rng: np.random.Generator
    book: CampaignBook
    calendar: DemandCalendar
    n_days: int
    product_ids_by_cat: dict[str, np.ndarray]
    product_prices_by_cat: dict[str, np.ndarray]
    product_weights_by_cat: dict[str, np.ndarray]
    category_alpha: np.ndarray
    newsletter_days: np.ndarray
    winback_days: np.ndarray
    promo_email_days: np.ndarray
    promo_email_ids: list[str]


def build_customer_context(rng: np.random.Generator, products: pd.DataFrame, book: CampaignBook,
                           start: pd.Timestamp, end: pd.Timestamp) -> CustomerContext:
    n_days = (end - start).days
    ids, prices, weights = {}, {}, {}
    for cat in CATEGORIES:
        sub = products.loc[products["category"] == cat]
        ids[cat] = sub["product_id"].to_numpy()
        prices[cat] = sub["list_price"].to_numpy()
        w = 1.0 / np.arange(1, len(sub) + 1) ** 0.8  # a few best sellers per category
        weights[cat] = w / w.sum()
    popularity = np.array([CATEGORY_SPECS[c][3] for c in CATEGORIES])
    first_tuesday = (7 - start.dayofweek + 1) % 7
    newsletter = np.arange(first_tuesday, n_days, 14) + 10 * HOUR
    months = pd.date_range(start, end - pd.Timedelta(days=1), freq="MS")
    winback = np.array([(m - start).days for m in months], dtype=float) + 11 * HOUR
    promo_days, promo_ids = [], []
    for _, row in book.table.loc[book.table["objective"] == "promotion"].iterrows():
        promo_days.append((row["start_date"] - start).days + 9 * HOUR)
        promo_ids.append(row["campaign_id"])
    return CustomerContext(
        rng=rng,
        book=book,
        calendar=DemandCalendar.build(start, end, book),
        n_days=n_days,
        product_ids_by_cat=ids,
        product_prices_by_cat=prices,
        product_weights_by_cat=weights,
        category_alpha=p.CATEGORY_AFFINITY_ALPHA * len(CATEGORIES) * popularity,
        newsletter_days=newsletter,
        winback_days=winback,
        promo_email_days=np.array(promo_days),
        promo_email_ids=promo_ids,
    )


class CustomerSimulator:
    """Simulates one customer's post-conversion lifecycle into the shared ``EventLog``."""

    def __init__(self, ctx: CustomerContext, log: EventLog, cust: int, person: int,
                 prospect: pd.Series, z: float, conversion: Conversion) -> None:
        self.ctx, self.log, self.rng = ctx, log, ctx.rng
        self.cust, self.person, self.z = cust, person, z
        self.region = prospect["region"]
        self.device = prospect["device_type"]
        self.opt_in = bool(prospect["email_opt_in"])
        rng = self.rng
        self.conversion = conversion
        self.since = conversion.t
        self.rate = (rng.gamma(p.ORDERS_PER_MONTH_SHAPE, p.ORDERS_PER_MONTH_SCALE)
                     * math.exp(p.ORDERS_INTENT_SLOPE * z) / 30.0)
        self.base_hazard = (p.BASE_DAILY_CHURN_HAZARD
                            * p.CHANNEL_CHURN_MULT[prospect["acquisition_channel"]]
                            * math.exp(p.CHURN_INTENT_SLOPE * z
                                       + rng.normal(0.0, p.CHURN_NOISE_SD)))
        self.hazard_mult = 1.0
        early = rng.random() < sigmoid(p.EARLY_CHURN_LOGIT + p.EARLY_CHURN_INTENT_SLOPE * z)
        self.forced_churn = (self.since + rng.exponential(p.EARLY_CHURN_MEAN_DAYS)
                             if early else math.inf)
        self.affinity = rng.dirichlet(ctx.category_alpha)
        self.basket_mean = p.BASKET_EXTRA_ITEMS * p.INCOME_BASKET_MULT[prospect["income_band"]]
        self.store_share = p.REGION_STORE_SHARE[self.region]
        self.app_share = p.DEVICE_APP_SHARE[self.device]
        self.unsubscribed_at = self.since + rng.exponential(365.0 / p.ANNUAL_UNSUBSCRIBE_RATE)
        self.member_plan: str | None = None
        self.next_renewal = math.inf
        self.last_sub_event = -math.inf
        self.ever_cancelled = False
        self.order_times: list[float] = []
        self.active_intervals: list[tuple[float, float]] = []
        self.winback_clicks: set[int] = set()

    # ------------------------------------------------------------ helpers
    def _next_order_time(self, t: float, rate: float) -> float:
        cal = self.ctx.calendar
        lam_max = rate * cal.max_mult
        while True:
            t += self.rng.exponential(1.0 / lam_max)
            if t >= self.ctx.n_days:
                return math.inf
            accept = cal.hourly[int(t * 24)] / cal.max_mult
            if self.rng.random() < accept and t - self.order_times[-1] > 0.5:
                return t

    def _order_channel(self) -> str:
        u = self.rng.random()
        if u < self.store_share:
            return "store"
        if u < self.store_share + (1 - self.store_share) * self.app_share:
            return "app"
        return "web"

    def _place_order(self, t: float, channel: str, session: int | None, campaign: str | None,
                     discount: float, basket_mult: float = 1.0) -> int:
        rng, ctx = self.rng, self.ctx
        n_items = 1 + int(rng.poisson(self.basket_mean * basket_mult))
        cats = rng.choice(len(CATEGORIES), size=n_items, p=self.affinity)
        picks: dict[tuple[str, int], int] = {}
        for c in cats:
            cat = CATEGORIES[c]
            k = int(rng.choice(len(ctx.product_ids_by_cat[cat]), p=ctx.product_weights_by_cat[cat]))
            picks[(cat, k)] = picks.get((cat, k), 0) + 1
        order_idx = len(self.log.orders)
        gross = disc = 0.0
        for (cat, k), qty in picks.items():
            price = float(ctx.product_prices_by_cat[cat][k])
            line_gross = round(qty * price, 2)
            line_disc = round(line_gross * discount, 2)
            self.log.lines.append((order_idx, ctx.product_ids_by_cat[cat][k], qty, price,
                                   line_disc, round(line_gross - line_disc, 2)))
            gross += line_gross
            disc += line_disc
        gross, disc = round(gross, 2), round(disc, 2)
        self.log.orders.append((self.cust, t, channel, session, campaign, n_items, gross, disc,
                                round(gross - disc, 2)))
        self.order_times.append(t)
        self._maybe_support(t, order_idx, channel, any(CATEGORIES[c] in SOFT_GOODS for c in cats))
        self._maybe_join(t)
        return order_idx

    def _maybe_support(self, t: float, order_idx: int | None, channel: str,
                       soft_goods: bool, reason: str | None = None) -> None:
        rng = self.rng
        if reason is None:
            shipped = channel != "store"
            prob = (p.SUPPORT_BASE_PROB + p.SUPPORT_SOFT_GOODS_PROB * soft_goods
                    + (p.REGION_DELIVERY_ISSUE[self.region] if shipped else -0.02))
            if rng.random() >= prob:
                return
            if shipped:
                reasons = ["delivery_delay", "return_request", "product_quality",
                           "account_access"]
                weights = [0.35 + 4 * p.REGION_DELIVERY_ISSUE[self.region],
                           0.30 + 0.15 * soft_goods, 0.2, 0.05]
            else:
                reasons = ["return_request", "product_quality", "account_access"]
                weights = [0.45, 0.4, 0.15]
            w = np.array(weights)
            reason = reasons[rng.choice(len(reasons), p=w / w.sum())]
        delay = {"delivery_delay": 4 + rng.exponential(3), "return_request": 6 + rng.exponential(6),
                 "product_quality": 3 + rng.exponential(8), "account_access": rng.exponential(10),
                 "billing": rng.exponential(2)}[reason]
        contact_t = t + delay + rng.random() * HOUR
        if contact_t >= self.ctx.n_days:
            return
        contact_channel = ("chat", "email", "phone")[rng.choice(3, p=[0.45, 0.35, 0.2])]
        hours = {"phone": 3.0, "chat": 8.0, "email": 28.0}[contact_channel]
        hours *= math.exp(rng.normal(0.0, 0.9)) * (1.5 if reason == "delivery_delay" else 1.0)
        resolved_t: float | None = contact_t + hours * HOUR
        if rng.random() < 0.04 or resolved_t >= self.ctx.n_days:
            resolved_t = None
        if resolved_t is None:
            latent_csat = 1 + int(rng.random() < 0.3)
        else:
            raw = (p.CSAT_BASE - p.CSAT_SLOWNESS * math.log1p(hours / 6.0)
                   + rng.normal(0.0, p.CSAT_NOISE_SD))
            latent_csat = int(min(5, max(1, round(raw))))
        reported = resolved_t is not None and rng.random() < p.CSAT_RESPONSE_RATE
        self.log.contacts.append((self.cust, order_idx, contact_t, contact_channel, reason,
                                  resolved_t, latent_csat if reported else None))
        if latent_csat <= 2:
            self.hazard_mult = min(self.hazard_mult * p.LOW_CSAT_CHURN_MULT, p.MAX_HAZARD_MULT)

    def _maybe_join(self, t: float) -> None:
        if self.member_plan is not None or t < self.last_sub_event:
            return
        logit = (p.MEMBERSHIP_JOIN_LOGIT + p.MEMBERSHIP_JOIN_INTENT_SLOPE * self.z
                 - 1.0 * self.ever_cancelled)
        if self.rng.random() >= sigmoid(logit):
            return
        join_t = t + self.rng.uniform(1, 30) * MINUTE
        if join_t >= self.ctx.n_days:
            return
        plan = "annual" if self.rng.random() < p.MEMBERSHIP_ANNUAL_SHARE else "monthly"
        self.member_plan = plan
        self.log.subscriptions.append((self.cust, join_t, "subscribe", plan,
                                       p.MEMBERSHIP_PRICE[plan]))
        self.last_sub_event = join_t
        self.next_renewal = join_t + p.MEMBERSHIP_PERIOD_DAYS[plan]

    def _renewal(self, t: float, active: bool) -> None:
        plan = self.member_plan
        assert plan is not None
        cancel_prob = (p.MEMBERSHIP_CANCEL_PROB_ACTIVE if active
                       else p.MEMBERSHIP_CANCEL_PROB_LAPSED)[plan]
        if self.rng.random() < cancel_prob:
            cancel_t = max(self.last_sub_event + HOUR, t - self.rng.uniform(1, 20))
            self.log.subscriptions.append((self.cust, cancel_t, "cancel", plan, 0.0))
            self.last_sub_event = t  # benefits run to the end of the paid period
            self.member_plan = None
            self.next_renewal = math.inf
            self.ever_cancelled = True
        else:
            self.log.subscriptions.append((self.cust, t, "renew", plan, p.MEMBERSHIP_PRICE[plan]))
            self.last_sub_event = t
            self.next_renewal = t + p.MEMBERSHIP_PERIOD_DAYS[plan]
            if self.rng.random() < 0.03:
                self._maybe_support(t, None, "web", False, reason="billing")

    def _reactivation_time(self, churn_t: float) -> tuple[float, int | None]:
        rng, ctx = self.rng, self.ctx
        if rng.random() >= p.REACTIVATION_PROB:
            return math.inf, None
        candidate = churn_t + rng.exponential(p.REACTIVATION_DELAY_MEAN)
        if not self.opt_in:
            return candidate, None
        eligible = max(candidate, self.order_times[-1] + p.WINBACK_ELIGIBLE_DAYS)
        k = int(np.searchsorted(ctx.winback_days, eligible))
        if k >= len(ctx.winback_days) or ctx.winback_days[k] >= self.unsubscribed_at:
            return math.inf, None
        return ctx.winback_days[k] + HOUR + rng.exponential(4.0), k

    # ------------------------------------------------------------ main loop
    def run(self) -> None:
        rng, ctx, book = self.rng, self.ctx, self.ctx.book
        conv = self.conversion
        day = int(conv.t)
        promo = book.promo_by_day[day]
        campaign, discount = None, 0.0
        if promo >= 0 and rng.random() < p.PROMO_USE_PROB:
            campaign, discount = book.promo_ids[promo], float(book.promo_discounts[promo])
        elif self.opt_in and rng.random() < p.WELCOME_OFFER_PROB:
            campaign, discount = book.welcome, book.welcome_discount
        if discount > 0:
            self.base_hazard *= p.DISCOUNTED_FIRST_ORDER_CHURN_MULT
        basket_mult = p.EXPERIMENT_BASKET_MULT if conv.treated else 1.0
        self._place_order(conv.t, conv.platform, conv.session, campaign, discount, basket_mult)

        t = conv.t
        active = True
        interval_start = t
        react_at, react_winback = math.inf, None
        while True:
            member = self.member_plan is not None
            if active:
                rate = self.rate * (p.MEMBER_FREQUENCY_MULT if member else 1.0)
                hazard = (self.base_hazard * self.hazard_mult
                          * (p.MEMBER_CHURN_MULT if member else 1.0))
                t_order = self._next_order_time(t, rate)
                t_churn = min(t + rng.exponential(1.0 / hazard), self.forced_churn)
            else:
                t_order, t_churn = react_at, math.inf
            t_next = min(t_order, t_churn, self.next_renewal)
            if t_next >= ctx.n_days:
                break
            t = t_next
            if t == self.next_renewal:
                self._renewal(t, active)
                continue
            if t == t_churn:
                active = False
                self.forced_churn = math.inf
                self.active_intervals.append((interval_start, t))
                react_at, react_winback = self._reactivation_time(t)
                continue
            # purchase
            order_campaign, discount, source = None, 0.0, None
            if not active:
                active = True
                interval_start = t
                self.hazard_mult *= p.REACTIVATED_CHURN_MULT
                if react_winback is not None:
                    self.winback_clicks.add(react_winback)
                    order_campaign, discount, source = book.winback, book.winback_discount, "email"
                react_at, react_winback = math.inf, None
            promo = book.promo_by_day[int(t)]
            if order_campaign is None and promo >= 0 and rng.random() < p.PROMO_USE_PROB:
                order_campaign = book.promo_ids[promo]
                discount = float(book.promo_discounts[promo])
                source = "email" if rng.random() < 0.5 else None
            channel = self._order_channel()
            session = None
            if channel != "store":
                device = "desktop" if channel == "web" and rng.random() < 0.5 else self.device
                if channel == "app" and device == "desktop":
                    device = "mobile"
                if source is None:
                    src, src_campaign = ("direct" if rng.random() < 0.7 else "organic_search"), None
                else:
                    src, src_campaign = source, order_campaign
                session = self.log.add_purchase_session(rng, self.person, self.cust, t, channel,
                                                        device, src, src_campaign)
            self._place_order(t, channel, session, order_campaign, discount)
        if active:
            self.active_intervals.append((interval_start, float(ctx.n_days)))
        self._browse_sessions()
        self._customer_emails()

    def _is_active(self, t: float) -> bool:
        return any(a <= t < b for a, b in self.active_intervals)

    def _browse(self, t: float, source: str, campaign: str | None, active: bool) -> None:
        rng = self.rng
        platform = "app" if rng.random() < self.app_share else "web"
        device = self.device if platform == "app" or self.device != "mobile" else (
            "mobile" if rng.random() < 0.6 else "desktop")
        if platform == "app" and device == "desktop":
            device = "mobile"
        logits = (1.2, -0.6 if active else -1.4, -0.2)
        depth = 1
        for lg in logits:
            if rng.random() >= sigmoid(lg):
                break
            depth += 1
        self.log.add_session(rng, self.person, self.cust, t, platform, device, source, campaign,
                             depth)

    def _browse_sessions(self) -> None:
        rng, n_days = self.rng, self.ctx.n_days
        dormant_start = self.since
        for a, b in self.active_intervals:
            if a > dormant_start:
                self._dormant_browse(dormant_start, a)
            for s in np.sort(rng.uniform(a, b, rng.poisson(1.3 * self.rate * (b - a)))):
                source = "direct" if rng.random() < 0.6 else "organic_search"
                self._browse(float(s), source, None, active=True)
            dormant_start = b
        if dormant_start < n_days:
            self._dormant_browse(dormant_start, n_days)

    def _dormant_browse(self, a: float, b: float) -> None:
        rng = self.rng
        for s in np.sort(rng.uniform(a, b, rng.poisson(0.004 * (b - a)))):
            self._browse(float(s), "direct", None, active=False)

    def _customer_emails(self) -> None:
        if not self.opt_in:
            return
        rng, ctx, book = self.rng, self.ctx, self.ctx.book
        stop = min(self.unsubscribed_at, float(ctx.n_days))

        def engage(t: float, campaign: str, click_mult: float = 1.0) -> None:
            state = "active" if self._is_active(t) else "lapsed"
            opened = rng.random() < p.NEWSLETTER_OPEN[state]
            clicked = opened and rng.random() < min(
                1.0, p.NEWSLETTER_CLICK_GIVEN_OPEN[state] * click_mult)
            self.log.add_touch(self.person, self.cust, campaign, "email", "email", t, opened,
                               clicked, 0.005)
            if clicked:
                self._browse(t + rng.exponential(2 * HOUR), "email", campaign,
                             active=state == "active")

        for t in ctx.newsletter_days:
            if self.since < t < stop:
                engage(float(t), book.newsletter)
        for t, cid in zip(ctx.promo_email_days, ctx.promo_email_ids, strict=True):
            if self.since < t < stop:
                engage(float(t), cid, click_mult=1.3)
        orders = np.array(self.order_times)
        for k, t in enumerate(ctx.winback_days):
            if not self.since < t < stop:
                continue
            last = orders[orders < t].max()
            if t - last < p.WINBACK_ELIGIBLE_DAYS:
                continue
            if k in self.winback_clicks:
                opened, clicked = True, True
            else:
                opened = rng.random() < 0.12
                clicked = opened and rng.random() < 0.15
            self.log.add_touch(self.person, self.cust, book.winback, "email", "email", float(t),
                               opened, clicked, 0.01)
