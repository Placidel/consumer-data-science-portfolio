"""Generative assumptions of the Northstar Consumer simulation.

These are the *inputs* of the simulation (the "true" data-generating process), not analytical
results. Keeping them in one place makes the simulation auditable and lets later sections check
that their estimators recover known structure (for example the embedded A/B test effect).

Conventions: effects named ``*_LOGIT`` are additive on the log-odds scale; ``*_MULT`` are
multiplicative on a rate. Time is measured in days.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_SEED = 20240101
DEFAULT_N_PROSPECTS = 40_000

# ---------------------------------------------------------------- prospect mix
CHANNEL_WEIGHTS = {  # share of new leads at the start of the history
    "paid_search": 0.24,
    "paid_social": 0.18,
    "display": 0.10,
    "affiliate": 0.08,
    "email": 0.12,
    "referral": 0.08,
    "organic_search": 0.20,
}
# Linear drift in channel share over the two years (multiplier at the end of the history).
CHANNEL_DRIFT = {"paid_social": 1.5, "display": 0.6, "referral": 1.3}
CHANNEL_INTENT_LOGIT = {
    "paid_search": 0.30,
    "paid_social": -0.35,
    "display": -0.70,
    "affiliate": 0.00,
    "email": 0.35,
    "referral": 0.75,
    "organic_search": 0.20,
}
# Cost of the sourcing touch per lead (USD, lognormal median).
CHANNEL_LEAD_COST = {
    "paid_search": 18.0,
    "paid_social": 12.0,
    "display": 7.0,
    "affiliate": 22.0,
    "email": 1.5,
    "referral": 10.0,
}
# Customers acquired through these channels churn at a different base rate.
CHANNEL_CHURN_MULT = {
    "paid_search": 1.0,
    "paid_social": 1.35,
    "display": 1.25,
    "affiliate": 1.15,
    "email": 0.9,
    "referral": 0.7,
    "organic_search": 0.85,
}

REGION_WEIGHTS = {"northeast": 0.2, "southeast": 0.24, "midwest": 0.2, "southwest": 0.14,
                  "west": 0.22}
REGION_INTENT_LOGIT = {"northeast": 0.05, "southeast": -0.10, "midwest": 0.0, "southwest": -0.05,
                       "west": 0.10}
REGION_STORE_SHARE = {"northeast": 0.22, "southeast": 0.25, "midwest": 0.35, "southwest": 0.2,
                      "west": 0.15}
# Extra probability that a shipped order generates a delivery complaint.
REGION_DELIVERY_ISSUE = {"northeast": 0.01, "southeast": 0.03, "midwest": 0.01,
                         "southwest": 0.05, "west": 0.02}

AGE_WEIGHTS = {"18-24": 0.14, "25-34": 0.25, "35-44": 0.22, "45-54": 0.17, "55-64": 0.13,
               "65+": 0.09}
AGE_INTENT_LOGIT = {"18-24": -0.25, "25-34": 0.15, "35-44": 0.20, "45-54": 0.05, "55-64": -0.05,
                    "65+": -0.30}
INCOME_WEIGHTS = {"low": 0.3, "middle": 0.5, "high": 0.2}
INCOME_INTENT_LOGIT = {"low": -0.20, "middle": 0.0, "high": 0.25}
INCOME_BASKET_MULT = {"low": 0.8, "middle": 1.0, "high": 1.35}
DEVICE_WEIGHTS = {"mobile": 0.58, "desktop": 0.34, "tablet": 0.08}
DEVICE_APP_SHARE = {"mobile": 0.45, "desktop": 0.0, "tablet": 0.25}
EMAIL_OPT_IN_RATE = 0.55
EMAIL_OPT_IN_LOGIT = 0.25
INTENT_NOISE_SD = 1.0

# ---------------------------------------------------------------- prospect journey
RESEARCH_DAYS_MEAN = 50.0  # how long an average lead keeps visiting before giving up
SESSIONS_LOG_RATE = 0.5  # extra organic sessions ~ Poisson(exp(0.5 + 0.35 * intent))
SESSIONS_INTENT_SLOPE = 0.35
NURTURE_EMAILS = 8  # weekly nurture emails to opted-in prospects while unconverted
RETARGET_PROB = 0.25  # cart abandoners on paid social/display who click a retargeting ad

# Stage-transition log-odds for prospect sessions: base, intent slope.
FUNNEL_BASE_LOGIT = {"product_view": 1.1, "add_to_cart": -1.3, "checkout_start": 0.0,
                     "purchase": 0.55}
FUNNEL_INTENT_SLOPE = {"product_view": 0.45, "add_to_cart": 0.65, "checkout_start": 0.40,
                       "purchase": 0.35}
MOBILE_CHECKOUT_LOGIT = -0.35  # small screens lose people at checkout
CLICK_SESSION_CART_LOGIT = 0.45  # sessions from a clicked email/retargeting ad
RETURN_VISIT_CART_LOGIT = 0.25  # each earlier session adds consideration (up to 4)
SAVED_CART_CHECKOUT_LOGIT = 0.6  # a cart saved in an earlier session makes checkout likelier

# ---------------------------------------------------------------- embedded experiment
EXPERIMENT_ID = "EXP001"
EXPERIMENT_START = pd.Timestamp("2025-03-03")
EXPERIMENT_END = pd.Timestamp("2025-05-25")  # inclusive last day
EXPERIMENT_TREATMENT_SHARE = 0.5
# True effect of the one-page checkout on checkout_start -> purchase (log-odds).
EXPERIMENT_PURCHASE_LOGIT = 0.30
# True side effect on basket size: fewer add-on items at checkout.
EXPERIMENT_BASKET_MULT = 0.90

# ---------------------------------------------------------------- customer lifecycle
ORDERS_PER_MONTH_SHAPE = 1.5  # gamma heterogeneity in purchase frequency
ORDERS_PER_MONTH_SCALE = 0.30
ORDERS_INTENT_SLOPE = 0.25
BASE_DAILY_CHURN_HAZARD = 0.0022  # ~6.4% monthly hazard for an average active customer
CHURN_INTENT_SLOPE = -0.25
CHURN_NOISE_SD = 0.45
EARLY_CHURN_LOGIT = 0.25  # one-and-done buyers
EARLY_CHURN_INTENT_SLOPE = -0.55
EARLY_CHURN_MEAN_DAYS = 20.0
DISCOUNTED_FIRST_ORDER_CHURN_MULT = 1.2
REACTIVATION_PROB = 0.25
REACTIVATION_DELAY_MEAN = 120.0
REACTIVATED_CHURN_MULT = 1.3
BASKET_EXTRA_ITEMS = 0.7  # items per order ~ 1 + Poisson(0.7 * income multiplier)
CATEGORY_AFFINITY_ALPHA = 0.6
PROMO_USE_PROB = 0.7
WELCOME_OFFER_PROB = 0.5

# Membership (Northstar Plus).
MEMBERSHIP_JOIN_LOGIT = -3.6
MEMBERSHIP_JOIN_INTENT_SLOPE = 0.5
MEMBERSHIP_ANNUAL_SHARE = 0.3
MEMBERSHIP_PRICE = {"monthly": 9.99, "annual": 89.0}
MEMBERSHIP_PERIOD_DAYS = {"monthly": 30, "annual": 365}
MEMBERSHIP_CANCEL_PROB_ACTIVE = {"monthly": 0.05, "annual": 0.25}
MEMBERSHIP_CANCEL_PROB_LAPSED = {"monthly": 0.45, "annual": 0.70}
MEMBER_FREQUENCY_MULT = 1.25
MEMBER_CHURN_MULT = 0.6

# Support.
SUPPORT_BASE_PROB = 0.07
SUPPORT_SOFT_GOODS_PROB = 0.03  # orders containing apparel/footwear (fit/returns)
LOW_CSAT_CHURN_MULT = 2.0  # a bad service experience (latent CSAT <= 2)
MAX_HAZARD_MULT = 6.0
CSAT_RESPONSE_RATE = 0.55
# Latent CSAT = round(CSAT_BASE - CSAT_SLOWNESS * log1p(resolution_hours / 6) + noise).
CSAT_BASE = 4.6
CSAT_SLOWNESS = 0.8
CSAT_NOISE_SD = 0.9

# Engagement with customer email.
NEWSLETTER_OPEN = {"active": 0.28, "lapsed": 0.07}
NEWSLETTER_CLICK_GIVEN_OPEN = {"active": 0.18, "lapsed": 0.05}
ANNUAL_UNSUBSCRIBE_RATE = 0.18
WINBACK_ELIGIBLE_DAYS = 90  # win-back email goes to opted-in customers idle this long

# ---------------------------------------------------------------- seasonality
# Weekday multipliers (Mon..Sun) and hourly activity profile for demand.
WEEKDAY_MULT = (0.92, 0.94, 0.95, 0.97, 1.02, 1.12, 1.08)
HOURLY_PROFILE = (
    0.15, 0.08, 0.05, 0.04, 0.05, 0.12, 0.30, 0.55, 0.75, 0.90, 1.00, 1.05,
    1.15, 1.10, 1.00, 1.00, 1.05, 1.15, 1.35, 1.55, 1.60, 1.40, 0.95, 0.45,
)
PROMO_DEMAND_MULT = 1.35
LEAD_GROWTH_END_MULT = 1.5  # new-lead volume grows ~50% over the two years
