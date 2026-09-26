"""Request and response contracts for the scoring API and the batch scorer.

A request carries the point-in-time feature record that sections 01 and 02 build for a scoring
date (``build_features``); the service does not look anything up. Validation is deliberately
strict because a malformed record does not crash a model, it just scores wrongly:

* unknown fields are rejected (a typo, or an outcome column such as ``converted``, never slips in);
* categorical values must be levels of the shared schema (the one-hot encoder would silently
  score an unknown level as "none of the above");
* numbers must be finite and inside their domain: counts are whole and non-negative, shares lie in
  [0, 1], and records outside the model's population (a lead older than the pipeline window, a
  customer with no order in the active window) are refused rather than extrapolated;
* identities that hold for every correctly built record are checked (e.g. 7-day sessions cannot
  exceed 30-day sessions; an open rate must equal opens / emails), which catches broken upstream
  feature pipelines.

Every record the section feature builders produce passes these checks (see the tests).
"""

from __future__ import annotations

from collections import Counter
from typing import Annotated, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from northstar.acquisition.dataset import PIPELINE_DAYS
from northstar.retention.dataset import ACTIVE_DAYS
from northstar.schema import AGE_BANDS, CATEGORIES, CHANNELS, DEVICE_TYPES, INCOME_BANDS, REGIONS

MAX_RECORDS = 1000
TOLERANCE = 1e-3  # for identities between derived float features

# Literal[tuple] is equivalent to listing the levels; the tuples are the shared schema's domains.
Channel = Literal[CHANNELS]
Region = Literal[REGIONS]
AgeBand = Literal[AGE_BANDS]
IncomeBand = Literal[INCOME_BANDS]
DeviceType = Literal[DEVICE_TYPES]

EntityId = Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_\-]+$")]
Count = Annotated[int, Field(ge=0, description="Whole, non-negative count.")]
Flag = Annotated[int, Field(ge=0, le=1, description="1 = yes, 0 = no.")]
Share = Annotated[float, Field(ge=0.0, le=1.0)]
Days = Annotated[float, Field(ge=0.0)]


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    acquisition_channel: Channel = Field(description="Channel that sourced the lead.")
    region: Region
    age_band: AgeBand
    income_band: IncomeBand
    device_type: DeviceType = Field(description="Device at sign-up.")
    email_opt_in: Flag


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


class LeadFeatures(_Record):
    """One open lead as of the scoring date (section 01 feature definitions)."""

    prospect_id: EntityId
    lead_age_days: float = Field(gt=0.0, le=PIPELINE_DAYS,
                                 description="Days since the lead was created (open pipeline "
                                             f"covers {PIPELINE_DAYS} days).")
    sessions_total: Count
    sessions_7d: Count
    sessions_30d: Count
    days_since_last_session: Days
    max_stage_reached: int = Field(ge=0, le=4, description="Deepest funnel stage before the "
                                   "scoring date: 1 session ... 4 checkout (5 = purchase would "
                                   "mean the lead already converted).")
    cart_sessions: Count
    cart_sessions_7d: Count
    checkout_sessions: Count
    pages_viewed: Count
    app_session_share: Share
    mobile_session_share: Share
    emails_received: Count = Field(description="Follow-up emails after lead creation.")
    emails_opened: Count
    emails_clicked: Count
    email_open_rate: Share
    retarget_clicks: Count

    @model_validator(mode="after")
    def _consistent(self) -> LeadFeatures:
        _require(self.sessions_7d <= self.sessions_30d <= self.sessions_total,
                 "sessions must satisfy sessions_7d <= sessions_30d <= sessions_total")
        _require(self.checkout_sessions <= self.cart_sessions <= self.sessions_total,
                 "sessions must satisfy checkout_sessions <= cart_sessions <= sessions_total")
        _require(self.cart_sessions_7d <= min(self.cart_sessions, self.sessions_7d),
                 "cart_sessions_7d cannot exceed cart_sessions or sessions_7d")
        _require((self.max_stage_reached >= 3) == (self.cart_sessions > 0),
                 "max_stage_reached >= 3 (cart) must coincide with cart_sessions > 0")
        _require((self.max_stage_reached >= 4) == (self.checkout_sessions > 0),
                 "max_stage_reached >= 4 (checkout) must coincide with checkout_sessions > 0")
        _require(self.days_since_last_session <= self.lead_age_days + TOLERANCE,
                 "days_since_last_session cannot exceed lead_age_days")
        _require(max(self.emails_opened, self.emails_clicked) <= self.emails_received,
                 "emails_opened and emails_clicked cannot exceed emails_received")
        expected = self.emails_opened / self.emails_received if self.emails_received else 0.0
        _require(abs(self.email_open_rate - expected) <= TOLERANCE,
                 "email_open_rate must equal emails_opened / emails_received (0 if none)")
        return self


class CustomerFeatures(_Record):
    """One active customer as of the scoring date (section 02 feature definitions)."""

    customer_id: EntityId
    tenure_days: float = Field(gt=0.0, description="Days since the first order.")
    days_since_last_order: float = Field(ge=0.0, lt=ACTIVE_DAYS,
                                         description="Active base: last order within "
                                                     f"{ACTIVE_DAYS} days.")
    orders_total: int = Field(ge=1)
    orders_90d: Count
    orders_prev_90d: Count = Field(description="Orders 90 to 180 days before the scoring date.")
    net_revenue_180d: float = Field(ge=0.0)
    avg_order_value: float = Field(ge=0.0)
    discount_share: Share
    first_order_discounted: Flag
    category_count: int = Field(ge=1, le=len(CATEGORIES))
    store_order_share: Share
    app_order_share: Share
    browse_sessions_30d: Count
    browse_sessions_90d: Count
    days_since_last_session: Days
    emails_received_90d: Count
    email_open_rate_90d: Share
    email_clicks_90d: Count
    plus_member: Flag
    plus_cancelled_180d: Flag
    support_contacts_180d: Count
    low_csat_contacts_180d: Count
    slow_resolution_contacts_180d: Count
    open_contacts: Count

    @model_validator(mode="after")
    def _consistent(self) -> CustomerFeatures:
        _require(self.days_since_last_order <= self.tenure_days + TOLERANCE,
                 "days_since_last_order cannot exceed tenure_days")
        recent = self.orders_90d + self.orders_prev_90d
        _require(1 <= recent <= self.orders_total,
                 "an active customer needs 1 <= orders_90d + orders_prev_90d <= orders_total")
        _require((self.orders_90d > 0) == (self.days_since_last_order < 90),
                 "orders_90d > 0 must coincide with days_since_last_order < 90")
        _require(self.store_order_share + self.app_order_share <= 1 + TOLERANCE,
                 "store_order_share + app_order_share cannot exceed 1")
        _require(self.browse_sessions_30d <= self.browse_sessions_90d,
                 "browse_sessions_30d cannot exceed browse_sessions_90d")
        _require(self.email_clicks_90d <= self.emails_received_90d,
                 "email_clicks_90d cannot exceed emails_received_90d")
        _require(max(self.low_csat_contacts_180d, self.slow_resolution_contacts_180d)
                 <= self.support_contacts_180d,
                 "low-CSAT and slow contacts are subsets of support_contacts_180d")
        return self


def _unique_ids(records: list, entity: str) -> None:
    counts = Counter(getattr(r, entity) for r in records)
    dupes = sorted(i for i, n in counts.items() if n > 1)
    _require(not dupes, f"duplicate {entity} values in one request: {dupes[:5]}")


class AcquisitionScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[LeadFeatures] = Field(min_length=1, max_length=MAX_RECORDS)

    @model_validator(mode="after")
    def _ids(self) -> AcquisitionScoreRequest:
        _unique_ids(self.records, "prospect_id")
        return self


class ChurnScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[CustomerFeatures] = Field(min_length=1, max_length=MAX_RECORDS)

    @model_validator(mode="after")
    def _ids(self) -> ChurnScoreRequest:
        _unique_ids(self.records, "customer_id")
        return self


# ---------------------------------------------------------------- responses
class ModelInfo(BaseModel):
    name: str
    version: str
    algorithm: str
    horizon_days: int
    deployable_from: str = Field(description="First scoring date whose training labels were "
                                             "all observed; earlier dates would leak.")


class LeadScore(BaseModel):
    prospect_id: str
    conversion_probability: float = Field(description="P(first order within the horizon).")
    reference_percentile: float = Field(description="Percent of training-period leads scoring "
                                                    "at or below this lead.")


class CustomerScore(BaseModel):
    customer_id: str
    churn_probability: float = Field(description="P(no order within the horizon).")
    reference_percentile: float = Field(description="Percent of training-period customers "
                                                    "scoring at or below this customer.")


class AcquisitionScoreResponse(BaseModel):
    model: ModelInfo
    predictions: list[LeadScore]


class ChurnScoreResponse(BaseModel):
    model: ModelInfo
    predictions: list[CustomerScore]


RECORD_MODELS: dict[str, type[_Record]] = {"acquisition_lead_score": LeadFeatures,
                                           "churn_risk": CustomerFeatures}


def records_frame(records: list[_Record]) -> pd.DataFrame:
    """Validated records as a frame; numeric features as float, like the feature builders."""
    frame = pd.DataFrame([r.model_dump() for r in records])
    numeric = frame.select_dtypes("number").columns
    frame[numeric] = frame[numeric].astype(float)
    return frame
