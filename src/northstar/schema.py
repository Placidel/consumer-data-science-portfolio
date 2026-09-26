"""Declarative schema for the shared Northstar Consumer data model.

The schema is the single source of truth for column order, types, nullability, allowed values,
primary keys and foreign keys. It drives validation (``northstar.validation``) and the rendered
data dictionary (``docs/data_dictionary.md``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from northstar.timeline import DATA_END, DATA_START, DEFAULT_CUTOFF, TIME_COLUMNS

CHANNELS = (
    "paid_search",
    "paid_social",
    "display",
    "affiliate",
    "email",
    "referral",
    "organic_search",
)
TRAFFIC_SOURCES = (*CHANNELS, "direct")
REGIONS = ("northeast", "southeast", "midwest", "southwest", "west")
AGE_BANDS = ("18-24", "25-34", "35-44", "45-54", "55-64", "65+")
INCOME_BANDS = ("low", "middle", "high")
DEVICE_TYPES = ("mobile", "desktop", "tablet")
PLATFORMS = ("web", "app")
ORDER_CHANNELS = ("web", "app", "store")
CATEGORIES = ("apparel", "footwear", "home", "beauty", "outdoor", "accessories")
CAMPAIGN_OBJECTIVES = ("acquisition", "nurture", "retargeting", "conversion", "retention",
                       "winback", "promotion")
TOUCH_TYPES = ("ad_click", "email", "referral_credit")
FUNNEL_STAGES = ("session_start", "product_view", "add_to_cart", "checkout_start", "purchase")
SUBSCRIPTION_EVENT_TYPES = ("subscribe", "renew", "cancel")
SUBSCRIPTION_PLANS = ("monthly", "annual")
CONTACT_CHANNELS = ("chat", "email", "phone")
CONTACT_REASONS = ("delivery_delay", "return_request", "product_quality", "billing",
                   "account_access")
VARIANTS = ("control", "treatment")


@dataclass(frozen=True)
class Column:
    name: str
    kind: str  # one of: string, timestamp, int, float, bool
    description: str
    nullable: bool = False
    allowed: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ForeignKey:
    column: str
    ref_table: str
    ref_column: str


@dataclass(frozen=True)
class Table:
    name: str
    description: str
    grain: str
    primary_key: tuple[str, ...]
    columns: tuple[Column, ...]
    foreign_keys: tuple[ForeignKey, ...] = field(default_factory=tuple)

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    @property
    def time_column(self) -> str | None:
        return TIME_COLUMNS.get(self.name)


def _id(name: str, description: str, nullable: bool = False) -> Column:
    return Column(name, "string", description, nullable=nullable)


TABLES: dict[str, Table] = {
    t.name: t
    for t in (
        Table(
            "products",
            "Static product catalog.",
            "one row per product (SKU)",
            ("product_id",),
            (
                _id("product_id", "Product identifier (`SKU####`)."),
                Column("product_name", "string", "Synthetic product label."),
                Column("category", "string", "Merchandise category.", allowed=CATEGORIES),
                Column("list_price", "float", "Regular unit price (USD)."),
                Column("unit_cost", "float", "Landed unit cost (USD); margin = price - cost."),
            ),
        ),
        Table(
            "campaigns",
            "Marketing programs: quarterly paid acquisition, always-on lifecycle email programs "
            "and dated promotions.",
            "one row per campaign",
            ("campaign_id",),
            (
                _id("campaign_id", "Campaign identifier (`CMP###`)."),
                Column("campaign_name", "string", "Human-readable campaign name."),
                Column("channel", "string", "Delivery channel.", allowed=CHANNELS),
                Column("objective", "string", "Business objective.", allowed=CAMPAIGN_OBJECTIVES),
                Column("start_date", "timestamp", "First active day (midnight)."),
                Column("end_date", "timestamp", "Last active day, inclusive (midnight)."),
                Column("discount_pct", "float", "Discount offered by the campaign (0 if none)."),
            ),
        ),
        Table(
            "experiments",
            "Registry of randomized experiments embedded in the data.",
            "one row per experiment",
            ("experiment_id",),
            (
                _id("experiment_id", "Experiment identifier."),
                Column("experiment_name", "string", "Short name."),
                Column("hypothesis", "string", "Pre-registered hypothesis."),
                Column("randomization_unit", "string", "Unit of assignment."),
                Column("eligible_population", "string", "Who can be assigned."),
                Column("start_date", "timestamp", "First day of the experiment."),
                Column("end_date", "timestamp", "Last day of the experiment, inclusive."),
                Column("treatment_share", "float", "Target share of units assigned to treatment."),
                Column("primary_metric", "string", "Pre-registered primary metric."),
                Column("guardrail_metrics", "string", "Pre-registered guardrail metrics."),
            ),
        ),
        Table(
            "prospects",
            "Every identified lead (sign-up, account creation or captured email). Contains only "
            "attributes known when the lead is created - no conversion outcome.",
            "one row per person, created at first identified visit",
            ("prospect_id",),
            (
                _id("prospect_id", "Person identifier (`P######`); stable across the lifecycle."),
                Column("created_at", "timestamp", "When the lead was created."),
                Column("acquisition_channel", "string", "Channel that sourced the lead.",
                       allowed=CHANNELS),
                _id("campaign_id", "Sourcing campaign (paid, email-list and referral leads).",
                    nullable=True),
                Column("region", "string", "Home region.", allowed=REGIONS),
                Column("age_band", "string", "Self-reported age band.", allowed=AGE_BANDS),
                Column("income_band", "string", "Modelled household income band.",
                       allowed=INCOME_BANDS),
                Column("device_type", "string", "Device used at lead creation.",
                       allowed=DEVICE_TYPES),
                Column("email_opt_in", "bool", "Consented to marketing email at creation."),
            ),
            (ForeignKey("campaign_id", "campaigns", "campaign_id"),),
        ),
        Table(
            "customers",
            "Prospects who placed a first order. Attributes are copied from the prospect record; "
            "no post-conversion aggregates (lifetime value, churn status) are stored.",
            "one row per customer",
            ("customer_id",),
            (
                _id("customer_id",
                    "Customer identifier (`C######`), issued in order of conversion."),
                _id("prospect_id", "Originating prospect."),
                Column("customer_since", "timestamp", "Timestamp of the first order."),
                Column("acquisition_channel", "string", "Channel that sourced the lead.",
                       allowed=CHANNELS),
                Column("region", "string", "Home region.", allowed=REGIONS),
                Column("age_band", "string", "Age band.", allowed=AGE_BANDS),
                Column("income_band", "string", "Income band.", allowed=INCOME_BANDS),
                Column("device_type", "string", "Device used at lead creation.",
                       allowed=DEVICE_TYPES),
                Column("email_opt_in", "bool", "Marketing email consent at lead creation."),
            ),
            (ForeignKey("prospect_id", "prospects", "prospect_id"),),
        ),
        Table(
            "marketing_touches",
            "Outbound and paid marketing interactions: sourcing ad clicks, nurture/retargeting, "
            "newsletters, promotions and win-back emails.",
            "one row per touch delivered to a person",
            ("touch_id",),
            (
                _id("touch_id", "Touch identifier."),
                _id("prospect_id", "Person touched."),
                _id("customer_id", "Set only when the person was already a customer at touch time.",
                    nullable=True),
                _id("campaign_id", "Campaign that generated the touch."),
                Column("channel", "string", "Delivery channel.", allowed=CHANNELS),
                Column("touch_type", "string", "Kind of interaction.", allowed=TOUCH_TYPES),
                Column("touch_at", "timestamp", "Delivery (email) or click (ads) time."),
                Column("opened", "bool", "Email opened (always true for ad clicks)."),
                Column("clicked", "bool", "Clicked through to the site/app."),
                Column("cost", "float", "Variable media/delivery cost (USD)."),
            ),
            (
                ForeignKey("prospect_id", "prospects", "prospect_id"),
                ForeignKey("customer_id", "customers", "customer_id"),
                ForeignKey("campaign_id", "campaigns", "campaign_id"),
            ),
        ),
        Table(
            "sessions",
            "Web and app visits by identified people (anonymous traffic is out of scope).",
            "one row per session",
            ("session_id",),
            (
                _id("session_id", "Session identifier."),
                _id("prospect_id", "Visitor."),
                _id("customer_id", "Set only when the session started after the first order.",
                    nullable=True),
                Column("session_start", "timestamp", "Session start."),
                Column("platform", "string", "Digital property.", allowed=PLATFORMS),
                Column("device_type", "string", "Device.", allowed=DEVICE_TYPES),
                Column("traffic_source", "string", "Attributed traffic source.",
                       allowed=TRAFFIC_SOURCES),
                _id("campaign_id", "Campaign that drove the visit, if any.", nullable=True),
                Column("pages_viewed", "int", "Page/screen views in the session."),
            ),
            (
                ForeignKey("prospect_id", "prospects", "prospect_id"),
                ForeignKey("customer_id", "customers", "customer_id"),
                ForeignKey("campaign_id", "campaigns", "campaign_id"),
            ),
        ),
        Table(
            "funnel_events",
            "Ordered funnel milestones within sessions. A session reaching stage k has exactly "
            "one event for each stage 1..k.",
            "one row per session x stage reached",
            ("event_id",),
            (
                _id("event_id", "Event identifier."),
                _id("session_id", "Session."),
                Column("event_ts", "timestamp", "Event time (non-decreasing along the funnel)."),
                Column("event_type", "string", "Funnel stage.", allowed=FUNNEL_STAGES),
                Column("stage_number", "int", "1 = session_start ... 5 = purchase."),
            ),
            (ForeignKey("session_id", "sessions", "session_id"),),
        ),
        Table(
            "orders",
            "Completed orders across web, app and store.",
            "one row per order",
            ("order_id",),
            (
                _id("order_id", "Order identifier."),
                _id("customer_id", "Purchasing customer."),
                Column("order_ts", "timestamp", "Order time (equals the purchase event time)."),
                Column("order_channel", "string", "Sales channel.", allowed=ORDER_CHANNELS),
                _id("session_id", "Purchase session for web/app orders; null for store orders.",
                    nullable=True),
                _id("campaign_id", "Discount campaign applied, if any.", nullable=True),
                Column("item_count", "int", "Total units."),
                Column("gross_amount", "float", "Sum of line list value (USD)."),
                Column("discount_amount", "float", "Total discount (USD)."),
                Column("net_amount", "float", "gross_amount - discount_amount (USD)."),
            ),
            (
                ForeignKey("customer_id", "customers", "customer_id"),
                ForeignKey("session_id", "sessions", "session_id"),
                ForeignKey("campaign_id", "campaigns", "campaign_id"),
            ),
        ),
        Table(
            "order_lines",
            "Products within orders.",
            "one row per order x product",
            ("order_line_id",),
            (
                _id("order_line_id", "Order line identifier."),
                _id("order_id", "Order."),
                _id("product_id", "Product."),
                Column("quantity", "int", "Units."),
                Column("unit_price", "float", "Unit list price at time of order (USD)."),
                Column("discount_amount", "float", "Line discount (USD)."),
                Column("net_amount", "float", "quantity * unit_price - discount_amount (USD)."),
            ),
            (
                ForeignKey("order_id", "orders", "order_id"),
                ForeignKey("product_id", "products", "product_id"),
            ),
        ),
        Table(
            "subscription_events",
            "Northstar Plus membership lifecycle (monthly $9.99 or annual $89 plans).",
            "one row per membership event",
            ("subscription_event_id",),
            (
                _id("subscription_event_id", "Event identifier."),
                _id("customer_id", "Member."),
                Column("event_ts", "timestamp", "Event time."),
                Column("event_type", "string", "subscribe / renew / cancel (cancel takes effect at "
                       "the end of the paid period).", allowed=SUBSCRIPTION_EVENT_TYPES),
                Column("plan", "string", "Membership plan.", allowed=SUBSCRIPTION_PLANS),
                Column("amount", "float", "Fee charged (0 for cancellations)."),
            ),
            (ForeignKey("customer_id", "customers", "customer_id"),),
        ),
        Table(
            "support_contacts",
            "Customer service contacts. Resolution time and CSAT become known only at "
            "`resolved_at`; `timeline.snapshot` masks them for contacts unresolved at the cutoff.",
            "one row per contact",
            ("contact_id",),
            (
                _id("contact_id", "Contact identifier."),
                _id("customer_id", "Customer."),
                _id("order_id", "Related order, if any.", nullable=True),
                Column("contact_ts", "timestamp", "Contact opened."),
                Column("contact_channel", "string", "Contact channel.", allowed=CONTACT_CHANNELS),
                Column("reason", "string", "Contact reason.", allowed=CONTACT_REASONS),
                Column("resolved_at", "timestamp", "Resolution time; null if never resolved.",
                       nullable=True),
                Column("csat_score", "int", "Post-resolution satisfaction 1-5; null when the "
                       "survey was not answered.", nullable=True),
            ),
            (
                ForeignKey("customer_id", "customers", "customer_id"),
                ForeignKey("order_id", "orders", "order_id"),
            ),
        ),
        Table(
            "experiment_assignments",
            "Randomized assignment of prospects to experiment arms (deterministic hash of "
            "experiment_id and prospect_id), logged at first eligible exposure.",
            "one row per experiment x prospect",
            ("experiment_id", "prospect_id"),
            (
                _id("experiment_id", "Experiment."),
                _id("prospect_id", "Assigned prospect."),
                Column("variant", "string", "Arm.", allowed=VARIANTS),
                Column("assigned_at", "timestamp", "First eligible session start in the window."),
            ),
            (
                ForeignKey("experiment_id", "experiments", "experiment_id"),
                ForeignKey("prospect_id", "prospects", "prospect_id"),
            ),
        ),
    )
}

TABLE_NAMES: tuple[str, ...] = tuple(TABLES)

# Relationship diagram shown in the data dictionary.
ENTITY_DIAGRAM = """\
campaigns ─┬─< prospects ──1:0..1── customers ─┬─< orders ──< order_lines >── products
           │        │                          ├─< subscription_events
           │        ├─< sessions ──< funnel_events
           │        ├─< marketing_touches      └─< support_contacts
           │        └─< experiment_assignments >── experiments
           └─< marketing_touches / sessions / orders (campaign_id)"""


def render_data_dictionary() -> str:
    """Markdown data dictionary generated from ``TABLES``."""
    lines = [
        "# Northstar Consumer data dictionary",
        "",
        "<!-- Generated by `northstar data-dictionary`; edit src/northstar/schema.py instead. -->",
        "",
        "Northstar Consumer is a fictional omnichannel retailer (web, app and physical stores) "
        "with a paid membership program, Northstar Plus. All records are synthetic: identifiers "
        "are sequential codes and no names, emails, addresses or other personal data exist.",
        "",
        "## Time conventions",
        "",
        f"- History covers `{DATA_START.date()}` to `{DATA_END.date()}` (end exclusive), "
        "24 months.",
        "- Timestamps are timezone-naive, in the business's reporting timezone, at "
        "one-second resolution.",
        f"- The default modelling cutoff is `{DEFAULT_CUTOFF.date()}`. Features use rows with "
        "`time < cutoff`; outcome labels use `cutoff <= time < cutoff + horizon`.",
        "- `northstar.timeline.snapshot(tables, cutoff)` returns the point-in-time view of every "
        "table (including masking support resolutions not yet known at the cutoff).",
        "- Prospects created close to the end of the data are right-censored: they have had less "
        "time to convert. Conversion analyses must use a fixed outcome window.",
        "",
        "## Entity relationships",
        "",
        "```text",
        ENTITY_DIAGRAM,
        "```",
        "",
        "`prospect_id` is the person key used across the whole lifecycle. `customer_id` is set on "
        "sessions and touches only once the person has converted, so filtering by time never "
        "reveals a future conversion.",
        "",
    ]
    for table in TABLES.values():
        lines += [
            f"## `{table.name}`",
            "",
            table.description,
            "",
            f"- Grain: {table.grain}",
            f"- Primary key: `{', '.join(table.primary_key)}`",
        ]
        if table.time_column:
            lines.append(f"- Time column: `{table.time_column}`")
        if table.name == "order_lines":
            lines.append("- Time column: inherited from `orders.order_ts`")
        for fk in table.foreign_keys:
            lines.append(f"- Foreign key: `{fk.column}` -> `{fk.ref_table}.{fk.ref_column}`")
        lines += ["", "| Column | Type | Nullable | Description |", "|---|---|---|---|"]
        for col in table.columns:
            desc = col.description
            if col.allowed:
                desc += " Values: " + ", ".join(f"`{v}`" for v in col.allowed) + "."
            lines.append(
                f"| `{col.name}` | {col.kind} | {'yes' if col.nullable else 'no'} | {desc} |"
            )
        lines.append("")
    return "\n".join(lines)
