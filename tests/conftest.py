from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from northstar.synthetic import generate

TEST_SEED = 7
TEST_N_PROSPECTS = 6_000


@pytest.fixture(scope="session")
def tables() -> dict[str, pd.DataFrame]:
    """A reduced but complete dataset shared by read-only tests (do not mutate)."""
    return generate(seed=TEST_SEED, n_prospects=TEST_N_PROSPECTS)


@pytest.fixture
def mutable_tables(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {name: df.copy() for name, df in tables.items()}


LIFECYCLE_TOY_PERIODS = pd.date_range("2024-01-01", periods=12, freq="MS")


def _lifecycle_toy_tables() -> dict[str, pd.DataFrame]:
    """Four people whose monthly states are worked out by hand in test_lifecycle_states.py.

    * P1/C1: orders 2024-01-10 and 2024-02-15, then lapses.
    * P2: lead created 2024-03-20, never orders.
    * P3/C3: orders on the 2nd of every month (becomes loyal).
    * P4/C4: orders 2024-01-03, lapses to churned, returns 2024-09-10.
    """
    ts = pd.Timestamp
    prospects = pd.DataFrame({
        "prospect_id": ["P1", "P2", "P3", "P4"],
        "created_at": [ts("2024-01-05"), ts("2024-03-20"), ts("2024-01-01"), ts("2024-01-01")],
        "acquisition_channel": ["email", "display", "referral", "paid_search"],
    })
    customers = pd.DataFrame({
        "customer_id": ["C1", "C3", "C4"], "prospect_id": ["P1", "P3", "P4"],
        "customer_since": [ts("2024-01-10"), ts("2024-01-02"), ts("2024-01-03")],
    })
    c3 = LIFECYCLE_TOY_PERIODS + pd.Timedelta(days=1)  # the 2nd of every month
    orders = pd.DataFrame({
        "customer_id": ["C1", "C1", *["C3"] * len(c3), "C4", "C4"],
        "order_ts": [ts("2024-01-10"), ts("2024-02-15"), *c3, ts("2024-01-03"),
                     ts("2024-09-10")],
        "net_amount": [50.0, 30.0, *[20.0] * len(c3), 40.0, 60.0],
    })
    return {"prospects": prospects, "customers": customers, "orders": orders}


@pytest.fixture
def toy_lifecycle() -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex]:
    """Hand-checkable lifecycle history and its twelve monthly periods (Jan - Dec 2024)."""
    return _lifecycle_toy_tables(), LIFECYCLE_TOY_PERIODS


@pytest.fixture
def dashboard_projects(tmp_path) -> Path:
    """A private copy of the committed section outputs the dashboard reads (figures skipped).

    Tests edit or delete files here to show the dashboard follows its inputs.
    """
    from northstar.dashboard.artifacts import SECTIONS, source_files
    from northstar.paths import PROJECTS_DIR

    root = tmp_path / "projects"
    for section in SECTIONS.values():
        for src in source_files(section, PROJECTS_DIR):
            dest = root / src.relative_to(PROJECTS_DIR)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
    return root


@pytest.fixture(scope="session")
def model_registry(tables, tmp_path_factory) -> Path:
    """Both served models trained once on the shared test data (read-only; copy to modify)."""
    from northstar.serving.specs import SPECS
    from northstar.serving.training import train_and_register

    root = tmp_path_factory.mktemp("models")
    for spec in SPECS.values():
        train_and_register(spec, tables, root,
                           data_info={"seed": TEST_SEED, "n_prospects": TEST_N_PROSPECTS})
    return root
