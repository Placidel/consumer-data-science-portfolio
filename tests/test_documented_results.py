"""Documented section 00 results must trace back to a reproducible run."""

from __future__ import annotations

import json

import pytest

from northstar.paths import FOUNDATION_DIR
from northstar.profile import build_profile, extract_generated_block, render_profile_markdown
from northstar.synthetic import generate
from northstar.synthetic import params as p

PROFILE = FOUNDATION_DIR / "outputs" / "data_profile.json"
README = FOUNDATION_DIR / "README.md"


def test_readme_results_block_matches_committed_profile():
    profile = json.loads(PROFILE.read_text())
    assert extract_generated_block(README) == render_profile_markdown(profile)


@pytest.mark.slow
def test_committed_profile_is_reproduced_by_default_generation():
    """Regenerating with the default seed reproduces every committed number."""
    committed = json.loads(PROFILE.read_text())
    assert committed["seed"] == p.DEFAULT_SEED
    assert committed["n_prospects"] == p.DEFAULT_N_PROSPECTS
    tables = generate(seed=p.DEFAULT_SEED, n_prospects=p.DEFAULT_N_PROSPECTS)
    fresh = build_profile(tables, {"seed": p.DEFAULT_SEED, "n_prospects": p.DEFAULT_N_PROSPECTS})
    assert json.loads(json.dumps(fresh)) == committed
