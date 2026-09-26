"""Reproducible synthetic data for the fictional retailer Northstar Consumer."""

from northstar.synthetic.generate import generate, normalize_dtypes
from northstar.synthetic.simulate import assignment_bucket, experiment_variant

__all__ = ["assignment_bucket", "experiment_variant", "generate", "normalize_dtypes"]
