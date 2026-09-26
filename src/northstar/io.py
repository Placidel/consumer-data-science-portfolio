"""Reading and writing the shared tables."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import pandas as pd

from northstar import __version__
from northstar.paths import default_data_dir
from northstar.schema import TABLE_NAMES
from northstar.synthetic import generate, normalize_dtypes
from northstar.synthetic import params as p
from northstar.timeline import DATA_END, DATA_START, DEFAULT_CUTOFF

MANIFEST = "manifest.json"


def content_hash(df: pd.DataFrame) -> str:
    """Hash of a table's values, column names and row order (independent of file encoding)."""
    h = hashlib.sha256()
    h.update("|".join(df.columns).encode())
    h.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    return h.hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_tables(tables: Mapping[str, pd.DataFrame], out_dir: Path, *, seed: int,
                 n_prospects: int) -> dict:
    """Write one Parquet file per table plus a manifest; returns the manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = {}
    for name in TABLE_NAMES:
        df = tables[name]
        path = out_dir / f"{name}.parquet"
        df.to_parquet(path, index=False, engine="pyarrow", compression="zstd")
        entries[name] = {
            "file": path.name,
            "rows": len(df),
            "columns": list(df.columns),
            "content_sha256": content_hash(df),
            "file_sha256": file_hash(path),
        }
    manifest = {
        "generator": "northstar.synthetic.generate",
        "package_version": __version__,
        "seed": seed,
        "n_prospects": n_prospects,
        "data_start": str(DATA_START.date()),
        "data_end_exclusive": str(DATA_END.date()),
        "default_cutoff": str(DEFAULT_CUTOFF.date()),
        "tables": entries,
    }
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def read_manifest(data_dir: Path | None = None) -> dict:
    path = (data_dir or default_data_dir()) / MANIFEST
    if not path.exists():
        raise FileNotFoundError(
            f"No generated data at {path.parent}. Run `northstar generate-data` first."
        )
    return json.loads(path.read_text())


def load_tables(data_dir: Path | None = None, names: Iterable[str] | None = None, *,
                generate_if_missing: bool = False) -> dict[str, pd.DataFrame]:
    """Load shared tables with canonical dtypes.

    With ``generate_if_missing=True`` the default dataset is generated first when absent, so
    downstream pipelines can run from a clean checkout with a single command.
    """
    data_dir = Path(data_dir) if data_dir is not None else default_data_dir()
    if not (data_dir / MANIFEST).exists():
        if not generate_if_missing:
            read_manifest(data_dir)  # raises a helpful FileNotFoundError
        tables = generate(seed=p.DEFAULT_SEED, n_prospects=p.DEFAULT_N_PROSPECTS)
        write_tables(tables, data_dir, seed=p.DEFAULT_SEED, n_prospects=p.DEFAULT_N_PROSPECTS)
    wanted = list(names) if names is not None else list(TABLE_NAMES)
    unknown = set(wanted) - set(TABLE_NAMES)
    if unknown:
        raise KeyError(f"Unknown tables: {sorted(unknown)}")
    return {
        name: normalize_dtypes(name, pd.read_parquet(data_dir / f"{name}.parquet"))
        for name in wanted
    }
