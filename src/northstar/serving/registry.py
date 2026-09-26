"""A local, file-based model registry with versioned artifacts and explicit failure modes.

Layout (``models/`` by default, or ``NORTHSTAR_MODEL_DIR``)::

    models/<model name>/<version>/model.joblib           fitted scikit-learn estimator
    models/<model name>/<version>/metadata.json          model card: data, split, metrics, env
    models/<model name>/<version>/reference_profile.json training distribution for monitoring
    models/<model name>/LATEST                           version served when none is pinned

Versions are content-addressed (``v<first usable date>-<fingerprint>``), so retraining on the same
data and code reproduces the same version, and any change to data, features, split, champion or
library version yields a new one. Registering writes to a temporary folder and renames it into
place before ``LATEST`` is switched, so a reader never sees a half-written version.

Loading refuses, with a message that says how to fix it, when the artifact is missing, its file
hash does not match the metadata, it was trained with a different scikit-learn, or its features do
not match the code that will build the request frames. Artifacts are pickles: only load them from a
registry you trust (the hash check detects corruption, not a malicious writer).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import joblib
import sklearn

from northstar.paths import default_model_dir
from northstar.serving.specs import ModelSpec

SCHEMA_VERSION = 1
MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"
PROFILE_FILE = "reference_profile.json"
LATEST_FILE = "LATEST"


class ArtifactError(RuntimeError):
    """A model artifact is missing, corrupt or incompatible with the running code."""


@dataclass(frozen=True)
class ModelArtifact:
    spec: ModelSpec
    version: str
    model: object
    metadata: dict
    profile: dict
    path: Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, obj: Mapping) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n")


def save_artifact(model: object, metadata: dict, profile: dict, root: Path | None = None,
                  set_latest: bool = True) -> Path:
    """Write a new version atomically and (by default) point ``LATEST`` at it."""
    root = Path(root) if root is not None else default_model_dir()
    name, version = metadata["name"], metadata["version"]
    model_root = root / name
    tmp = model_root / f".{version}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    joblib.dump(model, tmp / MODEL_FILE, compress=3)
    metadata = {**metadata, "artifact": {"file": MODEL_FILE,
                                         "sha256": _sha256(tmp / MODEL_FILE)}}
    _write_json(tmp / METADATA_FILE, metadata)
    _write_json(tmp / PROFILE_FILE, profile)
    dest = model_root / version
    if dest.exists():  # same content-addressed version: replace it
        shutil.rmtree(dest)
    os.replace(tmp, dest)
    if set_latest:
        _point_latest(model_root, version)
    return dest


def _point_latest(model_root: Path, version: str) -> None:
    pointer = model_root / f".{LATEST_FILE}.tmp"
    pointer.write_text(version + "\n")
    os.replace(pointer, model_root / LATEST_FILE)


def promote(spec: ModelSpec, version: str, root: Path | None = None) -> ModelArtifact:
    """Point ``LATEST`` at ``version`` (promotion or rollback) after verifying it loads."""
    root = Path(root) if root is not None else default_model_dir()
    artifact = load_artifact(spec, root, version)
    _point_latest(root / spec.name, version)
    return artifact


def list_versions(spec: ModelSpec, root: Path | None = None) -> list[str]:
    model_root = (Path(root) if root is not None else default_model_dir()) / spec.name
    if not model_root.is_dir():
        return []
    return sorted(p.name for p in model_root.iterdir()
                  if p.is_dir() and not p.name.startswith("."))


def resolve_version(spec: ModelSpec, root: Path | None = None, version: str | None = None
                    ) -> str:
    root = Path(root) if root is not None else default_model_dir()
    if version:
        return version
    pointer = root / spec.name / LATEST_FILE
    if not pointer.exists():
        raise ArtifactError(
            f"No registered '{spec.name}' model under {root}. Train one with "
            "`northstar train-models` (or set NORTHSTAR_MODEL_DIR to an existing registry).")
    return pointer.read_text().strip()


def load_artifact(spec: ModelSpec, root: Path | None = None, version: str | None = None
                  ) -> ModelArtifact:
    """Load and verify one model version (``LATEST`` unless ``version`` is given)."""
    root = Path(root) if root is not None else default_model_dir()
    version = resolve_version(spec, root, version)
    path = root / spec.name / version
    missing = [f for f in (MODEL_FILE, METADATA_FILE, PROFILE_FILE) if not (path / f).exists()]
    if missing:
        available = ", ".join(list_versions(spec, root)) or "none"
        raise ArtifactError(f"Model '{spec.name}' version '{version}' is incomplete or missing at "
                            f"{path} (missing {', '.join(missing)}; available versions: "
                            f"{available}). Retrain with `northstar train-models`.")
    try:
        metadata = json.loads((path / METADATA_FILE).read_text())
        profile = json.loads((path / PROFILE_FILE).read_text())
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"Unreadable metadata for '{spec.name}' {version}: {exc}") from exc

    problems = []
    if metadata.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"metadata schema {metadata.get('schema_version')} != {SCHEMA_VERSION}")
    if metadata.get("name") != spec.name or metadata.get("version") != version:
        problems.append("metadata name/version do not match the registry path")
    if metadata.get("features") != list(spec.features):
        problems.append("feature list differs from the code that builds request frames")
    trained_with = metadata.get("environment", {}).get("scikit_learn")
    if trained_with != sklearn.__version__:
        problems.append(f"trained with scikit-learn {trained_with}, running {sklearn.__version__}")
    if problems:
        raise ArtifactError(f"Model '{spec.name}' {version} is incompatible: "
                            f"{'; '.join(problems)}. Retrain with `northstar train-models`.")
    expected = metadata.get("artifact", {}).get("sha256")
    if _sha256(path / MODEL_FILE) != expected:
        raise ArtifactError(f"Model file for '{spec.name}' {version} does not match the hash "
                            "recorded in its metadata (corrupted or replaced); refusing to load.")

    try:
        model = joblib.load(path / MODEL_FILE)
    except Exception as exc:  # e.g. a class that moved or changed since the artifact was saved
        raise ArtifactError(f"Model '{spec.name}' {version} could not be unpickled "
                            f"({type(exc).__name__}: {exc}). Retrain with "
                            "`northstar train-models`.") from exc
    if not hasattr(model, "predict_proba"):
        raise ArtifactError(f"Model '{spec.name}' {version} has no predict_proba method.")
    return ModelArtifact(spec=spec, version=version, model=model, metadata=metadata,
                         profile=profile, path=path)
