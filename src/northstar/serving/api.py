"""FastAPI scoring service: health, model metadata and the lead and churn scoring endpoints.

Models are loaded once, when the app starts, from the local registry (``LATEST`` unless a version
is pinned). Requests only call ``predict_proba`` on those objects; nothing is fit at request time.

Failure behavior:

* An artifact that is missing or fails verification does not take the process down. The app
  starts, ``GET /health`` answers 503 and names the problem, and that model's endpoint answers 503.
  ``northstar serve`` runs the same checks first and refuses to start unless every model loads,
  so a deployment fails fast instead of serving half a contract.
* An invalid request answers 422 with one entry per problem (field path, error type and reason).
  Submitted values are not echoed back, so error responses and logs never carry customer data.

Logs record model, version, record count and latency per scoring call, never feature values.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse

from northstar import __version__
from northstar.paths import default_model_dir
from northstar.serving.registry import ArtifactError, ModelArtifact, load_artifact
from northstar.serving.schemas import (
    AcquisitionScoreRequest,
    AcquisitionScoreResponse,
    ChurnScoreRequest,
    ChurnScoreResponse,
    records_frame,
)
from northstar.serving.scoring import Scorer
from northstar.serving.specs import ACQUISITION, CHURN, SPECS, ModelSpec

logger = logging.getLogger("northstar.api")

VERSION_ENV = {ACQUISITION.name: "NORTHSTAR_ACQUISITION_MODEL_VERSION",
               CHURN.name: "NORTHSTAR_CHURN_MODEL_VERSION"}

Loader = Callable[[ModelSpec, Path, str | None], ModelArtifact]


@dataclass
class LoadedModels:
    scorers: dict[str, Scorer] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, model_dir: Path, versions: Mapping[str, str | None],
             loader: Loader = load_artifact) -> LoadedModels:
        loaded = cls()
        for name, spec in SPECS.items():
            try:
                loaded.scorers[name] = Scorer(loader(spec, model_dir, versions.get(name)))
            except ArtifactError as exc:
                loaded.errors[name] = str(exc)
                logger.error("model %s unavailable: %s", name, exc)
        return loaded

    def require(self, name: str) -> Scorer:
        if name not in self.scorers:
            raise HTTPException(status_code=503,
                                detail=f"Model '{name}' is not loaded: {self.errors.get(name)}")
        return self.scorers[name]


def get_models(request: Request) -> LoadedModels:
    return request.app.state.models


Models = Annotated[LoadedModels, Depends(get_models)]


def pinned_versions() -> dict[str, str | None]:
    """Versions pinned through the environment (unset = serve ``LATEST``)."""
    return {name: os.environ.get(env) or None for name, env in VERSION_ENV.items()}


def preflight(model_dir: Path | None = None, versions: Mapping[str, str | None] | None = None
              ) -> dict[str, str]:
    """Load every model once; returns {model name: problem} (empty when all load)."""
    model_dir = Path(model_dir) if model_dir is not None else default_model_dir()
    return LoadedModels.load(model_dir, versions if versions is not None else pinned_versions()
                             ).errors


def create_app(model_dir: Path | None = None, versions: Mapping[str, str | None] | None = None,
               loader: Loader = load_artifact) -> FastAPI:
    model_dir = Path(model_dir) if model_dir is not None else default_model_dir()
    versions = dict(versions) if versions is not None else pinned_versions()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.models = LoadedModels.load(model_dir, versions, loader)
        yield

    app = FastAPI(
        title="Northstar Consumer scoring API",
        version=__version__,
        lifespan=lifespan,
        description=("Scores open leads (30-day first-order probability) and active customers "
                     "(90-day churn probability) with the versioned champions of sections 01 "
                     "and 02. Requests carry point-in-time feature records; see "
                     "`projects/08_productionization/README.md` for intended use."),
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        detail = [{"loc": list(e.get("loc", ())), "type": e.get("type"), "msg": e.get("msg")}
                  for e in exc.errors()]
        return JSONResponse({"detail": detail}, status_code=422)

    @app.middleware("http")
    async def timing(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - started) * 1000
        response.headers["X-Process-Time-Ms"] = f"{elapsed:.1f}"
        logger.info("%s %s -> %s in %.1f ms", request.method, request.url.path,
                    response.status_code, elapsed)
        return response

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/docs")

    @app.get("/health", tags=["operations"])
    def health(loaded: Models) -> JSONResponse:
        """200 when every model is loaded; 503 with the reason otherwise."""
        body = {
            "status": "ok" if not loaded.errors else "unavailable",
            "models": {
                name: ({"loaded": True, "version": loaded.scorers[name].artifact.version}
                       if name in loaded.scorers else
                       {"loaded": False, "error": loaded.errors.get(name)})
                for name in SPECS},
        }
        return JSONResponse(body, status_code=200 if not loaded.errors else 503)

    @app.get("/v1/models", tags=["operations"])
    def model_cards(loaded: Models) -> dict:
        """Model card of every loaded model: population, intended use, split, metrics."""
        keys = ("name", "version", "title", "section", "algorithm", "algorithm_label", "entity",
                "score_field", "horizon_days", "population", "intended_use", "not_for",
                "features", "training", "selection", "validation_metrics", "holdout_metrics",
                "data", "environment")
        return {name: {k: s.artifact.metadata.get(k) for k in keys}
                for name, s in loaded.scorers.items()}

    def _score(loaded: LoadedModels, spec: ModelSpec, records: list) -> tuple[Scorer, list]:
        scorer = loaded.require(spec.name)
        started = time.perf_counter()
        scored = scorer.score(records_frame(records))
        logger.info("scored model=%s version=%s records=%d in %.1f ms", spec.name,
                    scorer.artifact.version, len(scored),
                    (time.perf_counter() - started) * 1000)
        return scorer, scored.to_dict(orient="records")

    @app.post(f"/v1/{ACQUISITION.route}/score", response_model=AcquisitionScoreResponse,
              tags=["scoring"])
    def score_leads(request: AcquisitionScoreRequest,
                    loaded: Models) -> AcquisitionScoreResponse:
        """Probability that each open lead places a first order within 30 days."""
        scorer, rows = _score(loaded, ACQUISITION, request.records)
        return AcquisitionScoreResponse(model=scorer.info, predictions=rows)

    @app.post(f"/v1/{CHURN.route}/score", response_model=ChurnScoreResponse, tags=["scoring"])
    def score_customers(request: ChurnScoreRequest,
                        loaded: Models) -> ChurnScoreResponse:
        """Probability that each active customer places no order within 90 days."""
        scorer, rows = _score(loaded, CHURN, request.records)
        return ChurnScoreResponse(model=scorer.info, predictions=rows)

    return app
