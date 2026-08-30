"""HTTP server for OpenCGM-StateEvent — thin wrapper over the Python `Analyser`.

The Python `opencgm_stateevent.infer.Analyser.analyse_day(readings) -> DayReport` is the
canonical API. This server exposes it as four endpoints so a non-Python client can score
a 24-hour CGM trace. It is intentionally small: ~120 lines, no auth, no rate limiting, no
database. The ONNX-based browser demo is separate; this server uses the canonical PyTorch
encoder + sklearn heads, so its outputs match the parity-checked Python pipeline exactly.

Endpoints:
  GET  /healthz       liveness probe
  GET  /v1/version    encoder + heads provenance
  GET  /v1/heads      every phenotype head with reliability metadata
  POST /v1/analyse    full DayReport from a 24h reading list

Inputs follow the conventions in `opencgm_stateevent.infer`:
  - timestamps are local-naive ISO8601 (e.g. "2024-03-14T13:25:00"); the encoder does not
    need a timezone offset because CGM windows are placed on the 5-minute grid by the CLI
    end of the input
  - glucose is mg/dL
  - "never interpolate CGM" is preserved: missing positions stay missing in the embedding

Run with:
    OPENCGM_CHECKPOINT=runs_5090/rawstats120/ckpt_ep040.pt \\
    OPENCGM_HEADS=artifacts/heads.pkl \\
    uvicorn api.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from opencgm_stateevent.infer import COVERAGE_TOLERANCE, Analyser

log = logging.getLogger("api.server")
log.setLevel(logging.INFO)

CHECKPOINT = Path(os.environ.get(
    "OPENCGM_CHECKPOINT",
    "runs_5090/rawstats120/ckpt_ep040.pt",
))
HEADS = Path(os.environ.get(
    "OPENCGM_HEADS",
    "artifacts/heads.pkl",
))


# -- Request schema -----------------------------------------------------------
class Reading(BaseModel):
    """One CGM reading. `t` is local-naive ISO8601; `mgdl` is mg/dL."""
    t: str = Field(..., description="Local-naive ISO8601 timestamp, no offset")
    mgdl: float = Field(..., description="Glucose value in mg/dL")


class AnalyseRequest(BaseModel):
    """Body for POST /v1/analyse. `readings` is the only required field."""
    readings: list[Reading] = Field(
        ...,
        description="CGM readings (any order; the Analyser sorts them).",
    )
    start: str | None = Field(
        None,
        description=(
            "Optional ISO8601 start of the 24h window. Default: window ends at the most "
            "recent reading + one 5-minute step."
        ),
    )


# -- Helpers ------------------------------------------------------------------
def _parse_iso(s: str) -> datetime:
    """Parse a local-naive ISO8601 string. `datetime.fromisoformat` is strict but accepts
    most ISO8601 in Python 3.11+. We don't accept an offset: this model consumes Dexcom-
    style local timestamps without a timezone, by design (the on-disk CGM is local)."""
    return datetime.fromisoformat(s)


# -- App + lifespan (load model once at startup) -------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not CHECKPOINT.exists():
        raise RuntimeError(
            f"checkpoint not found: {CHECKPOINT} "
            "(set OPENCGM_CHECKPOINT or place the released ckpt at the default path)"
        )
    log.info("loading checkpoint: %s", CHECKPOINT)
    app.state.analyser = Analyser.load(
        CHECKPOINT,
        heads=HEADS if HEADS.exists() else None,
    )
    log.info("loaded; %d heads", len(app.state.analyser.heads))
    yield


app = FastAPI(
    title="OpenCGM-StateEvent API",
    description=(
        "CGM phenotype scoring from the released OpenCGM-StateEvent encoder + heads. "
        "Apache-2.0 for code, CC-BY-NC-4.0 for weights. The Python package "
        "`opencgm_stateevent.infer.Analyser` is the canonical API; this server is a thin "
        "transport. Auto-generated OpenAPI schema at `/docs`."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# -- Endpoints ----------------------------------------------------------------
@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/v1/version")
def version() -> dict:
    """Encoder + heads provenance. Pin these SHAs into your consumer to detect drift."""
    a: Analyser = app.state.analyser
    return {
        "model": "OpenCGM-StateEvent",
        "checkpoint": str(CHECKPOINT),
        "encoder": {
            "weights_sha256": a.ref.weights_sha256,
            "architecture": a.ref.architecture,
            "backend": a.ref.backend,
            "dtype": a.ref.dtype,
            "epoch": a.ref.epoch,
            "seed": a.ref.seed,
        },
        "heads": {
            "path": str(HEADS) if HEADS.exists() else None,
            "n_heads": len(a.heads),
        },
        "license": {
            "code": "Apache-2.0",
            "weights": "CC-BY-NC-4.0",
        },
    }


@app.get("/v1/heads")
def heads() -> dict:
    """Every phenotype head with reliability metadata.

    The `coverage_p05` / `coverage_p95` band is the sampling density the head was fitted on;
    the applicability gate refuses to score a window outside `coverage_p05 - TOL` ...
    `coverage_p95 + TOL` (TOL = 0.15). ROC-AUC below the `signal_floor` is reported as
    `has_signal=false`.
    """
    a: Analyser = app.state.analyser
    out = {}
    for key, head in a.heads.items():
        out[key] = {
            "task": head["task"],
            "dataset": head["dataset"],
            "has_signal": bool(head.get("has_signal", False)),
            "reliability": {
                "roc_auc": head.get("roc_auc"),
                "roc_auc_sd": head.get("roc_auc_sd"),
                "roc_auc_subject": head.get("roc_auc_subject"),
                "n_subjects": head.get("n_subjects"),
                "coverage_p05": head.get("coverage_p05"),
                "coverage_p95": head.get("coverage_p95"),
                "coverage_tolerance": COVERAGE_TOLERANCE,
            },
        }
    return {"heads": out, "encoder": {
        "weights_sha256": a.ref.weights_sha256,
        "architecture": a.ref.architecture,
        "backend": a.ref.backend,
    }}


@app.post("/v1/analyse")
def analyse(req: AnalyseRequest) -> dict:
    """Score a 24-hour CGM trace. Returns the full DayReport (metrics + 128-d embedding +
    per-head phenotype scores + warnings).

    Errors:
      - 400: empty readings, malformed timestamp, or units check failed
        (see `opencgm_stateevent.infer.check_units` for the heuristic)
    """
    a: Analyser = app.state.analyser
    try:
        readings = [(_parse_iso(r.t), r.mgdl) for r in req.readings]
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"timestamp parse failed: {exc}",
        ) from exc
    if not readings:
        raise HTTPException(status_code=400, detail="readings list is empty")
    start = _parse_iso(req.start) if req.start else None
    try:
        report = a.analyse_day(readings, start=start)
    except ValueError as exc:
        # check_units / coverage gate / coverage-based errors all raise ValueError.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # DayReport.to_json() returns a string; we re-parse so FastAPI renders it with the
    # right content-type and pretty-prints it.
    return json.loads(report.to_json())
