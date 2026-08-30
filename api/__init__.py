"""HTTP inference server for OpenCGM-StateEvent.

Thin transport over `opencgm_stateevent.infer.Analyser`. The Python package is the canonical
API (`Analyser.analyse_day(readings) -> DayReport`); this server exposes it as four endpoints
so a non-Python client can score a 24-hour CGM trace.

Endpoints:
  GET  /healthz       liveness probe
  GET  /v1/version    encoder + heads provenance (model SHA, weights SHA, license tags)
  GET  /v1/heads      every phenotype head with reliability metadata
  POST /v1/analyse    full DayReport from a 24h reading list

Run locally:
    uvicorn api.server:app --host 0.0.0.0 --port 8000

Or with the included Dockerfile (Hugging Face Spaces, Render, Fly.io, Railway).
"""
