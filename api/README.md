# OpenCGM-StateEvent API

A small HTTP service that scores a 24-hour CGM trace using the released encoder and
phenotype heads. It is a **thin transport** over the existing Python API
(`opencgm_stateevent.infer.Analyser.analyse_day`); if you are a Python user, you should
import the package directly and skip this server.

The web demo in `web/` runs the encoder in the browser via ONNX Runtime Web and applies
the heads in TypeScript. This server uses the canonical PyTorch encoder and sklearn
pipelines instead, so its outputs match the parity-checked Python pipeline exactly.

License: Apache-2.0 (code) + CC-BY-NC-4.0 (weights).

---

## Quick start

```bash
# Local
uv sync
uv pip install -e ".[api]"

OPENCGM_CHECKPOINT=runs_5090/rawstats120/ckpt_ep040.pt \
OPENCGM_HEADS=artifacts/heads.pkl \
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

Then open <http://localhost:8000/docs> for the auto-generated OpenAPI / Swagger UI.

```bash
# Docker (mount the released artefacts at runtime — never bake them into the image)
docker build -t opencgm-stateevent-api -f api/Dockerfile .
docker run --rm -p 8000:8000 \
    -v "$(pwd)/runs_5090:/app/runs_5090:ro" \
    -v "$(pwd)/artifacts:/app/artifacts:ro" \
    -e OPENCGM_CHECKPOINT=/app/runs_5090/rawstats120/ckpt_ep040.pt \
    -e OPENCGM_HEADS=/app/artifacts/heads.pkl \
    opencgm-stateevent-api
```

---

## Endpoints

### `GET /healthz`

Liveness probe. Always returns `{"status": "ok"}` if the process is up.

### `GET /v1/version`

Encoder and heads provenance. Pin `encoder.weights_sha256` into your client to detect
drift on the next deploy.

```json
{
  "model": "OpenCGM-StateEvent",
  "checkpoint": "runs_5090/rawstats120/ckpt_ep040.pt",
  "encoder": {
    "weights_sha256": "225b7529d8dc73ac...",
    "architecture": {"raw_statistics": true, "use_circadian": true, ...},
    "backend": "pytorch",
    "dtype": "float32",
    "epoch": 40,
    "seed": 17
  },
  "heads": {
    "path": "artifacts/heads.pkl",
    "n_heads": 18
  },
  "license": {"code": "Apache-2.0", "weights": "CC-BY-NC-4.0"}
}
```

### `GET /v1/heads`

Every phenotype head with its reliability metadata. Use this to render your own UI.

```json
{
  "heads": {
    "cgmacros:insulin_resistance[cgmacros_dexcom]": {
      "task": "cgmacros:insulin_resistance",
      "dataset": "cgmacros_dexcom",
      "has_signal": true,
      "reliability": {
        "roc_auc": 0.87,
        "roc_auc_sd": 0.05,
        "roc_auc_subject": 0.84,
        "n_subjects": 45,
        "coverage_p05": 0.78,
        "coverage_p95": 0.99,
        "coverage_tolerance": 0.15
      }
    },
    ...
  }
}
```

A head with `has_signal: false` cleared the cross-validated signal floor and should not
be presented as a score to the user.

### `POST /v1/analyse`

Score a 24-hour CGM trace. Returns the full DayReport: tier-1 clinical metrics, the
128-d daily embedding, and every phenotype head's probability + reliability.

**Request body**

```json
{
  "readings": [
    {"t": "2024-03-14T00:00:00", "mgdl": 110},
    {"t": "2024-03-14T00:30:00", "mgdl": 115}
  ],
  "start": "2024-03-14T00:00:00"
}
```

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `readings` | `[{t, mgdl}]` | yes | `t` is local-naive ISO8601 (`"2024-03-14T13:25:00"`); `mgdl` is glucose in mg/dL. Out-of-range values are dropped before scoring. |
| `start` | ISO8601 string | no | Window start time. Default: `(most recent reading) - 24h + 5min`, so the window ends at the last reading. |

**Response** (abridged)

```json
{
  "start": "2024-03-14T00:00:00",
  "metrics": {
    "n_observed": 181,
    "coverage": 0.628,
    "mean_glucose": 142.2,
    "glucose_management_indicator": 6.71,
    "time_in_range": 0.99,
    "time_below_70": 0.0,
    "time_above_180": 0.01,
    "coefficient_of_variation": 0.09,
    "variability_is_stable": true
  },
  "embedding": [0.123, 0.456, "..."],
  "phenotypes": [
    {
      "task": "cgmacros:insulin_resistance",
      "dataset": "cgmacros_dexcom",
      "probability": 1.0,
      "predicted_class": 1,
      "reliability": 0.87,
      "reliability_sd": 0.05,
      "reliability_subject_level": 0.84,
      "n_subjects_learned_from": 45,
      "has_signal": true,
      "applicable": true,
      "applicability_note": "",
      "phrasing": "Days like this one are more common among people with cgmacros:insulin_resistance (100% model score; held-out ROC-AUC 0.87 from 45 subjects)."
    }
  ],
  "warnings": []
}
```

**Errors**

| Status | Cause |
|--------|-------|
| 400 | Empty readings list |
| 400 | Unparseable timestamp |
| 400 | `check_units()` rejected the input (almost certainly mmol/L confused for mg/dL) |

---

## Numeric API conventions

These follow the released `opencgm_stateevent.infer` module exactly; the API server does
not add any new behaviour on top of it.

- **Glucose is mg/dL.** The unit check rejects a stream where > 50% of readings are
  below 25, the most common mmol/L-vs-mg/dL mistake. Pass `mgdl` already in mg/dL; do
  not feed mmol/L values.
- **Timestamps are local-naive ISO8601.** No timezone offset. This matches Dexcom Clarity,
  LibreView, and most consumer CGM exports.
- **Gaps stay gaps.** A reading every 5 minutes is not synthesised from a reading every
  15 minutes. The 5-minute grid is filled by **median bucketing**, and unobserved
  positions carry `mask=0` all the way through the encoder.
- **Heads are population probes, not diagnoses.** Each `phenotype.probability` is a
  cohort-conditional estimate, not a probability that *this* person has the condition.
  The `phrasing` field is the only text the API emits and is worded accordingly.
- **Heads have applicability gates.** A head fitted on 5-minute Dexcom data is
  unreliable on a 1-minute LibreView stream where the coverage band differs. The
  applicability check uses `coverage ± 0.15` around the head's fitted `coverage_p05..p95`
  band. The check is observable: `phenotype.applicable` is `false` and `phrasing`
  explains why when it fails.

---

## cURL examples

```bash
# 1. Liveness
curl http://localhost:8000/healthz

# 2. Encoder provenance
curl http://localhost:8000/v1/version | jq '.encoder.weights_sha256'

# 3. List heads with reliability
curl http://localhost:8000/v1/heads | jq '.heads | to_entries | map({key, auc: .value.reliability.roc_auc, n: .value.reliability.n_subjects})'

# 4. Score a 24-hour trace
curl -X POST http://localhost:8000/v1/analyse \
  -H 'Content-Type: application/json' \
  -d '{
    "readings": [
      {"t": "2024-03-14T00:00:00", "mgdl": 110},
      {"t": "2024-03-14T00:05:00", "mgdl": 112},
      {"t": "2024-03-14T00:10:00", "mgdl": 115}
    ]
  }' | jq '.phenotypes[0:3] | map({task, probability, applicable, has_signal})'
```

---

## Deployment

The Docker image is the unit of deployment. Model weights are mounted at runtime, never
baked into the image.

| Target | How |
|--------|-----|
| **Hugging Face Spaces (Docker)** | New Space → Docker. Set `OPENCGM_CHECKPOINT` and `OPENCGM_HEADS` in the Space's Secrets to point at the uploaded HF files. |
| **Render** | New Web Service → Docker. Mount the artefact directory or use Render Disk; set the env vars. |
| **Fly.io** | `fly launch --dockerfile api/Dockerfile`. Set env vars; mount a volume if persisting heads. |
| **Railway** | New Project → Docker; same env vars. |
| **Bare metal** | `uvicorn api.server:app --workers 1 --host 0.0.0.0 --port 8000` behind nginx. |

A single uvicorn worker is intentional: the model is loaded once at startup and the
inference path is CPU-bound. Scaling is **horizontal** (more containers behind a load
balancer), not by adding workers inside one container, which would duplicate the model in
memory.

---

## Limitations (intentional)

- **No auth.** This is a research-API surface. Add an auth proxy (oauth2-proxy, Cloudflare
  Access, or a sidecar) before exposing it publicly.
- **No rate limiting.** Same reasoning. The model is small (~3 ms per request on CPU) so a
  single instance handles thousands of req/s, but a runaway client can OOM it.
- **No batching.** Each request is one window. If you call `analyse_day` in a Python loop
  in production, prefer the Python API directly so you reuse the loaded model.
- **No persistence.** The server holds no state between requests. The 24-hour window is
  derived from the request's `readings` only.

If any of these matter to your deployment, this server is a starting point, not a
final answer — fork it and add what you need.
