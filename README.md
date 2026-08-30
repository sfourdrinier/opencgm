# OpenCGM-StateEvent

**Independent public-data reconstruction of the GlucoFM dual-stream CGM method.**

A ~0.72M-parameter foundation model for continuous glucose monitoring, reimplemented from
*GlucoFM: A Dual-Stream Foundation Model for Continuous Glucose Monitoring*
([arXiv:2605.30865v2](https://arxiv.org/abs/2605.30865v2)) and trained on public CGM cohorts.

> **This is not Google's implementation or checkpoint.** Wear-CGM — 75,330 of the paper's
> 109,066 pretraining hours — is non-public, and several low-level choices are unspecified in
> the paper. This is a high-fidelity public-data reconstruction, not the authors' model.
> No official GlucoFM source or weights were available at the evidence cutoff of 2026-08-27.

## Method in one paragraph

A 24-hour CGM window on a 5-minute grid (288 positions, 24 hourly patches of 12). A learnable
one-sided **causal Gaussian filter** decomposes the signal into a slow **state** stream and a
residual **event** stream. Each is embedded to 64 dimensions and fused to a 128-dimensional
physiological token, gated against a circadian time embedding, and encoded by a 3-layer,
4-head Transformer. Training is JEPA-style: 50–60% of patches are hidden from the online
branch *before* normalization and filtering, and predicted against an EMA target, with an
additional temporal-dynamics loss on state/event transitions. Missing data is never
interpolated — a physical observation mask is carried end to end.

## Status

Five-seed evaluation complete; encoder and probe heads released. See
[`DECISIONS.md`](DECISIONS.md) for every choice the paper left open, and
[`REPRODUCE.md`](REPRODUCE.md) to run it yourself.

```bash
just status     # what is on disk, under which rights — derived from the filesystem
just gate       # lint + tests + status; must pass before any PR is done
```

## Why reproduce it

The paper reports strong results but published no code or weights. Independent of matching
their numbers, this project aims to contribute what the paper leaves open:

- a **pinned, single-command reproduction path** on consumer hardware (RTX 3090): every
  step is a `just` recipe, the corpus is fixed by a hashed manifest, and every run record
  names the checkpoint it read. See `REPRODUCE.md` for measured wall-clocks
- a resolved **assumption registry** for the 19 under-specified options the paper leaves open,
  each tagged in `bundle/glucofm_reference_config.yaml` and settled in `DECISIONS.md`
- **subgroup evaluation** by device, diabetes status, cadence, and density
- **external validation** on cohorts the model is licensed *not* to train on — zero leakage
  risk by construction
- an open checkpoint and local ONNX inference

## Repository

| Path | What |
|---|---|
| `bundle/` | frozen blueprint + reference config, and the amendments delta |
| `manifests/sources/registry.yaml` | every source, its lane, rights, and paper target |
| `src/opencgm_stateevent/` | the implementation |
| `DECISIONS.md` | every inferred choice, with rationale, written when made |

Raw data lives outside the repo and is never committed.

## Data and rights

Sources are separated into **lanes**, and a checkpoint's lane membership determines whether it
may be released:

| Lane | Meaning |
|---|---|
| A | the four public GlucoFM pretraining cohorts — the strict reproduction |
| B | downstream evaluation |
| C | public-plus, permissive, after rights and duplicate audit |
| D | PPG bridge for later teacher/student work |
| E | **evaluation only** — NC/ND/SA sources, never in a distributed checkpoint |

Lane E exists because restrictive licenses restrict *distribution* of derivatives, not local
evaluation. That makes those cohorts usable — and unusually valuable — as held-out validation.
See [`bundle/BLUEPRINT_AMENDMENTS.md`](bundle/BLUEPRINT_AMENDMENTS.md) A4.

Code is Apache-2.0. **Dataset and model-weight licensing are decided separately** and
per-source; see `NOTICE`.

## Not a medical device

Research software. Not for diagnosis, treatment, or dosing decisions. The original paper
describes GlucoFM as a research prototype.

## Citation

If you use this code, cite the original method alongside this reconstruction — see
`CITATION.cff`.
