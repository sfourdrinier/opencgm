---
license: other
license_name: cc-by-nc-4.0-encoder-and-cc-by-nc-sa-4.0-heads
license_link: https://github.com/sfourdrinier/opencgm/blob/main/LICENSE-HEADS
library_name: onnx
pipeline_tag: feature-extraction
tags:
  - continuous-glucose-monitoring
  - cgm
  - time-series
  - jepa
  - self-supervised
  - healthcare
  - onnx
---

# Model card — `opencgm_stateevent` encoder (v0.1.0)

## License

| Component | License |
|---|---|
| **Encoder** (`glucofm_encoder.onnx`) | **CC-BY-NC-4.0** (research / non-commercial) |
| **Probe heads** (`glucofm_heads.json`) | **CC-BY-NC-SA-4.0** — share-alike inherited from CGMacros (D025) |
| **Source code** (this repo, the Python trainer, the Next.js demo) | Apache-2.0 |
| **HF Hub `license:` tag** | `other` — the repo holds two artefacts under two licences |

See `LICENSE-WEIGHTS` for the full text and `LICENSE_NOTES.md` for the decision record.
A commercial license for the weights can be negotiated with the maintainer
(`@sfourdrinier` on GitHub).

## Summary

A 435,633-parameter dual-stream JEPA-style encoder for 24-hour continuous glucose
monitoring windows. Trained on a public-data corpus; not derived from any Google
implementation or checkpoint. See `LICENSE_NOTES.md` for weights licensing and
`manifests/sources/registry.yaml` for the data record.

## Intended use

- **In scope:** research on CGM-derived phenotypes (glycemic variability, insulin
  resistance proxies, beta-cell dysfunction proxies, glucotype clustering,
  hyperlipidaemia proxies, obesity proxies, diabetes risk, hypoglycaemia).
  Embeddings compose across days (`n_days = 1..7`) for cohort-level phenotypes.
- **Out of scope:** clinical diagnosis, treatment decisions, dosing recommendations,
  real-time insulin titration, alarm generation. **Not a medical device.**

## Training data

The encoder was pretrained on **Lane A only** — the four public cohorts the GlucoFM paper
names. These are the windows that are actually inside the released weights.

| Source | Windows | Subjects | Hours | Licence |
|---|---:|---:|---:|---|
| stanford | 171,140 | 56 | 8,761 | source repository states none |
| shanghai_t2dm | 159,119 | 100 | 12,414 | CC-BY-4.0 |
| big_ideas | 13,167 | 16 | 3,017 | ODC-By-1.0 |
| colas | 9,701 | 68 | 9,544 | CC-BY-4.0 |
| **total** | **353,127** | **240** | **33,736** | 30.9% of the paper's 109,066 h |

Windows overlap, so the window count is much larger than hours ÷ 24; hours are the
non-overlapping quantity and are what the 30.9% refers to.

**No other lane entered pretraining.** Hall (Lane B) and the Lane C and D cohorts are used for
downstream evaluation and for the PPG teacher-student pilot, not for training the encoder.
Lane E — cgmacros, uchtt1dm, glucofm_bench — is evaluation-only under non-commercial,
share-alike or no-derivatives terms.

The full corpus audit is in `manifests/sources/registry.yaml`; the lane rules are in
`DECISIONS.md` D002 and `bundle/BLUEPRINT_AMENDMENTS.md` A4.

## Architecture

| | value |
|---|---|
| Parameters | 435,633 in this released encoder; 732,593 trainable in the full pretraining model (encoder + predictor + transition heads) |
| Input | 24 h × 5 min grid (288 positions, 24 hourly patches × 12 steps) |
| Streams | state + event (causal Gaussian decomposition, learnable σ) |
| Embedding | 64 → 128-d fused physiological token, gated with circadian time embedding |
| Backbone | 3-layer, 4-head Transformer (paper §3.3; our D019) |
| Training objective | JEPA-style masked-token prediction + temporal-dynamics loss |
| Masking | 50-60 % of patches, sampled uniformly per window, masked *before* normalisation and filtering |
| EMA target | linear ramp 0.997 → 0.9994 per epoch |
| Reported in paper | 0.72 M — within 1.7 % |

## Provenance of *this* artifact

The weights in this repository are **one checkpoint**, not the five-seed ensemble the
evaluation below is computed over. Read this section before quoting a number against the
file you downloaded.

| | value |
|---|---|
| Checkpoint | `runs_5090/rawstats120/ckpt_ep040.pt` |
| Seed | 17 |
| Epoch | **40**, not 120 |
| `zero_empty_patches` | **`false`** — as trained (see below) |
| Exported file | `glucofm_encoder.onnx`, opset 17, 1,991,782 bytes |
| SHA-256 | `b1349deffd15ab62a5a98d7c7c4a7e143bcf1dee7ad745b1e05533ff54f34768` |

**Why epoch 40 and not 120.** Epoch 40 is the measured transfer peak at 30.9 % of the paper's
corpus; epoch 120 is mildly *worse* (−0.005 ROC, about 1 sd of the seed spread). That is the
expected consequence of giving a model more capacity than a reduced corpus can use, and it is
reported rather than hidden. See D024 and `findings/results_section.md`.

**Why `zero_empty_patches` is `false` here.** D020 — an empty patch emits nothing rather than a
learned bias — is tagged `PAPER_EXACT` and is now the repo default (`True`). This checkpoint
predates that change, so it was trained with the learned bias and is exported faithfully as
such. D020 was measured to be neutral (0.6780 vs 0.6818, well inside the seed sd), so the
released weights are not disadvantaged by it — but **if you rebuild the architecture from the
current repo defaults, you will not get an architecture that matches this file.** Construct it
with `zero_empty_patches=False`, or read the flags from `glucofm_encoder.onnx.meta.json`, which
records the exact architecture this artifact was exported under.

## Which probe heads are distributed

All 18 probe heads are published; 14 clear the signal floor. The bundle
(`glucofm_heads.json`) is licensed **CC-BY-NC-SA-4.0**, one step stricter than the encoder.

Eight heads are fitted on CGMacros, whose registry entry reads `CC-BY-NC-SA-4.0` with
`license_confidence: unverified` (open question Q4). Share-alike obliges an adapted work to be
offered under the same licence, so the bundle carries it. Keeping the heads as a separate
artefact from the encoder stops that term reaching the encoder, which never saw CGMacros and
stays CC-BY-NC-4.0.

Three heads are fitted on Stanford. They are published: Stanford is Lane A and already
inside the encoder, so withholding classifiers fitted on its labels would protect nothing.

Anything fitted on UCHTT1DM would stay out permanently — its no-derivatives term has no
labelling remedy. No such head exists today; the filter enforces it regardless.

See D025. Enforced by
`tests/unit/test_source_rights.py::test_published_heads_bundle_declares_the_licence_its_sources_impose`.

## Evaluation (the headline numbers)

The table below is the **five-seed, 120-epoch** protocol — the scientific result of the
project. It characterises the method, not the single file above. A single ep40 checkpoint
will land near, but not exactly on, these numbers.

5 seeds × 120 epochs, subject-disjoint 5×10-fold probe, macro across 14 dataset-task probes
(18 task-source combinations — two cohorts are measured on two sensors each):

| Method | ROC-AUC (mean ± sd) | PR-AUC (mean ± sd) |
|---|---|---|
| `clinical_metrics` baseline | 0.643 | — |
| `raw_masked` baseline | 0.607 | — |
| **`opencgm_mean` (this model)** | **0.670 ± 0.003** | **0.588 ± 0.003** |
| CGM-JEPA (the paper's central comparator) | 0.643 ± 0.004 | 0.566 ± 0.003 |

The paper reports 0.74 ROC across 14 tasks on its full corpus; we measure 0.670 on
30.9 % of the corpus. **Direction and ordering of effects survive; absolute magnitude
is smaller.** See `findings/results_section.md` §8–§9 for the gap analysis.

## Ablation evidence

The 30-run Tier-1 ablation (3 seeds × 10 conditions × 40 epochs) confirms the
load-bearing role of the dual-stream decomposition: removing the event stream costs
**−0.050 ROC** (largest single ablator), removing the state stream costs −0.014, and, the
circadian embedding costs −0.015. Single-stream baselines (`abl_raw`, `abl_event`,
`abl_state`) are the only ablations beyond −0.02.

The `abl_dense` arm (forced interpolation, which the production rule forbids) is
**statistically indistinguishable** from the observed mask at 40 epochs (+0.0005). The
never-interpolate rule is not a hair-shirt; it is the principled choice that does not
also need ablation to defend it.

## Caveats and limits

- **30.9 % of paper pretraining hours.** Wear-CGM (two unreleased Google/Fitbit cohorts
  corpus) is non-public. The released encoder cannot match the paper's absolute scores.
- **Stanford's source repository states no licence.** It is in the released weights, and
  recorded as such in `manifests/sources/registry.yaml`.
- **Single-seed multi-day, cross-dataset, and few-shot extensions.** These are
  documented in `findings/results_section.md` and will gain error bars when the
  multi-seed sweep finishes.
- **No real-time inference.** The encoder expects 24-hour windows; real-time
  per-reading inference is out of scope.

## Citation

```
@software{opencgm_stateevent_2026,
  title  = {OpenCGM-StateEvent: independent public-data reconstruction of GlucoFM},
  author = {Fourdrinier, Stephane},
  year   = {2026},
  url    = {https://github.com/sfourdrinier/opencgm},
  note   = {Apache-2.0; see CITATION.cff}
}
```

Plus cite the original method:

```
@article{glucofm_2026,
  title  = {GlucoFM: A Dual-Stream Foundation Model for Continuous Glucose Monitoring},
  url    = {https://arxiv.org/abs/2605.30865v2}
}
```
