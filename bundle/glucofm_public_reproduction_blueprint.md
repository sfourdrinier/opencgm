# OpenCGM-StateEvent
## Full Local Reproduction Blueprint for GlucoFM

**Version:** 1.0  
**Evidence cutoff:** August 27, 2026  
**Primary reference:** *GlucoFM: A Dual-Stream Foundation Model for Continuous Glucose Monitoring*, arXiv:2605.30865v2  
**Target hardware:** NVIDIA RTX 3090 first; RTX 5090 later; H100 optional  
**Primary goal:** Independently reproduce the disclosed GlucoFM method using every obtainable public source dataset, publish source code and appropriately licensed weights, and export the frozen encoder for local use in a local application.

> **Public positioning:** This must be described as an **independent public-data reconstruction of the GlucoFM method**, not Google's implementation or checkpoint. Wear-CGM is non-public, no official code or checkpoint was linked as of the evidence cutoff, and several low-level choices are not specified in the paper.

---

# 1. Executive decision

Build one codebase and three clearly separated model lanes.

## 1.1 Strict public-subset reproduction

**Checkpoint name:** `opencgm-stateevent-repro-public-strict`

Use only the four public pretraining sources used by GlucoFM:

1. ShanghaiT2DM pretraining entries
2. Stanford pretraining subset
3. BIG IDEAs
4. Colas

Target the paper's public subset:

- **285 dataset-defined records/entries**
- **33,736 CGM hours**
- approximately **30.9% of the paper's pretraining hours**
- approximately **59.7% of its dataset-defined records**

This is the checkpoint used for scientific comparison and the headline reproduction claim.

## 1.2 Public-plus permissive checkpoint

**Checkpoint name:** `opencgm-stateevent-public-plus`

Start from the same architecture and add newer public CGM sources only after original-source license, provenance, duplicate, consent, and weight-release reviews. This model is intended to become the strongest generally useful open checkpoint, but its results must never be mixed with the strict-reproduction numbers.

## 1.3 Research-only/restricted checkpoint

**Checkpoint name:** `opencgm-stateevent-research-restricted`

Optionally include noncommercial, share-alike, credentialed, DUA-governed, or otherwise restricted sources when their terms permit the experiment. Do not distribute this checkpoint or embed it in a commercial product unless the complete source-rights review permits that use.

## 1.4 Why the separation matters

Combining all datasets into one initial checkpoint would weaken both claims:

- Reviewers could not distinguish faithful GlucoFM reconstruction from benefits caused by extra data.
- A single ambiguous or noncommercial source could complicate an otherwise permissive weight release.

The strict checkpoint establishes credibility. The public-plus checkpoint establishes utility.

---

# 2. Evidence labels and scientific honesty

Every consequential implementation choice must be tagged in code, configuration, and documentation as one of:

- **`PAPER_EXACT`** — directly disclosed by GlucoFM v2.
- **`SOURCE_VERIFIED`** — verified from an original data repository, paper, license, or official project page.
- **`INFERRED_RECONSTRUCTION`** — needed to implement the method but not specified by the authors.
- **`PROPOSED_EXTENSION`** — intentionally beyond GlucoFM and excluded from the headline reproduction.

A guessed choice must never silently become “the GlucoFM recipe.” The resolved configuration for every run must contain an `evidence_status` field for each ambiguous option.

---

# 3. What is reproducible and what is not

## 3.1 Disclosed well enough to reproduce

The paper specifies:

- 24-hour chronological windows
- a 5-minute grid with 288 positions
- preservation of a physical observation mask and no default interpolation
- segmentation at gaps longer than one hour
- 24 one-hour patches of 12 grid positions
- state and event decomposition using a learnable one-sided causal Gaussian filter
- filter bounds, initialization, and kernel truncation
- patch-level state/event statistics and rate-of-change search horizon
- stream feature dimensions
- fused token dimension
- Transformer depth, width, heads, and feed-forward dimension
- masked-patch ratio
- EMA target momentum schedule
- masked contextual and temporal-dynamics losses
- loss weights
- CGM-specific augmentation types, probabilities, and ranges
- epoch count, global batch size, learning rates, weight decay, and H100 reference hardware
- frozen downstream pooling and major evaluation protocols

## 3.2 Not disclosed or unavailable

The following prevent an exact checkpoint reproduction:

- Wear-CGM, representing 75,330 of 109,066 pretraining hours, is non-public.
- No official GlucoFM source repository or public weights were linked or found as of August 27, 2026.
- The exact optimizer is not named.
- The exact learning-rate schedule, warmup, gradient clipping, mixed precision, dropout, random seeds, initialization, and framework are not named.
- “Mask-aware normalization” is shown as instance normalization, but its exact equation, epsilon, axes, and affine behavior are not specified.
- The exact convolution kernels, stream-embedder MLP depth, activations, transition-head widths, time-gate equation, and predictor internals are not specified.
- The exact interpretation of the 20–80% overlapping-window “coverage ratio” is not specified.
- The fixed window-sampling seed is not published.
- The paper says most sources use floor binning and some use nearest-index rounding, but does not map rules to datasets.
- Several downstream head details and hyperparameters are omitted.

## 3.3 Approved public claim

Use language equivalent to:

> We independently reconstructed the GlucoFM v2 method from the paper and trained it on the four public pretraining cohorts used by the authors. Because Wear-CGM and several low-level implementation details are unavailable, this is a high-fidelity public-data reproduction rather than the authors' checkpoint.

Do not write “we reproduced Google's model” without this qualification.

---

# 4. Completion criteria

The project is not complete merely because training runs.

## 4.1 Reproducibility

- Every downloadable source has a version, source URL/DOI, file size, SHA-256 checksum, acquisition date, and license record.
- Every derived artifact records source-manifest, preprocessing-config, code-commit, and split-manifest hashes.
- Subject/session split files are immutable and reviewable.
- A checkpoint resumes with optimizer, EMA target, scheduler, sampler, augmentation, and RNG states intact.
- At least five strict-reproduction seeds are published.
- Full environment locks and GPU/runtime details accompany every result.

## 4.2 Scientific fidelity

- Strict-source record/hour counts reconcile as closely as possible with the paper, with every discrepancy documented.
- Parameter counts are close to the rounded 0.72M trainable and 1.18M total counts, without distorting disclosed architecture dimensions merely to hit rounded totals.
- Representation-collapse metrics remain healthy.
- Learned Gaussian bandwidth behaves nontrivially and remains bounded.
- Dual-stream, stream-only, interpolation, augmentation, and temporal-loss ablations are completed.
- Subject, biological-person, visit, and duplicate-source leakage tests pass.
- Downstream folds are identical across representation comparisons.

## 4.3 Release readiness

- Original code uses an OSI-approved license, recommended Apache-2.0.
- Raw data are not repackaged unless explicitly permitted.
- Weight licensing is reviewed separately from code licensing.
- Model and dataset cards document provenance, restrictions, limitations, intended use, and non-medical status.
- Safetensors and ONNX exports pass numerical parity tests.
- A local-application adapter can run the encoder against a rolling 24-hour local CGM buffer.

---

# 5. Repository and implementation stack

## 5.1 Recommended project identity

- **Repository:** `opencgm-stateevent`
- **Model family:** `OpenCGM-StateEvent`
- **Subtitle:** “Independent public-data reconstruction of the GlucoFM dual-stream CGM method”

Avoid Google's logos, endorsement implications, and naming released weights simply `GlucoFM`.

## 5.2 Technology choices

Use Python/PyTorch for the scientific core. Use Bun/TypeScript where it helps orchestration, dashboards, schema generation, and Tauri integration.

Recommended stack:

- Python 3.11 or 3.12, pinned
- current stable PyTorch compatible with the selected CUDA toolkit, pinned
- `uv` and `uv.lock`
- NumPy, SciPy
- PyArrow and Polars or pandas
- Pydantic v2
- scikit-learn
- Safetensors
- ONNX and ONNX Runtime
- pytest, Hypothesis
- Ruff and Pyright or mypy
- optional MLflow/W&B, while retaining fully local JSONL/Parquet experiment records
- Bun/TypeScript for local tooling and Tauri bridge packages

Do not make a hosted tracking platform necessary to reproduce results.

## 5.3 Repository layout

```text
opencgm-stateevent/
├── README.md
├── LICENSE
├── NOTICE
├── CITATION.cff
├── CONTRIBUTING.md
├── SECURITY.md
├── pyproject.toml
├── uv.lock
├── bun.lock
├── justfile
├── configs/
│   ├── data/
│   │   ├── strict_public.yaml
│   │   ├── public_plus.yaml
│   │   └── research_restricted.yaml
│   ├── model/
│   │   ├── glucofm_v2_reconstruction_a.yaml
│   │   └── ablations/
│   ├── train/
│   │   ├── paper_minimal_3090.yaml
│   │   ├── modern_stable_3090.yaml
│   │   ├── paper_minimal_5090.yaml
│   │   └── h100_throughput.yaml
│   └── eval/
├── manifests/
│   ├── sources/
│   ├── files/
│   ├── licenses/
│   ├── identities/
│   ├── splits/
│   └── windows/
├── data/                       # gitignored
│   ├── raw/
│   ├── canonical/
│   ├── windows/
│   └── reports/
├── src/opencgm_stateevent/
│   ├── cli.py
│   ├── config.py
│   ├── provenance.py
│   ├── data/
│   │   ├── schema.py
│   │   ├── timestamps.py
│   │   ├── units.py
│   │   ├── segmentation.py
│   │   ├── grid.py
│   │   ├── windowing.py
│   │   ├── augmentations.py
│   │   ├── validation.py
│   │   └── datasets/
│   ├── model/
│   │   ├── masks.py
│   │   ├── normalization.py
│   │   ├── causal_gaussian.py
│   │   ├── statistics.py
│   │   ├── stream_embedder.py
│   │   ├── time_embedding.py
│   │   ├── encoder.py
│   │   ├── predictor.py
│   │   ├── transition.py
│   │   ├── ema.py
│   │   ├── losses.py
│   │   └── model.py
│   ├── train/
│   │   ├── sampler.py
│   │   ├── dataloader.py
│   │   ├── loop.py
│   │   ├── checkpoint.py
│   │   ├── diagnostics.py
│   │   └── distributed.py
│   ├── eval/
│   │   ├── embeddings.py
│   │   ├── linear_probe.py
│   │   ├── ppgr.py
│   │   ├── few_shot.py
│   │   ├── multiday.py
│   │   ├── cross_dataset.py
│   │   ├── fairness.py
│   │   └── statistics.py
│   └── export/
│       ├── safetensors.py
│       ├── onnx.py
│       └── parity.py
├── packages/
│   ├── canonical-cgm-schema/
│   └── local-app-opencgm/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── golden/
│   ├── leakage/
│   └── export/
├── reports/
└── model_cards/
```

---

# 6. Paper dataset map

## 6.1 Exact paper composition

| Role | Dataset | Native cadence | Paper records/entries | Paper duration | Public status |
|---|---|---:|---:|---:|---|
| Pretrain | Wear-CGM | 5 min | 192 | 75,330 h | Non-public |
| Pretrain | ShanghaiT2DM subset | 15 min | 44 entries | 12,414 h | Public |
| Pretrain | Stanford subset | 5 min | 19 | 8,761 h | Public |
| Pretrain | BIG IDEAs | 5 min | 16 | 3,017 h | Public |
| Pretrain | Colas | 5 min | 206 | 9,544 h | Public |
| Downstream | CGMacros | 5 and 15 min | 45 | 10,376 / 10,998 h | Public, terms need lane review |
| Downstream | ShanghaiT2DM subset | 15 min | 65 entries | 15,634 h | Public |
| Downstream | Stanford subset | 5 min | 37 | 27,571 h | Public |
| Downstream | Hall | 5 min | 56 | 7,090 h | Public |

Shanghai counts are recording-visit entries and may not equal unique biological individuals. Store both visit and person identities whenever the source allows reconstruction.

## 6.2 Source manifest contract

```yaml
dataset_id: big_ideas
role: strict_pretrain
source:
  title: BIG IDEAs Lab Glycemic Variability and Wearable Device Data
  version: 1.1.2
  doi: 10.13026/zthx-5212
  landing_page: https://physionet.org/content/big-ideas-glycemic-wearable/1.1.2/
  acquired_at_utc: null
license:
  identifier: ODC-By-1.0
  raw_redistribution: verify
  commercial_use: verify_source_content_rights
  model_weight_release: legal_review_required
files: []
selection:
  expected_records: 16
  expected_hours: 3017
  split_manifest: manifests/splits/big_ideas_strict.yaml
adapter:
  module: opencgm_stateevent.data.datasets.big_ideas
  glucose_unit: mg/dL
  native_cadence_minutes: 5
  binning_rule: unresolved
notes: []
```

Every file entry must contain path, bytes, SHA-256, source URL, and content role.


## 6.3 BIG IDEAs

**Strict source:** PhysioNet version 1.1.2, DOI `10.13026/zthx-5212`.

- The paper cites version 1.1.2.
- The dataset license is ODC Attribution 1.0 at the database level; separately review content/privacy and model-weight implications.
- Strict GlucoFM pretraining requires Dexcom files and basic identity metadata, not the much larger unrelated wearable channels.
- Expected paper result after preprocessing: 16 participants and 3,017 hours.

Acquisition plan:

1. Download the 1.1.2 landing-page files or use a documented PhysioNet downloader.
2. Acquire `Demographics.csv` and the per-participant Dexcom files for subjects 001–016.
3. Preserve raw byte hashes.
4. Parse source timestamps and values with no interpolation.
5. Reconcile native rows, valid rows, segment counts, aligned observations, and hours to the paper.
6. Record whether later version 1.1.3 modifies any CGM bytes before considering it for public-plus. Pin 1.1.2 for strict reproduction.

## 6.4 ShanghaiT2DM

**Source:** Figshare collection “Diabetes Datasets—ShanghaiT1DM and ShanghaiT2DM,” accompanying the Scientific Data article on Chinese diabetes datasets.

- T2DM source cohort: 100 participants, with some repeat recording visits represented as separate entries.
- Device/cadence: FreeStyle Libre, 15-minute readings.
- Source page identifies CC BY 4.0; retain attribution and verify any per-file exceptions.
- Paper split: 44 recording entries for pretraining, 65 entries for downstream evaluation.

Implementation requirements:

1. Build `canonical_person_id` and `recording_visit_id` separately.
2. Reconstruct the paper's pretraining/downstream assignments from source labels and metadata; freeze them in YAML.
3. Never interpolate the two missing 5-minute slots between native 15-minute readings. Map only real readings to the 5-minute grid.
4. Test floor and nearest-index mapping against timestamp residuals and expected counts.
5. Audit biological-person overlap between the paper-compatible entry split and downstream entries.
6. Publish a second, stronger biological-person-disjoint sensitivity evaluation where possible.

Expected strict pretraining target: 44 entries and 12,414 hours.

## 6.5 Stanford CGM database

**Source:** Stanford's public Continuous Glucose Monitoring Database associated with metabolic-subphenotype work.

Core files include:

- `data_cgm.csv`
- `data_meta.csv`
- phenotype/lipid/metabolomics files used for downstream outcomes

Paper allocation:

- 19 participants without relevant downstream clinical labels used for pretraining: 8,761 hours
- 37 participants used downstream: 27,571 hours

Rules:

1. Download original database files rather than a smoothed derivative copy.
2. Select participants through an explicit manifest.
3. Preserve raw sentinel values and document replacements.
4. Do not use cubic-spline-smoothed glucose in the strict pipeline.
5. Preserve internally consistent local time-of-day even if calendar dates were privacy shifted.
6. Review the database-specific terms separately; an open-access article license does not automatically answer every data or weight-release question.

## 6.6 Colas

**Source:** supporting material for the PLOS ONE paper “Detrended fluctuation analysis in the prediction of type 2 diabetes mellitus in patients at risk,” DOI `10.1371/journal.pone.0225817`.

- Native device/cadence: iPro at 5-minute intervals.
- Source paper cohort: 208 participants.
- GlucoFM reports 206 after preprocessing and 9,544 hours.

Rules:

1. Download the original PLOS supporting archive, not a transformed third-party copy.
2. Preserve original identifiers and timestamps.
3. Identify exactly why two records are excluded; create a rejection report instead of silently dropping them.
4. Verify units, duplicate timestamps, impossible readings, and sentinels.
5. Because many recordings are only around two days, audit boundary handling and legal 24-hour window starts carefully.
6. Confirm supporting-file rights before releasing a checkpoint trained on it.

## 6.7 Wear-CGM

The paper states that Wear-CGM is non-public. It contains two non-overlapping Google/Fitbit studies of healthy, non-diabetic US adults wearing Dexcom G6 Pro:

- Phase 1: 105 participants, approximately four weeks, typically 2–3 sensors per participant
- Phase 2: 87 participants, up to 15 days, standardized meal challenges and clinical measurements
- Combined: 192 participants and 75,330 CGM hours

Only CGM was used by GlucoFM even though additional wearable/nutrition/clinical information existed.

Practical access routes:

1. Contact the corresponding authors for a controlled DUA or collaboration.
2. Ask whether they can run the public reconstruction against Wear-CGM internally.
3. Ask whether a federated validation or author-run training experiment is possible.
4. Monitor the promised code/reproducibility release for an access pathway.
5. Do not infer access rights from the participant consent language.

A first-party replacement cohort at comparable scale is feasible:

```text
225 people × 14 days × 24 hours = 75,600 participant-hours
```

A strategically superior replacement study would collect synchronized CGM, raw PPG, accelerometry, temperature, HR/HRV, sleep/activity, meals, medication/insulin where relevant, finger-prick references, device events, exercise, and illness annotations. That would power both this CGM encoder and the later PPG-to-glucose student.

## 6.8 Downstream datasets

### CGMacros

- PhysioNet 1.0.0
- 45 participants
- paired Dexcom and Libre streams plus nutrition/context
- use for the paper-style downstream and PPGR tasks
- the accessible PhysioNet `LICENSE.txt` is empty as of the evidence cutoff, so do not repeat an unverified license label; review the dataset landing page, publication, author terms, and PhysioNet metadata before training or weight release
- keep it evaluation-only until rights are resolved

### Hall

- PLOS Biology: “Glucotypes reveal new patterns of glucose dysregulation,” DOI `10.1371/journal.pbio.2005143`
- paper target: 56 people and 7,090 hours
- use original supporting data for downstream classification
- consider for public-plus only after confirming supporting-file terms and duplicate relationships

## 6.9 Newer public-data discovery resources

### GlucoFM-Bench, June 2026

GlucoFM-Bench evaluates 15 public CGM datasets totaling 1,117 individuals across T1D, T2D, prediabetes, and no-diabetes cohorts. It provides a public standardized dataset and code for forecasting.

Use it to:

- discover source datasets
- validate adapters and expected population metadata
- add forecasting and subgroup benchmarks

Do not use its standardized/interpolated arrays as the strict GlucoFM pretraining input. Acquire original source data, preserve physical missingness, and run this project's own preprocessing. Aggregated licensing metadata is not a substitute for source-level review.

Candidate original sources include BIG IDEAs, D1NAMO, HUPA-UCM, Colas, ShanghaiT1DM, ShanghaiT2DM, Bris-T1D Open, T1DM-UOM, AZT1D, Hall, and others. Maintain a duplicate graph so the same participants are never counted twice through both original and aggregate sources.

### MetaboNet, 2026

MetaboNet consolidates T1D datasets containing overlapping CGM and insulin data. Its paper reports 3,135 participants and 1,228 participant-years of overlap, split between a fully public subset and DUA-governed sources.

Use it for a later T1D/public-plus checkpoint, not the strict reproduction. Requirements:

- trace every record back to the constituent source
- obey constituent rights and DUAs
- deduplicate against directly ingested datasets
- preserve missingness rather than using benchmark-imputed arrays
- keep T1D-heavy results separate from mixed-population results
- add insulin/carbohydrate channels only in a separately named multimodal extension

### FairGlucose, August 2026

FairGlucose introduces a 300-patient cohort balanced across 12 demographic strata and emphasizes that aggregate performance can hide subgroup disparities. Even when its raw data access and rights are unsuitable for pretraining, its evaluation methodology should be reflected in the public-plus release: report results by diabetes type, age, gender, glycemic range, device, and dataset wherever source metadata and sample sizes permit.

### PhysioCGM, November 2025

PhysioCGM is a CC0 multimodal dataset containing synchronized Dexcom G6 CGM, raw Empatica E4 PPG/EDA/accelerometry/temperature, Zephyr ECG/respiration/accelerometry, and timestamps from 10 participants with T1D for up to 17 days. The Figshare release is approximately 23 GB.

Use it in two ways:

- CGM-only public-plus pretraining after deduplication and quality review
- a high-value paired bridge for the later CGM-teacher to wearable-student project

Keep the raw multimodal and authors' prepacked five-minute clips. Build alignment from source timestamps independently and compare against the supplied preprocessing rather than assuming it is exact.

### Paired smartwatch PPG + CGM dataset, June 2026

The exploratory PPG/glucose study associated with arXiv:2606.15927 releases two-week paired smartwatch PPG and CGM data from five volunteers through Zenodo. It is too small to materially change CGM-only pretraining, but it is timely for teacher/student prototyping and external validation. Its paper explicitly reports dangerous errors in an early model, reinforcing that it is a research dataset rather than evidence of clinical readiness. Verify the current Zenodo version and license before use.

### Capillary-reference PPG dataset, August 2026

A newer open PPG dataset contains 125 multichannel PPG samples from 24 volunteers annotated with capillary glucose. It is not continuous CGM and therefore does not belong in GlucoFM reproduction, but it can become a small auxiliary benchmark for the later non-invasive project. Keep it out of the current model's training corpus.

### Synthetic datasets

Synthetic CGM is useful for stress tests, CI fixtures, privacy-safe examples, and an explicit synthetic-pretraining ablation. Never report synthetic hours as equivalent to human participant-hours and never mix them into the headline checkpoint without separate labeling.

## 6.10 Dataset release lanes

| Source category | Strict checkpoint | Public-plus | Research restricted | Raw redistribution |
|---|---:|---:|---:|---:|
| Four public GlucoFM pretrain cohorts | yes | yes after rights audit | yes | source-specific |
| Wear-CGM | unavailable | unavailable | collaboration only | no |
| CGMacros | evaluation initially | no until resolved | possibly | source-specific |
| Open GlucoFM-Bench originals | no | after source audit | yes where allowed | source-specific |
| Controlled/DUA datasets | no | generally no | DUA-dependent | usually no |
| MetaboNet public constituents | no | after constituent audit | yes | constituent-specific |
| Synthetic CGM | ablation only | optional separately labeled | yes | generator terms |

---

# 7. Rights and open-weight strategy

Treat four objects independently:

1. **Original code** — independently written from the paper; Apache-2.0 recommended.
2. **Raw datasets** — remain under each source's terms.
3. **Processed windows/features** — may still be restricted and should not be assumed redistributable.
4. **Model weights** — require a source-by-source decision; public download does not automatically imply unrestricted commercial weight release.

Required release process:

- generate a source-contribution manifest for every checkpoint
- list every source, version, license/terms URL, and inclusion fraction
- document whether raw or processed data are redistributed
- review ambiguous sources before public weight publication
- assign a weight license separately from the code license
- publish limitations and non-medical-use language

Recommended code license: **Apache-2.0**. Include `NOTICE` and `CITATION.cff` crediting the GlucoFM paper without implying endorsement.

Privacy requirements:

- never commit participant-level source data
- keep stable pseudonymous identities locally for leakage prevention
- publish only source-permitted samples or synthetic examples
- avoid publishing small-subgroup embeddings that increase re-identification risk
- document deletion/retraining procedures for future first-party data
- perform basic memorization and nearest-neighbor audits before release

This blueprint is an engineering and research plan, not legal advice.

---

# 8. Canonical data and provenance model

## 8.1 Immutable raw layer

```text
data/raw/<dataset>/<source-version>/...
```

Never modify raw files. Record:

- source URL/DOI
- acquisition timestamp
- source version
- bytes
- SHA-256
- license/terms reference
- downloader/tool version

## 8.2 Canonical reading schema

Use Arrow/Parquet with at least:

| Field | Type | Meaning |
|---|---|---|
| `dataset_id` | string | namespaced source identifier |
| `source_version` | string | exact source release |
| `source_subject_id` | string | original pseudonymous ID |
| `canonical_subject_id` | string | project namespaced ID |
| `biological_person_id` | string nullable | links repeat visits where source permits |
| `session_id` | string | visit/sensor session |
| `device_family` | string nullable | Dexcom, Libre, iPro, etc. |
| `timestamp_original` | string | exact source text |
| `timestamp_utc` | timestamp nullable | verified UTC |
| `local_datetime` | timestamp | local wall-clock used for circadian phase |
| `utc_offset_minutes` | int16 nullable | known offset |
| `glucose_original` | float32 nullable | original numeric value |
| `unit_original` | string | mg/dL or mmol/L |
| `glucose_mg_dl` | float32 nullable | canonical value |
| `quality_flag` | string nullable | source quality/sentinel |
| `is_real_measurement` | bool | never true for inserted grid positions |
| `source_file` | string | raw relative path |
| `source_row` | int64 | source row number |
| `license_id` | string | rights manifest key |

Use `1 mmol/L = 18.0182 mg/dL`. Preserve original values and sentinels in the audit layer.

## 8.3 Canonical session record

```yaml
session_id: shanghai_t2dm/person_023/visit_02
canonical_subject_id: shanghai_t2dm/person_023
biological_person_id: shanghai_t2dm/person_023
start_local: null
end_local: null
native_cadence_minutes: 15
device_family: libre
raw_rows: 0
valid_rows: 0
observed_hours: 0
long_gap_count: 0
selected_for: strict_pretrain
circadian_confidence: verified_local_time
```

## 8.4 Derived window record

```text
window_id
source dataset/version
canonical subject/person/session/segment IDs
24-hour local start
circadian start index
values float32[288]
physical mask bool[288]
observed count
coverage fraction
split
source/preprocessing/window-manifest hashes
```

Cache deterministic unaugmented windows. Apply stochastic augmentations online.

## 8.5 Timezone policy

Absolute time-of-day is a model input, so timezone handling is not incidental.

- Prefer source-provided local timestamps.
- Validate UTC/local offsets when both exist.
- Preserve local hour when dates are privacy-shifted.
- Do not invent a timezone from geography without documenting the inference.
- Handle daylight-saving repeated/skipped hours explicitly.
- Mark `circadian_confidence` as verified, preserved-but-shifted, inferred, or unavailable.
- If true time-of-day is unavailable, keep the source for a no-circadian ablation rather than silently fabricating it.


---

# 9. Segmentation, gridding, and window construction

## 9.1 Continuous-segment rule — `PAPER_EXACT`

For ordered readings within a subject/session:

- timestamp gap **≤ 60 minutes**: remain in the same continuous segment; absent grid positions later have mask zero
- timestamp gap **> 60 minutes**: start a new segment
- a 24-hour window must never cross a segment boundary

```python
def segment_readings(rows, max_internal_gap_minutes=60):
    rows = sort_and_resolve_duplicates(rows)
    segments, current = [], []
    for row in rows:
        if current and minutes(row.timestamp - current[-1].timestamp) > 60:
            segments.append(current)
            current = []
        current.append(row)
    if current:
        segments.append(current)
    return segments
```

The exact boundary matters: a 60-minute gap remains internal because the paper says gaps longer than one hour create boundaries.

## 9.2 Chronological grid — `PAPER_EXACT`

- window length: 24 hours
- grid interval: 5 minutes
- sequence length: `L = 288`
- patches: `P = 24`
- positions per patch: `K = 12`

For first window timestamp `t₁`:

```text
s = floor((60 × local_hour(t₁) + local_minute(t₁)) / 5)
a_j = (s + j) mod 288
```

For source reading `tᵢ`:

```text
uᵢ = (tᵢ - t₁) / 5 minutes
jᵢ = B(uᵢ)
```

`B` is floor for most datasets and nearest-index rounding where it better matches the source timestamp convention. Exclude indices outside `[0,287]`. Average multiple real readings assigned to the same position.

Missing positions may be stored as zero for tensors, but the physical observation mask is authoritative. There is no default interpolation or imputation.

## 9.3 Dataset-specific binning audit

The paper does not publish the mapping of dataset to floor/nearest. Implement both and generate a report containing:

- timestamp residual histogram relative to five-minute boundaries
- collisions
- out-of-range assignments
- recovered native cadence
- observed-position count
- segment/window count
- paper-hour reconciliation

Define nearest-rounding ties explicitly. Do not accidentally rely on Python banker's rounding.

Freeze the chosen rule in the source manifest. A later official code release should be compared against this registry.

## 9.4 Overlapping pretraining windows

The paper states:

- fixed seed
- random overlapping 24-hour windows
- per-segment coverage ratio between 20% and 80%

It does not define “coverage ratio.” Implement and compare two candidates.

### Candidate A: legal-start fraction — default initial reconstruction

1. Enumerate every legal five-minute start whose 24-hour interval lies in the segment.
2. Sample `c ~ Uniform(0.20, 0.80)` per segment.
3. Select `max(1, round(c × N_legal_starts))` unique starts without replacement.
4. Use a stable seed derived from global seed + dataset + subject + session + segment.
5. Sort selected starts before serializing the manifest.

### Candidate B: union-of-timeline coverage

1. Sample `c ~ Uniform(0.20,0.80)`.
2. Select legal starts until the union of covered source timeline reaches `c` of coverable segment duration.
3. Remove duplicate windows.

Generate paper-count, training-volume, overlap, and runtime estimates for both. Choose the candidate whose scale and behavior are most plausible, record the decision, and ask the authors this exact question. Never hide the ambiguity.

## 9.5 Downstream windows — `PAPER_EXACT`

- non-overlapping 24-hour windows
- subject-grouped splits
- no subject crosses train/test folds
- split identities before window extraction

For repeat visits, run both:

- paper-compatible entry-level allocation
- stronger biological-person-disjoint sensitivity analysis

## 9.6 Base validity policy

The paper does not publish a minimum-density threshold for ordinary windows.

Reference behavior:

- reject windows with zero observations
- reject impossible timestamps/units
- otherwise retain the window and record density
- do not introduce a hidden minimum count
- test density thresholds only as explicit ablations

The augmentation condition “more than 200 observations” is not a base inclusion criterion.

---

# 10. Mask system and operation ordering

Maintain distinct masks with explicit names:

1. `physical_mask [B,288]` — real source measurements after grid mapping
2. `augmented_mask [B,288]` — physical mask after structural augmentation
3. `context_patch_mask [B,24]` — patches hidden from online JEPA branch; true means hidden
4. `online_visible_mask [B,288]` — augmented mask with hidden patches removed
5. `target_visible_mask [B,288]` — complete augmented physical support used by EMA target branch in the reference view policy
6. `roc_valid_mask [B,288]` — branch-specific valid rate-of-change pairs
7. `patch_density [B,24]` — target/physical observation fraction per patch

## 10.1 Critical no-leakage ordering

The paper explicitly says selected patches are hidden from the visible signal used for statistics and filtering. Therefore:

```text
aligned raw window
→ physical CGM augmentation
→ sample JEPA patch mask
→ remove masked patches from online-visible mask
→ online mask-aware normalization
→ online causal Gaussian filter
→ online state/event statistics and embeddings
→ replace hidden physiological patch tokens with learned mask token
→ online context Transformer
```

Target branch:

```text
same aligned/augmented physical view
→ full target-visible mask
→ target mask-aware normalization
→ EMA causal Gaussian filter and stream embedders
→ EMA context Transformer
→ stop-gradient targets
```

An implementation that computes full-day normalization/filtering and masks only after tokenization leaks masked-patch information and is wrong.

## 10.2 Fill-value invariance

Tests must prove that changing the tensor fill value at `mask=0` positions does not alter valid normalization, filter, statistics, tokens, or losses beyond numerical tolerance.

---

# 11. Mask-aware normalization and statistics

## 11.1 Reference normalization — `INFERRED_RECONSTRUCTION`

The paper names mask-aware normalization and diagrams instance normalization, but omits the equation. Use observed-only per-window instance normalization:

```text
n = Σⱼ Mⱼ
μ = Σⱼ Mⱼ Xⱼ / (n + ε)
v = Σⱼ Mⱼ (Xⱼ - μ)² / (n + ε)
scale = sqrt(v + ε)
X̃ⱼ = Mⱼ (Xⱼ - μ) / max(scale, scale_min)
```

Defaults:

- compute separately for every sample and branch-visible mask
- no learned affine parameters
- `ε = 1e-6`
- `scale_min = 1e-4`
- all-zero input returns zeros plus an invalid diagnostic

Required ablations:

- observed-only per-window instance norm
- dataset-global train-only normalization
- observed-only per-patch norm
- no normalization
- target branch using its own full statistics versus shared online statistics

The first is the reference because it most closely matches the paper diagram and wording.

## 11.2 Patch density — `PAPER_EXACT`

For patch `i`:

```text
dᵢ = (1/12) Σⱼ∈patchᵢ Mⱼ
```

Use physical/target density for contextual weighting. Use adjacent densities in temporal-transition weighting.

## 11.3 State patch statistics — `PAPER_EXACT`

Observed-only mean and standard deviation over the state source in each one-hour patch:

```text
μᵢ = Σ Mⱼ Xⱼ / (Σ Mⱼ + ε)
σᵢ = sqrt(Σ Mⱼ (Xⱼ - μᵢ)² / (Σ Mⱼ + ε))
```

Empty patch outputs are zero and carry a validity diagnostic.

## 11.4 Rate of change — `PAPER_EXACT`

For every observed position `j`, search backward for the nearest observed predecessor within nine grid steps:

```text
b = min { k ∈ [1,9] : Mⱼ₋ₖ = 1 }
rⱼ = (Xⱼ - Xⱼ₋ᵦ) / b
```

If no predecessor exists, set `rⱼ=0` and `roc_validⱼ=0`. Otherwise set valid true. The strict unit is normalized-glucose change per five-minute grid step, matching the paper equation; a per-minute conversion is an extension.

Compute event patch mean/std over valid RoC entries only.

## 11.5 Intra-patch trend difference — `INFERRED_RECONSTRUCTION`

The paper gives a 16-dimensional trend-difference feature and the diagram labels a patch-level `Diff` path, but does not define its source sequence.

Reference:

```text
diffᵢ,ₖ = stateᵢ,ₖ - stateᵢ,ₖ₋₁     for k=1..11
validᵢ,ₖ = Mᵢ,ₖ × Mᵢ,ₖ₋₁
```

Do not bridge patch boundaries. Process the 11 values or prepend a masked zero to length 12. Add first-to-last slope and raw-signal difference as labeled ablations.

---

# 12. Causal Gaussian state/event decomposition

## 12.1 Exact equations

For normalized signal `X̃`, branch mask `M`, and grid position `j`:

```text
Stateⱼ = [Σᵣ₌₀ᴿ Kσ(r) Mⱼ₋ᵣ X̃ⱼ₋ᵣ]
         / [Σᵣ₌₀ᴿ Kσ(r) Mⱼ₋ᵣ + ε]
```

Ignore negative indices.

```text
Kσ(r) = exp(-r² / (2σ²)) / Σᵤ₌₀ᴿ exp(-u² / (2σ²))
```

Paper settings:

- `σ_min = 2`
- `σ_max = 12`
- `R = ceil(3 × σ_max) = 36`
- initial `σ = 6`
- five-minute steps, corresponding to an approximate 10–60 minute scale range

Constrain the learnable bandwidth:

```text
σ = 2 + 10 × sigmoid(ρ)
ρ₀ = logit((6-2)/(12-2)) = logit(0.4) ≈ -0.4054651081
```

Event residual:

```text
Event = (X̃ - State) ⊙ M
```

## 12.2 Implementation requirements

- differentiable PyTorch implementation
- FP32 kernel and denominator under autocast
- strictly one-sided/causal
- masked denominator so absent points do not pull toward the fill value
- zero output where support denominator is zero
- online `ρ` receives a separate learning rate
- EMA target has a separate copied `ρ` updated by EMA, never optimizer gradients
- ordinary `conv1d` implementation is sufficient for length 288 and radius 36

## 12.3 Golden tests

- an impulse at `t+1` cannot affect state at `t`
- all-observed output matches a direct-loop implementation
- random masks match a high-precision NumPy reference
- fill-value invariance
- sigma initialization equals six within tolerance
- gradients reach `ρ`
- sigma never escapes `[2,12]`
- denominator-zero positions produce finite zeros

## 12.4 Diagnostics

Log:

- online/target sigma
- rho gradient norm
- zero-support fraction
- state/event variance ratio
- state/event frequency-energy summaries
- event sparsity and kurtosis
- correlation of state with hourly glucose mean
- correlation of event with short-term changes

---

# 13. Stream embedder reconstruction

## 13.1 Paper-disclosed dimensions

State patch:

- waveform feature: 64
- intra-patch trend-difference feature: 16
- projected mean/std feature: 48
- concatenate 128 and project to 64-dimensional state token

Event patch:

- residual waveform feature: 48
- rate-of-change waveform feature: 48
- projected RoC mean/std feature: 32
- concatenate 128 and project to 64-dimensional event token

Fusion/time:

- state 64 + event 64 → physiological token 128
- circular time features → 128
- learned patch-position embeddings combined with circadian embedding through a learnable gate

Context/target encoder:

- three Transformer layers
- hidden dimension 128
- four attention heads
- feed-forward dimension 256

Predictor:

- one Transformer layer

## 13.2 Reference Reconstruction A

This is a defensible default, not claimed as the authors' unpublished implementation.

### State waveform branch

```text
[B,24,12]
→ reshape [B×24,1,12]
→ Conv1d(1,64,kernel=3,padding=1,bias=true)
→ GELU
→ masked mean over patch length
→ [B,24,64]
```

### State difference branch

```text
adjacent state differences + validity mask
→ Conv1d(1,16,kernel=3,padding=1,bias=true)
→ GELU
→ masked mean
→ [B,24,16]
```

### State statistics branch

```text
[state mean,state std]
→ Linear(2,48)
→ GELU
→ [B,24,48]
```

Concatenate and project:

```text
64+16+48=128
→ Linear(128,64)
→ LayerNorm(64)
→ state tokens
```

### Event residual branch

```text
event residual
→ Conv1d(1,48,kernel=3,padding=1)
→ GELU
→ masked mean
→ [B,24,48]
```

### Event RoC branch

```text
rate-of-change + validity mask
→ Conv1d(1,48,kernel=3,padding=1)
→ GELU
→ masked mean
→ [B,24,48]
```

### Event statistics branch

```text
[RoC mean,RoC std]
→ Linear(2,32)
→ GELU
→ [B,24,32]
```

Concatenate and project:

```text
48+48+32=128
→ Linear(128,64)
→ LayerNorm(64)
→ event tokens
```

### Fusion

```text
concat(state,event) [B,24,128]
→ Linear(128,128)
→ LayerNorm(128)
→ physiological token
```

## 13.3 Convolution/missingness modes

Reference strict mode:

- zero masked input values
- ordinary convolution
- zero all outputs for empty patches after the bias-bearing convolution
- masked mean pooling

A `partial_conv` mode that normalizes by local observed support is scientifically interesting but is an ablation because it adds an unpublished operation.

## 13.4 Circular time and position

For absolute five-minute index `a`:

```text
time(a) = [sin(2πa/288), cos(2πa/288)]
```

Reference representative patch time: patch start.

```text
aᵢ = (circadian_start_index + 12i) mod 288
time₂ → Linear(2,128)
```

Learned position embedding: `[24,128]`.

Reference gate:

```text
g = sigmoid(g_raw), g_raw ∈ R¹²⁸ initialized to 0
τᵢ = position_embeddingᵢ + g ⊙ time_embeddingᵢ
encoder_inputᵢ = physiological_tokenᵢ + τᵢ
```

Required ambiguity tests:

- patch start versus center time
- scalar versus vector gate
- additive versus convex blend

## 13.5 Transformer internals — inferred defaults

- pre-norm
- GELU
- dropout 0.1
- LayerNorm epsilon `1e-5`
- biases enabled
- no class token
- full attention across 24 patches

Run pre/post-norm and dropout 0/0.1 as architecture-resolution ablations.

## 13.6 Mask token

Selected patches are removed from online signal processing and their physiological tokens are replaced before context encoding.

Reference:

```text
masked_encoder_inputᵢ = learned_mask_token + τᵢ
```

Retain time/position so the predictor knows the missing patch's temporal location.

## 13.7 Parameter-count constraint

The paper reports approximately:

- 0.72M trainable parameters
- 1.18M total during pretraining, including EMA target

A plausible implementation using the modules above, ordinary PyTorch Transformer layers, a one-layer predictor, and two hidden-256 transition MLPs lands near 0.73M trainable and 1.17M total. That is close but does not prove exact internal identity.

Required action:

- print a component-level parameter report in CI
- keep transition width, embedder depth, gate shape, and norms configurable
- use the paper count as a constraint, not as permission to modify disclosed dimensions
- rerun parity when official code appears

---

# 14. Context encoder, EMA target, predictor, and transitions

## 14.1 Online and target encoders

The EMA target is initialized from the online encoder and not updated by backpropagation.

```text
θ_target ← m θ_target + (1-m) θ_online
```

Momentum follows a cosine schedule from 0.997 to 1.0 over all optimizer steps:

```python
progress = global_step / max_steps
m = 1.0 - (1.0 - 0.997) * (math.cos(math.pi * progress) + 1.0) / 2.0
```

Update target parameters after the optimizer step. The target copy should include all learned online encoder components required to generate targets: Gaussian bandwidth, stream embedders, fusion/time modules, and three-layer Transformer.

## 14.2 Predictor

Reference one-layer Transformer:

- dimension 128
- four heads
- FFN 256
- same inferred norm/activation/dropout conventions as context encoder
- input/output shape `[B,24,128]`

Loss is evaluated only at context-masked positions.

## 14.3 Transition heads

Use online pre-Transformer state/event tokens and temporal embedding:

```text
Ŝᵢ₊₁ = Sᵢ + gS([Sᵢ,Eᵢ,τᵢ])
Êᵢ₊₁ = Eᵢ + gE([Eᵢ,Sᵢ,τᵢ])
```

Targets are EMA pre-Transformer state/event tokens at the next patch, preventing future contextual attention from leaking into transition targets.

Reference head:

```text
input 256
→ Linear(256,256)
→ GELU
→ Linear(256,64)
```

Keep hidden width configurable for parameter-count matching.


---

# 15. Pretraining objectives

## 15.1 Patch masking — `PAPER_EXACT`

- sample a mask ratio uniformly from `[0.50,0.60]` for each sample
- convert it to an integer number of masked patches using a documented rule
- select patch indices without replacement
- target branch sees complete physical/augmented observations
- online branch removes those patches before normalization, filtering, and statistics

Reference integer rule:

```text
n_masked = round(mask_ratio × 24)
```

Clamp to retain at least one visible and one masked patch. Log requested and realized mask-ratio distributions.

## 15.2 Masked contextual representation loss — `PAPER_EXACT`

Let:

- `Z_pred_i` = predictor output at patch `i`
- `Z_target_i` = detached EMA contextual target
- `c_i` = 1 if patch is context-masked
- `d_i` = physical observation density

```text
L_MCR = Σᵢ dᵢ cᵢ SmoothL1(Z_predᵢ,Z_targetᵢ)
        / (Σᵢ dᵢ cᵢ + ε)
```

Reference details:

- mean over latent dimension
- SmoothL1 `beta=1.0`
- zero differentiable loss if denominator is zero
- target detached explicitly

## 15.3 Transition weight — `PAPER_EXACT`

For adjacent patches `i` and `i+1`:

```text
qᵢ = (1 - cᵢ) dᵢ dᵢ₊₁
```

This excludes transitions whose starting online patch is masked and down-weights sparse adjacent patches. Follow the published equation; do not add a separate destination-context-mask term in the strict path.

## 15.4 Temporal dynamics loss — `PAPER_EXACT`

```text
L_TD = 0.5 × [
    Σᵢ qᵢ SmoothL1(Ŝᵢ₊₁,S_targetᵢ₊₁) / (Σᵢ qᵢ + ε)
  + Σᵢ qᵢ SmoothL1(Êᵢ₊₁,E_targetᵢ₊₁) / (Σᵢ qᵢ + ε)
]
```

## 15.5 Total loss — `PAPER_EXACT`

```text
L = λ_MCR L_MCR + λ_TD L_TD
λ_MCR = 1.0
λ_TD = 1.0
```

Required temporal-weight sweep:

```text
λ_TD ∈ {0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0}
```

The paper reports broad strength around approximately 0.6–1.0, useful as a directional check rather than a guaranteed public-subset result.

## 15.6 Collapse and health diagnostics

Log per epoch and periodically per step:

- total, MCR, state-TD, event-TD losses
- losses by source dataset
- target and online latent standard deviation by dimension
- effective rank and covariance off-diagonal magnitude
- pairwise cosine similarity
- contextual/state/event token norms
- predictor-target similarity at masked patches
- state/event token variance and cross-correlation
- gradient norms by component
- realized context-mask and observation-density distributions
- learned sigma and rho gradients

Warn when effective rank collapses or most dimensions have near-zero variance. Do not add VICReg-style variance penalties to the strict path unless collapse is empirically demonstrated; that would be an extension.

---

# 16. CGM-aware augmentations

## 16.1 Ordering and probability decay — `PAPER_EXACT`

- four candidate augmentations
- evaluate them in random order
- after one applies, multiply every subsequent application probability by `0.25`
- value perturbations preserve the mask
- structural perturbations modify the mask

Reference pseudocode:

```python
operations = rng.permutation([
    baseline_wander,
    compression_drop,
    decimation,
    disconnection,
])
scale = 1.0
for op in operations:
    if rng.random() < op.base_probability * scale:
        values, mask = op(values, mask, rng)
        scale *= 0.25
```

This applies the reduction repeatedly after multiple successful operations. Add a one-time-only interpretation only as an ablation.

The paper does not disclose whether online and target receive the same augmentation view. Reference policy:

1. create one physically augmented values/mask view
2. give it to both branches
3. add JEPA patch masking only to online branch

Compare online-only and independent-view variants later.

## 16.2 Baseline wander

Paper parameters:

- base probability `0.25`
- sinusoidal amplitude uniform 5–15 mg/dL
- frequency uniform 0.5–2 cycles per 24-hour window

Reference missing choice: random phase uniform `[0,2π)`.

```text
δⱼ = A sin(2π f j/288 + φ)
X'ⱼ = Xⱼ + Mⱼ δⱼ
```

Do not alter unobserved positions. Do not clip in the strict path because clipping is not specified; report implausible post-augmentation ranges.

## 16.3 Compression-like drop

Paper parameters:

- base probability `0.10`
- contiguous duration 6–12 grid positions
- V-shaped attenuation
- minimum multiplier uniform 0.4–0.7

Reference:

- uniformly sample legal start and integer length
- build a piecewise-linear multiplier `1 → minimum → 1`
- multiply observed values
- keep mask unchanged
- explicitly test odd and even lengths

## 16.4 Decimation

Paper parameters:

- base probability `0.40`
- apply only when observed count is greater than 200
- choose random offset 0, 1, or 2
- retain every third five-minute grid position matching the offset
- remove other observed positions to simulate a 15-minute-like pattern

```text
keepⱼ = Mⱼ AND (j mod 3 = offset)
```

Apply by absolute grid index, not by every third row in a compressed list of observations.

## 16.5 Disconnection blocks

Paper parameters:

- base probability `0.05`
- remove 1–3 blocks
- each block length 2–12 grid positions

Reference:

- integer counts/lengths sampled uniformly
- starts sampled uniformly from legal indices
- overlapping blocks allowed unless official code shows otherwise
- removed positions receive mask zero and tensor fill value

## 16.6 Augmentation tests

- deterministic with supplied seed
- no canonical-window mutation
- value perturbations leave mask unchanged
- structural operations never increase observed count
- decimation is skipped at observed count ≤200
- all blocks stay inside `[0,287]`
- missing-value fill cannot influence valid outputs
- Monte Carlo frequencies match expected random-order behavior
- branch-view policies are covered by golden tests

---

# 17. Training recipe

## 17.1 Paper-minimal headline profile

### Published

- epochs: 120
- global batch size: 128
- base learning rate: `1e-4`
- weight decay: `1e-2`
- Gaussian-bandwidth learning rate: `1e-3`
- authors trained on one NVIDIA H100

### Reference inferred choices

- optimizer: AdamW
- betas: `(0.9,0.999)`
- epsilon: `1e-8`
- constant learning rate
- no warmup
- no gradient clipping unless instability is observed
- bandwidth `rho` weight decay: zero
- other parameters receive `1e-2` weight decay
- SmoothL1 beta: 1.0
- mixed precision appropriate to GPU

Optimizer groups:

```python
optimizer = AdamW([
    {
        "params": ordinary_trainable_parameters,
        "lr": 1e-4,
        "weight_decay": 1e-2,
    },
    {
        "params": [model.online_encoder.gaussian.rho],
        "lr": 1e-3,
        "weight_decay": 0.0,
    },
], betas=(0.9, 0.999), eps=1e-8)
```

Target parameters are excluded from the optimizer.

## 17.2 Modern-stable profile

Keep separate from headline results:

- AdamW betas `(0.9,0.95)`
- five-epoch linear warmup
- cosine base-LR decay
- gradient-norm clipping at 1.0
- BF16 where supported
- no change to global batch 128

Use this if it improves stability, but label it a modernized optimization recipe.

## 17.3 Seed policy

Recommended release seeds:

```text
17, 29, 43, 71, 101
```

Use hash/counter-derived RNG streams for:

- window sampling
- DataLoader workers
- augmentation order and parameters
- patch masking
- folds and few-shot subsets

Record Python, NumPy, PyTorch CPU, all CUDA, sampler, augmentation, and patch-mask states in checkpoints.

## 17.4 Epoch definition

Freeze the sampled strict training-window manifest before the headline run. One epoch is one full pass over it. Do not silently resample a new set of source windows per epoch. A dynamic-window experiment can be added later under a different configuration.

## 17.5 Checkpoint schedule

Retain epochs:

```text
0, 1, 5, 10, 20, 40, 60, 80, 100, 120
```

Also maintain an atomic latest-resume checkpoint.

Every checkpoint includes:

- online encoder
- EMA target encoder
- predictor and transition heads
- optimizer/scheduler/GradScaler
- epoch/global step
- all RNG states
- sampler state
- resolved config
- Git SHA and dirty flag
- package lock hash
- source, canonical, window, and split manifest hashes
- learned sigma
- aggregate diagnostics

Release Safetensors normally contain only the frozen online encoder, with a separate pretraining checkpoint if rights and repository size permit.

## 17.6 Validation during pretraining

Create a subject-disjoint diagnostic holdout stratified by dataset where possible. Since the paper does not specify this holdout, it cannot be used to manufacture a hidden “best seed.”

Track:

- all losses and per-dataset losses
- sigma behavior
- representation health
- mask/density distributions
- short fixed linear-probe diagnostics at predeclared epochs
- nearest-neighbor embedding inspection with privacy-safe aggregate output

---

# 18. Hardware profiles

## 18.1 RTX 3090 first

The 24 GB 3090 is sufficient for the published global batch of 128. The model has only 24 Transformer tokens and approximately 0.72M trainable parameters. Data loading and Python overhead are more likely bottlenecks than VRAM.

Reference settings:

- FP16 autocast
- GradScaler enabled
- micro/global batch 128 initially
- no accumulation unless profiling proves necessary
- four to eight DataLoader workers, tuned
- pinned memory
- deterministic eager mode first
- `torch.compile` off for initial parity
- canonical windows memory-mapped or Arrow-backed
- TF32 disabled in the conservative run; optional documented throughput run

Bring-up sequence:

1. CPU unit/golden tests.
2. Overfit 32 windows and confirm losses fall.
3. One-epoch smoke run on a 1% subject subset.
4. Five epochs on the complete strict corpus.
5. Review loss, sigma, collapse diagnostics, and throughput.
6. Complete one 120-epoch seed.
7. Freeze pipeline and run four more seeds.

Do not promise a wall-clock time before the exact overlapping-window manifest is known. Provide a profiling command that runs 500–1,000 optimizer steps and extrapolates with explicit limitations.

## 18.2 RTX 5090 later

- use BF16 autocast
- retain global batch 128 for comparability
- use speed for multi-seed, assumption-resolution, ablation, and public-plus runs
- enable `torch.compile` only after eager/compiled forward, gradient, and checkpoint parity tests
- do not use FP8 for headline reproduction
- publish throughput separately without changing scientific config

## 18.3 H100 optional

An H100 is not required for memory or basic feasibility. Use it only when rapid turnaround across many full seeds/sweeps matters, the public-plus corpus grows substantially, or a final hardware-matching run is useful. Use BF16, not FP8, for fidelity.

## 18.4 Multi-GPU

Not needed for strict reproduction. If used later:

- DistributedDataParallel
- preserve global batch 128
- deterministic sampler shards
- well-defined synchronized EMA semantics
- short parity comparison with single GPU

## 18.5 Performance benchmark command

Implement:

```bash
uv run opencgm benchmark-train \
  --config configs/train/paper_minimal_3090.yaml \
  --steps 1000 \
  --warmup-steps 100 \
  --report reports/throughput/3090.json
```

Report examples/second, windows/second, GPU utilization, peak VRAM, CPU load, data wait, step distribution, and estimated full-run duration based on the frozen manifest.


---

# 19. Frozen representation and downstream evaluation

## 19.1 Encoder output

After pretraining, discard the EMA target, predictor, and transition heads. The frozen online encoder produces:

```python
@dataclass
class EncoderOutput:
    contextual_tokens: Tensor   # [B,24,128]
    daily_embedding: Tensor     # [B,128]
    state_tokens: Tensor        # [B,24,64]
    event_tokens: Tensor        # [B,24,64]
    state_signal: Tensor        # [B,288]
    event_signal: Tensor        # [B,288]
    patch_density: Tensor       # [B,24]
    sigma: Tensor               # scalar
    quality: dict[str, Tensor]
```

The paper's daily representation is the unweighted global mean of all 24 contextual patch representations:

```text
z_day = (1/24) Σᵢ zᵢ
```

Do not density-weight the headline embedding. Add weighted pooling separately.

## 19.2 Embedding cache provenance

Every cached embedding records:

- model weight hash
- resolved model config hash
- preprocessing hash
- source window ID
- code Git SHA
- output dtype
- ONNX/PyTorch backend

Invalidate the cache on any mismatch.

## 19.3 Paper label reconstruction

These thresholds reproduce the paper's evaluation labels and are research definitions, not standalone diagnostic criteria.

### CGMacros

- diabetes risk: normoglycemia / prediabetes / T2D, three classes
- insulin resistance: `HOMA-IR = fasting insulin µU/mL × fasting glucose mg/dL / 405 > 2.9`
- obesity: `BMI ≥ 30`
- hyperlipidemia: total cholesterol ≥240, LDL ≥160, or triglycerides ≥200 mg/dL
- pair Dexcom and Libre from the same person in the same split

### Hall

- diabetes risk: prediabetes or diabetes positive versus normoglycemia
- glucotype: severe positive versus low/moderate non-severe
- insulin resistance: SSPG >120, or HOMA-IR >2.9 when SSPG unavailable
- hyperlipidemia: same cholesterol/LDL/triglyceride thresholds

### Stanford

- insulin resistance: source SSPG-derived classes
- beta-cell dysfunction: dysfunction versus normal using median disposition index definition from source processing
- diabetes risk: HbA1c ≥5.7% positive

### ShanghaiT2DM

- hypoglycemia: source clinical yes/no
- insulin resistance: convert fasting insulin pmol/L to µU/mL by dividing by 6.945; HOMA-IR >2.9
- hyperlipidemia: convert cholesterol and LDL mmol/L ×38.67; triglycerides mmol/L ×88.57; apply thresholds 240/160/200 mg/dL
- 65 labeled sessions from 58 biological participants; run paper entry-level and person-disjoint sensitivity protocols

## 19.4 Subject-disjoint linear probing — `PAPER_EXACT`

- frozen encoder
- non-overlapping 24-hour windows
- scikit-learn logistic regression
- L2 regularization
- `lbfgs`
- maximum 1,000 iterations
- fixed seed
- five-fold subject-grouped cross-validation
- ten repeated fold assignments
- identical folds for every compared representation
- fit scaling and classifier only on training subjects
- report PR-AUC, ROC-AUC, and Macro-F1

Unpublished choices:

- exact `C`
- class weighting
- feature scaling details
- threshold behavior

Reference pipeline:

```python
Pipeline([
    ("scale", StandardScaler()),
    ("classifier", LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        class_weight=None,
        random_state=seed,
    )),
])
```

Publish sensitivity for no scaling, `class_weight="balanced"`, and a training-only `C` grid. The headline uses predeclared `C=1.0` unless official code clarifies it.

## 19.5 Statistical comparison

To mirror the paper:

- retain all unrounded fold-level scores
- macro-average task/dataset scores per repeat/fold
- form paired differences against baselines
- use two-sided Nadeau–Bengio corrected repeated-cross-validation paired t-tests
- use `n_test/n_train = 1/4`
- calculate 95% confidence intervals
- Holm-adjust across planned comparator × metric tests

Also report raw fold distributions and nonparametric bootstrap sensitivity; avoid relying on p-values alone.

## 19.6 Baselines

Minimum practical baseline set:

1. hand-engineered daily CGM metrics
2. raw flattened 288-vector with mask/density
3. small supervised 1D CNN/Transformer trained per task
4. state-only OpenCGM encoder
5. event-only OpenCGM encoder
6. fused model without temporal loss
7. fused model with interpolation rather than mask-aware pipeline
8. CGM-JEPA reconstruction, if licensing/time permits
9. CGMformer official weights where applicable
10. current general time-series baselines used in GlucoFM-Bench for forecasting, not as substitutes for the representation benchmark

Do not delay first open release for every paper baseline. The must-have comparison is against transparent classical/raw/ablation baselines and the closest obtainable CGM-specific models.

## 19.7 Few-shot adaptation — `PAPER_EXACT`

Frozen embeddings, same repeated five-fold subject groups.

Limited subjects:

```text
K ∈ {1,2,3,4,5} support subjects per class
```

Train on all windows from selected support subjects.

Limited observations:

```text
fractions ∈ {1%,5%,10%,20%,30%,40%,50%}
```

Retain all training subjects but sample that fraction of each subject's windows.

- five support samplings per fold/configuration
- evaluate held-out subjects
- report PR-AUC, ROC-AUC, Macro-F1
- share exact support manifests across models

## 19.8 Cross-dataset transfer — `PAPER_EXACT`

- freeze encoder
- train logistic regression on all labeled source-dataset windows
- evaluate directly on target dataset
- target labels never used for training, validation, threshold selection, or model selection
- harmonize binary diabetes-risk labels across CGMacros, Stanford, Hall
- use source-provided compatible insulin-resistance labels
- publish every direction, not only favorable ones

Add calibration and distribution-shift reports by device/cadence.

## 19.9 Multiday observation — `PAPER_EXACT`

- extract one embedding per valid non-overlapping day
- fixed eligible anchor episode per subject
- `Kmax=7` for Stanford, CGMacros, ShanghaiT2DM
- `Kmax=4` for Hall
- enumerate adjacent K-day subwindows inside the fixed anchor
- aggregate using:

```text
mean pooling
concat(mean pooling, elementwise max pooling)
```

- one representation and prediction per subject/start, avoiding pseudo-replication
- ten repeated five-fold stratified subject-level evaluations
- report paired PR-AUC change relative to K=1

## 19.10 PPGR reconstruction

Use CGMacros original raw Dexcom and Libre exports, not its interpolated CGM columns.

Paper protocol:

- strictly causal 24-hour input window before meal onset
- exclude if another logged meal occurs in `(0,120]` minutes
- 874 meals from 34 participants per sensor after filtering
- meal-start baseline = latest raw observation at or before meal
- maximum baseline age: five minutes Dexcom, 15 minutes Libre
- post-meal targets = nearest raw readings within 2.5 minutes Dexcom or 7.5 minutes Libre
- no target interpolation
- train/evaluate sensors separately at native cadence
- two-hidden-layer MLP predicts 24 glucose changes at five-minute horizons from 5–120 minutes
- five subject-disjoint folds and ten head initializations

Context progression:

1. frozen representation only
2. recent CGM context
3. meal nutrition
4. subject/context features

Endpoints:

- trajectory MAE in mg/dL
- positive incremental AUC MAE in mg/dL·h
- peak-rise MAE in mg/dL
- peak-time MAE in minutes

The paper omits some MLP widths/training choices; place them in the assumption registry. Keep this evaluation after the representation benchmark, not on the critical path to the first pretraining checkpoint.

## 19.11 Fairness and subgroup reporting — current best practice

For public-plus evaluation, report when metadata/sample sizes permit:

- diabetes type/status
- age bands
- sex/gender as recorded
- device family
- native cadence
- dataset/source
- glycemic range
- missingness/density quartile
- recording duration

Show subgroup sample sizes and confidence intervals. Avoid claims from very small groups. FairGlucose's 2026 findings make aggregate-only reporting insufficient for any deployment-oriented release.

---

# 20. Required ablation matrix

Prioritize ablations by claim value.

## Tier 1 — required for first paper/release

1. fused state+event versus raw single stream
2. state only
3. event only
4. no temporal-dynamics loss
5. no augmentations
6. dense interpolation versus physical-mask/no-interpolation
7. fixed sigma=6 versus learned sigma
8. no absolute circadian embedding
9. public strict data versus each leave-one-dataset-out model

## Tier 2 — resolves unpublished choices

1. floor versus nearest binning by dataset
2. overlapping-window sampler Candidate A versus B
3. optimizer AdamW versus Adam
4. constant LR versus warmup+cosine
5. pre-norm versus post-norm
6. dropout 0 versus 0.1
7. ordinary masked convolution versus partial convolution
8. patch-start versus patch-center time
9. scalar versus vector gate
10. shared augmentation view versus online-only versus independent branch views
11. transition hidden widths selected around parameter-count parity
12. normalization variants

## Tier 3 — useful extensions

1. public-plus scaling curve
2. native multi-day encoder
3. causal/streaming encoder
4. device embeddings
5. population/status embeddings
6. CGM + insulin/meals/activity
7. synthetic-data contribution
8. uncertainty/calibration heads

Predeclare the matrix and avoid choosing only the best-looking runs after inspection.

---

# 21. Testing and quality gates

## 21.1 Data tests

- schema and unit validation
- monotonic timestamps after duplicate resolution
- no cross-session merges
- gap boundary exactness
- source row accounting
- grid collision averaging
- floor/nearest golden cases
- no inserted value marked as a real observation
- 15-minute sources remain sparse on five-minute grid
- paper record/hour reconciliation report
- source/version/checksum drift detection

## 21.2 Leakage tests

- no canonical subject across train/test
- no biological person across leakage-safe train/test
- CGMacros Dexcom/Libre pairs remain together
- duplicate fingerprints across source datasets
- no overlapping temporal evidence in downstream non-overlapping windows
- scalers/label thresholds fitted or applied without test leakage
- target dataset unused in cross-dataset model selection

## 21.3 Model tests

- shape/dtype/device coverage
- causal Gaussian impulse test
- fill-value invariance
- empty/sparse patch finite outputs
- mask-before-filter no-leakage test
- online/target equality at initialization
- no target gradients
- EMA update equation and end momentum
- loss manual-reference equivalence
- temporal targets use pre-Transformer tokens
- transition weights match equation
- deterministic augmentations and masking
- parameter-count report

## 21.4 Training tests

- 32-window overfit
- restart produces identical next step
- seed repeatability on deterministic profile
- no NaN/Inf under FP16/BF16
- sigma gradients and bounds
- representation-health thresholds
- multi-worker sampler determinism
- DDP parity if introduced

## 21.5 Export tests

Golden batch comparisons among:

- eager PyTorch FP32
- mixed-precision PyTorch
- Safetensors reload
- ONNX Runtime CPU
- ONNX Runtime GPU where available
- optional quantized ONNX

Acceptance:

- daily-embedding cosine similarity >0.99999 for FP32 ONNX
- max/mean absolute errors within declared tolerance
- identical mask/time semantics
- deterministic output for fixed input
- no unsupported operator fallbacks in intended runtime

---

# 22. Local application integration

## 22.1 Deployment artifact

Export only the frozen online encoder:

```text
aligned values [1,288] float32
physical mask [1,288] bool/int
circadian start index scalar/int
→ contextual tokens [1,24,128]
→ daily embedding [1,128]
→ state/event tokens and signals
→ quality metadata
```

Recommended package:

```text
packages/local-app-opencgm/
├── model/
│   ├── encoder.onnx
│   ├── encoder.safetensors
│   ├── model_config.json
│   ├── preprocessing_contract.json
│   ├── model_card.md
│   └── checksums.json
├── src/
│   ├── index.ts
│   ├── schema.ts
│   ├── rolling-window.ts
│   ├── preprocess.ts
│   └── outputs.ts
└── rust/
    ├── mod.rs
    └── ort_encoder.rs
```

## 22.2 Ingestion contract

Local-app, xDrip+ and direct-CGM adapters should emit canonical events:

```ts
export interface CgmReading {
  sourceId: string;
  deviceFamily?: "dexcom" | "libre" | "ipro" | "other";
  sensorId?: string;
  timestampUtc?: string;
  localDateTime: string;
  utcOffsetMinutes?: number;
  glucoseMgDl: number;
  quality?: string;
  isBackfilled?: boolean;
}
```

The bridge owns:

- stable local time handling
- duplicates and backfill replacement
- sensor/session boundaries
- rolling 24-hour ring buffer
- five-minute grid and physical mask
- long-gap segmentation
- model version and provenance

Do not make the UI reconstruct preprocessing independently from the Python reference. Generate a language-neutral preprocessing contract and golden fixtures consumed by both implementations.

## 22.3 Runtime options

Preferred first implementation:

- Rust Tauri command using ONNX Runtime
- local CPU inference
- no cloud requirement
- cache embedding by window/model/preprocessing hash

Alternative:

- sidecar Python service for early validation only
- replace with ONNX/Rust for distribution

## 22.4 Product surfaces

Useful, non-diagnostic UI:

- current 24-hour state trend and event residual
- observation-density heatmap
- missingness/device timeline
- daily embedding history
- similarity to the user's prior days
- cluster/exploration tools
- model-quality warnings
- export of latent features for local research

Avoid showing phenotype probabilities as clinical conclusions. The paper itself describes GlucoFM as a research prototype, not a diagnostic or treatment system.

## 22.5 Streaming caveat

The model consumes an independently constructed 24-hour window. It is not inherently a causal real-time foundation model simply because the Gaussian filter is causal; the Transformer can attend across all 24 patches in the provided window. For live use, the current window contains only past/current observations, but embeddings at older positions may be recomputed as more context arrives.

A true streaming/past-only extension should be a separate model/config.

---

# 23. Connection to the later PPG-to-glucose project

This project models **CGM from CGM**. The later project estimates glucose from PPG and other wearable signals.

Use the frozen CGM encoder as a teacher on paired data:

```text
actual synchronized CGM
→ OpenCGM-StateEvent teacher
→ state/event/context latents
                    │
                    │ representation alignment/distillation
                    ▼
PPG + ACC + temperature + HR/HRV + context
→ wearable student
→ current glucose, direction, trajectory, uncertainty, teacher-latent estimates
```

Advantages:

- CGM-only data trains a much stronger glucose representation than paired PPG-CGM data alone.
- The student can learn slow state and acute-event targets in addition to raw mg/dL.
- Paired datasets such as BIG IDEAs can bridge the branches.

Guardrail: distillation improves the learning objective but cannot manufacture glucose information absent from wearable signals. Validate subject-independent physiological signal, confounds, device/site dependence, and uncertainty rigorously.

For real-time student training, teacher targets must be generated causally from past/current CGM only. Do not use future-day information that would leak into deployment.

---

# 24. End-to-end implementation plan by PR

## PR 0 — Repository, locks, provenance, and CI

Deliver:

- repository layout
- Apache-2.0/NOTICE/CITATION
- `uv` lock and optional Bun workspace
- config loader with evidence-status validation
- provenance/hash utilities
- CPU CI and optional GPU workflow
- lint/type/test commands

Acceptance:

- clean checkout executes unit-test skeleton
- resolved config is deterministic and hashable
- run metadata captures Git/environment state

## PR 1 — Source manifests and rights registry

Deliver:

- source/license manifest schemas
- entries for all strict and downstream sources
- downloader interfaces with manual-access instructions
- SHA-256 file inventory
- raw data never committed

Acceptance:

- manifests validate
- source version drift is detected
- every planned dataset has a rights status and owner for unresolved questions

## PR 2 — Dataset adapters and canonical schema

Deliver:

- BIG IDEAs, Shanghai, Stanford, Colas adapters
- downstream CGMacros/Hall adapters
- unit conversion and timestamp modules
- canonical Parquet output
- source reconciliation reports

Acceptance:

- row-level traceability to source files
- no silent data loss
- unit/time/sentinel decisions documented
- expected subject/session counts approached and discrepancies explained

## PR 3 — Segmentation, grid, and window manifests

Deliver:

- >1-hour segment boundaries
- floor/nearest mapper
- collision averaging
- 288-grid values/mask
- overlapping sampler candidates A/B
- non-overlapping downstream windows
- fixed split/window manifests

Acceptance:

- golden cases pass
- no interpolation
- 15-minute sources visibly retain sparse physical masks
- complete public-hour/window report

## PR 4 — Augmentations and deterministic RNG

Deliver:

- four CGM augmentations
- random-order probability decay
- branch-view policies
- hash-derived RNG streams
- visualization/reporting utility

Acceptance:

- Monte Carlo tests
- seed determinism across workers
- no canonical-data mutation

## PR 5 — Gaussian and statistics foundation

Deliver:

- observed instance normalization
- causal Gaussian filter
- state/event residuals
- patch densities/statistics/RoC/differences
- NumPy reference implementation

Acceptance:

- causality and fill invariance
- gradient/sigma tests
- sparse/empty support finite

## PR 6 — Stream embedders and Transformer encoder

Deliver:

- state/event embedders
- fusion/time/position gate
- mask token
- three-layer online and target encoders
- component parameter report
- configurable ambiguity options

Acceptance:

- exact disclosed shapes
- target copied correctly
- output schema and golden fixtures
- parameter count plausibly close to paper

## PR 7 — Predictor, transitions, EMA, and objectives

Deliver:

- one-layer predictor
- transition heads
- MCR and TD losses
- cosine EMA schedule
- collapse diagnostics

Acceptance:

- manual loss tests
- no target gradients
- transition targets pre-Transformer
- EMA resume parity

## PR 8 — Training system and 3090 bring-up

Deliver:

- paper-minimal and modern-stable profiles
- mixed precision
- atomic checkpoint/resume
- local metrics
- benchmark command
- 32-window overfit and subset smoke runs

Acceptance:

- one 3090 epoch completes cleanly
- restart reproduces next step
- five-epoch full-corpus health review passes

## PR 9 — Full strict pretraining

Deliver:

- first complete 120-epoch seed
- frozen online release weights
- loss/representation/sigma reports
- four additional seeds after pipeline freeze

Acceptance:

- all runs use identical manifests/config family
- no collapse
- variance across seeds reported
- no cherry-picked “best only” release

## PR 10 — Downstream benchmark

Deliver:

- embedding cache
- all paper label builders
- repeated grouped linear probes
- statistics
- few-shot, cross-dataset, multiday
- optional PPGR

Acceptance:

- split manifests identical across methods
- all 14 dataset-task evaluations reproduced where source labels allow
- fold-level results published

## PR 11 — Required ablations

Deliver Tier 1 matrix and prioritized Tier 2 ambiguity experiments.

Acceptance:

- predeclared experiment table
- failed/negative results retained
- claims tied to confidence intervals and effect consistency

## PR 12 — Public-plus data expansion

Deliver:

- source-level rights/dedup reviews
- newly approved adapters
- scaling-curve runs
- source leave-one-out analysis
- subgroup reports

Acceptance:

- no duplicate participants across sources
- checkpoint's rights lane is internally consistent
- strict model/results remain untouched

## PR 13 — Export and Local application integration

Deliver:

- Safetensors and ONNX
- parity suite
- canonical TypeScript/Rust schemas
- rolling-window module
- Rust ORT command
- example Tauri panel

Acceptance:

- local inference on fixture and real owned CGM stream
- no cloud dependency
- model/preprocessing/version provenance visible

## PR 14 — Open release

Deliver:

- code and model cards
- rights/provenance manifest
- reproducibility command sequence
- release artifacts/checksums
- technical report/preprint or detailed repository report
- X launch materials

Acceptance:

- fresh machine reproduction test
- independent reviewer follows setup
- limitations prominently stated
- all claims trace to published experiments

---

# 25. Command-line contract

Target commands:

```bash
# Environment
uv sync --extra cuda

# Inspect rights and acquisition status
uv run opencgm data status

# Validate downloaded source files
uv run opencgm data verify --manifest manifests/sources/strict_public.yaml

# Canonicalize
uv run opencgm data canonicalize --dataset big_ideas
uv run opencgm data canonicalize --dataset shanghai_t2dm
uv run opencgm data canonicalize --dataset stanford
uv run opencgm data canonicalize --dataset colas

# Reconcile source counts/hours
uv run opencgm data reconcile --config configs/data/strict_public.yaml

# Build fixed windows
uv run opencgm windows build --config configs/data/strict_public.yaml --seed 17

# Inspect alternative overlap semantics
uv run opencgm windows compare-samplers --config configs/data/strict_public.yaml

# Model/parameter report
uv run opencgm model inspect --config configs/model/glucofm_v2_reconstruction_a.yaml

# Smoke train
uv run opencgm train --config configs/train/paper_minimal_3090.yaml --max-steps 1000

# Full run
uv run opencgm train --config configs/train/paper_minimal_3090.yaml --seed 17

# Frozen embeddings and evaluation
uv run opencgm embed --checkpoint runs/<id>/encoder.safetensors
uv run opencgm eval linear-probe --config configs/eval/paper_linear_probe.yaml
uv run opencgm eval few-shot --config configs/eval/few_shot.yaml
uv run opencgm eval cross-dataset --config configs/eval/cross_dataset.yaml
uv run opencgm eval multiday --config configs/eval/multiday.yaml

# Export
uv run opencgm export onnx --checkpoint runs/<id>/encoder.safetensors
uv run opencgm export verify --artifact exports/<id>/encoder.onnx
```

Every command writes a machine-readable run record and refuses to proceed on manifest/hash mismatches unless an explicit override is recorded.

---

# 26. Risk register

| Risk | Severity | Mitigation |
|---|---:|---|
| Wear-CGM unavailable | high | strict public claim; public-plus; request collaboration; first-party cohort |
| Overlap sampler ambiguity | high | two implementations, reconciliation, author question, publish manifest |
| Dataset identity/session leakage | high | person graph, immutable splits, entry/person sensitivity runs |
| Weight-release rights unclear | high | separate lanes, source review, no raw redistribution, counsel before release |
| Official code later differs | medium | modular assumption registry and parity branch |
| Small public corpus limits results | high | five seeds, scaling curves, public-plus, avoid overclaiming |
| 15-minute source handling differs | medium | no-interpolation strict path, floor/nearest audit |
| Representation collapse | medium | EMA/loss tests, latent diagnostics, overfit/smoke gates |
| Downstream threshold mismatch | medium | label unit tests and source-derived manifests |
| Device/population bias | high | cross-dataset, device, diabetes-status, subgroup reports |
| Training appears easy and hides data bugs | high | reconciliation and leakage gates before full run |
| Tauri preprocessing diverges | high | generated contract and cross-language golden fixtures |
| Public post overstates reproduction | medium | approved claim language and limitations in launch materials |

---

# 27. Questions to send the authors

Ask concise, implementation-specific questions:

1. What optimizer, betas, epsilon, schedule, and warmup were used?
2. How exactly is the 20–80% per-segment pretraining-window coverage ratio defined?
3. What fixed seed and start-time resolution were used for window sampling?
4. Which datasets use floor versus nearest grid assignment, and what tie rule is used?
5. What is the exact mask-aware normalization equation, epsilon, and branch behavior?
6. What are the convolution kernel sizes, layers, activations, and pooling rules in each stream embedder?
7. How is the time/position gate parameterized and initialized?
8. What are predictor and transition-head MLP details?
9. Are physical augmentations shared between online and target views?
10. What dropout, norm ordering, initialization, and SmoothL1 beta are used?
11. Does EMA copy the entire preprocessing/tokenizer/encoder path including Gaussian rho?
12. Can Wear-CGM be accessed through a DUA, collaboration, federated run, or author-run validation?
13. Will the promised release include weights, exact source-subset manifests, and dataset binning rules?

Store answers verbatim in an `author_clarifications.md` provenance file and release a new config rather than rewriting old runs.

---

# 28. Open release and X launch

## 28.1 Minimum credible release package

- complete code
- environment lock
- strict source/download guide
- immutable preprocessing and split manifests
- at least five strict seeds
- fold-level downstream results
- Tier 1 ablations
- public-data encoder weights
- ONNX artifact
- Local demo application
- detailed model/data cards
- rights/provenance matrix
- known differences from paper
- one-command smoke reproduction

## 28.2 Claims worth making

Strong and honest:

- independent reconstruction completed within days/weeks of the paper
- trained locally on an RTX 3090/5090 rather than requiring H100-scale memory
- faithful state/event, mask-aware, circadian, JEPA, and dynamics implementation
- public-data checkpoint and reproducible manifests
- local ONNX integration with direct CGM ingestion in a local application
- explicit negative and ambiguity results

Avoid:

- “exact Google checkpoint reproduction”
- “medical glucose intelligence” without validation/regulatory context
- claiming public+ gains as strict reproduction gains
- implying PPG-based non-invasive glucose estimation has already been solved

## 28.3 Suggested technical launch sequence

1. Repository and architecture explainer.
2. Data/provenance thread explaining why Wear-CGM prevents exact reproduction.
3. First strict training curves and learned sigma/state/event visualizations.
4. Five-seed downstream/ablation results.
5. Open weights and ONNX release.
6. A local demo reading the user's own CGM data.
7. Follow-up roadmap: larger public+ model, multi-day/streaming, then paired PPG teacher/student work.

A compelling post will come from transparent evidence, not hype: a tiny paper-defined model, real public-data engineering, independent reproduction, open weights, and an immediate local application.

---

# 29. Final handoff checklist

Before implementation begins:

- [ ] Create repository and assign package/project name.
- [ ] Freeze this blueprint and reference config in version control.
- [ ] Confirm strict source URLs/versions and acquisition steps.
- [ ] Complete source-level rights spreadsheet.
- [ ] Create author-question issue/email.
- [ ] Implement source manifest schemas before parsing data.
- [ ] Implement canonical identity graph before splitting.
- [ ] Choose and document initial overlap sampler candidate.
- [ ] Build mask-before-filter golden tests before model training.
- [ ] Implement parameter-count and representation-health reports.
- [ ] Run 3090 smoke gates before any full run.
- [ ] Freeze strict manifests/config before five-seed training.
- [ ] Keep public-plus experiments in separate run namespace.
- [ ] Do not publish weights until the checkpoint rights lane is reviewed.
- [ ] Export Tauri fixtures from the Python reference, not a rewritten algorithm.
- [ ] Publish limitations beside results, not buried in an appendix.

---

# 30. Source index

Primary method:

- Google Research blog: https://research.google/blog/glucofm-foundation-model-for-continuous-glucose-monitoring/
- GlucoFM v2 abstract: https://arxiv.org/abs/2605.30865v2
- GlucoFM v2 PDF: https://arxiv.org/pdf/2605.30865v2

Strict public sources:

- BIG IDEAs PhysioNet: https://physionet.org/content/big-ideas-glycemic-wearable/1.1.2/
- Shanghai Figshare collection: https://figshare.com/collections/Diabetes_Datasets-ShanghaiT1DM_and_ShanghaiT2DM/5137813
- Stanford CGM database: https://cgmdb.stanford.edu/data/
- Associated Stanford code: https://github.com/aametwally/Metabolic_Subphenotype_Predictor
- Colas PLOS article/data: https://doi.org/10.1371/journal.pone.0225817

Downstream/public discovery:

- CGMacros PhysioNet: https://physionet.org/content/cgmacros/1.0.0/
- Hall PLOS Biology: https://doi.org/10.1371/journal.pbio.2005143
- GlucoFM-Bench: https://arxiv.org/abs/2606.06881
- GlucoFM-Bench dataset: https://huggingface.co/datasets/glucofmbench/GlucoFM-Bench
- MetaboNet: https://arxiv.org/abs/2601.11505
- MetaboNet site: https://metabo-net.org/
- FairGlucose: https://arxiv.org/abs/2608.18296
- PhysioCGM data: https://doi.org/10.6084/m9.figshare.28136294
- PhysioCGM paper: https://doi.org/10.1038/s41597-025-06090-6
- Paired smartwatch PPG + CGM: https://arxiv.org/abs/2606.15927
- Paired smartwatch dataset: https://zenodo.org/records/20577959
- Multichannel PPG + capillary glucose: https://zenodo.org/records/21978226

Interoperability reference:

- DIAX diabetes exchange format: https://arxiv.org/abs/2604.11944

Verify every URL, source version, and license again on the day of acquisition/release.

