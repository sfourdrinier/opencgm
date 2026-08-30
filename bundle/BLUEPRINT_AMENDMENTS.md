# Blueprint amendments

Corrections and extensions to `glucofm_public_reproduction_blueprint.md` v1.0, found during
acquisition. The blueprint is frozen; this file is the delta. Each amendment carries an
evidence label in the blueprint's own scheme (§2).

---

## A1 — Shanghai Figshare collection ID is wrong · `SOURCE_VERIFIED`

Blueprint §30 gives `figshare.com/collections/.../5137813`. That collection resolves to an
unrelated polymer-chemistry article.

**Correct source:** Figshare article **20444397 v3**, DOI `10.6084/m9.figshare.20444397`,
CC BY 4.0, `data.zip` 3.42 MB, 133 files covering both ShanghaiT1DM and ShanghaiT2DM.

---

## A2 — Stanford's official download page serves the wrong file · `SOURCE_VERIFIED`

Blueprint §6.5 lists `data_cgm.csv` among the core files to acquire from
`cgmdb.stanford.edu/data/`. That file is **not** the CGM time series. Its schema is:

```
"glucose","subject","foods","mitigator","food","rep","mins_since_start"
79.6583355058983,"XB68","Beans","","Beans",1,-25
```

Meal-challenge response curves, indexed by minutes-relative-to-meal, with spline-smoothed
float glucose. This is exactly the "cubic-spline-smoothed glucose" that §6.5 rule 4 forbids in
the strict pipeline. Ingesting it would have silently violated the blueprint's own rule.

**Correct source:** `data/filtered_cgm_03222026.csv` in the companion repo
`aametwally/Metabolic_Subphenotype_Predictor`, pinned at commit
`94d647944c001dfa34492fd593b6e4804c1f45c4` (do not track `main`). Integer glucose, true 5-min
cadence, real second-offsets:

```
"timestamp","glucose_value","subject"
"2017-11-29 15:01:51","114","S01"
```

Reconciles at **56 subjects** = the paper's 19 pretrain + 37 downstream, exactly.

Keep `data_meta.csv` from the cgmdb page — it carries HbA1c, SSPG, DI, the downstream labels
§19.3 needs.

---

## A3 — Hall's CGM series is a gzipped file with no extension · `SOURCE_VERIFIED`

PLOS Biology supporting file **s010** is gzip-compressed despite being served without a `.gz`
extension. Slots s001–s009 are TIFF figures. Decompressed schema:

```
DisplayTime	GlucoseValue	subjectId	InternalTime
```

Also note `DisplayTime` 2014 vs `InternalTime` 2016 on the same row — the privacy date-shift
§8.5 anticipates. Preserve local hour; set `circadian_confidence: preserved_but_shifted`.

---

## A4 — Restrictive licenses mean *evaluation-only*, not *excluded* · `SOURCE_VERIFIED`

**This is the substantive amendment. It changes the lane model in §6.10.**

The blueprint's release-lane table treats restrictively-licensed sources as largely binary:
usable or not. That conflates two very different acts, and the conflation costs us data we are
entitled to use.

### The distinction

Creative Commons 4.0 ND licenses grant, in §2(a)(1)(B), the right to:

> "produce and reproduce, **but not Share**, Adapted Material for NonCommercial purposes only."

So *NoDerivatives* restricts **distribution** of a derivative — not its **creation**. And
evaluation does not even create adapted material: running a frozen encoder over a dataset to
compute PR-AUC yields measurements about the data, which are facts, not an adaptation of it.

The same logic applies to *ShareAlike*. SA's copyleft attaches to distributed adapted
material. A checkpoint you never publish triggers nothing; metrics you publish are facts.

### Consequence — add Lane E to §6.10

| Lane | Meaning |
|---|---|
| A strict public | the four GlucoFM pretraining cohorts |
| B public-plus | permissive sources, after rights + dedup audit |
| C research-restricted | DUA/credentialed, per-agreement |
| **E evaluation-only (new)** | **NC / ND / SA sources. Never enters any distributed checkpoint. Freely usable as held-out validation.** |

Lane E is not a consolation prize. **External validation is where license-encumbered data is
most valuable**, precisely because the encoder never trained on it. A cohort we are forbidden
to train on is a cohort with zero leakage risk by construction — the cleanest possible
generalization test, and free of the "did it memorize?" objection that dogs held-out splits
carved from the training corpus.

### Sources reclassified into Lane E

| Source | License | Was | Now |
|---|---|---|---|
| UCHTT1DM | CC BY-NC-ND 4.0 | excluded entirely | eval-only, **acquired** |
| CGMacros | CC BY-NC-SA 4.0 | "evaluation initially" | eval-only, confirmed |
| GlucoFM-Bench | CC BY-NC-SA 4.0 | reference | eval-only + source index |
| D1NAMO | CC BY-SA 4.0 | public-plus | eval-only unless SA is cleared |

### Hard boundaries that do not move

- No Lane E source may contribute to a **distributed** checkpoint.
- NC additionally bars **commercial** use. If a downstream application ever ships commercially, nothing
  Lane E may be embedded in it — including a privately-trained model.
- Raw redistribution stays verbatim-only, attributed, NonCommercial.
- Record Lane E membership in each checkpoint's source-contribution manifest (§7) even when
  the contribution is evaluation-only, so the provenance record stays complete.

Engineering judgment, not legal advice. Have counsel confirm before any weight release.

---

## A5 — UCHTT1DM as a Wear-CGM population proxy · `PROPOSED_EXTENSION`

Beyond the license question, this cohort is worth using for a scientific reason.

Wear-CGM is 192 **healthy, non-diabetic** adults and 69% of the paper's pretraining hours. Our
strict public corpus is overwhelmingly diabetic and prediabetic — that population skew is the
single largest uncontrolled difference between our reconstruction and the paper's, and §26
lists it as a high-severity risk with no concrete mitigation.

UCHTT1DM contains **11 healthy subjects** alongside 9 with T1DM. It is far too small to
correct the skew by training (2,605 h ≈ 7.7% of the strict corpus), but as a *validation*
cohort it directly probes the question the skew raises: does an encoder pretrained mostly on
dysglycemic data still produce sane representations of normal physiology?

It also shifts two axes the strict corpus holds nearly fixed:

- **Device:** Guardian Sensor 3, absent from the strict corpus (Dexcom / Libre / iPro only).
  A free cross-device generalization test for §19.11 fairness reporting.
- **Geography:** Chilean cohort, against a corpus that is US and Chinese.

Proposed use: an external-validation report alongside the Tier 1 ablations — healthy-vs-T1DM
separability from frozen embeddings, and representation-health diagnostics (§15.6) computed on
a population the model never saw. Report it separately from the paper-comparable numbers.

Secondary value: Fitbit HR at 5 s paired with 5-min CGM makes it a small Lane D
teacher/student bridge (§23), subject to the same no-distribution constraint.

---

## A6 — CGMacros ships no raw export; its published CGM columns are interpolated

**Blueprint reference:** §19.10, "Use CGMacros original raw Dexcom and Libre exports, not its
interpolated CGM columns."

**Finding:** PhysioNet CGMacros 1.0.0 contains no raw export. Each subject folder holds meal
photographs and a single `CGMacros-0NN.csv` on a **1-minute** grid, in which *both* CGM columns
have been linearly interpolated from their native cadence:

* `Dexcom GL` advances by a constant 3.6 mg/dL per row between observations — 18 mg/dL spread
  across five 1-minute steps.
* `Libre GL` advances by a constant 0.1333 mg/dL per row — 2.0 mg/dL across fifteen steps.

The instruction in §19.10 cannot be followed as written against this release.

**Why it cannot be ignored:** the physical observation mask is authoritative throughout this
project, and every density weight in §15 depends on it. Reading the published column would report
a Dexcom day as 288 observations when 57 were measured and 231 manufactured. Density weighting
would then be uniform across sources that genuinely differ in cadence, which is precisely the
distortion the mask-aware design exists to prevent. This is the same class of defect as A2
(Stanford's spline-smoothed `data_cgm.csv`).

**Resolution:** recover the native observations rather than resample or exclude. Linear
interpolation is piecewise linear, so the original observations are exactly the points where the
slope changes, plus the endpoints. `readers.recover_observations` implements this, and
`tests/golden/test_cgmacros.py` verifies it three ways on real subjects:

1. re-interpolating the recovered points reproduces the published file to under 1e-9 — which can
   only hold if the recovered points are the originals;
2. recovered timestamps land on the sensor's native grid for >99% of gaps, with median gap
   exactly 5 minutes (Dexcom) and 15 minutes (Libre);
3. the recovered count is of order 100k readings against 660k file rows.

Yield: 45 subjects on both sensors, 105,771 Dexcom and 41,510 Libre observations. The subject
count matches §6.8's "45 participants".

**Known limitation, stated rather than hidden:** three collinear observations are
indistinguishable from two interpolated ones, so a run of exactly constant slope loses its middle
point. This under-counts observations, never over-counts, so it can only make density look lower
than the truth — the safe direction for a mask-aware model. Measured effect is under 1% of gaps.

**Rights:** unchanged. CGMacros remains Lane E, evaluation-only, pending Q4.

---

## A7 — Lane D becomes a teacher-student pilot, not a probe-on-encoder · `PROPOSED_EXTENSION`

**Blueprint reference:** §6.9 (PPG+CGM paired dataset), §23 (PPG-bridge / teacher-student use
of the frozen CGM encoder).

**Finding.** The PPG+CGM paired Zenodo record (DOI 10.5281/zenodo.20577959) contains 5
participants (P001..P005), 3327 BVP JSON segments of ~30 min × 64 Hz raw photoplethysmography
from an Empatica Embrace Plus smartwatch, paired with per-subject `P00x_glucose.csv` containing
CGM in mmol/L at ~15-min cadence. Total archive size 3.3 GB.

The registry already classifies it as `lane: D, role: ppg_bridge` (with the capillary Stuba
record parked as `ppg_auxiliary` next to it). The blueprint itself, at §6.9, says the dataset
is "too small to materially change CGM-only pretraining, but it is timely for teacher/student
prototyping and external validation", and §23 defines the canonical use as "Use the frozen CGM
encoder as a teacher on paired data." The lane system pre-decided the question.

**Why not pretraining integration (Lane A).** Adding 5 subjects to the strict pretraining
corpus would subject-disjoint-collapsed 5-fold to 1 test subject per fold — that is not a
held-out evaluation, it is a per-subject anecdote. 5 subjects × many days is exactly the
recipe for subject-identity encoding. Pretraining on this data would either not move the
headline (it is too small) or risk moving it backward (it is too narrow). Pretending to
"bump the corpus percentage to 63%" by adding 5 Empatica-wearing volunteers to a 250-subject
strict corpus is not a publishable claim — it is a quietly-inflated denominator.

**The pilot.**

*Teacher.* Frozen strict-pretrain **epoch 40** checkpoint (`runs_5090/rawstats120/
ckpt_ep040.pt`, seed 17), mean-pooled 256-dim `opencgm_mean` head, eval mode, gradients off.
Rationale: D024 — ep40 is the measured peak at 30.9% corpus; ep120 is mildly worse by ~1 sd.

*Student input.* Each BVP JSON segment is split into 5-minute patches matching the CGM grid
(5-min × 5 subjects' CGM windows). The per-subject glucose CSV supplies the aligned CGM
window at the same timestamps. The BVP-segment → CGM-window alignment is the natural unit
that lets the student produce one token per CGM timestep without re-encoding 64 Hz raw signal
into a CGM-rate architecture.

*Student encoder.* A small 1D conv stack on 64 Hz BVP that downsamples to ~1 Hz features
(HR-rate summary), then per-patch mean-pool → one student token per CGM timestep.
**No attempt to inject 64 Hz raw samples into the existing CGM encoder** — the rate mismatch
is 3,840× and would force architectural surgery that breaks the teacher-student framing.

*Two heads, both required.*

1. **Teacher-latent head.** 256-dim projection, loss = 0.5·MSE + 0.5·(1−cosine) on the
   teacher's tokens at the same timestamps, masked by the CGM observation mask (mask
   before loss, same rule as §10.1). This is the headline pilot number.
2. **Direct glucose head.** Per-token mmol/L regression with causal Gaussian NLL (§12),
   masked. Sanity check, not the headline.

*Evaluation.* Mean-pool student tokens across the window → cosine alignment to the
teacher-window-embedding (also mean-pooled) and RMSE/MAE on glucose. **Never interpolate
CGM** when computing alignment or loss — the observation mask is authoritative on both sides.

*Validation protocol.*

- Subject-disjoint 5-fold × 5 student seeds (student-side seeds 1003, 1019, 1043, 1071, 1103
  — distinct from the CGM pretraining seeds).
- Two baselines alongside the student: (a) identity (student latent = population-mean teacher
  latent; the "no transfer" floor); (b) per-subject ridge regression from BVP-summary
  features to CGM at each timestamp (the natural non-deep baseline).
- Per-fold numbers and a subject-level bootstrap CI. 5 subjects is small; a fold-mean CI
  would be misleading.
- Auxiliary probe (not headline): does the student latent, fed into the existing 14-task
  probe infrastructure, transfer to a small downstream task?

**What this amendment does NOT do.**

- Does not modify the strict encoder. The teacher stays frozen. KR5 ("improvements live in
  separate checkpoints and separate tables") applies without contortion.
- Does not change the strict corpus percentage (still 30.9%). The headline 0.679 ± 0.011
  is unaffected. That is the outcome.
- Does not promote `ppg_cgm_paired` out of `weight_release: permissive_pending_review`.
  The pilot checkpoint is internal/research-only until the published paper's authors' terms
  on paired wearable data are confirmed.
- Does not include the capillary Stuba dataset. Lane D registration for it remains
  `ppg_auxiliary`, not `ppg_bridge`. Out of scope for GlucoFM reproduction and for this
  teacher-student pilot.

**Cross-references.** Companion decisions: D023 (the choice itself), D024 (ep40 vs ep120).
Pilot code lives under `src/opencgm_stateevent/ppg/` and `scripts/ppg_teacher_student.py`;
outputs in `reports/eval/ppg_pilot/`; write-up in `findings/ppg_pilot.md`.
