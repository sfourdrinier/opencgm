default:
    @just --list

# Project state, derived from disk. Trust this over any prose.
status:
    @uv run opencgm data status

verify dataset="":
    @uv run opencgm data verify {{ if dataset != "" { "--dataset " + dataset } else { "" } }}

check config:
    @uv run opencgm config-check {{config}}

test:
    @uv run pytest -q

# The gate must cover everything that ships, not just src/ and tests/. `scripts/`, `api/`
# and `app.py` were outside it, which is how 52 lint errors accumulated there unnoticed.
lint:
    @uv run ruff check .
    @uv run ruff format --check src tests

fmt:
    @uv run ruff format src tests

# Manifest every source's raw bytes under data/raw/ into manifests/sources/.
manifest-sources:
    @uv run python scripts/generate_source_manifests.py

# Build the strict window cache (353,127 windows) from the frozen manifest.
# Required before any pretraining on a cold checkout. ~25 min.
build-windows *ARGS:
    @uv run python scripts/build_windows.py {{ARGS}}

# Strict pretraining sweep — 5 seeds × 120 epochs.
# 3090: ~30 h serial. 5090: ~10 h with 3-way concurrency.
# Resumable from ckpt_last.pt; safe across crashes (resume_gate.py verifies).
pretrain-sweep:
    @bash scripts/pretrain_sweep.sh

# CGM-JEPA comparator pretraining sweep (5 seeds).
# Source-verified port of github.com/cruiseresearchgroup/CGM-JEPA master @ 2026-05-11.
pretrain-cgmjepa:
    @bash scripts/pretrain_cgmjepa_sweep.sh

# Downstream eval — the headline (5 seeds × 5-fold × 10 repeats).
eval-headline:
    @bash scripts/evaluate_seed.sh 17 120 full
    @bash scripts/evaluate_seed.sh 29 120 full
    @bash scripts/evaluate_seed.sh 43 120 full
    @bash scripts/evaluate_seed.sh 71 120 full
    @bash scripts/evaluate_seed.sh 101 120 full

# CGM-JEPA comparator eval.
eval-cgmjepa:
    @bash scripts/evaluate_cgmjepa_seed.sh 17
    @bash scripts/evaluate_cgmjepa_seed.sh 29
    @bash scripts/evaluate_cgmjepa_seed.sh 43
    @bash scripts/evaluate_cgmjepa_seed.sh 71
    @bash scripts/evaluate_cgmjepa_seed.sh 101

# Paired 5-seed CGM-JEPA vs GlucoFM head-to-head.
head-to-head:
    @uv run python scripts/aggregate_cgmjepa_vs_glucofm.py

# Few-shot evaluation (k = 1, 5, 10, 20 per class).
eval-few-shot:
    @uv run python scripts/evaluate_few_shot.py \
        --checkpoint runs_5090/rawstats120/ckpt_ep040.pt

# Cross-dataset transfer (38 cohort-pair transfers).
eval-cross-dataset:
    @uv run python scripts/evaluate_cross_dataset.py \
        --checkpoint runs_5090/rawstats120/ckpt_ep040.pt

# Multiday pooling (n = 1, 2, 3, 5, 7 days).
eval-multiday:
    @uv run python scripts/evaluate_multiday.py \
        --checkpoint runs_5090/rawstats120/ckpt_ep040.pt

# PPGR head (paper §4.3).
eval-ppgr:
    @uv run python scripts/ppgr.py \
        --checkpoint runs_5090/rawstats120/ckpt_ep040.pt

# Permutation test for headline numbers.
permutation-test:
    @uv run python scripts/permutation_test.py

# Tier-1 ablation sweep — 3 seeds × 10 ablations × 40 epochs.
# 5090 only; ~8 h with 5-way concurrency.
ablation-sweep:
    @bash scripts/sweep_tier1_ablations.sh

# §23 PPG teacher-student pilot (D023, A7).
# Frozen strict-ep40 encoder as teacher; small 116K-param student encoder.
# 5-fold × 5-seed; per-subject bootstrap CI.
ppg-pilot:
    @uv run python scripts/ppg_teacher_student.py \
        --teacher-ckpt runs_5090/rawstats120/ckpt_ep040.pt

ppg-pilot-eval:
    @uv run python scripts/evaluate_ppg_pilot.py

# Aggregate the Tier-1 ablation matrix into findings/tier1_ablations.md.
ablation-aggregate:
    @uv run python scripts/aggregate_tier1_ablations.py

# --- Fast tier ---------------------------------------------------------------------------
# One seed, 40 epochs: the smoke test from REPRODUCE.md. Not a publishable number --
# the headline is 5 seeds × 120 epochs. ~1.5 h on a 3090.
fast-pretrain:
    @SEEDS=17 EPOCHS=40 TAG=fast bash scripts/pretrain_sweep.sh

# Evaluate the fast-tier checkpoint with the full probe protocol. ~20 min.
fast-eval:
    @bash scripts/evaluate_seed.sh 17 40 full

# Full eval suite — runs everything that produces a number we report.
# Outputs land in reports/eval/.
eval-all:
    @bash scripts/run_eval_all.sh

# Fit probe heads + reliability + coverage bands.
fit-heads:
    @uv run python scripts/fit_heads.py \
        --checkpoint runs_5090/rawstats120/ckpt_ep040.pt

# Stage the encoder + rights-filtered heads into web/public/models/.
# Run after any re-export; the published bundle is what the site and API serve.
publish-web-assets:
    @uv run python scripts/publish_web_assets.py

# Regenerate the worked example on /example from a CGM export.
# The output is one JSON file; swapping or removing the example is a one-file change.
example-analysis CSV:
    @uv run python scripts/build_example_analysis.py --csv {{CSV}} --days 5

# Assemble the Hugging Face repo under dist/hf. Does not upload; prints the command.
stage-hf: publish-web-assets
    @uv run python scripts/stage_huggingface.py

# The Next.js site and the public HTTP API, on every interface.
web: publish-web-assets
    @cd web && npm install --no-audit --no-fund && npx next dev -H 0.0.0.0 -p 3000

# Launch the Streamlit real-data demo.
app:
    @uv run streamlit run app.py --server.port 8765

# Everything that must pass before a PR is considered done.
gate: lint test status

# Full reproduction tier — the publishable-number tier.
# Pretrain + eval + ablations + heads + manifests.
all: status build-windows pretrain-sweep pretrain-cgmjepa eval-all ablation-sweep fit-heads manifest-sources
