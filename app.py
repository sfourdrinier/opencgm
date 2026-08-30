"""Streamlit demo: GlucoFM-on-real-data.

    streamlit run app.py

What this is
------------
A real-data front-end for the trained GlucoFM encoder. Accept a 24h (or longer) CGM CSV from a
Dexcom Clarity export, a Libre View export, or any plain `timestamp,glucose_mg_dl` file, and
produce:

  * coverage-corrected CGM metrics (TIR, GMI, CV, dawn rise, overnight mean, longest stable run)
  * research-only phenotype probabilities from the heads above the signal floor
  * explicit reliability and population-size for every phenotype
  * the measured reliability of every score, next to it

What this is not
----------------
A diagnostic tool. A medical device. A replacement for a clinician. The probe heads are linear
logistic regressions fitted on cohorts of 29-100 subjects; their probabilities are population
associations, not predictions about a single person.

Standing rule (mirrors the research code)
-----------------------------------------
Gaps stay gaps. The model was trained with an explicit observation mask and never sees
interpolated values. The CSV reader drops missing rows and marks missing five-minute positions
as unobserved. Filling them would feed the model a kind of input it has never seen and quietly
degrade every number below.

Sample data
-----------
A "Load sample data" button uses `data/canonical/windows/strict_seed17.values.npy` window 0 —
24h of real CGM from one of the public cohorts, with the mask preserved. That makes the demo
work without anyone uploading anything sensitive.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from opencgm_stateevent.infer import Analyser  # noqa: E402

# --- Cached loaders ---------------------------------------------------------

@st.cache_resource(show_spinner="Loading model + heads...")
def load_analyser(checkpoint: str, heads_path: str) -> Analyser:
    return Analyser.load(Path(checkpoint), heads=Path(heads_path), device="cpu")


@st.cache_data(show_spinner=False)
def load_sample_window() -> pd.DataFrame:
    """24h sample window from a real public cohort, mask preserved."""
    values = np.load(REPO / "data/canonical/windows/strict_seed17.values.npy",
                     mmap_mode="r")
    mask = np.load(REPO / "data/canonical/windows/strict_seed17.mask.npy",
                   mmap_mode="r")
    # window 0 = first 24h of first subject
    v = np.asarray(values[0], dtype=np.float32)
    m = np.asarray(mask[0], dtype=bool)
    grid = pd.date_range("2024-06-01 00:00", periods=288, freq="5min")
    df = pd.DataFrame({"timestamp": grid, "glucose_mg_dl": v})
    df = df[m].reset_index(drop=True)
    return df


# --- Streamlit UI -----------------------------------------------------------

st.set_page_config(
    page_title="GlucoFM — real-data demo",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("GlucoFM real-data demo")
st.caption(
    "Independent public-data reconstruction of GlucoFM (arXiv:2605.30865v2). "
    "Apache-2.0. Not a medical device."
)

with st.sidebar:
    st.subheader("Model")
    checkpoint = st.text_input(
        "Checkpoint",
        value="runs_5090/rawstats120/ckpt_ep040.pt",
        help="Path to a trained GlucoFM encoder checkpoint. Heads below were fitted against "
             "this exact checkpoint (the SHA-256 is checked at load time).",
    )
    heads_path = st.text_input(
        "Heads",
        value="artifacts/heads.pkl",
        help="Pickled fitted probe heads + their cross-validated reliability.",
    )
    st.divider()
    st.subheader("What this is")
    st.markdown(
        """
- Trained on **353,127 public CGM windows** (33,736 h; 30.9% of the paper's pretraining hours).
- **Five seeds**, subject-disjoint 5-fold x 10 repeats evaluation.
- CGM-JEPA comparator (the paper's own baseline) trained faithfully from authors' source —
  the paper's +4.11 PR-AUC margin **direction is preserved**, magnitude smaller at our
  corpus size (see `findings/head_to_head.md`).
        """
    )
    st.divider()
    st.markdown("**Caveats**")
    st.markdown(
        """
- Probe heads are linear regressions on cohorts of 29-100 subjects. Probabilities are
  population associations, **not** individual predictions.
- A head whose cross-validated ROC-AUC is below the **0.55 signal floor** is hidden.
- Coverage < 30% on a window = the model refuses; the underlying CGM metrics are still shown.
        """
    )

# Input section
st.header("1. Load CGM data")
col1, col2 = st.columns([2, 1])

with col1:
    uploaded = st.file_uploader(
        "Upload a CGM CSV",
        type=["csv"],
        help="Dexcom Clarity, Libre View, or any file with a timestamp + glucose_mg_dl column. "
             "Headers are sniffed; you can override below if it guesses wrong.",
    )
    time_col = st.text_input("Timestamp column (auto-detected if blank)", value="")
    glucose_col = st.text_input("Glucose column (auto-detected if blank)", value="")

with col2:
    use_sample = st.button("Load sample window (real public CGM, 24h)")
    st.caption(
        "Sample = one 24h window from the strict-pretraining corpus, with the mask preserved. "
        "No data leaves your machine."
    )

if uploaded is None and not use_sample:
    st.info("Upload a CSV or click **Load sample window** to see the demo.")
    st.stop()

if use_sample:
    df = load_sample_window()
    src = "sample"
else:
    raw = uploaded.read()
    uploaded.seek(0)
    # sniff / read_csv needs a path with file contents
    tmp = REPO / "artifacts" / "_upload.csv"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(raw)
    try:
        # we have to call read_csv but want column overrides from the UI
        # so re-implement the small bit that matters
        frame = pd.read_csv(tmp)
        from scripts.analyse import GLUCOSE_HINTS, TIME_HINTS, sniff
        tc = time_col or sniff(list(frame.columns), TIME_HINTS, "time")
        gc = glucose_col or sniff(list(frame.columns), GLUCOSE_HINTS, "glucose")
        st.caption(f"Reading time from `{tc}`, glucose from `{gc}`.")
        times = pd.to_datetime(frame[tc], errors="coerce", format="mixed")
        values = pd.to_numeric(frame[gc], errors="coerce")
        ok = times.notna() & values.notna()
        df = pd.DataFrame({
            "timestamp": times[ok].dt.tz_localize(None),
            "glucose_mg_dl": values[ok].astype(float),
        }).sort_values("timestamp").reset_index(drop=True)
        if not len(df):
            st.error("No usable rows after dropping NaNs.")
            st.stop()
        src = uploaded.name
    finally:
        tmp.unlink(missing_ok=True)

st.success(f"{len(df):,} readings loaded from {src}.")

if uploaded is not None:
    with st.expander("First 10 rows"):
        st.dataframe(df.head(10), use_container_width=True)

span = df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]
st.write(
    f"Span: **{span.days} days, {span.seconds // 3600} hours** "
    f"({df['glucose_mg_dl'].min():.0f} – {df['glucose_mg_dl'].max():.0f} mg/dL)"  # noqa: RUF001 - en dash is deliberate in display copy
)

# Run inference
st.header("2. Analysis")
analyser = load_analyser(checkpoint, heads_path)
readings = list(
        zip(df["timestamp"].dt.to_pydatetime(), df["glucose_mg_dl"].astype(float), strict=True)
    )

with st.spinner("Encoding + scoring..."):
    reports = (
        analyser.analyse_stream(readings, days=min(max(span.days, 1), 14))
        if span.days >= 1 else [analyser.analyse_day(readings)]
    )

if not reports:
    st.error("No window had enough readings to analyse.")
    st.stop()

# Pick which day(s) to show
if len(reports) == 1:
    day_idx = 0
else:
    day_labels = [f"{r.start:%a %d %b}" for r in reports]
    day_idx = st.selectbox("Window", range(len(reports)),
                            format_func=lambda i: day_labels[i])
report = reports[day_idx]

# CGM metrics
st.subheader("CGM metrics")
m = report.metrics
metric_cols = st.columns(4)
metric_cols[0].metric("Coverage", f"{m.coverage:.0%}",
                       help=f"{m.n_observed}/288 five-minute positions observed")
metric_cols[1].metric("Mean glucose", f"{m.mean_glucose:.0f} mg/dL",
                       help=f"GMI {m.glucose_management_indicator:.1f}%")
metric_cols[2].metric("Time in range", f"{m.time_in_range:.0%}",
                       help="70-180 mg/dL")
metric_cols[3].metric("Variability (CV)",
                       f"{m.coefficient_of_variation:.0%}",
                       delta="stable" if m.variability_is_stable else "above threshold",
                       delta_color="normal" if m.variability_is_stable else "inverse")

band_cols = st.columns(4)
band_cols[0].metric("Below 70", f"{m.time_below_70:.0%}",
                     help=f"Below 54: {m.time_below_54:.0%}")
band_cols[1].metric("Above 180", f"{m.time_above_180:.0%}",
                     help=f"Above 250: {m.time_above_250:.0%}")
band_cols[2].metric("Range", f"{m.min_glucose:.0f} – {m.max_glucose:.0f} mg/dL")  # noqa: RUF001 - en dash is deliberate in display copy
band_cols[3].metric("Overnight mean", f"{m.overnight_mean:.0f} mg/dL",
                     help=f"Dawn rise {m.dawn_rise:+.1f} mg/dL")

st.caption(f"Longest steady run: **{m.longest_stable_hours:.1f} h**.")

# Phenotype scores
st.subheader("Research phenotype signals")
if not report.phenotypes:
    st.warning("No fitted heads loaded.")
else:
    usable = [p for p in report.phenotypes if p.has_signal and p.applicable]
    st.caption(
        f"{len(usable)} of {len(report.phenotypes)} heads above the 0.55 signal floor "
        "and applicable to this window's coverage."
    )
    rows = []
    for p in usable:
        rows.append({
            "Task": p.task,
            "Probability": f"{p.probability:.0%}",
            "Reliability (cv ROC-AUC)": f"{p.reliability:.2f}",
            "±": f"{p.reliability_sd:.2f}",
            "Subjects learned from": p.n_subjects_learned_from,
            "Phrasing": p.population_phrasing,
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.warning(
            "Coverage on this window fell outside every head's fitted band. "
            "If the upload was a short clip (<24h), pick the most-recent 24h window."
        )

# Warnings
if report.warnings:
    st.subheader("Warnings")
    for w in report.warnings:
        st.warning(w)

# Export
st.subheader("Export")
buf = io.StringIO()
buf.write(report.to_json())
st.download_button(
    "Download JSON report",
    data=buf.getvalue(),
    file_name=f"glucofm_report_{report.start:%Y%m%d_%H%M}.json",
    mime="application/json",
)

# Footer
st.divider()
st.markdown(
    """
**Underlying paper:** GlucoFM (Google Research / UNSW Sydney), arXiv:2605.30865v2.
**Reproduction:** independent public-data reconstruction, Apache-2.0.
**Not a medical device.** All phenotype probabilities are population associations derived from
linear probes fitted on cohorts of 29-109 subjects; reliability varies across heads. See
`reports/user_facing_capabilities.md` for the full reliability table.
"""
)
