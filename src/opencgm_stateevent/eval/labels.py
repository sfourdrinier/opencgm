"""Downstream label reconstruction. Blueprint §19.3 — `PAPER_EXACT` thresholds.

Fourteen dataset-task pairs: CGMacros x4, Hall x4, Stanford x3, ShanghaiT2DM x3.

Every threshold here is a research definition taken from the blueprint, not a diagnostic
criterion, and every one is traceable to a §19.3 line. Where §19.3 names a threshold we use it
verbatim; where it names a unit conversion we apply that conversion and nothing else. Two places
required a reading beyond the text, and both are marked `INFERRED_RECONSTRUCTION` in
`TASKS` and recorded in `DECISIONS.md`.

Labels are per subject, not per window. A subject's label attaches to all of that subject's
windows, and subject-grouped splitting (§9.5) is what keeps that from leaking.
"""

from __future__ import annotations

import io
import sqlite3
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")

# §19.3, shared across CGMacros, Hall and ShanghaiT2DM
CHOLESTEROL_MG_DL = 240.0
LDL_MG_DL = 160.0
TRIGLYCERIDE_MG_DL = 200.0
HOMA_IR_THRESHOLD = 2.9
SSPG_THRESHOLD = 120.0  # §19.3 Hall
BMI_OBESE = 30.0  # §19.3 CGMacros
HBA1C_RISK_PERCENT = 5.7  # §19.3 Stanford

#: Stanford visit whose venous draw accompanies the CGM recording. See D017.
CGM_MATCHED_EXPERIMENT = "venous_with_matching_cgm_and_with_planned_athome_cgm"

# Unit conversions, §19.3 ShanghaiT2DM
PMOL_L_PER_MICRO_U_ML = 6.945
MMOL_L_TO_MG_DL_CHOLESTEROL = 38.67
MMOL_L_TO_MG_DL_TRIGLYCERIDE = 88.57


@dataclass(frozen=True)
class Task:
    """One dataset-task pair."""

    dataset: str
    name: str
    evidence: str
    n_classes: int = 2

    @property
    def key(self) -> str:
        return f"{self.dataset}:{self.name}"


TASKS: tuple[Task, ...] = (
    # CGMacros — §19.3
    Task("cgmacros", "diabetes_risk", "INFERRED_RECONSTRUCTION", n_classes=3),
    Task("cgmacros", "insulin_resistance", "PAPER_EXACT"),
    Task("cgmacros", "obesity", "PAPER_EXACT"),
    Task("cgmacros", "hyperlipidemia", "PAPER_EXACT"),
    # Hall — §19.3
    Task("hall", "diabetes_risk", "PAPER_EXACT"),
    Task("hall", "glucotype", "PAPER_EXACT"),
    Task("hall", "insulin_resistance", "PAPER_EXACT"),
    Task("hall", "hyperlipidemia", "PAPER_EXACT"),
    # Stanford — §19.3
    Task("stanford", "insulin_resistance", "SOURCE_VERIFIED"),
    Task("stanford", "beta_cell_dysfunction", "SOURCE_VERIFIED"),
    Task("stanford", "diabetes_risk", "PAPER_EXACT"),
    # ShanghaiT2DM — §19.3
    Task("shanghai_t2dm", "hypoglycemia", "SOURCE_VERIFIED"),
    Task("shanghai_t2dm", "insulin_resistance", "PAPER_EXACT"),
    Task("shanghai_t2dm", "hyperlipidemia", "PAPER_EXACT"),
)


def homa_ir(insulin_micro_u_ml: pd.Series, glucose_mg_dl: pd.Series) -> pd.Series:
    """`HOMA-IR = insulin uU/mL * glucose mg/dL / 405`. §19.3, CGMacros."""
    return insulin_micro_u_ml * glucose_mg_dl / 405.0


def hyperlipidemia_from_mg_dl(
    cholesterol: pd.Series, ldl: pd.Series, triglycerides: pd.Series
) -> pd.Series:
    """Positive if any of the three exceeds its threshold. §19.3.

    Kept as a single function because all three datasets share the thresholds, and a copy that
    drifts in one of them would be invisible.
    """
    hit = (
        (cholesterol >= CHOLESTEROL_MG_DL)
        | (ldl >= LDL_MG_DL)
        | (triglycerides >= TRIGLYCERIDE_MG_DL)
    )
    # A subject with no lipid panel at all is unlabelled, not negative.
    known = cholesterol.notna() | ldl.notna() | triglycerides.notna()
    return hit.where(known)


# --- CGMacros ------------------------------------------------------------------------------


def cgmacros_bio() -> pd.DataFrame:
    z = zipfile.ZipFile(RAW / "cgmacros/1.0.0/CGMacros_dateshifted365.zip")
    name = next(n for n in z.namelist() if n.endswith("bio.csv"))
    bio = pd.read_csv(io.BytesIO(z.read(name)))
    bio.columns = [c.strip() for c in bio.columns]
    return bio


def cgmacros_labels() -> pd.DataFrame:
    """Four tasks. §19.3.

    `diabetes_risk` is three-class normoglycemia/prediabetes/T2D. The blueprint names the classes
    but not their cut points, so the standard HbA1c bands are used: <5.7, 5.7-6.4, >=6.5.
    INFERRED_RECONSTRUCTION, recorded as D014.
    """
    bio = cgmacros_bio()
    # The bio table numbers subjects 1..45; the per-subject folders are `CGMacros-001`, and the
    # window subject id comes from the folder. Zero-pad so the join key is the source's own
    # identifier on both sides.
    subject = bio["subject"].astype(int).map(lambda n: f"{n:03d}")
    a1c = pd.to_numeric(bio["A1c PDL (Lab)"], errors="coerce")
    fasting_glucose = pd.to_numeric(bio["Fasting GLU - PDL (Lab)"], errors="coerce")
    insulin = pd.to_numeric(bio["Insulin"], errors="coerce")

    risk = pd.Series(np.select([a1c >= 6.5, a1c >= 5.7], [2, 1], default=0), dtype="float")
    risk = risk.where(a1c.notna())

    return pd.DataFrame({
        "subject": subject,
        "diabetes_risk": risk,
        "insulin_resistance": (homa_ir(insulin, fasting_glucose) > HOMA_IR_THRESHOLD).where(
            insulin.notna() & fasting_glucose.notna()
        ),
        "obesity": (pd.to_numeric(bio["BMI"], errors="coerce") >= BMI_OBESE).where(
            pd.to_numeric(bio["BMI"], errors="coerce").notna()
        ),
        "hyperlipidemia": hyperlipidemia_from_mg_dl(
            pd.to_numeric(bio["Cholesterol"], errors="coerce"),
            pd.to_numeric(bio["LDL (Cal)"], errors="coerce"),
            pd.to_numeric(bio["Triglycerides"], errors="coerce"),
        ),
    })


# --- Hall ----------------------------------------------------------------------------------


def hall_clinical() -> pd.DataFrame:
    with sqlite3.connect(RAW / "hall/plos_pbio_2005143/S5_database.sqlite") as c:
        return pd.read_sql("select * from clinical", c)


def hall_labels() -> pd.DataFrame:
    """Four tasks. §19.3.

    Insulin resistance prefers SSPG and falls back to HOMA-IR only where SSPG is missing, exactly
    as §19.3 words it. In this release SSPG is present for all 57 subjects, so the fallback never
    fires — it is kept because the rule is the specification, not the data.
    """
    d = hall_clinical()
    sspg = pd.to_numeric(d["SSPG"], errors="coerce")
    insulin = pd.to_numeric(d["insulin"], errors="coerce")
    fbg = pd.to_numeric(d["FBG"], errors="coerce")

    by_sspg = sspg > SSPG_THRESHOLD
    by_homa = homa_ir(insulin, fbg) > HOMA_IR_THRESHOLD
    insulin_resistance = by_sspg.where(sspg.notna(), by_homa.where(insulin.notna() & fbg.notna()))

    return pd.DataFrame({
        "subject": d["userID"].astype(str),
        # "prediabetes or diabetes positive versus normoglycemia"
        "diabetes_risk": d["diagnosis"].map(
            {"non-diabetic": False, "pre-diabetic": True, "diabetic": True}
        ),
        # "severe positive versus low/moderate non-severe"
        "glucotype": d["glucotype"].map(
            {"low": False, "moderate": False, "severe": True}
        ),
        "insulin_resistance": insulin_resistance,
        "hyperlipidemia": hyperlipidemia_from_mg_dl(
            pd.to_numeric(d["Tchol"], errors="coerce"),
            pd.to_numeric(d["LDL"], errors="coerce"),
            pd.to_numeric(d["Trg"], errors="coerce"),
        ),
    })


# --- Stanford ------------------------------------------------------------------------------


def _stanford_dir() -> Path:
    return next((RAW / "stanford").glob("github_*"))


def stanford_labels() -> pd.DataFrame:
    """Three tasks. §19.3.

    Insulin resistance and beta-cell dysfunction use the source's own derived classes
    (`sspg_2_classes`, `di_2_classes_median`) rather than re-deriving them from raw values, which
    is what §19.3 asks for: "source SSPG-derived classes" and "median disposition index
    definition from source processing".
    """
    root = _stanford_dir()
    chars = pd.read_csv(root / "filtered_study_participants_characteristics.csv")
    tests = pd.read_csv(root / "filtered_metabolic_tests.csv")
    # Both tables carry one row per experiment type, so some subjects appear twice, and in
    # `characteristics` the repeated rows hold genuinely different lab values from different
    # visits. Prefer the draw taken alongside the CGM recording — it is the one contemporaneous
    # with the windows being labelled. D017.
    chars = (
        chars.assign(_cgm=chars["ExperimentType"].eq(CGM_MATCHED_EXPERIMENT))
        .sort_values("_cgm", ascending=False)
        .groupby("SubjectID", as_index=False)
        .first()
        .drop(columns="_cgm")
    )
    # `filtered_metabolic_tests` carries one row per experiment type, so 12 subjects appear twice.
    # Merging as-is duplicates every one of their windows and silently doubles their weight in a
    # probe. The duplicate rows agree wherever both are non-null — verified across all four
    # columns, zero conflicts — so collapsing on the first non-null value is lossless.
    tests = tests.groupby("SubjectID", as_index=False).first()
    d = chars.merge(tests, on="SubjectID", how="outer")

    def positive_class(series: pd.Series) -> pd.Series:
        """Map a two-level source class to a boolean, whatever the level names are."""
        text = series.astype("string").str.strip().str.upper()
        return text.map({"IR": True, "IS": False, "DYSFUNCTION": True, "NORMAL": False})

    hba1c = pd.to_numeric(d["HbA1c"], errors="coerce")
    return pd.DataFrame({
        "subject": d["SubjectID"].astype(str),
        "insulin_resistance": positive_class(d["sspg_2_classes"]),
        "beta_cell_dysfunction": positive_class(d["di_2_classes_median"]),
        "diabetes_risk": (hba1c >= HBA1C_RISK_PERCENT).where(hba1c.notna()),
    })


# --- ShanghaiT2DM --------------------------------------------------------------------------


def shanghai_summary() -> pd.DataFrame:
    z = zipfile.ZipFile(RAW / "shanghai/figshare_20444397_v3/data.zip")
    name = next(n for n in z.namelist() if "T2DM_Summary" in n)
    return pd.read_excel(io.BytesIO(z.read(name)))


def shanghai_labels() -> pd.DataFrame:
    """Three tasks. §19.3.

    Both unit conversions are the blueprint's: insulin pmol/L / 6.945 -> µU/mL, and lipids
    mmol/L x 38.67 (cholesterol, LDL) or x 88.57 (triglycerides) -> mg/dL. Fasting plasma glucose
    is already mg/dL in this source, so it is not converted.
    """
    d = shanghai_summary()
    d.columns = [c.strip() for c in d.columns]
    subject_col = next(c for c in d.columns if "Patient Number" in c or c.strip() == "ID")

    insulin = pd.to_numeric(d["Fasting Insulin (pmol/L)"], errors="coerce")
    fpg = pd.to_numeric(d["Fasting Plasma Glucose (mg/dl)"], errors="coerce")
    ir = homa_ir(insulin / PMOL_L_PER_MICRO_U_ML, fpg) > HOMA_IR_THRESHOLD

    hypo = d["Hypoglycemia (yes/no)"].astype("string").str.strip().str.lower()

    # ShanghaiT2DM labels an *entry* -- one patient's one visit -- not a person. §19.3 counts
    # "65 labeled sessions from 58 biological participants". The entry is the join key; the
    # patient is the grouping key, so repeat visits from one person cannot straddle a fold.
    entry = d[subject_col].astype(str)
    person = entry.str.split("_").str[0]
    visit = entry.str.split("_").str[1]

    return pd.DataFrame({
        "subject": person,
        "entry": person + "/visit_" + visit,
        "hypoglycemia": hypo.map({"yes": True, "no": False}),
        "insulin_resistance": ir.where(insulin.notna() & fpg.notna()),
        "hyperlipidemia": hyperlipidemia_from_mg_dl(
            pd.to_numeric(d["Total Cholesterol (mmol/L)"], errors="coerce")
            * MMOL_L_TO_MG_DL_CHOLESTEROL,
            pd.to_numeric(d["Low-Density Lipoprotein Cholesterol (mmol/L)"], errors="coerce")
            * MMOL_L_TO_MG_DL_CHOLESTEROL,
            pd.to_numeric(d["Triglyceride (mmol/L)"], errors="coerce")
            * MMOL_L_TO_MG_DL_TRIGLYCERIDE,
        ),
    })


def _ensure_entry(frame: pd.DataFrame) -> pd.DataFrame:
    """Give every table an `entry` column. Only ShanghaiT2DM labels below the person."""
    if "entry" not in frame.columns:
        frame = frame.copy()
        frame.insert(1, "entry", frame["subject"])
    return frame


BUILDERS = {
    "cgmacros": cgmacros_labels,
    "hall": hall_labels,
    "stanford": stanford_labels,
    "shanghai_t2dm": shanghai_labels,
}


def build_all() -> dict[str, pd.DataFrame]:
    return {name: _ensure_entry(fn()) for name, fn in BUILDERS.items()}


def coverage() -> pd.DataFrame:
    """One row per dataset-task: how many subjects carry a usable label, and the class balance.

    Printed before any probe runs. A task with a handful of positives cannot support a five-fold
    subject-grouped split, and it is better to say so up front than to report a confidence
    interval spanning the whole range.
    """
    rows = []
    frames = build_all()
    for task in TASKS:
        col = frames[task.dataset][task.name]
        labelled = col.notna()
        values = col[labelled]
        if task.n_classes > 2:
            balance = ", ".join(
                f"{int(k)}:{int(v)}" for k, v in sorted(values.value_counts().items())
            )
        else:
            positives = int(values.astype(bool).sum())
            balance = f"{positives}+/{len(values) - positives}-"
        rows.append({
            "task": task.key,
            "evidence": task.evidence,
            "subjects": len(col),
            "labelled": int(labelled.sum()),
            "balance": balance,
            "minority": (
                int(values.value_counts().min()) if len(values) else 0
            ),
        })
    return pd.DataFrame(rows)
