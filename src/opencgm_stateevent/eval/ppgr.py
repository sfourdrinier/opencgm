"""Postprandial glycaemic response. Paper §4.3, blueprint §19.10.

Given a meal and the 24 hours of CGM that preceded it, predict what the person's glucose does for
the next two hours. This is the one downstream task that is directly useful to a person wearing a
sensor and that requires no health claim: the prediction is checkable by the user within two hours,
and being wrong is embarrassing rather than harmful.

The paper's setup, reproduced here:

    "For each event, a strictly causal 24-hour pre-meal CGM window is encoded into a frozen
    representation, and the task predicts the subsequent two-hour glucose change relative to the
    observed meal-start glucose. Dexcom and Libre are modeled separately using the same MLP with
    two hidden layers at their native target cadences: 24 five-minute outputs for Dexcom and 8
    fifteen-minute outputs for Libre. The representation is combined cumulatively with one hour of
    pre-meal CGM, meal nutrition, fasting glucose, and BMI and diabetes status."

Two properties carry the validity of the whole exercise and are enforced rather than documented:

**Strict causality.** The encoder window ends at the meal timestamp and never includes it. A single
grid position of leakage would let the model see the meal's own rise and would improve every metric
below while making them meaningless.

**No overlapping meal.** Events with another logged meal inside the two-hour horizon are dropped,
because the target would then be a response to two meals attributed to one. The paper does the
same, and it is most of the difference between 1,706 logged meals and the ~874 per sensor it keeps.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.grid import build_window
from ..data.timestamps import GRID_MINUTES
from .labels import RAW, cgmacros_bio

CGMACROS_ROOT = RAW / "cgmacros"

#: Native output cadence per sensor, minutes, and the number of steps in two hours. §4.3.
HORIZON_MINUTES = 120
SENSORS = {
    "dexcom": {"column": "Dexcom GL", "cadence": 5, "steps": 24},
    "libre": {"column": "Libre GL", "cadence": 15, "steps": 8},
}

#: One hour of pre-meal CGM accompanies the representation, at the sensor's own cadence.
PREMEAL_MINUTES = 60

#: A window this empty cannot support a 24-hour representation. Matches the density of the
#: sparsest cohort we train on, so it excludes broken recordings rather than sparse sensors.
MIN_WINDOW_COVERAGE = 0.25


@dataclass(frozen=True)
class MealEvent:
    subject: str
    at: datetime
    calories: float
    carbs: float
    protein: float
    fat: float
    fiber: float
    amount_consumed: float

    @property
    def nutrition(self) -> np.ndarray:
        """Grams and kcal, scaled by the fraction the participant actually ate."""
        eaten = self.amount_consumed / 100.0 if self.amount_consumed > 0 else 1.0
        return np.array([
            self.calories * eaten, self.carbs * eaten, self.protein * eaten,
            self.fat * eaten, self.fiber * eaten,
        ], dtype=np.float64)


def _subject_frames(root: Path = CGMACROS_ROOT):
    """Yield (subject, frame) for each CGMacros participant, timestamps parsed."""
    archive = root / "1.0.0/CGMacros_dateshifted365.zip"
    with zipfile.ZipFile(archive) as z:
        names = sorted(
            n for n in z.namelist()
            if n.endswith(".csv") and "/CGMacros-" in n and "DataDictionary" not in n
        )
        for name in names:
            subject = name.rsplit("/", 1)[-1].removesuffix(".csv").removeprefix("CGMacros-")
            frame = pd.read_csv(io.BytesIO(z.read(name)))
            frame.columns = [c.strip() for c in frame.columns]
            frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], errors="coerce")
            yield subject, frame.dropna(subset=["Timestamp"])


def read_meals(root: Path = CGMACROS_ROOT) -> list[MealEvent]:
    """Every logged meal, before any filtering."""
    events: list[MealEvent] = []
    for subject, frame in _subject_frames(root):
        events.extend(read_meals_for(frame, subject))
    return events


@dataclass
class PPGRDataset:
    """One row per usable meal event, for one sensor."""

    sensor: str
    subjects: np.ndarray            # [N]
    event_times: np.ndarray         # [N]      meal timestamps, the key for cross-sensor matching
    window_values: np.ndarray       # [N,288]  the causal 24h pre-meal window, mg/dL
    window_mask: np.ndarray         # [N,288]
    circadian: np.ndarray           # [N]
    premeal: np.ndarray             # [N,K]    one hour before the meal, at native cadence
    nutrition: np.ndarray           # [N,5]
    subject_context: np.ndarray     # [N,3]    fasting glucose, BMI, diabetes status
    baseline_glucose: np.ndarray    # [N]      glucose at the meal
    target: np.ndarray              # [N,steps] change from baseline over the next two hours

    def __len__(self) -> int:
        return len(self.subjects)


def build(sensor: str = "dexcom", root: Path = CGMACROS_ROOT) -> PPGRDataset:
    """Assemble the PPGR dataset for one sensor.

    Every exclusion is counted and reported by `scripts/ppgr.py`, because "874 of 1,706 events"
    is only interpretable alongside why the rest went.
    """
    spec = SENSORS[sensor]
    column, cadence, steps = spec["column"], spec["cadence"], spec["steps"]
    horizon = timedelta(minutes=HORIZON_MINUTES)

    bio = cgmacros_bio()
    bio_index = {
        f"{int(row['subject']):03d}": row for _, row in bio.iterrows()
        if pd.notna(row.get("subject"))
    }

    rows: dict[str, list] = {k: [] for k in (
        "subject", "at", "values", "mask", "circadian", "premeal", "nutrition",
        "context", "baseline", "target",
    )}

    for subject, frame in _subject_frames(root):
        series = frame[["Timestamp", column]].dropna()
        if series.empty:
            continue
        # The published file is on a 1-minute grid with the CGM columns linearly interpolated;
        # keep only the positions the sensor actually measured. See D016.
        native = series[series["Timestamp"].dt.minute % cadence == 0]
        times = native["Timestamp"].dt.to_pydatetime()
        values = pd.to_numeric(native[column], errors="coerce").to_numpy(dtype=float)
        keep = np.isfinite(values)
        times, values = np.asarray(times)[keep], values[keep]
        if len(times) < 2:
            continue

        # Derived from the same parser as the events themselves. Filtering the frame separately
        # is how this went wrong once already: `str(NaN)` is `"nan"`, which is not the empty
        # string, so every CGM row counted as a logged meal and every event looked like it had a
        # second meal inside its horizon. The dataset came out empty rather than wrong, which was
        # luck -- a subtler version of the same slip would have silently dropped half the events.
        events = read_meals_for(frame, subject)
        meal_times = sorted(e.at for e in events)
        by_time = pd.Series(values, index=pd.DatetimeIndex(times)).sort_index()

        for event in events:
            # 1. no second meal inside the horizon: the target must be one meal's response.
            if any(event.at < other <= event.at + horizon for other in meal_times):
                continue

            # 2. the target: two hours of change from the meal-start value, at native cadence.
            grid = [event.at + timedelta(minutes=cadence * (k + 1)) for k in range(steps)]
            future = by_time.reindex(pd.DatetimeIndex(grid), method="nearest",
                                     tolerance=pd.Timedelta(minutes=cadence))
            baseline_at = by_time.reindex(pd.DatetimeIndex([event.at]), method="nearest",
                                          tolerance=pd.Timedelta(minutes=cadence))
            if future.isna().any() or baseline_at.isna().any():
                continue
            baseline = float(baseline_at.to_numpy()[0])

            # 3. one hour of pre-meal CGM, strictly before the meal.
            back = [event.at - timedelta(minutes=cadence * (k + 1))
                    for k in range(PREMEAL_MINUTES // cadence)]
            recent = by_time.reindex(pd.DatetimeIndex(back), method="nearest",
                                     tolerance=pd.Timedelta(minutes=cadence))
            if recent.isna().any():
                continue

            # 4. the 24-hour encoder window, ending strictly before the meal.
            start = event.at - timedelta(hours=24)
            inside = (times >= start) & (times < event.at)
            if not inside.any():
                continue
            window = build_window(
                list(times[inside]), list(values[inside]), start,
                dataset_id="cgmacros", canonical_subject_id=subject, session_id=sensor,
            )
            if window.coverage < MIN_WINDOW_COVERAGE:
                continue

            info = bio_index.get(subject)
            if info is None:
                continue
            fasting = pd.to_numeric(info.get("Fasting GLU - PDL (Lab)"), errors="coerce")
            bmi = pd.to_numeric(info.get("BMI"), errors="coerce")
            a1c = pd.to_numeric(info.get("A1c PDL (Lab)"), errors="coerce")
            if not (np.isfinite(fasting) and np.isfinite(bmi) and np.isfinite(a1c)):
                continue

            rows["subject"].append(subject)
            rows["at"].append(event.at)
            rows["values"].append(window.values)
            rows["mask"].append(window.mask)
            rows["circadian"].append(window.circadian_start_index)
            rows["premeal"].append(recent.to_numpy(dtype=float)[::-1])  # oldest first
            rows["nutrition"].append(event.nutrition)
            rows["context"].append([float(fasting), float(bmi), float(a1c >= 6.5)])
            rows["baseline"].append(baseline)
            rows["target"].append(future.to_numpy(dtype=float) - baseline)

    return PPGRDataset(
        sensor=sensor,
        subjects=np.array(rows["subject"]),
        event_times=np.array(rows["at"]),
        window_values=np.array(rows["values"], dtype=np.float32),
        window_mask=np.array(rows["mask"], dtype=bool),
        circadian=np.array(rows["circadian"], dtype=np.int64),
        premeal=np.array(rows["premeal"], dtype=np.float64),
        nutrition=np.array(rows["nutrition"], dtype=np.float64),
        subject_context=np.array(rows["context"], dtype=np.float64),
        baseline_glucose=np.array(rows["baseline"], dtype=np.float64),
        target=np.array(rows["target"], dtype=np.float64),
    )


def read_meals_for(frame: pd.DataFrame, subject: str) -> list[MealEvent]:
    """The meal events of one already-loaded participant frame."""
    meals = frame[frame["Meal Type"].astype(str).str.strip().ne("") & frame["Meal Type"].notna()]
    out = []
    for _, row in meals.iterrows():
        def value(column, default=0.0, row=row):
            v = pd.to_numeric(row.get(column), errors="coerce")
            return float(v) if pd.notna(v) else default

        out.append(MealEvent(
            subject=subject, at=row["Timestamp"].to_pydatetime(),
            calories=value("Calories"), carbs=value("Carbs"), protein=value("Protein"),
            fat=value("Fat"), fiber=value("Fiber"),
            amount_consumed=value("Amount Consumed", 100.0),
        ))
    return out


# --- endpoints ---------------------------------------------------------------------------------
# Four endpoints, computed on the *change* curve. §4.3 reports MAE on each.


def positive_iauc(curve: np.ndarray, cadence_minutes: int) -> np.ndarray:
    """Positive incremental area under the change curve, mg/dL x hours.

    Only the excursion above baseline counts, which is what "positive" means here: a dip below
    baseline is not negative postprandial response, it is a different phenomenon.
    """
    hours = cadence_minutes / 60.0
    return np.clip(curve, 0.0, None).sum(axis=-1) * hours


def peak_rise(curve: np.ndarray) -> np.ndarray:
    return curve.max(axis=-1)


def peak_time(curve: np.ndarray, cadence_minutes: int) -> np.ndarray:
    """Minutes from the meal to the maximum."""
    return (curve.argmax(axis=-1) + 1) * cadence_minutes


def endpoints(curve: np.ndarray, cadence_minutes: int) -> dict[str, np.ndarray]:
    return {
        "iauc": positive_iauc(curve, cadence_minutes),
        "peak_rise": peak_rise(curve),
        "peak_time": peak_time(curve, cadence_minutes),
    }


def score(predicted: np.ndarray, actual: np.ndarray, cadence_minutes: int) -> dict[str, float]:
    """MAE on the trajectory and on each of the three derived endpoints. §4.3."""
    out = {"trajectory_mae": float(np.abs(predicted - actual).mean())}
    a = endpoints(predicted, cadence_minutes)
    b = endpoints(actual, cadence_minutes)
    for name in a:
        out[f"{name}_mae"] = float(np.abs(a[name] - b[name]).mean())
    return out


def grid_minutes_sanity() -> int:
    """The encoder grid is 5 minutes; the Libre target cadence is a multiple of it."""
    return GRID_MINUTES


def build_matched(root: Path = CGMACROS_ROOT) -> dict[str, PPGRDataset]:
    """Both sensors, restricted to the events usable in each. §4.3.

    The paper evaluates "paired Dexcom and Libre recordings [...] 874 matched meal events per
    sensor". Matching matters for the comparison it draws: the two sensors are scored separately
    and then averaged with equal weight, and that average only means something if both are
    answering about the same meals. Without it, a sensor whose harder events happened to drop out
    would look better for a reason that has nothing to do with the sensor.
    """
    built = {name: build(name, root) for name in SENSORS}
    keys = [
        {(subject, at) for subject, at in zip(d.subjects, d.event_times, strict=True)}
        for d in built.values()
    ]
    shared = set.intersection(*keys)
    return {name: _restrict(d, shared) for name, d in built.items()}


def _restrict(dataset: PPGRDataset, keys: set) -> PPGRDataset:
    keep = np.array([
        (subject, at) in keys
        for subject, at in zip(dataset.subjects, dataset.event_times, strict=True)
    ])
    return PPGRDataset(
        sensor=dataset.sensor,
        subjects=dataset.subjects[keep],
        event_times=dataset.event_times[keep],
        window_values=dataset.window_values[keep],
        window_mask=dataset.window_mask[keep],
        circadian=dataset.circadian[keep],
        premeal=dataset.premeal[keep],
        nutrition=dataset.nutrition[keep],
        subject_context=dataset.subject_context[keep],
        baseline_glucose=dataset.baseline_glucose[keep],
        target=dataset.target[keep],
    )
