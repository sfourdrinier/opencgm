"""Run the trained model on one person's CGM stream. The entry point for an application.

Everything else in this package is built for reproducible research on fixed cohorts. This module
is the other direction: arbitrary readings in, a report out, with no reference to any dataset.

    from opencgm_stateevent.infer import Analyser

    analyser = Analyser.load("runs/rawstats120/ckpt_last.pt", heads="artifacts/heads.pkl")
    report = analyser.analyse_day(readings)      # readings: [(datetime, mg/dL), ...]
    print(report.to_json())

Three design rules, each of which the research code already obeys and which matter more here
because a live sensor is messier than a curated dataset:

**Gaps stay gaps.** A real sensor drops out -- showers, compression lows, warm-up periods. The
model was trained with an explicit observation mask and never sees interpolated values, so
readings are placed on the 5-minute grid and missing positions are marked missing. Filling them
would feed the model a kind of input it has never seen and quietly degrade every number below.

**Reliability travels with the score.** Each phenotype probability is returned with the
cross-validated ROC-AUC of the head that produced it, the number of subjects that head was
learned from, and a flag for whether it cleared the signal floor. A score without those is not
interpretable, and the spread across tasks is large enough that a single "AI confidence" number
would be misleading.

**Nothing here is a diagnosis.** The heads are subject-grouped linear probes fitted on cohorts of
29 to 109 people. They estimate a population association. The wording of every field reflects
that, and `PhenotypeScore.population_phrasing` gives text that is accurate to what was computed.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

from .data.grid import build_window
from .data.timestamps import GRID_MINUTES as MINUTES_PER_STEP
from .data.timestamps import SEQUENCE_LENGTH
from .eval.embed import load_encoder

#: Standard clinical thresholds, mg/dL. These are consensus targets, not model outputs.
HYPO_2, HYPO_1, TARGET_HIGH, HYPER_2 = 54.0, 70.0, 180.0, 250.0

#: Physiologically possible mg/dL. Values outside this cannot be a human glucose reading, and the
#: commonest cause by far is an mmol/L export: 4-12 mmol/L is 72-216 mg/dL, and feeding those
#: numbers through as mg/dL describes a person in fatal hypoglycaemia while every other check
#: passes. D019 made this sharper rather than milder -- the model now reads absolute level, so a
#: unit error is no longer a harmless rescaling the instance normalisation would absorb.
PLAUSIBLE_MG_DL = (20.0, 600.0)
#: Above this fraction of readings below 25 mg/dL, an mmol/L stream is the only sane explanation.
MMOL_SUSPICION = 0.5

#: A day this empty cannot support any of the numbers below. 24 of 288 positions is two hours of
#: a 5-minute sensor; below that even Time in Range is a statement about the gaps, not the person.
MIN_OBSERVED = 24


@dataclass
class DayMetrics:
    """Deterministic clinical summary. Arithmetic on the trace, no model involved."""

    n_observed: int
    coverage: float
    mean_glucose: float
    glucose_management_indicator: float
    std_glucose: float
    coefficient_of_variation: float
    min_glucose: float
    max_glucose: float
    time_in_range: float
    time_below_70: float
    time_below_54: float
    time_above_180: float
    time_above_250: float
    mean_absolute_rate_of_change: float
    overnight_mean: float
    dawn_rise: float
    longest_stable_hours: float

    @property
    def variability_is_stable(self) -> bool:
        """CV below 36% is the consensus stability threshold."""
        return self.coefficient_of_variation < 0.36


@dataclass
class PhenotypeScore:
    """One head's output, with everything needed to interpret it."""

    task: str
    dataset: str
    probability: float
    predicted_class: int
    #: cross-validated ROC-AUC of this head on held-out subjects
    reliability: float
    reliability_sd: float
    reliability_subject_level: float
    n_subjects_learned_from: int
    has_signal: bool
    #: False when this day's sampling density is outside the range the head was fitted on.
    applicable: bool = True
    applicability_note: str = ""

    @property
    def population_phrasing(self) -> str:
        """Text that matches what was actually computed.

        A subject-grouped probe estimates how common a label is among people whose days look like
        this one. It does not estimate whether this person has the condition, and phrasing it that
        way would overstate a 29-subject linear probe.
        """
        if not self.applicable:
            return (
                f"{self.task} is not applicable to this recording: {self.applicability_note}"
            )
        if not self.has_signal:
            return (
                f"No reliable signal for {self.task} was found in our cohort "
                f"(held-out ROC-AUC {self.reliability:.2f}); this score should not be read."
            )
        return (
            f"Days like this one are more common among people with {self.task} "
            f"({self.probability:.0%} model score; held-out ROC-AUC {self.reliability:.2f} "
            f"from {self.n_subjects_learned_from} subjects)."
        )


@dataclass
class DayReport:
    start: datetime
    metrics: DayMetrics
    embedding: list[float]
    phenotypes: list[PhenotypeScore] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_json(self, **kwargs) -> str:
        payload = asdict(self)
        payload["start"] = self.start.isoformat()
        payload["phenotypes"] = [
            {**asdict(p), "phrasing": p.population_phrasing} for p in self.phenotypes
        ]
        payload["metrics"]["variability_is_stable"] = self.metrics.variability_is_stable
        return json.dumps(payload, indent=2, **kwargs)


#: How far outside a head's fitted coverage band a day may sit before the head refuses it. The
#: bands are far apart in practice -- 0.31 for 15-minute sensors against 0.84-1.00 for 5-minute
#: ones -- so this tolerance separates sampling regimes without rejecting an ordinary sensor gap.
COVERAGE_TOLERANCE = 0.15


def _applicable(head: dict, coverage: float) -> tuple[bool, str]:
    """Is this head fitted on data resembling this day's sampling density?"""
    low = head.get("coverage_p05")
    high = head.get("coverage_p95")
    if low is None or high is None:
        return True, ""  # a head from before the band was recorded; do not silently reject it
    if low - COVERAGE_TOLERANCE <= coverage <= high + COVERAGE_TOLERANCE:
        return True, ""
    return False, (
        f"fitted on recordings with {low:.0%}-{high:.0%} of positions observed, but this day "
        f"has {coverage:.0%}; the sampling rates differ too much for the score to mean anything"
    )


def check_units(values: np.ndarray) -> None:
    """Reject a stream that is almost certainly in mmol/L, rather than analysing it as mg/dL."""
    if not len(values):
        return
    low, high = PLAUSIBLE_MG_DL
    fraction_tiny = float((values < 25.0).mean())
    if fraction_tiny > MMOL_SUSPICION:
        raise ValueError(
            f"{fraction_tiny:.0%} of readings are below 25, which is not survivable in mg/dL "
            f"but is ordinary in mmol/L (median {np.median(values):.1f}). This model expects "
            f"mg/dL. Multiply by 18.0182 to convert, or pass values already in mg/dL."
        )
    outside = float(((values < low) | (values > high)).mean())
    if outside > 0.05:
        raise ValueError(
            f"{outside:.0%} of readings fall outside {low:.0f}-{high:.0f} mg/dL, so this is "
            f"probably not a mg/dL glucose series (median {np.median(values):.1f})."
        )


def _stable_run_hours(values: np.ndarray, mask: np.ndarray) -> float:
    """Longest stretch of consecutive observed positions inside the target range."""
    ok = mask & (values >= HYPO_1) & (values <= TARGET_HIGH)
    best = run = 0
    for flag in ok:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best * MINUTES_PER_STEP / 60.0


def compute_metrics(
    values: np.ndarray, mask: np.ndarray, circadian_start: int = 0
) -> DayMetrics:
    """Tier-1 clinical metrics from the masked trace. Unobserved positions never contribute.

    `circadian_start` is the grid index of the window's first position, so that "overnight" means
    midnight to 06:00 on the clock rather than the first quarter of whatever window happened to
    be built. A live window ends at the most recent reading and therefore starts at an arbitrary
    hour: for a report ending at 14:00, grid positions 0-71 are 14:05-20:05 of the previous day,
    and calling that "overnight mean" would be simply false.
    """
    observed = values[mask]
    n = int(mask.sum())
    mean = float(observed.mean()) if n else float("nan")
    sd = float(observed.std(ddof=1)) if n > 1 else float("nan")

    index = np.flatnonzero(mask)
    if len(index) > 1:
        rates = np.diff(values[index]) / (np.diff(index) * MINUTES_PER_STEP)
        roc = float(np.abs(rates).mean())
    else:
        roc = float("nan")

    # Absolute clock position of every grid slot, so the windows below are real hours.
    clock = (circadian_start + np.arange(SEQUENCE_LENGTH)) % SEQUENCE_LENGTH
    night_slots = clock < 72                      # 00:00-06:00
    dawn_slots = (clock >= 60) & (clock < 72)     # 05:00-06:00

    night = mask & night_slots
    overnight = float(values[night].mean()) if night.any() else float("nan")
    late = mask & dawn_slots
    dawn = (
        float(values[late].mean() - values[night].min())
        if late.any() and night.any() else float("nan")
    )

    def fraction(low: float, high: float) -> float:
        if not n:
            return float("nan")
        return float(((observed >= low) & (observed < high)).sum() / n)

    return DayMetrics(
        n_observed=n,
        coverage=n / SEQUENCE_LENGTH,
        mean_glucose=mean,
        # Bergenstal et al. 2018, the standard mg/dL form.
        glucose_management_indicator=3.31 + 0.02392 * mean if n else float("nan"),
        std_glucose=sd,
        coefficient_of_variation=sd / mean if n > 1 and mean > 0 else float("nan"),
        min_glucose=float(observed.min()) if n else float("nan"),
        max_glucose=float(observed.max()) if n else float("nan"),
        time_in_range=fraction(HYPO_1, TARGET_HIGH),
        time_below_70=fraction(0.0, HYPO_1),
        time_below_54=fraction(0.0, HYPO_2),
        time_above_180=fraction(TARGET_HIGH, np.inf),
        time_above_250=fraction(HYPER_2, np.inf),
        mean_absolute_rate_of_change=roc,
        overnight_mean=overnight,
        dawn_rise=dawn,
        longest_stable_hours=_stable_run_hours(values, mask),
    )


class Analyser:
    """A loaded encoder plus, optionally, the fitted phenotype heads."""

    def __init__(self, model, ref, heads: dict | None = None, device: str = "cpu") -> None:
        self.model = model
        self.ref = ref
        self.heads = heads or {}
        self.device = device

    @classmethod
    def load(
        cls,
        checkpoint: str | Path,
        *,
        heads: str | Path | None = None,
        device: str = "cpu",
    ) -> Analyser:
        """CPU by default: one day is a single 288-point forward pass through 733k parameters."""
        model, ref = load_encoder(Path(checkpoint), device=device)
        bundle = None
        if heads is not None and Path(heads).exists():
            with Path(heads).open("rb") as fh:
                bundle = pickle.load(fh)
            # Weight identity alone is not enough, for exactly the reason D019 established: the
            # same state dictionary produces different embeddings under different architecture
            # flags, so a head fitted against a normalised-statistics encoder would pass a
            # weights-only check against a raw-statistics one and then score in the wrong space.
            fitted = bundle.get("encoder", {})
            for field in ("weights_sha256", "architecture", "dtype", "backend"):
                expected, actual = fitted.get(field), getattr(ref, field, None)
                if expected is not None and expected != actual:
                    raise ValueError(
                        f"these heads were fitted against a different encoder ({field}: "
                        f"{expected!r} != {actual!r}). Re-run scripts/fit_heads.py against this "
                        f"checkpoint; a head is only meaningful in the embedding space it was "
                        f"fitted in."
                    )
        return cls(model, ref, bundle["heads"] if bundle else None, device=device)

    def window_from(
        self, readings: list[tuple[datetime, float]], start: datetime | None = None
    ) -> tuple[np.ndarray, np.ndarray, int, datetime]:
        """Place readings on the 5-minute grid. Gaps stay gaps.

        With no explicit `start`, the window is the 24 hours ending at the most recent reading,
        which is what a live application wants.
        """
        if not readings:
            raise ValueError("no readings")
        check_units(np.array([v for _, v in readings], dtype=float))
        readings = sorted(readings)
        if start is None:
            start = readings[-1][0] - timedelta(hours=24) + timedelta(minutes=MINUTES_PER_STEP)
        window = build_window(
            [t for t, _ in readings], [float(v) for _, v in readings], start,
            dataset_id="live", canonical_subject_id="live", session_id="live",
        )
        return window.values, window.mask, window.circadian_start_index, window.start_local

    @torch.no_grad()
    def embed(self, values: np.ndarray, mask: np.ndarray, circadian: int) -> np.ndarray:
        out = self.model.encode(
            torch.from_numpy(values).unsqueeze(0).to(self.device),
            torch.from_numpy(mask).unsqueeze(0).to(self.device),
            torch.tensor([circadian], dtype=torch.long, device=self.device),
        )
        return out.contextual_tokens.mean(dim=1).squeeze(0).float().cpu().numpy()

    def analyse_day(
        self, readings: list[tuple[datetime, float]], start: datetime | None = None
    ) -> DayReport:
        values, mask, circadian, window_start = self.window_from(readings, start)
        warnings: list[str] = []
        n = int(mask.sum())
        if n < MIN_OBSERVED:
            warnings.append(
                f"only {n} of 288 grid positions carry a reading; every number below is "
                f"unreliable at this coverage"
            )
        elif n < SEQUENCE_LENGTH * 0.5:
            warnings.append(
                f"sparse day: {n}/288 positions observed ({n / SEQUENCE_LENGTH:.0%} coverage)"
            )

        metrics = compute_metrics(values, mask, circadian_start=circadian)
        embedding = self.embed(values, mask, circadian)

        coverage = float(metrics.coverage)
        scores = []
        for key, head in self.heads.items():
            applicable, note = _applicable(head, coverage)
            if not applicable:
                scores.append(PhenotypeScore(
                    task=key, dataset=head["dataset"], probability=float("nan"),
                    predicted_class=-1, reliability=head["roc_auc"],
                    reliability_sd=head["roc_auc_sd"],
                    reliability_subject_level=head["roc_auc_subject"],
                    n_subjects_learned_from=head["n_subjects"], has_signal=head["has_signal"],
                    applicable=False, applicability_note=note,
                ))
                continue
            pipeline = head["pipeline"]
            proba = pipeline.predict_proba(embedding.reshape(1, -1))[0]
            # `classes_` is the authoritative mapping; argmax over columns is only the same thing
            # while labels happen to be contiguous 0..K-1.
            classes = getattr(pipeline, "classes_", np.arange(len(proba)))
            scores.append(PhenotypeScore(
                task=key,
                dataset=head["dataset"],
                probability=float(proba[1]) if len(proba) == 2 else float(proba.max()),
                predicted_class=int(classes[int(proba.argmax())]),
                reliability=head["roc_auc"],
                reliability_sd=head["roc_auc_sd"],
                reliability_subject_level=head["roc_auc_subject"],
                n_subjects_learned_from=head["n_subjects"],
                has_signal=head["has_signal"],
            ))
        scores.sort(key=lambda s: (-s.reliability if (s.has_signal and s.applicable) else 1e9))

        return DayReport(
            start=window_start, metrics=metrics, embedding=embedding.tolist(),
            phenotypes=scores, warnings=warnings,
        )

    def analyse_stream(
        self, readings: list[tuple[datetime, float]], *, days: int = 14
    ) -> list[DayReport]:
        """Split a long stream into consecutive 24-hour windows, most recent last.

        Non-overlapping and anchored to the most recent reading, so the last window is the same
        one `analyse_day` would produce.
        """
        if not readings:
            return []
        readings = sorted(readings)
        end = readings[-1][0]
        reports = []
        for k in range(days - 1, -1, -1):
            start = end - timedelta(hours=24 * (k + 1)) + timedelta(minutes=MINUTES_PER_STEP)
            chunk = [(t, v) for t, v in readings if start <= t < start + timedelta(hours=24)]
            if len(chunk) < MIN_OBSERVED:
                continue
            reports.append(self.analyse_day(chunk, start=start))
        return reports


def similarity(reports: list[DayReport]) -> np.ndarray:
    """Cosine similarity between days, in embedding space.

    This is what "today resembles your usual Tuesday" reduces to. It asserts nothing clinical --
    it is a distance between two representations, and it is the one learned quantity in this
    module that needs no cohort to justify it.
    """
    matrix = np.array([r.embedding for r in reports], dtype=np.float64)
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.where(norm > 0, norm, 1.0)
    return normalized @ normalized.T
