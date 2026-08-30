// Ranking a day against the training corpus.
//
// The probe heads are unregularised logistic classifiers fitted in 128 dimensions on a few
// hundred days. Their raw output saturates: most inputs come back near 0 or 1, including
// inputs drawn from the cohort a head was fitted on. Held-out discrimination is only
// 0.64-0.88 ROC-AUC. So the raw score cannot be presented as a probability.
//
// A rank can be. ROC-AUC is precisely a statement about ranking -- the chance a positive case
// is ordered above a negative one -- so "this day scores higher than 82% of corpus days" is
// supported by the same evidence that gives the head its AUC. That is what this module
// computes, against percentile breakpoints built by scripts/build_reference_distribution.py
// over 20,000 windows of the pretraining corpus.

export type ReferenceHead = {
  /** 101 breakpoints: the 0th through 100th percentile of this head's score on the corpus. */
  breakpoints: number[];
  median: number;
};

export type Reference = {
  n_reference_windows: number;
  sampled_from: number;
  cohort_mix: Record<string, number>;
  heads: Record<string, ReferenceHead>;
};

let cache: Reference | null = null;

export async function loadReference(): Promise<Reference | null> {
  if (cache) return cache;
  try {
    const res = await fetch("/models/reference.json", { cache: "force-cache" });
    if (!res.ok) return null;
    cache = (await res.json()) as Reference;
    return cache;
  } catch {
    // The site must still work without it; ranks are an enhancement, not a dependency.
    return null;
  }
}

/** Percentile of `score` within the corpus distribution, 0-100, or null if unknown. */
export function percentileOf(ref: Reference, key: string, score: number): number | null {
  const head = ref.heads[key];
  if (!head) return null;
  const b = head.breakpoints;

  // Breakpoints are non-decreasing; find the last one at or below the score.
  let lo = 0;
  let hi = b.length - 1;
  if (score <= b[0]!) return 0;
  if (score >= b[hi]!) return 100;
  while (lo < hi - 1) {
    const mid = (lo + hi) >> 1;
    if (b[mid]! <= score) lo = mid;
    else hi = mid;
  }
  // Linear interpolation between adjacent breakpoints.
  const span = b[hi]! - b[lo]!;
  const frac = span > 0 ? (score - b[lo]!) / span : 0;
  return Math.max(0, Math.min(100, lo + frac));
}

/**
 * How much a rank from this head is worth reading.
 *
 * If the corpus scores are themselves bunched into a tiny range, small differences in a day's
 * score swing the percentile wildly and the rank is noise dressed as precision. The
 * interquartile spread of the reference distribution is the cheapest guard against that.
 */
export function rankIsInformative(ref: Reference, key: string): boolean {
  const head = ref.heads[key];
  if (!head) return false;
  const iqr = head.breakpoints[75]! - head.breakpoints[25]!;
  return iqr > 0.02;
}

/** Plain-English reading of a percentile. Deliberately coarse: the underlying head is weak. */
export function describeRank(pct: number): string {
  if (pct >= 90) return "higher than almost all corpus days";
  if (pct >= 75) return "higher than most corpus days";
  if (pct >= 60) return "somewhat above the corpus middle";
  if (pct > 40) return "near the corpus middle";
  if (pct > 25) return "somewhat below the corpus middle";
  if (pct > 10) return "lower than most corpus days";
  return "lower than almost all corpus days";
}
