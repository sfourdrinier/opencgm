// The summary a clinic already computes for a day of CGM.
//
// One implementation, used by the HTTP API and by the in-browser analysis. They previously
// would have had two, which is how a number ends up meaning one thing on a page and another
// in a response.
//
// Everything is computed from observed samples only. Gaps are excluded rather than
// interpolated, so these figures are conditional on what the sensor actually recorded -- the
// same rule the model itself follows.

export type ClinicalMetrics = {
  n_observed: number;
  mean_mg_dl: number;
  sd_mg_dl: number;
  coefficient_of_variation: number;
  gmi_percent: number;
  time_below_70: number;
  time_in_range_70_180: number;
  time_above_180: number;
  min_mg_dl: number;
  max_mg_dl: number;
};

export function clinicalMetrics(
  values: ArrayLike<number>,
  mask: ArrayLike<number>,
): ClinicalMetrics | null {
  const obs: number[] = [];
  for (let i = 0; i < values.length; i += 1) if (mask[i]) obs.push(values[i]!);
  if (obs.length === 0) return null;

  const mean = obs.reduce((a, b) => a + b, 0) / obs.length;
  const sd = Math.sqrt(obs.reduce((a, b) => a + (b - mean) ** 2, 0) / obs.length);
  const frac = (lo: number, hi: number) =>
    obs.filter((v) => v >= lo && v < hi).length / obs.length;

  return {
    n_observed: obs.length,
    mean_mg_dl: +mean.toFixed(1),
    sd_mg_dl: +sd.toFixed(1),
    coefficient_of_variation: +(sd / mean).toFixed(4),
    // Glucose Management Indicator: the ADA's mapping from mean glucose to an estimated
    // HbA1c. It is a restatement of the mean, not an independent measurement.
    gmi_percent: +(3.31 + 0.02392 * mean).toFixed(2),
    time_below_70: +frac(0, 70).toFixed(4),
    time_in_range_70_180: +frac(70, 180).toFixed(4),
    time_above_180: +frac(180, 1000).toFixed(4),
    min_mg_dl: Math.min(...obs),
    max_mg_dl: Math.max(...obs),
  };
}
