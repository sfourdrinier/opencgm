// Synthetic example days, so the demo works for someone who has never worn a sensor.
//
// These are SIMULATED, not real patient traces. That is a deliberate choice: several of the
// public cohorts carry licence terms whose redistribution status we have not finished
// verifying (see the Stanford entry in the repository's rights registry), and shipping a real
// person's day to make a demo prettier is not a trade worth making.
//
// The generator is a crude two-compartment caricature: a slow baseline that drifts with the
// circadian clock, plus meal excursions that rise fast and decay slowly. It is good enough to
// exercise the model end to end and to show what a gap looks like. It is not a physiological
// claim about anything.

import type { Reading } from "../types";

const STEP_MS = 5 * 60 * 1000;
const N = 288;

type Meal = { atStep: number; peak: number; riseSteps: number; decaySteps: number };

type Profile = {
  id: string;
  label: string;
  blurb: string;
  baseline: number;
  circadianAmp: number;
  meals: Meal[];
  noise: number;
  /** step ranges the "sensor" failed to record, as [start, end) */
  gaps: [number, number][];
};

export const PROFILES: Profile[] = [
  {
    id: "steady",
    label: "A steady day",
    blurb:
      "Glucose stays in a narrow band, meals cause modest bumps that settle within two hours.",
    baseline: 95,
    circadianAmp: 6,
    meals: [
      { atStep: 90, peak: 32, riseSteps: 7, decaySteps: 22 },
      { atStep: 150, peak: 26, riseSteps: 6, decaySteps: 20 },
      { atStep: 228, peak: 30, riseSteps: 8, decaySteps: 26 },
    ],
    noise: 2.2,
    gaps: [[36, 48]],
  },
  {
    id: "swings",
    label: "A day with big swings",
    blurb:
      "The same three meals, but each excursion is larger and takes much longer to come back down.",
    baseline: 118,
    circadianAmp: 12,
    meals: [
      { atStep: 88, peak: 88, riseSteps: 9, decaySteps: 46 },
      { atStep: 152, peak: 72, riseSteps: 8, decaySteps: 42 },
      { atStep: 226, peak: 95, riseSteps: 10, decaySteps: 54 },
    ],
    noise: 3.4,
    gaps: [[132, 141]],
  },
  {
    id: "patchy",
    label: "A day the sensor half-missed",
    blurb:
      "A steady day with four hours of missing readings. The model is told the gap is a gap — it is never filled in.",
    baseline: 101,
    circadianAmp: 7,
    meals: [
      { atStep: 92, peak: 38, riseSteps: 7, decaySteps: 24 },
      { atStep: 224, peak: 34, riseSteps: 8, decaySteps: 28 },
    ],
    noise: 2.6,
    gaps: [
      [108, 156],
      [12, 20],
    ],
  },
];

/** Deterministic small PRNG so a given profile always renders the same day. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function buildSampleDay(profile: Profile, startT: number): Reading[] {
  const rand = mulberry32(
    [...profile.id].reduce((acc, ch) => (acc * 31 + ch.charCodeAt(0)) >>> 0, 7),
  );
  const readings: Reading[] = [];

  for (let i = 0; i < N; i += 1) {
    if (profile.gaps.some(([a, b]) => i >= a && i < b)) continue;

    // Circadian drift: lowest in the small hours, highest late afternoon.
    const phase = (2 * Math.PI * i) / N;
    let v = profile.baseline + profile.circadianAmp * Math.sin(phase - Math.PI / 2);

    for (const meal of profile.meals) {
      const d = i - meal.atStep;
      if (d < 0) continue;
      const rise = 1 - Math.exp(-d / meal.riseSteps);
      const decay = Math.exp(-d / meal.decaySteps);
      v += meal.peak * rise * decay * Math.E ** 0.5;
    }

    v += (rand() - 0.5) * 2 * profile.noise;
    readings.push({ t: startT + i * STEP_MS, mgdl: Math.max(40, Math.min(400, v)) });
  }
  return readings;
}

export function sampleCsv(readings: Reading[]): string {
  const rows = ["Timestamp,Glucose Value (mg/dL)"];
  for (const r of readings) {
    rows.push(`${new Date(r.t).toISOString().slice(0, 19).replace("T", " ")},${r.mgdl.toFixed(0)}`);
  }
  return rows.join("\n");
}
