// Place a stream of `Reading`s onto the 5-minute grid the encoder expects.
//
// The encoder input is `[B, 288]`:
//
//   - 288 positions = 24 hours × 12 steps/hour (one step every 5 minutes)
//   - mask is 1.0 where the user has a reading in the corresponding 5-minute bucket,
//     0.0 otherwise
//   - we NEVER interpolate, NEVER carry-forward, NEVER fill
//
// If multiple readings fall in the same bucket, we take the median. (For Dexcom at
// 5-minute cadence this is impossible by construction; for LibreView 1-minute readings
// it's the cleanest collapse rule.)
//
// Circadian start: the encoder's positional embedding depends on the window start time
// of day. For a midnight-aligned window (`startT % 86_400_000 === 0` for a day boundary
// at midnight local), circadian_start = 0. For a scrollable window, the consumer passes
// the window's hour-of-day offset directly.

import type { Reading, Window } from "../types";

export const SEQUENCE_LENGTH = 288;
export const MINUTES_PER_STEP = 5;
export const STEP_MS = MINUTES_PER_STEP * 60 * 1000;

/** Place readings into a 24-hour window starting at `startT`. */
export function buildWindow(readings: Reading[], startT: number, stepMs = STEP_MS): Window {
  const values = new Float32Array(SEQUENCE_LENGTH);
  const mask = new Float32Array(SEQUENCE_LENGTH);
  // Bucket → array of values in that bucket (so we can median-collapse later).
  const buckets: number[][] = new Array(SEQUENCE_LENGTH).fill(0).map(() => []);

  for (const r of readings) {
    if (r.t < startT) continue;
    const offset = Math.floor((r.t - startT) / stepMs);
    if (offset < 0 || offset >= SEQUENCE_LENGTH) continue;
    buckets[offset]!.push(r.mgdl);
  }

  for (let i = 0; i < SEQUENCE_LENGTH; i += 1) {
    const bucket = buckets[i]!;
    if (bucket.length === 0) continue;
    values[i] = median(bucket);
    mask[i] = 1.0;
  }

  return {
    values,
    mask,
    startT,
    // For a window starting at midnight local, circadianStart = 0
    // The Python build_window returns this same offset (see scripts/analyse.py).
    circadianStart: Math.floor(((startT % (24 * 60 * 60 * 1000)) / STEP_MS) % SEQUENCE_LENGTH),
  };
}

/** Find the most-recent 24-hour window ending at `endT`. */
export function mostRecentWindow(
  readings: Reading[],
  endT: number = Date.now(),
  stepMs = STEP_MS,
): Window {
  const startT = endT - SEQUENCE_LENGTH * stepMs;
  return buildWindow(readings, startT, stepMs);
}

function median(xs: number[]): number {
  if (xs.length === 1) return xs[0]!;
  const sorted = [...xs].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 === 0
    ? (sorted[mid - 1]! + sorted[mid]!) / 2
    : sorted[mid]!;
}

/** Coverage = fraction of positions observed (sum of mask / 288). */
export function coverageOf(mask: Float32Array): number {
  let s = 0;
  for (let i = 0; i < mask.length; i += 1) s += mask[i] ?? 0;
  return s / mask.length;
}
