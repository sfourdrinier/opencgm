// Primitives shared by every 24-hour chart on the site.
//
// All of them plot 288 five-minute slots with a validity mask, and all of them must break the
// line where the sensor recorded nothing rather than drawing across the gap. That logic was
// written five times; it lives here now.

export const SLOTS = 288;

export type Point = { i: number; v: number };

/**
 * Split a 288-slot series into runs of consecutive observed samples.
 *
 * One path per run is what keeps a gap looking like a gap. A single path across the whole
 * series would connect the last reading before an outage to the first one after it, which is
 * the visual form of the interpolation this project refuses to do everywhere else.
 */
export function contiguousRuns(
  values: ArrayLike<number | null>,
  isObserved: (i: number) => boolean,
): Point[][] {
  const runs: Point[][] = [];
  let cur: Point[] = [];
  for (let i = 0; i < SLOTS; i += 1) {
    const v = values[i];
    if (isObserved(i) && v != null) {
      cur.push({ i, v });
    } else if (cur.length) {
      runs.push(cur);
      cur = [];
    }
  }
  if (cur.length) runs.push(cur);
  return runs;
}

/** Runs from a values array that uses `null` for unobserved slots. */
export function runsFromNullable(values: (number | null)[]): Point[][] {
  return contiguousRuns(values, (i) => values[i] != null);
}

/** Runs from a values/mask pair, the shape the model itself uses. */
export function runsFromMask(values: ArrayLike<number>, mask: ArrayLike<number>): Point[][] {
  return contiguousRuns(values, (i) => Boolean(mask[i]));
}

/** An SVG path for one run, given axis mappings. */
export function pathFor(
  run: Point[],
  x: (i: number) => number,
  y: (v: number) => number,
): string {
  return run
    .map((p, k) => `${k === 0 ? "M" : "L"}${x(p.i).toFixed(1)},${y(p.v).toFixed(1)}`)
    .join(" ");
}

/** Contiguous spans of unobserved slots, for drawing the grey blocks. */
export function gapSpans(isObserved: (i: number) => boolean): { start: number; end: number }[] {
  const spans: { start: number; end: number }[] = [];
  let start: number | null = null;
  for (let i = 0; i <= SLOTS; i += 1) {
    const missing = i < SLOTS && !isObserved(i);
    if (missing && start === null) start = i;
    if (!missing && start !== null) {
      spans.push({ start, end: i - 1 });
      start = null;
    }
  }
  return spans;
}

/** Hour labels every six hours, the axis every one of these charts wants. */
export const HOUR_TICKS = [0, 72, 144, 216, 287] as const;

export function hourLabel(i: number): string {
  return String(Math.round((i / SLOTS) * 24)).padStart(2, "0");
}
