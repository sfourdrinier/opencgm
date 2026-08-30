// Parse Dexcom Clarity / LibreView / Libre CSV exports into normalized `Reading[]`.
//
// We support two on-disk column layouts:
//
//   1. **Dexcom Clarity:** `Device Timestamp` (local-naive), `Historic Glucose mmol/L`
//      or `Glucose Value (mg/dL)`. Optionally an `Event Type` column; we keep only EGV.
//
//   2. **LibreView / Libre:** `Device Timestamp` (local-naive), one of
//      `Historic Glucose mg/dL`, `Scan Glucose mg/dL`, `Strip Glucose mg/dL`.
//
// Local-naive timestamps carry no offset. We treat them as the user's chosen timezone
// (`userTimeZone`) and convert to absolute milliseconds. The consumer must specify the
// timezone once per session; we never silently assume UTC, because that misplaces the
// window boundary and the head coverage band.
//
// Out-of-range values are dropped at the parser level, not stored as NaN — a downstream
// "0.0" mask entry would be silently indistinguishable from a real observed 0.
//
// We never interpolate. Sparse reads → sparse window → low coverage → applicability gate.

import Papa from "papaparse";
import type { Reading } from "../types";

export type ParseResult = {
  readings: Reading[];
  /** row count BEFORE filtering (for the UI) */
  total: number;
  /** number of rows dropped for missing timestamp or out-of-range glucose */
  dropped: number;
};

const MG_DL_PLAUSIBLE = [20, 600] as const; // see src/opencgm_stateevent/data/grid.py
const MMOL_TO_MGDL = 18.0182;

// Exports in the wild do not use the vendor's column names. A Heart Studio export of a Dexcom
// sensor, for instance, carries BOTH `sensor_timestamp_seconds` -- a monotonic counter that
// starts near zero when the sensor is activated -- and `observed_at_utc`, the actual wall
// clock. Matching on the word "timestamp" picks the counter, and every reading then lands in
// 1970 without anything visibly failing.
//
// So a column is only accepted as the timestamp if its *values* look like datetimes. Bare
// integers are rejected however promising the header sounds.
const TIME_NAME_HINTS = [
  "observed_at_utc",
  "observed_at",
  "device timestamp",
  "timestamp (yyyy-mm-ddthh:mm:ss)",
  "date and time",
  "device time",
  "datetime",
  "timestamp",
  "time",
  "date",
];

/** Columns whose names suggest an epoch or an offset rather than a wall clock. */
const TIME_NAME_ANTIHINTS = ["_seconds", "_ms", "_epoch", "_delay", "duration", "elapsed"];

/** True if the sampled cell values look like parseable datetimes rather than bare numbers. */
function looksLikeDatetimeColumn(values: string[]): boolean {
  const sample = values.filter((v) => v && v.trim() !== "").slice(0, 25);
  if (sample.length === 0) return false;
  let ok = 0;
  for (const raw of sample) {
    const v = raw.trim();
    // A bare integer or float is a counter or an epoch, not a wall clock we can trust.
    if (/^-?\d+(\.\d+)?$/.test(v)) continue;
    if (!/\d{4}|\d{1,2}[/-]\d{1,2}/.test(v)) continue;
    if (Number.isFinite(Date.parse(v)) || /\d{1,2}:\d{2}/.test(v)) ok += 1;
  }
  return ok >= Math.max(1, Math.floor(sample.length * 0.8));
}

/** ISO-8601 strings carrying `Z` or an explicit +hh:mm offset are already absolute. */
export function hasExplicitOffset(value: string): boolean {
  return /(?:Z|[+-]\d{2}:?\d{2})$/.test(value.trim());
}

/**
 * Choose the timestamp column by name preference, then confirm by inspecting values.
 *
 * `rows` is a sample of parsed rows; only their values are read.
 */
export function pickTimestampColumn(
  header: string[],
  rows: Record<string, string>[],
  override?: string,
): string | undefined {
  if (override) {
    const exact = header.find((h) => h.trim().toLowerCase() === override.toLowerCase());
    if (exact) return exact;
  }

  const valuesFor = (col: string) => rows.map((r) => r[col] ?? "");
  const scored: { col: string; rank: number }[] = [];

  for (const col of header) {
    const name = col.trim().toLowerCase();
    if (TIME_NAME_ANTIHINTS.some((a) => name.includes(a))) continue;
    if (!looksLikeDatetimeColumn(valuesFor(col))) continue;
    const rank = TIME_NAME_HINTS.findIndex((h) => name === h || name.includes(h));
    scored.push({ col, rank: rank === -1 ? TIME_NAME_HINTS.length : rank });
  }

  scored.sort((a, b) => a.rank - b.rank);
  // Fall back to any datetime-shaped column even if its name says nothing useful.
  return scored[0]?.col;
}
const DEXCOM_MGDL_HINTS = [
  "glucose value (mg/dl)",
  "historic glucose mg/dl",
  "glucose_mg_dl",
  "glucose",
  "sgv",
  "value",
  "mg/dl",
];
const DEXCOM_MMOL_HINTS = [
  "glucose value (mmol/l)",
  "historic glucose mmol/l",
  "glucose_mmol_l",
  "mmol/l",
];
const EVENT_HINTS = ["event type", "event_subtype"];

function pickColumn(
  header: string[],
  hints: readonly string[],
  override?: string,
): string | undefined {
  if (override) {
    const exact = header.find((h) => h.trim().toLowerCase() === override.toLowerCase());
    if (exact) return exact;
  }
  for (const hint of hints) {
    const match = header.find((h) => h.trim().toLowerCase() === hint);
    if (match) return match;
  }
  return undefined;
}

function isPlausibleMgdl(v: number): boolean {
  return Number.isFinite(v) && v >= MG_DL_PLAUSIBLE[0] && v <= MG_DL_PLAUSIBLE[1];
}

/** Parse a Dexcom/Libre CSV file. Streams large files via Papa's `step` callback. */
export async function parseCgmCsv(
  file: File,
  userTimeZone: string,
  options: {
    timestampColumn?: string;
    glucoseColumn?: string;
    onProgress?: (rows: number) => void;
  } = {},
): Promise<ParseResult> {
  // Column choice needs to see values, not just the header, so read a small preview first.
  // Streaming row-by-row cannot do this: by the time row 1 arrives there is nothing to
  // compare it against, which is why detection used to run on header names alone.
  const preview = await file.slice(0, 64 * 1024).text();
  const head = Papa.parse<Record<string, string>>(preview, {
    header: true,
    skipEmptyLines: true,
    preview: 30,
  });
  const header = head.meta.fields ?? [];
  const sample = head.data;

  const timeCol = pickTimestampColumn(header, sample, options.timestampColumn);
  const mgdlCol = pickColumn(header, DEXCOM_MGDL_HINTS, options.glucoseColumn);
  const mmolCol = pickColumn(header, DEXCOM_MMOL_HINTS);
  const eventCol = pickColumn(header, EVENT_HINTS);

  if (!timeCol) {
    throw new Error(
      `No column in this file holds recognisable dates. Columns found: ${header.join(", ")}`,
    );
  }
  if (!mgdlCol && !mmolCol) {
    throw new Error(
      `No column in this file holds glucose readings. Columns found: ${header.join(", ")}`,
    );
  }
  const glucoseCol = mgdlCol ?? mmolCol!;
  const glucoseIsMmol = !mgdlCol && !!mmolCol;

  // Sensors report during warm-up, and those readings are not measurements. Drop them when
  // the export says so.
  const stateCol = header.find((h) => h.trim().toLowerCase() === "algorithm_state");
  const reliableCol = header.find((h) => h.trim().toLowerCase() === "reliable");

  const readings: Reading[] = [];
  let total = 0;
  let dropped = 0;

  return new Promise((resolve, reject) => {
    Papa.parse<Record<string, string>>(file, {
      header: true,
      skipEmptyLines: true,
      worker: true,
      step: ({ data }) => {
        total += 1;
        const ts = data[timeCol];
        const gRaw = data[glucoseCol];
        if (!ts || !gRaw) {
          dropped += 1;
          return;
        }

        if (eventCol) {
          const ev = data[eventCol];
          // Dexcom `Event Type` rows: calibrations, insulin, carbs. Not glucose traces.
          if (ev && ev !== "EGV" && ev !== "") return;
        }
        if (stateCol && (data[stateCol] ?? "").includes("warmup")) {
          dropped += 1;
          return;
        }
        if (reliableCol && (data[reliableCol] ?? "").trim() === "0" && !stateCol) {
          dropped += 1;
          return;
        }

        const g = Number(gRaw);
        // Unit inference: an explicitly-named mmol/L column, or a value too small to be mg/dL.
        const mgdl = glucoseIsMmol || g <= 40 ? g * MMOL_TO_MGDL : g;
        if (!isPlausibleMgdl(mgdl)) {
          dropped += 1;
          return;
        }

        // An ISO string ending in `Z` or `+02:00` already names an instant. Passing it
        // through the local-naive path would shift it by the user's offset a second time.
        const epochMs = hasExplicitOffset(ts)
          ? Date.parse(ts)
          : parseLocalNaiveTimestamp(ts, userTimeZone);
        if (!Number.isFinite(epochMs)) {
          dropped += 1;
          return;
        }

        readings.push({ t: epochMs, mgdl });
        if (options.onProgress && total % 500 === 0) options.onProgress(total);
      },
      complete: () => {
        readings.sort((a, b) => a.t - b.t);
        resolve({ readings, total, dropped });
      },
      error: (err: Error) => reject(err),
    });
  });
}

/**
 * Parse a local-naive timestamp (`"2024-03-14 13:25:00"` or `"2024-03-14T13:25:00"`) as
 * the user's specified timezone, returning milliseconds since epoch.
 *
 * Uses `Intl.DateTimeFormat` to derive the offset for the given timezone on the given
 * date. SVG-spec compliant: a date at 2024-03-14 13:25 in America/Los_Angeles is the
 * right UTC instant whether DST is in effect or not.
 */
export function parseLocalNaiveTimestamp(localStr: string, timeZone: string): number {
  // Normalize "YYYY-MM-DD HH:MM:SS" → "YYYY-MM-DDTHH:MM:SS"
  const iso = localStr.includes("T") ? localStr : localStr.replace(" ", "T");

  // Extract Y/M/D/H/M/S and ask Intl what UTC offset that wall-time has in `timeZone`.
  const m = iso.match(/^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})/);
  if (!m) return NaN;
  // `m` matched, so groups 1..6 exist; the defaults are only here to satisfy
  // `noUncheckedIndexedAccess`, which cannot know that.
  const [, y = "0", mo = "1", d = "1", h = "0", mi = "0", s = "0"] = m;
  const utcDate = new Date(Date.UTC(+y, +mo - 1, +d, +h, +mi, +s));

  // Trick: format the wall-time in `timeZone` to figure out what hour it lands on in
  // UTC, then offset back. This is the canonical timezone-naive-to-absolute conversion
  // pattern. (Libraries like date-fns-tz do the same.)
  const dtf = new Intl.DateTimeFormat("en-US", {
    timeZone,
    hour12: false,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
  const parts = dtf.formatToParts(utcDate);
  const get = (t: string): string => parts.find((p) => p.type === t)?.value ?? "";
  const tzAsUtc = Date.UTC(
    +get("year"),
    +get("month") - 1,
    +get("day"),
    +get("hour") % 24, // "24" appears in some Intl impls at midnight
    +get("minute"),
    +get("second"),
  );
  // The difference between the formatted-as-UTC and the original wall-time-as-UTC is the
  // timezone's offset. The actual absolute time is the wall-time minus that offset.
  return Date.UTC(+y, +mo - 1, +d, +h, +mi, +s) - (tzAsUtc - utcDate.getTime());
}
