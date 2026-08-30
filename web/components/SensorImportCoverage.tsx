"use client";

import type { SensorImportResult } from "@/lib/sensor/contracts";

export type SensorImportCoverageSummary = {
  readonly readings: number;
  readonly reliable: number;
  readonly duplicates: number;
  readonly range: string;
  readonly cadence: string;
  readonly gaps: string;
  readonly completeness: SensorImportResult["completeness"];
};

const dateTime = new Intl.DateTimeFormat("en-GB", {
  timeZone: "UTC",
  day: "2-digit",
  month: "short",
  year: "numeric",
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
});

function finiteTimes(result: SensorImportResult): number[] {
  return result.records
    .map((record) => record.atMs)
    .filter((value): value is number => Number.isFinite(value))
    .sort((a, b) => a - b);
}

function cadenceLabel(times: readonly number[]): string {
  const intervals = times.slice(1).flatMap((time, index) => {
    const previous = times[index];
    if (previous === undefined || time <= previous) return [];
    return [time - previous];
  });
  if (!intervals.length) return times.length === 1 ? "one reading" : "not estimated";
  const median = [...intervals].sort((a, b) => a - b)[Math.floor(intervals.length / 2)] ?? intervals[0] ?? 0;
  const minutes = Math.round(median / 60_000);
  return minutes < 1 ? "less than a minute" : `about every ${minutes} min`;
}

function gapLabel(times: readonly number[]): string {
  const intervals = times.slice(1).flatMap((time, index) => {
    const previous = times[index];
    if (previous === undefined || time <= previous) return [];
    return [time - previous];
  });
  if (!intervals.length) return "No interval to estimate gaps from.";
  const sorted = [...intervals].sort((a, b) => a - b);
  const typical = sorted[Math.floor(sorted.length / 2)] ?? sorted[0] ?? 0;
  const gaps = intervals.filter((interval) => interval > Math.max(typical * 1.8, 15 * 60_000));
  if (!gaps.length) return "No unusually long gaps detected.";
  const longest = Math.round(Math.max(...gaps) / 60_000);
  return `${gaps.length} gap${gaps.length === 1 ? "" : "s"}; longest about ${longest} min.`;
}

export function sensorImportCoverage(result: SensorImportResult): SensorImportCoverageSummary {
  const times = finiteTimes(result);
  const first = times[0];
  const last = times[times.length - 1];
  const sameDay = first !== undefined && last !== undefined && new Date(first).toISOString().slice(0, 10) === new Date(last).toISOString().slice(0, 10);
  const range = first === undefined || last === undefined
    ? "No dated readings"
    : sameDay
      ? `${dateTime.format(first)}–${new Intl.DateTimeFormat("en-GB", { timeZone: "UTC", hour: "2-digit", minute: "2-digit", hour12: false }).format(last)}`
      : `${dateTime.format(first)}–${dateTime.format(last)}`;
  return {
    readings: result.metadata.readingCount,
    reliable: result.records.filter((record) => record.reliable && record.mgdl !== null).length,
    duplicates: result.metadata.duplicateCount,
    range,
    cadence: cadenceLabel(times),
    gaps: gapLabel(times),
    completeness: result.completeness,
  };
}

export function SensorImportCoverage({ summary }: { summary: SensorImportCoverageSummary }) {
  return (
    <section className="border border-rule bg-paper-raised p-5" aria-labelledby="sensor-coverage-title">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 id="sensor-coverage-title" className="text-lg font-semibold text-ink">Coverage</h2>
        <span className={summary.completeness === "complete" ? "text-sm text-accent" : "text-sm font-medium text-warn"}>
          {summary.completeness === "complete" ? "Complete" : "Partial history"}
        </span>
      </div>
      <dl className="mt-4 grid gap-4 text-sm sm:grid-cols-2 lg:grid-cols-3">
        <div><dt className="text-ink-faint">Readings</dt><dd className="tnum mt-1 text-ink">{summary.readings.toLocaleString()} <span className="text-ink-faint">({summary.reliable.toLocaleString()} usable)</span></dd></div>
        <div><dt className="text-ink-faint">Range (UTC)</dt><dd className="tnum mt-1 text-ink">{summary.range}</dd></div>
        <div><dt className="text-ink-faint">Cadence</dt><dd className="tnum mt-1 text-ink">{summary.cadence}</dd></div>
        <div><dt className="text-ink-faint">Gaps</dt><dd className="mt-1 text-ink">{summary.gaps}</dd></div>
        <div><dt className="text-ink-faint">Duplicate rows</dt><dd className="tnum mt-1 text-ink">{summary.duplicates.toLocaleString()}</dd></div>
        <div><dt className="text-ink-faint">Completeness</dt><dd className="mt-1 text-ink">{summary.completeness === "complete" ? "Available history was read" : "Some history was unavailable"}</dd></div>
      </dl>
    </section>
  );
}
