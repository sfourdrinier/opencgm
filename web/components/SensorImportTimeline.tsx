"use client";

import { useMemo, useState } from "react";
import type { ImportedSensorReading } from "@/lib/sensor/contracts";

type TimelinePoint = { readonly atMs: number; readonly value: number; readonly index: number };

function pointsFor(records: readonly ImportedSensorReading[]): TimelinePoint[] {
  return records
    .map((record, index) => ({ atMs: record.atMs, value: record.mgdl, index }))
    .filter((point): point is TimelinePoint => Number.isFinite(point.atMs) && point.value !== null && Number.isFinite(point.value))
    .sort((a, b) => a.atMs - b.atMs || a.index - b.index);
}

function runsFor(points: readonly TimelinePoint[]): TimelinePoint[][] {
  if (points.length < 2) return points.length ? [Array.from(points)] : [];
  const intervals = points.slice(1).flatMap((point, index) => {
    const previous = points[index];
    return previous === undefined ? [] : [point.atMs - previous.atMs];
  }).filter((interval) => interval > 0);
  const typical = intervals.length ? ([...intervals].sort((a, b) => a - b)[Math.floor(intervals.length / 2)] ?? intervals[0] ?? 5 * 60_000) : 5 * 60_000;
  const threshold = Math.max(typical * 1.8, 15 * 60_000);
  const runs: TimelinePoint[][] = [];
  let current: TimelinePoint[] = [];
  points.forEach((point, index) => {
    const previous = points[index - 1];
    if (previous !== undefined && point.atMs - previous.atMs > threshold && current.length) {
      runs.push(current);
      current = [];
    }
    current.push(point);
  });
  if (current.length) runs.push(current);
  return runs;
}

function formatTime(atMs: number): string {
  return new Intl.DateTimeFormat("en-GB", {
    timeZone: "UTC",
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(atMs);
}

export function SensorImportTimeline({ records }: { records: readonly ImportedSensorReading[] }) {
  const points = useMemo(() => pointsFor(records), [records]);
  const runs = useMemo(() => runsFor(points), [points]);
  const [hover, setHover] = useState<TimelinePoint | null>(null);
  const first = points[0]?.atMs ?? 0;
  const last = points[points.length - 1]?.atMs ?? first + 1;
  const span = Math.max(1, last - first);
  const observed = points.map((point) => point.value);
  const lo = Math.min(50, ...observed) - 8;
  const hi = Math.max(220, ...observed) + 8;
  const width = 900;
  const height = 230;
  const pad = { left: 46, right: 14, top: 14, bottom: 30 };
  const x = (atMs: number) => pad.left + ((atMs - first) / span) * (width - pad.left - pad.right);
  const y = (value: number) => pad.top + (1 - (value - lo) / (hi - lo)) * (height - pad.top - pad.bottom);
  const onPointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const atMs = first + Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)) * span;
    const nearest = points.reduce<TimelinePoint | null>((best, point) => {
      if (best === null || Math.abs(point.atMs - atMs) < Math.abs(best.atMs - atMs)) return point;
      return best;
    }, null);
    setHover(nearest);
  };

  return (
    <section className="border border-rule bg-paper-raised p-5" aria-labelledby="sensor-timeline-title">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 id="sensor-timeline-title" className="text-lg font-semibold text-ink">Sensor timeline</h2>
        <p className="text-xs text-ink-faint">Gaps remain gaps; no values are filled in.</p>
      </div>
      <div className="relative mt-4 touch-none" onPointerMove={onPointerMove} onPointerLeave={() => setHover(null)}>
        {hover ? (
          <div className="pointer-events-none absolute left-2 top-2 z-10 border border-rule-strong bg-paper-raised px-3 py-2 text-xs shadow-sm" role="status">
            <span className="tnum font-medium text-ink">{formatTime(hover.atMs)} UTC</span>
            <span className="ml-3 tnum text-ink-soft">{hover.value.toFixed(0)} mg/dL</span>
          </div>
        ) : null}
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full" role="img" aria-label="Glucose sensor timeline with gaps">
          <rect x={pad.left} y={y(180)} width={width - pad.left - pad.right} height={Math.max(0, y(70) - y(180))} fill="var(--color-accent-soft)" />
          {[70, 180].map((value) => <line key={value} x1={pad.left} x2={width - pad.right} y1={y(value)} y2={y(value)} stroke="var(--color-rule-strong)" strokeDasharray="4 4" />)}
          {runsFor(points).slice(0, -1).map((run, index) => {
            const next = runs[index + 1]?.[0];
            const end = run[run.length - 1];
            if (!next || !end) return null;
            return <rect key={`${end.atMs}-${next.atMs}`} x={x(end.atMs)} y={pad.top} width={Math.max(2, x(next.atMs) - x(end.atMs))} height={height - pad.top - pad.bottom} fill="var(--color-rule)" opacity={0.68} />;
          })}
          {runs.map((run) => {
            const firstPoint = run[0];
            if (!firstPoint) return null;
            return <path key={firstPoint.atMs} d={run.map((point, index) => `${index === 0 ? "M" : "L"}${x(point.atMs).toFixed(1)},${y(point.value).toFixed(1)}`).join(" ")} fill="none" stroke="var(--color-accent)" strokeWidth={2.4} strokeLinecap="round" strokeLinejoin="round" />;
          })}
          {hover ? <line x1={x(hover.atMs)} x2={x(hover.atMs)} y1={pad.top} y2={height - pad.bottom} stroke="var(--color-ink)" strokeWidth={1.2} opacity={0.5} /> : null}
          {hover ? <circle cx={x(hover.atMs)} cy={y(hover.value)} r={4} fill="var(--color-paper-raised)" stroke="var(--color-accent)" strokeWidth={2} /> : null}
          <text x={pad.left} y={height - 8} fontSize="11" fill="var(--color-ink-faint)">{points[0] ? formatTime(points[0].atMs) : "No dated readings"}</text>
          <text x={width - pad.right} y={height - 8} textAnchor="end" fontSize="11" fill="var(--color-ink-faint)">{points.length ? formatTime(points[points.length - 1]?.atMs ?? first) : ""}</text>
          {[70, 180].map((value) => <text key={`label-${value}`} x={pad.left - 8} y={y(value) + 4} textAnchor="end" fontSize="11" fill="var(--color-ink-faint)">{value}</text>)}
        </svg>
      </div>
    </section>
  );
}
