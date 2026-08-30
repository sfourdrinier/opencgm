"use client";

import { HOUR_TICKS, gapSpans, hourLabel, pathFor, runsFromMask } from "@/lib/chart";
import type { Window } from "@/lib/types";

// A 24-hour glucose trace drawn as SVG.
//
// The one rule this chart exists to honour: a gap in the data is drawn as a gap. Charting
// libraries connect across missing points by default, which quietly turns "not recorded"
// into a confident straight line -- the same mistake the model itself refuses to make.

const LOW = 70;
const HIGH = 180;

export function DayChart({ window, height = 220 }: { window: Window; height?: number }) {
  const width = 900;
  const padL = 44;
  const padR = 12;
  const padT = 12;
  const padB = 26;

  const observed: number[] = [];
  for (let i = 0; i < 288; i += 1) {
    if (window.mask[i]) observed.push(window.values[i]!);
  }
  const lo = Math.min(50, ...observed) - 8;
  const hi = Math.max(220, ...observed) + 8;

  const x = (i: number) => padL + (i / 287) * (width - padL - padR);
  const y = (v: number) => padT + (1 - (v - lo) / (hi - lo)) * (height - padT - padB);

  const runs = runsFromMask(window.values, window.mask);

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="w-full"
      role="img"
      aria-label="24-hour glucose trace"
    >
      {/* target band */}
      <rect
        x={padL}
        y={y(HIGH)}
        width={width - padL - padR}
        height={Math.max(0, y(LOW) - y(HIGH))}
        fill="var(--color-accent-soft)"
      />
      <line x1={padL} x2={width - padR} y1={y(HIGH)} y2={y(HIGH)} stroke="var(--color-rule-strong)" strokeDasharray="3 3" />
      <line x1={padL} x2={width - padR} y1={y(LOW)} y2={y(LOW)} stroke="var(--color-rule-strong)" strokeDasharray="3 3" />

      {/* y labels */}
      {[LOW, HIGH].map((v) => (
        <text key={v} x={padL - 8} y={y(v) + 4} textAnchor="end" fontSize="11" fill="var(--color-ink-faint)">
          {v}
        </text>
      ))}

      {/* hour ticks every 6 h */}
      {HOUR_TICKS.map((i) => (
        <g key={i}>
          <line x1={x(i)} x2={x(i)} y1={height - padB} y2={height - padB + 4} stroke="var(--color-rule-strong)" />
          <text x={x(i)} y={height - 8} textAnchor="middle" fontSize="11" fill="var(--color-ink-faint)">
            {hourLabel(i)}:00
          </text>
        </g>
      ))}

      {/* missing spans, shown explicitly */}
      {gapSpans((i) => Boolean(window.mask[i])).map((g) => (
        <rect
          key={g.start}
          x={x(g.start)}
          y={padT}
          width={Math.max(1, x(g.end) - x(g.start))}
          height={height - padT - padB}
          fill="var(--color-rule)"
          opacity={0.55}
        />
      ))}

      {/* the trace, one path per observed run */}
      {runs.map((run) => (
        <path
          key={run[0]!.i}
          d={pathFor(run, x, y)}
          fill="none"
          stroke="var(--color-accent)"
          strokeWidth={1.8}
          strokeLinejoin="round"
          strokeLinecap="round"
        />
      ))}
    </svg>
  );
}
