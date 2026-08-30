"use client";

import Link from "next/link";
import { gapSpans, pathFor, runsFromMask, runsFromNullable } from "@/lib/chart";
import type { ExampleDay, ExampleProbe } from "@/lib/example";
import { cohortLabel, taskLabel } from "@/lib/example";

// Charts specific to the worked example. Kept apart from the generic chart components so the
// demo and the example can diverge without either dragging the other around.

const LO = 55;
const HI = 225;

/**
 * The week, five days side by side on one shared scale.
 *
 * A shared y-axis is the whole point. Auto-scaling each day to its own range makes a 143 mg/dL
 * day and a 218 mg/dL day fill the same frame, which hides the only difference between them a
 * reader can see unaided.
 */
export function WeekAtAGlance({
  days,
  selected,
  onSelect,
  secondStat,
  href,
}: {
  days: ExampleDay[];
  /** Omit both to render read-only, which is how the front page uses it. */
  selected?: number;
  onSelect?: (i: number) => void;
  /** Second figure under each day. TIR is the one that makes the days look alike. */
  secondStat?: "peak" | "tir";
  /** When set, each tile links to `${href}?day=<index>` instead of selecting in place. */
  href?: string;
}) {
  const interactive = typeof onSelect === "function";
  const stat = secondStat ?? "peak";
  const w = 220;
  const h = 110;
  const padB = 4;
  const x = (i: number) => (i / 287) * w;
  const y = (v: number) => 4 + (1 - (v - LO) / (HI - LO)) * (h - 4 - padB);

  return (
    <div className="grid gap-3 grid-cols-2 sm:grid-cols-3 lg:grid-cols-6">
      {days.map((d, di) => {
        const runs = runsFromNullable(d.values_mg_dl);

        return (
          <Cell
            key={d.label}
            interactive={interactive}
            selected={di === selected}
            onSelect={() => onSelect?.(di)}
            href={href ? `${href}?day=${di}` : undefined}
            label={`${d.weekday}, ${(d.coverage * 100).toFixed(0)}% recorded`}
          >
            <div className="flex items-baseline justify-between">
              <span className="text-sm font-medium text-ink">{d.weekday.slice(0, 3)}</span>
              <span className="tnum text-xs text-ink-faint">
                {(d.coverage * 100).toFixed(0)}%
              </span>
            </div>
            <svg viewBox={`0 0 ${w} ${h}`} className="mt-1 w-full" aria-hidden="true">
              <rect x={0} y={y(180)} width={w} height={Math.max(0, y(70) - y(180))} fill="var(--color-accent-soft)" />
              {gapSpans((i) => d.values_mg_dl[i] != null).map((g) => (
                <rect
                  key={g.start}
                  x={x(g.start)}
                  y={4}
                  width={Math.max(0.8, x(g.end) - x(g.start) + w / 288)}
                  height={h - 4 - padB}
                  fill="var(--color-rule)"
                  opacity={0.7}
                />
              ))}
              {runs.map((r) => (
                <path
                  key={r[0]!.i}
                  d={pathFor(r, x, y)}
                  fill="none"
                  stroke="var(--color-accent)"
                  strokeWidth={1.2}
                />
              ))}
            </svg>
            <div className="tnum mt-1 flex justify-between text-xs text-ink-faint">
              <span>avg {d.clinical?.mean_mg_dl.toFixed(0)}</span>
              <span>
                {stat === "tir"
                  ? `in range ${((d.clinical?.time_in_range_70_180 ?? 0) * 100).toFixed(0)}%`
                  : `peak ${d.clinical?.max_mg_dl.toFixed(0)}`}
              </span>
            </div>
          </Cell>
        );
      })}
    </div>
  );
}

/** A day tile: a link, a button, or plain markup, depending on what the caller wants. */
function Cell({
  interactive,
  selected,
  onSelect,
  href,
  label,
  children,
}: {
  interactive: boolean;
  selected: boolean;
  onSelect: () => void;
  href?: string;
  label: string;
  children: React.ReactNode;
}) {
  const cls = `block border p-2 text-left transition ${
    selected ? "border-accent bg-accent-soft/40" : "border-rule bg-paper-raised"
  }${interactive || href ? " hover:border-accent" : ""}`;
  if (href) {
    return (
      <Link href={href} className={cls} aria-label={`See ${label} in the worked example`}>
        {children}
      </Link>
    );
  }
  if (!interactive) return <div className={cls}>{children}</div>;
  return (
    <button type="button" onClick={onSelect} className={cls}>
      {children}
    </button>
  );
}

/** Each day placed by how much it resembles the rest of the week. */
export function TypicalityAxis({ days, similarity }: { days: ExampleDay[]; similarity: number[][] }) {
  const points = days
    .map((d, i) => {
      const others = similarity[i]!.filter((_, j) => j !== i);
      return { day: d, mean: others.reduce((a, b) => a + b, 0) / others.length };
    })
    .sort((a, b) => b.mean - a.mean);

  const lo = Math.min(...points.map((p) => p.mean)) - 0.008;
  const hi = Math.max(...points.map((p) => p.mean)) + 0.004;
  const pos = (m: number) => ((hi - m) / (hi - lo)) * 100;

  // Days with near-identical scores land on top of each other, which made the labels
  // unreadable. Anything within 7% of the axis width goes on the next tier down.
  const tiers: number[] = [];
  const placed: { x: number; tier: number }[] = [];
  for (const p of points) {
    const x = pos(p.mean);
    let tier = 0;
    while (placed.some((q) => q.tier === tier && Math.abs(q.x - x) < 7)) tier += 1;
    placed.push({ x, tier });
    tiers.push(tier);
  }
  const maxTier = Math.max(...tiers);
  const rowH = 17;

  return (
    <div className="mt-4">
      <div className="relative" style={{ height: 34 + (maxTier + 1) * rowH }}>
        <div className="absolute left-0 right-0 top-3 h-px bg-rule-strong" />
        {points.map((p, i) => {
          const x = Math.max(3, Math.min(97, pos(p.mean)));
          return (
            <div key={p.day.label} className="absolute" style={{ left: `${x}%`, top: 0 }}>
              <span className="block h-2.5 w-2.5 -translate-x-1/2 translate-y-1.5 rounded-full bg-accent" />
              <span
                className="absolute block -translate-x-1/2 whitespace-nowrap text-center"
                style={{ top: 18 + tiers[i]! * rowH }}
              >
                <span className="block text-xs text-ink-soft">{p.day.weekday.slice(0, 3)}</span>
                <span className="tnum block text-[10px] text-ink-faint">{p.mean.toFixed(3)}</span>
              </span>
              {tiers[i]! > 0 ? (
                <span
                  className="absolute left-0 w-px -translate-x-1/2 bg-rule-strong"
                  style={{ top: 14, height: 6 + tiers[i]! * rowH }}
                />
              ) : null}
            </div>
          );
        })}
      </div>
      <div className="flex justify-between text-xs text-ink-faint">
        <span>most like the rest of the week</span>
        <span>least like it</span>
      </div>
    </div>
  );
}

/** One rail per question, one dot per cohort's classifier. The spread is the point. */
export function ProbeDotPlot({ probes }: { probes: ExampleProbe[] }) {
  const byTask = new Map<string, ExampleProbe[]>();
  for (const p of probes) {
    const label = taskLabel(p.task);
    byTask.set(label, [...(byTask.get(label) ?? []), p]);
  }

  return (
    <div className="mt-4 space-y-4">
      {[...byTask.entries()].map(([label, ps]) => (
        <div key={label} className="grid grid-cols-1 gap-1 sm:grid-cols-[13rem_1fr] sm:gap-4">
          <div className="text-sm text-ink">
            {label}
            <span className="ml-2 text-xs text-ink-faint">
              {ps.length} classifier{ps.length === 1 ? "" : "s"}
            </span>
          </div>
          <div className="relative h-9">
            <div className="absolute left-0 right-0 top-4 h-1.5 bg-rule/70" />
            <div className="absolute top-2.5 left-1/2 h-4.5 w-px bg-rule-strong" />
            {ps.map((p) =>
              p.percentile == null ? null : (
                <div
                  key={p.key}
                  className="absolute -translate-x-1/2"
                  style={{ left: `${Math.max(1, Math.min(99, p.percentile))}%`, top: 6 }}
                  title={`${cohortLabel(p.cohort)} · ${p.n_subjects} people · ROC-AUC ${p.roc_auc.toFixed(2)} · ${p.percentile.toFixed(0)}th percentile`}
                >
                  <span
                    className="block h-3 w-3 rounded-full border-2"
                    style={{
                      borderColor: "var(--color-accent)",
                      background: p.roc_auc >= 0.75 ? "var(--color-accent)" : "transparent",
                    }}
                  />
                  <span className="mt-0.5 block -translate-x-1/2 whitespace-nowrap text-[10px] text-ink-faint" style={{ marginLeft: "50%" }}>
                    {cohortLabel(p.cohort)}
                  </span>
                </div>
              ),
            )}
          </div>
        </div>
      ))}
      <div className="grid grid-cols-1 gap-1 text-xs text-ink-faint sm:grid-cols-[13rem_1fr] sm:gap-4">
        <div />
        <div className="flex justify-between">
          <span>lower than most corpus days</span>
          <span>higher than most</span>
        </div>
      </div>
    </div>
  );
}

/**
 * Where the corpus scores actually pile up, and where this day landed among them.
 *
 * This is the picture that makes "the raw score is not a probability" obvious. For most of
 * these classifiers the corpus scores bunch at 0 and at 1 with very little in between, so a
 * score of 0.996 can sit barely above the middle of the pile.
 */
export function CalibrationStrip({ probe }: { probe: ExampleProbe }) {
  if (!probe.corpus_deciles) return null;
  const d = probe.corpus_deciles;
  // Each segment spans 5 percentiles of the corpus; its width on a 0-1 score axis shows how
  // thinly the corpus is spread there.
  return (
    <div>
      <div className="flex h-7 w-full overflow-hidden border border-rule">
        {d.slice(0, -1).map((v, i) => {
          const width = Math.max(0, d[i + 1]! - v) * 100;
          return (
            <div
              key={i}
              className="h-full"
              style={{
                width: `${width}%`,
                background: "var(--color-accent)",
                opacity: 0.15 + (i / d.length) * 0.55,
              }}
              title={`${i * 5}th-${(i + 1) * 5}th percentile of corpus days: score ${v.toFixed(3)}-${d[i + 1]!.toFixed(3)}`}
            />
          );
        })}
      </div>
      <div className="relative mt-1 h-6">
        <div
          className="absolute -translate-x-1/2 text-center"
          style={{ left: `${Math.max(2, Math.min(98, probe.score * 100))}%` }}
        >
          <span className="block h-3 w-px bg-ink" style={{ marginLeft: "50%" }} />
          <span className="tnum whitespace-nowrap text-[10px] text-ink">
            this day {probe.score.toFixed(3)}
          </span>
        </div>
      </div>
      <div className="tnum flex justify-between text-[10px] text-ink-faint">
        <span>score 0</span>
        <span>score 1</span>
      </div>
    </div>
  );
}


/**
 * How 18 shipped classifiers become the handful of rows above.
 *
 * Without this the page shows five rows and a reader reasonably assumes five classifiers.
 * Three of the rows carry three each, four of the eighteen never clear chance, and several
 * only accept sparse recordings. Each narrowing is a real editorial decision and worth seeing.
 */
export function ProbeFunnel({
  shipped,
  withSignal,
  applicable,
  questions,
}: {
  shipped: number;
  withSignal: number;
  applicable: number;
  questions: number;
}) {
  const steps = [
    { n: shipped, label: "classifiers shipped" },
    { n: withSignal, label: "better than chance" },
    { n: applicable, label: "accept this day" },
    { n: questions, label: "distinct questions" },
  ];
  return (
    <div className="mt-4 flex flex-wrap items-stretch gap-px border border-rule bg-rule">
      {steps.map((s, i) => (
        <div key={s.label} className="flex-1 bg-paper-raised px-4 py-3">
          <div className="tnum text-xl font-semibold text-ink">
            {s.n}
            {i > 0 ? (
              <span className="ml-1.5 align-middle text-xs font-normal text-ink-faint">
                of {steps[i - 1]!.n}
              </span>
            ) : null}
          </div>
          <div className="mt-0.5 text-xs text-ink-soft">{s.label}</div>
        </div>
      ))}
    </div>
  );
}


/**
 * One day's slow and fast streams, drawn small and without interaction.
 *
 * Used on the front page beside the prose that describes the decomposition. The streams are
 * precomputed in the example JSON, so this costs a visitor nothing beyond the bytes already
 * loaded for the week strip.
 */
export function StreamsThumb({ day }: { day: ExampleDay }) {
  const w = 520;
  const h = 90;
  const mid = h / 2;
  const observed = day.state.filter((_, i) => day.mask[i]);
  const bound = Math.max(0.5, ...observed.map(Math.abs), ...day.event.filter((_, i) => day.mask[i]).map(Math.abs)) * 1.1;
  const x = (i: number) => (i / 287) * w;
  const y = (v: number) => mid - (v / bound) * (mid - 6);
  const line = (s: number[], color: string, sw: number) =>
    runsFromMask(s, day.mask).map((r) => (
      <path key={r[0]!.i} d={pathFor(r, x, y)} fill="none" stroke={color} strokeWidth={sw} />
    ));
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full" role="img" aria-label="Slow and fast streams for one day">
      <line x1={0} x2={w} y1={mid} y2={mid} stroke="var(--color-rule-strong)" />
      {line(day.event, "var(--color-low)", 1)}
      {line(day.state, "var(--color-accent)", 1.8)}
    </svg>
  );
}
