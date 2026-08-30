"use client";

import { useState, useSyncExternalStore } from "react";
import {
  Crosshair,
  HoverCard,
  slotTime,
  useChartHover,
} from "@/components/ChartHover";
import {
  HOUR_TICKS,
  gapSpans,
  hourLabel,
  pathFor,
  runsFromMask,
  runsFromNullable,
} from "@/lib/chart";
import {
  CalibrationStrip,
  ProbeDotPlot,
  ProbeFunnel,
  TypicalityAxis,
  WeekAtAGlance,
} from "@/components/ExampleCharts";
import {
  type ExampleAnalysis,
  cohortLabel,
  dayName,
  readWeek,
  taskLabel,
} from "@/lib/example";

// The worked example. Everything here was produced by running the released encoder over one
// real five-day recording; nothing is illustrative or drawn by hand.

function Trace({ values, height = 150 }: { values: (number | null)[]; height?: number }) {
  const { ref, hover, onMove, onLeave } = useChartHover();
  const w = 900;
  const padL = 40;
  const padB = 20;
  const obs = values.filter((v): v is number => v != null);
  const lo = Math.min(55, ...obs) - 6;
  const hi = Math.max(185, ...obs) + 6;
  const x = (i: number) => padL + (i / 287) * (w - padL - 8);
  const y = (v: number) => 8 + (1 - (v - lo) / (hi - lo)) * (height - 8 - padB);

  const runs = runsFromNullable(values);

  const hv = hover ? values[hover.index] : null;

  return (
    <div
      ref={ref}
      className="relative touch-none"
      onPointerMove={onMove}
      onPointerLeave={onLeave}
    >
      <HoverCard
        hover={hover}
        title={hover ? slotTime(hover.index) : ""}
        rows={[
          {
            label: "Glucose",
            value: hv == null ? "not recorded" : `${hv.toFixed(0)} mg/dL`,
            color: "var(--color-accent)",
          },
        ]}
      />
      <svg viewBox={`0 0 ${w} ${height}`} className="w-full" role="img" aria-label="Glucose trace">
      <rect x={padL} y={y(180)} width={w - padL - 8} height={Math.max(0, y(70) - y(180))} fill="var(--color-accent-soft)" />
      {[70, 180].map((v) => (
        <g key={v}>
          <line x1={padL} x2={w - 8} y1={y(v)} y2={y(v)} stroke="var(--color-rule-strong)" strokeDasharray="3 3" />
          <text x={padL - 6} y={y(v) + 4} textAnchor="end" fontSize="10" fill="var(--color-ink-faint)">{v}</text>
        </g>
      ))}
      {gapSpans((i) => values[i] != null).map((g) => (
        <rect
          key={g.start}
          x={x(g.start)}
          y={8}
          width={Math.max(1, x(g.end) - x(g.start) + (w - padL - 8) / 288)}
          height={height - 8 - padB}
          fill="var(--color-rule)"
          opacity={0.6}
        />
      ))}
      {HOUR_TICKS.map((i) => (
        <text key={i} x={x(i)} y={height - 5} textAnchor="middle" fontSize="10" fill="var(--color-ink-faint)">
          {hourLabel(i)}
        </text>
      ))}
      {runs.map((r) => (
        <path key={r[0]!.i} d={pathFor(r, x, y)} fill="none" stroke="var(--color-accent)" strokeWidth={1.6} />
      ))}
      {hover ? <Crosshair x={x(hover.index)} top={8} bottom={height - padB} /> : null}
      {hover && hv != null ? (
        <circle cx={x(hover.index)} cy={y(hv)} r={3} fill="var(--color-accent)" />
      ) : null}
      </svg>
    </div>
  );
}

function Streams({ state, event, mask }: { state: number[]; event: number[]; mask: number[] }) {
  const { ref, hover, onMove, onLeave } = useChartHover();
  const w = 900;
  const h = 110;
  const padL = 40;
  const mid = h / 2;
  const vals = state.concat(event).filter((_, i) => mask[i % 288]);
  const bound = Math.max(0.5, ...vals.map(Math.abs)) * 1.1;
  const x = (i: number) => padL + (i / 287) * (w - padL - 8);
  const y = (v: number) => mid - (v / bound) * (h / 2 - 8);
  const line = (s: number[], color: string, sw: number) =>
    runsFromMask(s, mask).map((r) => (
      <path key={r[0]!.i} d={pathFor(r, x, y)} fill="none" stroke={color} strokeWidth={sw} />
    ));
  const observed = hover ? mask[hover.index] === 1 : false;

  return (
    <div
      ref={ref}
      className="relative touch-none"
      onPointerMove={onMove}
      onPointerLeave={onLeave}
    >
      <HoverCard
        hover={hover}
        title={hover ? slotTime(hover.index) : ""}
        rows={
          !hover
            ? []
            : observed
              ? [
                  {
                    label: "Slow (state)",
                    value: state[hover.index]!.toFixed(2),
                    color: "var(--color-accent)",
                  },
                  {
                    label: "Fast (event)",
                    value: event[hover.index]!.toFixed(2),
                    color: "var(--color-low)",
                  },
                ]
              : [{ label: "", value: "not recorded" }]
        }
      />
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full" role="img" aria-label="State and event streams">
      <line x1={padL} x2={w - 8} y1={mid} y2={mid} stroke="var(--color-rule-strong)" />
      {line(event, "var(--color-low)", 1)}
      {line(state, "var(--color-accent)", 1.8)}
      {hover ? <Crosshair x={x(hover.index)} top={4} bottom={h - 4} /> : null}
      {hover && observed ? (
        <>
          <circle cx={x(hover.index)} cy={y(state[hover.index]!)} r={3} fill="var(--color-accent)" />
          <circle cx={x(hover.index)} cy={y(event[hover.index]!)} r={2.5} fill="var(--color-low)" />
        </>
      ) : null}
      </svg>
    </div>
  );
}

export function ExampleAnalysisView({
  data,
  owner = "third-party",
}: {
  data: ExampleAnalysis;
  /** "own" when the reader loaded the file themselves, which changes two sentences. */
  owner?: "third-party" | "own";
}) {
  const mine = owner === "own";
  // Open on the least typical complete day: it is the one with something to look at, and a
  // partial day would make the split view mostly empty.
  const initial = (() => {
    const w = readWeek(data);
    const i = w.leastTypical ? data.days.indexOf(w.leastTypical) : -1;
    return i >= 0 ? i : data.days.findIndex((d) => d.coverage >= 0.9);
  })();
  // A tile on the front page links here with ?day=<index>. Read the location as an external
  // snapshot so the query can seed state without synchronously setting state in an effect.
  const search = useSyncExternalStore(
    () => () => {},
    () => globalThis.location?.search ?? "",
    () => "",
  );
  const queryDay = new URLSearchParams(search).get("day");
  const queryIndex = queryDay == null ? Number.NaN : Number.parseInt(queryDay, 10);
  const [selected, setSelected] = useState(
    Number.isInteger(queryIndex) && queryIndex >= 0 && queryIndex < data.days.length
      ? queryIndex
      : initial < 0
        ? 0
        : initial,
  );
  const day = data.days[selected]!;

  const week = readWeek(data);
  const sparse = week.sparse[0];
  // The two illustrations for "the raw score is not a probability": the most saturated score
  // whose percentile is nonetheless unremarkable, and the score that reads high but ranks low.
  const scored = data.days.flatMap((d) =>
    d.probes.filter((pr) => pr.percentile != null).map((pr) => ({ day: d, probe: pr })),
  );
  const saturated = scored
    .filter((s) => s.probe.score > 0.95 && (s.probe.percentile ?? 100) < 75)
    .sort((a, b) => (a.probe.percentile ?? 0) - (b.probe.percentile ?? 0))[0];
  const misleadingLow = scored
    .filter((s) => s.probe.score > 0.4 && (s.probe.percentile ?? 100) < 25)
    .sort((a, b) => (a.probe.percentile ?? 0) - (b.probe.percentile ?? 0))[0];

  return (
    <div className="mt-8 space-y-12">
      {/* ------------------------------------------------------- 1. the week */}
      <section>
        <h2 className="text-lg font-semibold text-ink">
          1. {mine ? "Your days, as your app would show them" : "The week, as their own app would show it"}
        </h2>
        <p className="mt-2 max-w-3xl text-sm text-ink-soft">
          {data.n_days} {data.n_days === 1 ? "day" : "days"} on one shared scale.
          {mine ? " Only days with enough readings to score are shown." : ""} Average glucose
          between{" "}
          {Math.min(...data.days.map((d) => d.clinical?.mean_mg_dl ?? 0)).toFixed(0)} and{" "}
          {Math.max(...data.days.map((d) => d.clinical?.mean_mg_dl ?? 0)).toFixed(0)} mg/dL, and
          at least{" "}
          {(
            Math.min(...week.complete.map((d) => d.clinical?.time_in_range_70_180 ?? 1)) * 100
          ).toFixed(0)}
          % of every complete day inside the 70&ndash;180 band a clinician would target. By those numbers
          the days are interchangeable. Everything below is what the model adds on top of them.
        </p>
        <div className="mt-4">
          <WeekAtAGlance days={data.days} selected={selected} onSelect={setSelected} />
        </div>
        <p className="mt-2 text-xs text-ink-faint">
          Click a day to follow it through the rest of the page. Grey blocks are stretches the
          sensor did not record.
        </p>
      </section>

      {/* ------------------------------------------------- 2. the selected day */}
      <section>
        <h2 className="text-lg font-semibold text-ink">
          2. {day.weekday} in detail, and split in two
        </h2>
        <p className="mt-2 max-w-3xl text-sm text-ink-soft">
          Before the model reads anything it separates the slow drift through the day (green)
          from what is left over (rust). Meals and exercise land in the rust line. This split is
          the only part of the design that measurably matters: remove it and the model loses six
          times more accuracy than the noise between training runs.
        </p>
        <div className="mt-4 border border-rule bg-paper-raised p-4">
          <Trace values={day.values_mg_dl} />
          <div className="mt-2 border-t border-rule pt-2">
            <Streams state={day.state} event={day.event} mask={day.mask} />
          </div>
        </div>
        {day.clinical ? (
          <div className="mt-4 grid grid-cols-2 gap-x-8 gap-y-2 text-sm sm:grid-cols-4">
            {[
              ["Average", `${day.clinical.mean_mg_dl} mg/dL`],
              ["Variability (CV)", day.clinical.coefficient_of_variation.toFixed(3)],
              ["Time in 70-180", `${(day.clinical.time_in_range_70_180 * 100).toFixed(0)}%`],
              ["Range", `${day.clinical.min_mg_dl}-${day.clinical.max_mg_dl} mg/dL`],
            ].map(([k, v]) => (
              <div key={k}>
                <div className="tnum text-base text-ink">{v}</div>
                <div className="text-xs text-ink-faint">{k}</div>
              </div>
            ))}
          </div>
        ) : null}
      </section>

      {/* ------------------------------------------------------- 3. the vector */}
      <section>
        <h2 className="text-lg font-semibold text-ink">3. What the model actually returns</h2>
        <p className="mt-2 max-w-3xl text-sm text-ink-soft">
          This, and nothing else: 128 numbers. Every claim further down is a small classifier
          reading them — the model itself has never been told what diabetes is.
        </p>
        <div className="mt-4 flex h-14 items-center gap-[1px] border border-rule bg-paper-raised p-3">
          {day.embedding.map((v, i) => {
            const mag = Math.min(1, Math.abs(v) / 3);
            return (
              <div
                key={i}
                className="flex-1"
                style={{
                  height: `${10 + mag * 90}%`,
                  background: v >= 0 ? "var(--color-accent)" : "var(--color-low)",
                  opacity: 0.35 + mag * 0.65,
                }}
                title={`dimension ${i}: ${v.toFixed(3)}`}
              />
            );
          })}
        </div>
        <p className="mt-2 max-w-3xl text-xs text-ink-faint">
          To the eye the five days&rsquo; bar codes look nearly the same. The differences that
          matter are small and spread across many of the numbers, which is why the next section
          measures the distance between them instead of eyeballing it.
        </p>
      </section>

      {/* --------------------------------------------------- 4. which days alike */}
      <section>
        <h2 className="text-lg font-semibold text-ink">4. Which days were alike</h2>
        <p className="mt-2 max-w-3xl text-sm text-ink-soft">
          No classifier is involved here — this is the raw distance between the 128-number
          summaries. It is also the thing a CGM app does not do: it compares the shape of a
          whole day rather than its summary statistics.
        </p>
        <TypicalityAxis days={data.days} similarity={data.similarity} />
        <p className="mt-3 max-w-3xl text-sm text-ink-soft">
          {week.mostAlike ? (
            <>
              {dayName(week.mostAlike.a)} and {dayName(week.mostAlike.b)} read as near-twins to
              the model ({week.mostAlike.similarity.toFixed(3)}), despite peaks of{" "}
              {week.mostAlike.a.clinical?.max_mg_dl.toFixed(0)} and{" "}
              {week.mostAlike.b.clinical?.max_mg_dl.toFixed(0)} mg/dL.{" "}
            </>
          ) : null}
          {week.leastTypical ? (
            <>
              {dayName(week.leastTypical)} is the least like the other complete days, and the
              trace shows why: it climbs to{" "}
              {week.leastTypical.clinical?.max_mg_dl.toFixed(0)} mg/dL, and its variability is{" "}
              {week.leastTypical.clinical?.coefficient_of_variation.toFixed(2)} against{" "}
              {(
                week.complete
                  .filter((d) => d !== week.leastTypical)
                  .reduce((a, d) => a + (d.clinical?.coefficient_of_variation ?? 0), 0) /
                Math.max(1, week.complete.length - 1)
              ).toFixed(2)}{" "}
              for the rest. Time in range barely separates them.{" "}
            </>
          ) : null}
          {sparse ? (
            <>
              {dayName(sparse)} sits furthest out mostly because{" "}
              {(100 - sparse.coverage * 100).toFixed(0)}% of it is missing.
            </>
          ) : null}
        </p>
        <details className="mt-4">
          <summary className="cursor-pointer text-xs text-ink-faint hover:text-accent">
            Every pair, as a matrix
          </summary>
          <div className="mt-3 overflow-x-auto">
            <table className="border-collapse text-xs">
              <thead>
                <tr>
                  <th className="p-1" />
                  {data.days.map((d) => (
                    <th key={d.label} className="p-1 font-normal text-ink-faint">
                      {d.weekday.slice(0, 3)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.similarity.map((row, i) => (
                  <tr key={i}>
                    <td className="whitespace-nowrap p-1 pr-2 text-ink-faint">
                      {data.days[i]!.weekday.slice(0, 3)}
                    </td>
                    {row.map((s, j) => (
                      <td
                        key={j}
                        className="h-8 w-8 p-0"
                        title={`${data.days[i]!.weekday} vs ${data.days[j]!.weekday}: ${s.toFixed(3)}`}
                        style={{ background: `rgba(31, 95, 78, ${Math.max(0, (s - 0.85) / 0.15).toFixed(3)})` }}
                      />
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      </section>

      {/* ------------------------------------------------------ 5. classifiers */}
      <section>
        <h2 className="text-lg font-semibold text-ink">
          5. What the classifiers say about {day.weekday}
        </h2>
        <p className="mt-2 max-w-3xl text-sm text-ink-soft">
          The model itself answers no questions — it returns the 128 numbers above. The
          questions come from {data.heads_shipped} classifiers fitted on top of them, covering
          seven subjects in all: insulin resistance, diabetes risk, blood lipids, body-mass
          category, glucose variability subtype, insulin production, and low-glucose episodes.
          Fewer than seven get answered on any one day, and which ones depends on how much of
          the day the sensor recorded.
        </p>
        <ProbeFunnel
          shipped={data.heads_shipped}
          withSignal={data.heads_with_signal}
          applicable={day.probes.length}
          questions={new Set(day.probes.map((p) => taskLabel(p.task))).size}
        />
        <p className="mt-4 max-w-3xl text-sm text-ink-soft">
          {data.heads_shipped - data.heads_with_signal} of the eighteen never beat chance in
          cross-validation and are switched off — one of them,{" "}
          {data.heads_below_floor[0]
            ? `${cohortLabel(data.heads_below_floor[0].cohort)} ${taskLabel(data.heads_below_floor[0].task).toLowerCase()}`
            : "the weakest"}
          , scores{" "}
          <span className="tnum">{data.heads_below_floor[0]?.roc_auc.toFixed(2)}</span>, which is
          worse than guessing. The rest are filtered by how much of the day was recorded. What
          remains is drawn below: each dot is one classifier placing this day against{" "}
          {data.reference_windows.toLocaleString()} days from the training data. Filled dots are
          the more accurate ones. Hover a dot for its cohort and accuracy.
        </p>
        {day.probes.length === 0 ? (
          <p className="mt-4 text-sm text-ink-soft">
            No classifier will score this day.
          </p>
        ) : (
          <>
            <ProbeDotPlot probes={day.probes} />
            <p className="mt-5 max-w-3xl text-sm text-ink-soft">
              The spread within a row is the finding, not a flaw in the drawing. Three
              classifiers were trained to answer &ldquo;diabetes risk&rdquo; on three different
              cohorts, and they place the same day at very different points. Averaged over every
              question and cohort the model beats a strong hand-built baseline by a small,
              repeatable margin. On any single question, the disagreement you can see here is
              why no individual result survives statistical correction. Read the rows together,
              never one at a time.
            </p>
            <details className="mt-4">
              <summary className="cursor-pointer text-xs text-ink-faint hover:text-accent">
                The same rows as a table, with cohort sizes and accuracies
              </summary>
              <div className="mt-3 overflow-x-auto">
                <table className="w-full min-w-[38rem] border-collapse text-sm">
                  <thead>
                    <tr className="border-b border-rule-strong text-left text-ink-soft">
                      <th className="py-2 pr-4 font-medium">Question</th>
                      <th className="py-2 pr-4 font-medium">Trained on</th>
                      <th className="py-2 pr-4 font-medium">Percentile</th>
                      <th className="py-2 font-medium">Accuracy</th>
                    </tr>
                  </thead>
                  <tbody>
                    {day.probes.map((p) => (
                      <tr key={p.key} className="border-b border-rule">
                        <td className="py-2 pr-4 text-ink">{taskLabel(p.task)}</td>
                        <td className="py-2 pr-4 text-xs text-ink-faint">
                          {cohortLabel(p.cohort)}, {p.n_subjects} people
                        </td>
                        <td className="tnum py-2 pr-4 text-ink">
                          {p.percentile == null ? "—" : `${Math.round(p.percentile)}th`}
                        </td>
                        <td className="tnum py-2 text-xs text-ink-faint">
                          {p.roc_auc.toFixed(2)} ROC-AUC
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          </>
        )}
      </section>

      {/* ------------------------------------------- 6. why no percentages */}
      {saturated || misleadingLow ? (
        <section className="border border-warn/30 bg-warn-soft/50 p-6">
          <h2 className="text-lg font-semibold text-ink">
            6. Why this page never shows you a percentage
          </h2>
          <p className="mt-2 max-w-3xl text-sm text-ink-soft">
            The raw output of these classifiers is a number between 0 and 1, and it is tempting
            to read it as a probability. Here is what that number is actually made of. Each bar
            below is 5% of the {data.reference_windows.toLocaleString()} reference days; where
            the bars are wide, corpus days are spread out, and where they are invisible,
            thousands of days share almost the same score.
          </p>

          {saturated ? (
            <div className="mt-6">
              <p className="text-sm font-medium text-ink">
                {taskLabel(saturated.probe.task)} on {dayName(saturated.day)}: raw score{" "}
                <span className="tnum">{saturated.probe.score.toFixed(3)}</span> — which an app would
                show as &ldquo;{(saturated.probe.score * 100).toFixed(1)}%&rdquo;
              </p>
              <div className="mt-3">
                <CalibrationStrip probe={saturated.probe} />
              </div>
              <p className="mt-2 max-w-3xl text-sm text-ink-soft">
                Most of the reference corpus scores at least that high. The day is
                unremarkable: the {Math.round(saturated.probe.percentile ?? 0)}th percentile.
              </p>
            </div>
          ) : null}

          {misleadingLow ? (
            <div className="mt-8">
              <p className="text-sm font-medium text-ink">
                {taskLabel(misleadingLow.probe.task)} on {dayName(misleadingLow.day)}: raw score{" "}
                <span className="tnum">{misleadingLow.probe.score.toFixed(3)}</span> — reads like
                &ldquo;{(misleadingLow.probe.score * 100).toFixed(0)}%&rdquo;
              </p>
              <div className="mt-3">
                <CalibrationStrip probe={misleadingLow.probe} />
              </div>
              <p className="mt-2 max-w-3xl text-sm text-ink-soft">
                Most reference days score higher. It is the{" "}
                {Math.round(misleadingLow.probe.percentile ?? 0)}th percentile.
              </p>
            </div>
          ) : null}

          <p className="mt-8 max-w-3xl text-sm text-ink-soft">
            These classifiers push almost everything toward 0 or 1 regardless of the input. The
            ranking carries information; the raw number does not. If a product ever shows you a
            percentage built from a day of glucose data, this is what is underneath it.
          </p>
        </section>
      ) : null}

      {/* ------------------------------------------- 7. the day it declined */}
      {(() => {
        const partial = week.sparse;
        if (!partial.length) return null;
        const fewest = partial.reduce((a, b) => (a.probes.length <= b.probes.length ? a : b));
        const most = partial.reduce((a, b) => (a.probes.length >= b.probes.length ? a : b));
        const inversion =
          fewest !== most && fewest.coverage > most.coverage && fewest.probes.length === 0;

        return (
          <section>
            <h2 className="text-lg font-semibold text-ink">
              7. The days the model would not score
            </h2>
            <p className="mt-2 max-w-3xl text-sm text-ink-soft">
              {dayName(fewest)} recorded{" "}
              <span className="tnum">{(fewest.coverage * 100).toFixed(0)}%</span> of the day and{" "}
              {fewest.probes.length === 0
                ? "no classifier would score it at all"
                : `only ${fewest.probes.length} classifiers would score it`}
              . The rule is mechanical: a classifier only scores days whose coverage falls
              inside the range it saw during its own training, so it is never asked to
              extrapolate to a recording unlike anything it learned from.
            </p>
            {inversion ? (
              <p className="mt-3 max-w-3xl text-sm text-ink-soft">
                This produces a result that looks backwards and is not.{" "}
                {dayName(most)} recorded{" "}
                <span className="tnum">{(most.coverage * 100).toFixed(0)}%</span> — less than{" "}
                {dayName(fewest)} — and {most.probes.length} classifiers accepted it. The
                classifiers do not share one threshold. Those fitted on intermittently scanned
                sensors accept days between roughly 14% and 48% coverage; those fitted on
                near-continuous recordings want 59% or more. {dayName(fewest)} falls in the gap
                between the two, so nothing claims it. A single &ldquo;enough data?&rdquo; cutoff
                would have hidden that; per-classifier bands make it visible.
              </p>
            ) : null}
            <p className="mt-3 max-w-3xl text-sm text-ink-soft">
              A tool that returned a confident number for these days would be inventing one.
            </p>
          </section>
        );
      })()}
    </div>
  );
}
