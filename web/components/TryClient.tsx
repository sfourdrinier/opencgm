"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { CsvFormat } from "@/components/CsvFormat";
import { DayChart } from "@/components/DayChart";
import { ExampleAnalysisView } from "@/components/ExampleAnalysis";
import { buildWindow, coverageOf } from "@/lib/csv/grid";
import { parseCgmCsv } from "@/lib/csv/parse";
import { type LiveDay, buildAnalysis } from "@/lib/analysis";
import type { ExampleAnalysis } from "@/lib/example";
import { loadHeadsBundle } from "@/lib/model/heads";
import { decompose, embedBatch } from "@/lib/model/loadOnnx";
import { type Reference, loadReference } from "@/lib/model/rank";
import { PROFILES, buildSampleDay } from "@/lib/sample/days";
import type { HeadsBundle, Reading, Window } from "@/lib/types";

// The demo used to give a thinner analysis than the worked example: one day, a probe table,
// and a separate similarity panel. It now builds the same object /example is generated into
// and hands it to the same view, so someone running their own export sees exactly what the
// example promised them -- every day, the decomposition, the rankings, the refusals.

const SAMPLE_DAY_START = Date.UTC(2026, 0, 15, 0, 0, 0);
const STEP_MS = 5 * 60 * 1000;
const N = 288;

/** Days a classifier could plausibly score. Below this the analysis is mostly gaps. */
const MIN_COVERAGE = 0.25;

/**
 * How many days to analyse from one file.
 *
 * The streams model runs once per day, in WASM, so a 90-day export would mean ninety
 * sequential inferences and a page that appears to hang. Two weeks is enough to see whether
 * weekdays differ from weekends, which is the question multiple days are useful for, and
 * completes in a few seconds. The most recent days are kept: someone looking at their own
 * data is usually asking about now.
 */
const MAX_DAYS = 14;

type Status = "idle" | "loading-model" | "running" | "done" | "error";

/** Local-midnight starts for every day an upload covers, oldest first. */
function availableDays(readings: Reading[]): number[] {
  const starts = new Set<number>();
  for (const r of readings) {
    const d = new Date(r.t);
    d.setHours(0, 0, 0, 0);
    starts.add(d.getTime());
  }
  return [...starts].sort((a, b) => a - b);
}

function coverageAt(readings: Reading[], startT: number): number {
  const seen = new Set<number>();
  for (const r of readings) {
    const idx = Math.floor((r.t - startT) / STEP_MS);
    if (idx >= 0 && idx < N) seen.add(idx);
  }
  return seen.size / N;
}

/** Everything the run produced, as one file, for taking into a notebook. */
function downloadAnalysis(analysis: ExampleAnalysis) {
  const blob = new Blob([JSON.stringify(analysis, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `opencgm-analysis-${new Date().toISOString().slice(0, 10)}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

export function TryClient() {
  const [profileId, setProfileId] = useState(PROFILES[0]!.id);
  const [readings, setReadings] = useState<Reading[] | null>(null);
  const [dayStarts, setDayStarts] = useState<number[]>([SAMPLE_DAY_START]);
  const [sourceLabel, setSourceLabel] = useState<string>(PROFILES[0]!.label);
  const [status, setStatus] = useState<Status>("idle");
  const [progress, setProgress] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<ExampleAnalysis | null>(null);
  const [elapsedMs, setElapsedMs] = useState<number | null>(null);
  const [dragging, setDragging] = useState(false);
  /** Scoreable days beyond MAX_DAYS that were left out, so the page can say so. */
  const [trimmed, setTrimmed] = useState(0);

  const bundleRef = useRef<HeadsBundle | null>(null);
  const referenceRef = useRef<Reference | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  // A simulated day so the page is useful with no interaction and no file.
  useEffect(() => {
    const profile = PROFILES.find((p) => p.id === profileId) ?? PROFILES[0]!;
    setReadings(buildSampleDay(profile, SAMPLE_DAY_START));
    setDayStarts([SAMPLE_DAY_START]);
    setTrimmed(0);
    setSourceLabel(profile.label);
    setAnalysis(null);
    setError(null);
  }, [profileId]);

  const usable = useMemo(() => {
    if (!readings) return [];
    const scoreable = dayStarts.filter((s) => coverageAt(readings, s) >= MIN_COVERAGE);
    return scoreable.slice(-MAX_DAYS);
  }, [readings, dayStarts]);

  // The first usable day, drawn before anything runs so the reader sees what will be read.
  const preview: Window | null = useMemo(
    () => (readings && usable.length ? buildWindow(readings, usable[0]!) : null),
    [readings, usable],
  );

  const run = useCallback(
    async (rs: Reading[], starts: number[]) => {
      setError(null);
      setAnalysis(null);
      setStatus("loading-model");
      const t0 = performance.now();
      try {
        if (!bundleRef.current) bundleRef.current = await loadHeadsBundle();
        if (!referenceRef.current) referenceRef.current = await loadReference();

        setStatus("running");
        const windows = starts.map((s) => buildWindow(rs, s));

        setProgress(`Encoding ${windows.length} day${windows.length === 1 ? "" : "s"}…`);
        const embeddings = await embedBatch(windows);

        // The streams come from a second graph, fetched once and then run per day.
        const streams: (Awaited<ReturnType<typeof decompose>> | null)[] = [];
        for (let i = 0; i < windows.length; i += 1) {
          setProgress(`Splitting day ${i + 1} of ${windows.length}…`);
          try {
            streams.push(await decompose(windows[i]!));
          } catch {
            streams.push(null);
          }
        }

        const live: LiveDay[] = windows.map((w, i) => ({
          window: w,
          embedding: embeddings[i]!,
          streams: streams[i] ?? null,
          startT: starts[i]!,
        }));

        setAnalysis(buildAnalysis(live, bundleRef.current, referenceRef.current));
        setElapsedMs(performance.now() - t0);
        setStatus("done");
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        setStatus("error");
      } finally {
        setProgress("");
      }
    },
    [],
  );

  const onFile = useCallback(
    async (file: File) => {
      setError(null);
      setAnalysis(null);
      try {
        const tz = Intl.DateTimeFormat().resolvedOptions().timeZone;
        const parsed = await parseCgmCsv(file, tz);
        if (!parsed.readings.length) {
          setError("No usable rows. The file needs a time column and a glucose column.");
          return;
        }
        const starts = availableDays(parsed.readings);
        const scoreable = starts.filter((s) => coverageAt(parsed.readings, s) >= MIN_COVERAGE);
        const keep = scoreable.slice(-MAX_DAYS);
        setReadings(parsed.readings);
        setDayStarts(starts);
        setTrimmed(scoreable.length - keep.length);
        setSourceLabel(
          `${file.name} — ${parsed.readings.length.toLocaleString()} readings over ${starts.length} day${starts.length === 1 ? "" : "s"}`,
        );
        if (keep.length === 0) {
          setError(
            `None of the ${starts.length} days in this file has at least ${MIN_COVERAGE * 100}% of its readings.`,
          );
          return;
        }
        // Run immediately, over everything. Making someone press a second button to see the
        // thing they came for is friction with nothing on the other side of it.
        void run(parsed.readings, keep);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    },
    [run],
  );

  const coverage = preview ? coverageOf(preview.mask) : 0;
  const busy = status === "loading-model" || status === "running";

  return (
    <div className="mt-8 space-y-6">
      {/* ------------------------------------------------------------- load */}
      <section className="border border-rule bg-paper-raised p-6">
        <h2 className="text-lg font-semibold text-ink">Load a day of readings</h2>

        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            const f = e.dataTransfer.files?.[0];
            if (f) void onFile(f);
          }}
          className={`mt-4 border-2 border-dashed px-6 py-8 text-center transition ${
            dragging ? "border-accent bg-accent-soft/50" : "border-rule-strong bg-paper"
          }`}
        >
          <p className="text-base font-medium text-ink">Drop your CGM export here</p>
          <p className="mx-auto mt-2 max-w-xl text-sm text-ink-soft">
            A CSV from Dexcom Clarity, FreeStyle Libre, or anything with a time column and a
            glucose column. Every day in the file is analysed, and it is all read in this tab —
            nothing is uploaded.
          </p>
          <button
            type="button"
            onClick={() => fileRef.current?.click()}
            className="mt-4 bg-accent px-5 py-2.5 text-sm font-medium text-white hover:bg-accent-ink"
          >
            Choose a file
          </button>
          <input
            ref={fileRef}
            type="file"
            accept=".csv,text/csv"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) void onFile(f);
            }}
          />
        </div>

        <div className="mt-6 border-t border-rule pt-5">
          <p className="text-sm text-ink-soft">
            No sensor of your own? Run one of these simulated days instead.
          </p>
          <div className="mt-3 flex flex-wrap gap-2">
            {PROFILES.map((pr) => (
              <button
                key={pr.id}
                type="button"
                onClick={() => setProfileId(pr.id)}
                className={`border px-4 py-2 text-sm transition ${
                  profileId === pr.id && sourceLabel === pr.label
                    ? "border-accent bg-accent-soft text-accent-ink"
                    : "border-rule-strong text-ink-soft hover:border-accent hover:text-accent"
                }`}
              >
                {pr.label}
              </button>
            ))}
          </div>
          <p className="mt-2 text-xs text-ink-faint">
            {PROFILES.find((pr) => pr.id === profileId && sourceLabel === pr.label)?.blurb ??
              sourceLabel}
          </p>
        </div>
      </section>

      <CsvFormat />

      {/* --------------------------------------------------------- what runs */}
      {preview && !analysis ? (
        <section className="border border-rule bg-paper-raised p-6">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <h2 className="text-lg font-semibold text-ink">
              {usable.length > 1 ? `${usable.length} days ready` : "The day the model will read"}
            </h2>
            <p className="tnum text-sm text-ink-faint">
              {(coverage * 100).toFixed(1)}% of this day recorded
            </p>
          </div>
          <div className="mt-4">
            <DayChart window={preview} />
          </div>
          <p className="mt-3 max-w-3xl text-xs text-ink-faint">
            Grey bands are stretches with no readings. They stay gaps — nothing is filled in.
            The green band is the 70&ndash;180&nbsp;mg/dL range clinicians usually target.
            {dayStarts.length - trimmed > usable.length
              ? ` ${dayStarts.length - trimmed - usable.length} day${dayStarts.length - trimmed - usable.length === 1 ? "" : "s"} in this file have too few readings to analyse.`
              : ""}
            {trimmed > 0
              ? ` This file holds ${trimmed} more day${trimmed === 1 ? "" : "s"} than the ${MAX_DAYS}-day limit; the most recent ${MAX_DAYS} are used.`
              : ""}
          </p>

          <button
            type="button"
            onClick={() => readings && void run(readings, usable)}
            disabled={busy}
            className="mt-5 bg-accent px-5 py-2.5 text-sm font-medium text-white hover:bg-accent-ink disabled:opacity-60"
          >
            {busy
              ? progress || "Working…"
              : usable.length > 1
                ? `Analyse all ${usable.length} days`
                : "Analyse this day"}
          </button>
          {status === "loading-model" ? (
            <span className="ml-3 text-xs text-ink-faint">
              downloading the encoder, once per visit
            </span>
          ) : null}
        </section>
      ) : null}

      {busy && analysis === null && !preview ? (
        <p className="text-sm text-ink-soft">{progress || "Working…"}</p>
      ) : null}

      {error ? (
        <div className="border border-low/40 bg-low/5 px-5 py-4 text-sm text-ink-soft">
          <strong className="text-ink">The model could not run on this input.</strong> {error}
        </div>
      ) : null}

      {/* ------------------------------------- the same analysis /example gets */}
      {analysis ? (
        <section>
          <div className="flex flex-wrap items-baseline justify-between gap-3 border-b border-rule pb-3">
            <h2 className="text-lg font-semibold text-ink">Your analysis</h2>
            <div className="flex items-baseline gap-4">
              {elapsedMs ? (
                <p className="tnum text-sm text-ink-faint">
                  {(elapsedMs / 1000).toFixed(1)} s, entirely in this tab
                </p>
              ) : null}
              <button
                type="button"
                onClick={() => downloadAnalysis(analysis)}
                className="border border-rule-strong px-3 py-1 text-xs text-ink-soft hover:border-accent hover:text-accent"
              >
                Download as JSON
              </button>
            </div>
          </div>
          <ExampleAnalysisView data={analysis} owner="own" />
        </section>
      ) : null}
    </div>
  );
}
