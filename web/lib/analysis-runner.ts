import { buildWindow, coverageOf } from "./csv/grid";
import { buildAnalysis, type LiveDay } from "./analysis";
import type { ExampleAnalysis } from "./example";
import { loadHeadsBundle } from "./model/heads";
import { decompose, embedBatch, type Streams } from "./model/loadOnnx";
import { loadReference, type Reference } from "./model/rank";
import type { HeadsBundle, Reading, Window } from "./types";

export const MIN_COVERAGE = 0.25;
export const MAX_DAYS = 14;
const STEP_MS = 5 * 60 * 1000;
const SEQUENCE_LENGTH = 288;

export type AnalysisRunResult = {
  readonly analysis: ExampleAnalysis;
  readonly usableDayStarts: readonly number[];
  readonly trimmedDayCount: number;
};

export type AnalysisDaySelection = {
  readonly allDayStarts: readonly number[];
  readonly scoreableDayStarts: readonly number[];
  readonly usableDayStarts: readonly number[];
  readonly trimmedDayCount: number;
};

export type AnalysisRunnerDependencies = {
  readonly loadHeadsBundle: () => Promise<HeadsBundle>;
  readonly loadReference: () => Promise<Reference | null>;
  readonly embedBatch: (windows: Window[]) => Promise<Float32Array[]>;
  readonly decompose: (window: Window) => Promise<Streams>;
  readonly buildAnalysis: typeof buildAnalysis;
};

const defaultDependencies: AnalysisRunnerDependencies = {
  loadHeadsBundle,
  loadReference,
  embedBatch,
  decompose,
  buildAnalysis,
};

/** Local-midnight starts for every day an upload covers, oldest first. */
export function availableDays(readings: readonly Reading[]): number[] {
  const starts = new Set<number>();
  for (const reading of readings) {
    const day = new Date(reading.t);
    day.setHours(0, 0, 0, 0);
    starts.add(day.getTime());
  }
  return [...starts].sort((a, b) => a - b);
}

/** Fraction of the 5-minute positions in one civil day represented by readings. */
export function coverageAt(readings: readonly Reading[], startT: number): number {
  const seen = new Set<number>();
  for (const reading of readings) {
    const index = Math.floor((reading.t - startT) / STEP_MS);
    if (index >= 0 && index < SEQUENCE_LENGTH) seen.add(index);
  }
  return seen.size / SEQUENCE_LENGTH;
}

export function selectAnalysisDays(readings: readonly Reading[]): AnalysisDaySelection {
  const allDayStarts = availableDays(readings);
  const scoreableDayStarts = allDayStarts.filter(
    (startT) => coverageAt(readings, startT) >= MIN_COVERAGE,
  );
  const usableDayStarts = scoreableDayStarts.slice(-MAX_DAYS);
  return {
    allDayStarts,
    scoreableDayStarts,
    usableDayStarts,
    trimmedDayCount: scoreableDayStarts.length - usableDayStarts.length,
  };
}

export async function runReadingsAnalysis(
  readings: readonly Reading[],
  onProgress: (message: string) => void,
  dependencies: AnalysisRunnerDependencies = defaultDependencies,
): Promise<AnalysisRunResult> {
  const selection = selectAnalysisDays(readings);
  const { allDayStarts, usableDayStarts } = selection;
  if (!usableDayStarts.length) {
    throw new Error(
      `None of the ${allDayStarts.length} days in this file has at least ${MIN_COVERAGE * 100}% of its readings.`,
    );
  }

  onProgress("Loading model…");
  const bundle = await dependencies.loadHeadsBundle();
  const reference = await dependencies.loadReference();
  const windows = usableDayStarts.map((startT) => buildWindow([...readings], startT));

  onProgress(`Encoding ${windows.length} day${windows.length === 1 ? "" : "s"}…`);
  const embeddings = await dependencies.embedBatch(windows);
  if (embeddings.length !== windows.length) {
    throw new Error("encoder returned an unexpected number of embeddings");
  }

  const streams: (Streams | null)[] = [];
  for (let i = 0; i < windows.length; i += 1) {
    onProgress(`Splitting day ${i + 1} of ${windows.length}…`);
    try {
      streams.push(await dependencies.decompose(windows[i]!));
    } catch {
      streams.push(null);
    }
  }

  const live: LiveDay[] = windows.map((window, index) => {
    const embedding = embeddings[index];
    if (!embedding) throw new Error("encoder returned a missing embedding");
    return {
      window,
      embedding,
      streams: streams[index] ?? null,
      startT: usableDayStarts[index]!,
    };
  });

  return {
    analysis: dependencies.buildAnalysis(live, bundle, reference),
    usableDayStarts,
    trimmedDayCount: selection.trimmedDayCount,
  };
}

export { coverageOf };
