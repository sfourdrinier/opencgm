import assert from "node:assert/strict";
import test from "node:test";
import { buildWindow } from "./csv/grid";
import { buildAnalysis, type LiveDay } from "./analysis";
import { runReadingsAnalysis, type AnalysisRunnerDependencies } from "./analysis-runner";
import type { ExampleAnalysis } from "./example";
import type { HeadsBundle, Reading, Window } from "./types";

const DAY_MS = 24 * 60 * 60 * 1000;
const STEP_MS = 5 * 60 * 1000;

function localMidnight(year: number, month: number, day: number): number {
  const value = new Date(year, month, day);
  value.setHours(0, 0, 0, 0);
  return value.getTime();
}

function readingsForDay(startT: number, count: number, value = 100): Reading[] {
  return Array.from({ length: count }, (_, i) => ({
    t: startT + i * STEP_MS,
    mgdl: value + (i % 3),
  }));
}

function emptyBundle(): HeadsBundle {
  return {
    encoder: {
      checkpoint: "test",
      epoch: 0,
      seed: 0,
      weightsSha256: "test",
      architecture: {},
    },
    signalFloor: 0,
    heads: {},
  };
}

function fakeAnalysis(): ExampleAnalysis {
  return {
    note: "test",
    source: "test",
    n_days: 0,
    heads_shipped: 0,
    heads_with_signal: 0,
    signal_floor: 0,
    heads_below_floor: [],
    days: [],
    similarity: [],
    reference_windows: 0,
  };
}

function dependencies(
  build: AnalysisRunnerDependencies["buildAnalysis"] = () => fakeAnalysis(),
): AnalysisRunnerDependencies {
  return {
    loadHeadsBundle: async () => emptyBundle(),
    loadReference: async () => null,
    embedBatch: async (windows) => windows.map(() => new Float32Array(128)),
    decompose: async () => ({ state: new Float32Array(288), event: new Float32Array(288) }),
    buildAnalysis: build,
  };
}

test("groups readings by civil day and keeps exactly the newest 14 scoreable days", async () => {
  const all: Reading[] = [];
  for (let i = 0; i < 18; i += 1) {
    const start = localMidnight(2026, 0, 1 + i);
    all.push(...readingsForDay(start, 72, i));
  }
  all.push(...readingsForDay(localMidnight(2026, 0, 19), 71));
  const result = await runReadingsAnalysis(all, () => undefined, dependencies());

  assert.equal(result.usableDayStarts.length, 14);
  assert.deepEqual(result.usableDayStarts, Array.from({ length: 14 }, (_, i) =>
    localMidnight(2026, 0, 5 + i),
  ));
  assert.equal(result.trimmedDayCount, 4);
});

test("refuses an input whose days are below the 25 percent coverage floor", async () => {
  const sparse = readingsForDay(localMidnight(2026, 0, 1), 71);
  await assert.rejects(
    runReadingsAnalysis(sparse, () => undefined, dependencies()),
    /at least 25%/,
  );
});

test("reports model and per-day progress while preserving buildAnalysis output", async () => {
  const start = localMidnight(2026, 0, 1);
  const readings = readingsForDay(start, 288);
  const progress: string[] = [];
  const build = (days: LiveDay[], bundle: HeadsBundle, reference: null): ExampleAnalysis =>
    buildAnalysis(days, bundle, reference);
  const deps = dependencies(build);
  const result = await runReadingsAnalysis(readings, (message) => progress.push(message), deps);
  const window = buildWindow(readings, start);
  const expected = buildAnalysis(
    [{
      window,
      embedding: new Float32Array(128),
      streams: { state: new Float32Array(288), event: new Float32Array(288) },
      startT: start,
    }],
    emptyBundle(),
    null,
  );

  assert.deepEqual(result.analysis, expected);
  assert.match(progress[0] ?? "", /Loading model/);
  assert.ok(progress.some((message) => /Encoding 1 day/.test(message)));
  assert.ok(progress.some((message) => /Splitting day 1 of 1/.test(message)));
});

test("keeps civil-day boundaries instead of merging adjacent dates", async () => {
  const first = localMidnight(2026, 0, 1);
  const readings = [
    ...readingsForDay(first, 72),
    ...readingsForDay(localMidnight(2026, 0, 2), 72),
  ];
  const result = await runReadingsAnalysis(readings, () => undefined, dependencies());
  assert.deepEqual(result.usableDayStarts, [first, localMidnight(2026, 0, 2)]);
});
