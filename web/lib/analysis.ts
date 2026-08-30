// Turn live browser results into the shape /example renders.
//
// The worked example is generated offline by scripts/build_example_analysis.py and rendered
// by ExampleAnalysisView. Everything that view needs, the demo already computes in the
// browser: the gridded window, the two streams, the embedding, the probe scores, and the
// corpus percentiles. Assembling them into the same object means someone running their own
// export gets the same analysis, drawn by the same components, rather than a thinner
// second-class version of it.

import { coverageOf } from "./csv/grid";
import { clinicalMetrics } from "./clinical";
import type { ExampleAnalysis, ExampleDay, ExampleProbe } from "./example";
import type { Streams } from "./model/loadOnnx";
import type { HeadsBundle, Window } from "./types";
import { type Reference, percentileOf, rankIsInformative } from "./model/rank";

export type LiveDay = {
  window: Window;
  embedding: Float32Array;
  streams: Streams | null;
  /** Local midnight the window starts at, used only for the weekday label. */
  startT: number;
};

const COVERAGE_TOLERANCE = 0.15;

function probesFor(
  bundle: HeadsBundle,
  reference: Reference | null,
  embedding: Float32Array,
  coverage: number,
): ExampleProbe[] {
  const out: ExampleProbe[] = [];
  for (const [key, head] of Object.entries(bundle.heads)) {
    if (!head.meta.reliability.hasSignal) continue;
    const { coverageP05, coverageP95 } = head.meta.applicability;
    if (coverage < coverageP05 - COVERAGE_TOLERANCE) continue;
    if (coverage > coverageP95 + COVERAGE_TOLERANCE) continue;

    // Same arithmetic the ranking column uses: scale, then the classifier, then the
    // positive-class or winning-class score.
    const { mean, scale } = head.scale;
    const { coef, intercept } = head.classifier;
    const z = coef.map((row, k) => {
      let acc = intercept[k] ?? 0;
      for (let i = 0; i < row.length; i += 1) {
        acc += (((embedding[i] ?? 0) - (mean[i] ?? 0)) / (scale[i] ?? 1)) * (row[i] ?? 0);
      }
      return acc;
    });
    let score: number;
    if (z.length === 1) {
      score = 1 / (1 + Math.exp(-(z[0] ?? 0)));
    } else {
      const max = Math.max(...z);
      const e = z.map((v) => Math.exp(v - max));
      const sum = e.reduce((a, b) => a + b, 0);
      score = Math.max(...e.map((v) => v / sum));
    }

    const usable = reference ? rankIsInformative(reference, key) : false;
    const pct = reference && usable ? percentileOf(reference, key, score) : null;
    const deciles = reference?.heads[key]?.breakpoints ?? null;

    out.push({
      key,
      task: head.meta.task,
      cohort: head.meta.dataset,
      score,
      percentile: pct,
      roc_auc: head.meta.reliability.rocAuc ?? 0,
      n_subjects: head.meta.reliability.nSubjects,
      corpus_deciles: deciles ? deciles.filter((_, i) => i % 5 === 0) : null,
    });
  }
  out.sort((a, b) => b.roc_auc - a.roc_auc);
  return out;
}

function cosine(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i += 1) {
    dot += a[i]! * b[i]!;
    na += a[i]! * a[i]!;
    nb += b[i]! * b[i]!;
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
}

export function buildAnalysis(
  days: LiveDay[],
  bundle: HeadsBundle,
  reference: Reference | null,
): ExampleAnalysis {
  const heads = Object.values(bundle.heads);
  const belowFloor = heads
    .filter((h) => !h.meta.reliability.hasSignal)
    .map((h) => ({
      key: `${h.meta.dataset}:${h.meta.task}`,
      task: h.meta.task,
      cohort: h.meta.dataset,
      roc_auc: h.meta.reliability.rocAuc ?? 0,
    }))
    .sort((a, b) => a.roc_auc - b.roc_auc);

  const built: ExampleDay[] = days.map((d, i) => {
    const coverage = coverageOf(d.window.mask);
    const clinical = clinicalMetrics(d.window.values, d.window.mask);
    return {
      label: `Day ${i + 1}`,
      weekday: new Date(d.startT).toLocaleDateString(undefined, { weekday: "long" }),
      coverage,
      values_mg_dl: Array.from(d.window.values, (v, j) =>
        d.window.mask[j] ? +v.toFixed(1) : null,
      ),
      mask: Array.from(d.window.mask, (m) => (m ? 1 : 0)),
      state: d.streams ? Array.from(d.streams.state) : [],
      event: d.streams ? Array.from(d.streams.event) : [],
      embedding: Array.from(d.embedding),
      clinical,
      probes: probesFor(bundle, reference, d.embedding, coverage),
    };
  });

  const similarity = days.map((a) => days.map((b) => +cosine(a.embedding, b.embedding).toFixed(4)));

  return {
    note: "Computed in this browser from the file you loaded. Nothing was uploaded.",
    source: "your own recording",
    n_days: built.length,
    heads_shipped: heads.length,
    heads_with_signal: heads.filter((h) => h.meta.reliability.hasSignal).length,
    signal_floor: bundle.signalFloor,
    heads_below_floor: belowFloor,
    days: built,
    similarity,
    reference_windows: reference?.n_reference_windows ?? 0,
  };
}
