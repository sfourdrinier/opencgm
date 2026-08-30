// Apply the fitted logistic-regression heads to a 128-d encoder embedding.
//
// The shipped `glucofm_heads.json` (produced by `scripts/export_heads_json.py`) is loaded
// once at app start and cached. Each head is a `StandardScaler → LogisticRegression`
// pipeline that we evaluate in plain TypeScript — no external dependencies.
//
// Output is a `PhenotypeScore` per head, sorted by reliability descending. The
// applicability gate refuses to score a window whose coverage falls outside the head's
// training coverage band; that prevents the Libre-vs-Dexcom nonsense measured in
// `scripts/fit_heads.py`.

import type {
  HeadsBundle,
  Head,
  PhenotypeScore,
  EmbeddingResult,
  EncodedDay,
} from "../types";
import { coverageOf } from "../csv/grid";

/** Load the JSON bundle. Cached at the page level; call once per page load. */
let bundleCache: HeadsBundle | null = null;
/** The on-disk shape written by `scripts/export_heads_json.py`.
 *
 * The exporter writes flat snake_case, because it mirrors the Python bundle and its layout is
 * pinned by `tests/export/test_onnx_parity.py`. The UI wants a nested camelCase `meta`. That
 * translation happens here, in one place, rather than by bending either side out of shape.
 */
type RawHead = {
  scale: { mean: number[]; scale: number[] };
  classifier: { coef: number[][]; intercept: number[]; classes: number[] };
  task: string;
  dataset: string;
  source: string;
  n_classes: number;
  reliability: {
    roc_auc: number | null;
    roc_auc_sd: number | null;
    roc_auc_subject: number | null;
    n_subjects: number;
    n_windows: number;
    n_folds: number;
    has_signal: boolean;
  };
  applicability: {
    coverage_p05: number;
    coverage_p95: number;
    coverage_median: number;
  };
  class_balance: number[];
};

type RawBundle = {
  encoder: {
    checkpoint: string;
    epoch: number;
    seed: number;
    weights_sha256: string;
    architecture: string | Record<string, unknown>;
  };
  signal_floor: number;
  heads: Record<string, RawHead>;
};

function adaptHead(raw: RawHead): Head {
  return {
    meta: {
      task: raw.task,
      dataset: raw.dataset,
      source: raw.source,
      nClasses: raw.n_classes,
      reliability: {
        rocAuc: raw.reliability.roc_auc,
        rocAucSd: raw.reliability.roc_auc_sd,
        rocAucSubject: raw.reliability.roc_auc_subject,
        nSubjects: raw.reliability.n_subjects,
        nWindows: raw.reliability.n_windows,
        nFolds: raw.reliability.n_folds,
        hasSignal: raw.reliability.has_signal,
      },
      applicability: {
        coverageP05: raw.applicability.coverage_p05,
        coverageP95: raw.applicability.coverage_p95,
        coverageMedian: raw.applicability.coverage_median,
      },
      classBalance: raw.class_balance,
      confidence: confidenceFromRoc(raw.reliability.roc_auc),
    },
    scale: {
      mean: new Float32Array(raw.scale.mean),
      scale: new Float32Array(raw.scale.scale),
    },
    classifier: {
      coef: raw.classifier.coef.map((row) => new Float32Array(row)),
      intercept: new Float32Array(raw.classifier.intercept),
      classes: raw.classifier.classes.slice(),
    },
  };
}

export async function loadHeadsBundle(): Promise<HeadsBundle> {
  if (bundleCache) return bundleCache;
  const res = await fetch("/models/heads.json", { cache: "force-cache" });
  if (!res.ok) throw new Error(`failed to load heads.json: ${res.status}`);
  const raw = (await res.json()) as RawBundle;

  const heads: Record<string, Head> = {};
  for (const [key, rawHead] of Object.entries(raw.heads)) {
    heads[key] = adaptHead(rawHead);
  }

  const architecture =
    typeof raw.encoder.architecture === "string"
      ? (JSON.parse(raw.encoder.architecture) as Record<string, unknown>)
      : raw.encoder.architecture;

  bundleCache = {
    encoder: {
      checkpoint: raw.encoder.checkpoint,
      epoch: raw.encoder.epoch,
      seed: raw.encoder.seed,
      weightsSha256: raw.encoder.weights_sha256,
      architecture,
    },
    signalFloor: raw.signal_floor,
    heads,
  };
  return bundleCache;
}

function confidenceFromRoc(roc: number | null): number {
  if (roc == null || !Number.isFinite(roc)) return 0;
  // 0.5 → 0 confidence, 1.0 → 1 confidence, clipped
  return Math.max(0, Math.min(1, (roc - 0.5) * 2));
}

/** Numerically-stable sigmoid. */
function sigmoid(z: number): number {
  if (z >= 0) return 1 / (1 + Math.exp(-z));
  const e = Math.exp(z);
  return e / (1 + e);
}

/** Numerically-stable softmax along the last axis. */
function softmax(logits: Float32Array, k: number): Float32Array {
  const out = new Float32Array(k);
  let max = -Infinity;
  for (let i = 0; i < k; i += 1) if (logits[i]! > max) max = logits[i]!;
  let sum = 0;
  for (let i = 0; i < k; i += 1) {
    out[i] = Math.exp(logits[i]! - max);
    sum += out[i]!;
  }
  for (let i = 0; i < k; i += 1) out[i]! /= sum;
  return out;
}

/** Apply one head to a 128-d embedding. Returns the per-class probability vector. */
export function applyHead(head: Head, embedding: Float32Array): Float32Array {
  const { mean, scale } = head.scale;
  const { coef, intercept } = head.classifier;
  const k = coef.length;
  const logits = new Float32Array(k);

  for (let i = 0; i < k; i += 1) {
    const row = coef[i]!;
    let z = intercept[i]!;
    for (let j = 0; j < 128; j += 1) {
      // x_scaled = (x - mean) / scale; then z += coef @ x_scaled
      const x = ((embedding[j] ?? 0) - (mean[j] ?? 0)) / (scale[j] ?? 1);
      z += (row[j] ?? 0) * x;
    }
    logits[i] = z;
  }

  if (k === 2) {
    // binary → single probability for class 1 (matches Analyser.analyse_day logic)
    const out = new Float32Array(2);
    out[1] = sigmoid(logits[0]!); // LogReg binary fits one set of coef for class 1
    out[0] = 1 - out[1];
    return out;
  }
  return softmax(logits, k);
}

/** Coverage-based applicability gate (per blueprint §19.4 and the Streamlit demo). */
export function isApplicable(head: Head, coverage: number): boolean {
  const { coverageP05, coverageP95 } = head.meta.applicability;
  const TOL = 0.15; // matches the Python `COVERAGE_TOLERANCE` in src/.../infer.py
  return coverage >= coverageP05 - TOL && coverage <= coverageP95 + TOL;
}

/** Compute all phenotype scores for one embedding. */
export function scoreAll(
  bundle: HeadsBundle,
  result: EmbeddingResult,
  phrasing: (score: PhenotypeScore) => string = defaultPhrasing,
): EncodedDay {
  const coverage = coverageOf(result.window.mask);
  const scores: PhenotypeScore[] = [];
  for (const [key, head] of Object.entries(bundle.heads)) {
    const applicable = isApplicable(head, coverage);
    let probability = 0;
    let predictedClass = 0;
    if (applicable && head.meta.reliability.hasSignal) {
      const proba = applyHead(head, result.embedding);
      // Binary → report class 1; multiclass → report max
      let maxIdx = 0;
      let maxVal = -Infinity;
      for (let i = 0; i < proba.length; i += 1) {
        if (proba[i]! > maxVal) {
          maxVal = proba[i]!;
          maxIdx = i;
        }
      }
      probability = proba[maxIdx] ?? 0;
      predictedClass = head.classifier.classes[maxIdx] ?? 0;
    }
    const score: PhenotypeScore = {
      key,
      task: head.meta.task,
      meta: head.meta,
      probability,
      predictedClass,
      coverage,
      applicable,
      phrasing: "",
    };
    score.phrasing = phrasing(score);
    scores.push(score);
  }
  // Sort by reliability (then signal) descending so the most reliable heads float to the top.
  scores.sort((a, b) => {
    const aR = a.meta.reliability.hasSignal ? a.meta.confidence : -1;
    const bR = b.meta.reliability.hasSignal ? b.meta.confidence : -1;
    return bR - aR;
  });
  return { ...result, scores };
}

/** Plain-English phrasing per score. Matches the Streamlit demo's `population_phrasing`. */
export function defaultPhrasing(score: PhenotypeScore): string {
  if (!score.applicable) {
    return `Coverage ${(score.coverage * 100).toFixed(0)}% is outside the ${score.meta.applicability.coverageP05.toFixed(0)}–${score.meta.applicability.coverageP95.toFixed(0)}% band this head was trained on. Score withheld.`;
  }
  if (!score.meta.reliability.hasSignal) {
    return `This head showed no measurable signal (ROC-AUC ${(score.meta.reliability.rocAuc ?? 0).toFixed(2)}) in cross-validation; treat as uninformative.`;
  }
  const p = score.probability;
  const pct = (p * 100).toFixed(0);
  const auc = score.meta.reliability.rocAuc?.toFixed(2) ?? "?";
  const subj = score.meta.reliability.nSubjects;
  return `${pct}% confidence in the model's prediction for this ${score.meta.task} signal (cross-validated ROC-AUC ${auc}, ${subj} training subjects).`;
}
