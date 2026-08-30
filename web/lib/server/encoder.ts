// Server-side encoder session, shared by every API route.
//
// onnxruntime-node runs the same graph the browser runs through onnxruntime-web. Both read
// `public/models/encoder.onnx`, so a result from the API and a result from the demo page are
// produced by the same bytes -- which is checked by the SHA-256 reported at /api/v1/version.

import { readFileSync } from "node:fs";
import { join } from "node:path";
import * as ort from "onnxruntime-node";

const MODELS_DIR = join(process.cwd(), "public", "models");

export type EncoderMeta = {
  checkpoint: string;
  epoch: number;
  seed: number;
  opset: number;
  size_bytes: number;
  sha256: string;
  architecture: Record<string, unknown>;
  mask_convention: string;
  units: string;
};

let sessionPromise: Promise<ort.InferenceSession> | null = null;
let metaCache: EncoderMeta | null = null;
let headsCache: unknown = null;

export function encoderMeta(): EncoderMeta {
  if (!metaCache) {
    metaCache = JSON.parse(
      readFileSync(join(MODELS_DIR, "encoder.meta.json"), "utf8"),
    ) as EncoderMeta;
  }
  return metaCache;
}

export function headsBundle(): {
  encoder: Record<string, unknown>;
  signal_floor: number;
  heads: Record<string, RawHead>;
  withheld?: Record<string, string>;
  withheld_note?: string;
} {
  if (!headsCache) {
    headsCache = JSON.parse(readFileSync(join(MODELS_DIR, "heads.json"), "utf8"));
  }
  return headsCache as ReturnType<typeof headsBundle>;
}

export type RawHead = {
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
  applicability: { coverage_p05: number; coverage_p95: number; coverage_median: number };
  class_balance: number[];
};

async function session(): Promise<ort.InferenceSession> {
  if (!sessionPromise) {
    sessionPromise = ort.InferenceSession.create(join(MODELS_DIR, "encoder.onnx"));
  }
  return sessionPromise;
}

/** Encode one 24-hour window. `values` is mg/dL; `mask` is 1 = observed, 0 = not. */
export async function encode(
  values: Float32Array,
  mask: Float32Array,
  circadianStart: number,
): Promise<Float32Array> {
  const s = await session();
  const out = await s.run({
    values: new ort.Tensor("float32", values, [1, 288]),
    mask: new ort.Tensor("float32", mask, [1, 288]),
    circadian_start: new ort.Tensor("int64", BigInt64Array.from([BigInt(circadianStart)]), [1]),
  });
  const emb = out["embedding"];
  if (!emb) throw new Error("encoder returned no 'embedding' output");
  return Float32Array.from(emb.data as Float32Array);
}

/** Apply one head. Returns the per-class probability vector (uncalibrated -- see the docs). */
export function applyHead(head: RawHead, embedding: Float32Array): number[] {
  const { mean, scale } = head.scale;
  const { coef, intercept } = head.classifier;
  const z = coef.map((row, k) => {
    let acc = intercept[k] ?? 0;
    for (let i = 0; i < row.length; i += 1) {
      acc += ((embedding[i] ?? 0) - (mean[i] ?? 0)) / (scale[i] ?? 1) * (row[i] ?? 0);
    }
    return acc;
  });
  if (z.length === 1) {
    const p = 1 / (1 + Math.exp(-(z[0] ?? 0)));
    return [1 - p, p];
  }
  const max = Math.max(...z);
  const e = z.map((v) => Math.exp(v - max));
  const sum = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / sum);
}
