// Shared types across the demo.
//
// Units: the encoder expects glucose in **mg/dL** (matches the released ONNX).
// Time: all timestamps are local-naive ISO strings; the consumer must specify the
// timezone once per session. We never assume UTC silently.

export type Reading = {
  /** milliseconds since Unix epoch in the user's specified timezone */
  t: number;
  /** glucose value in mg/dL */
  mgdl: number;
};

export type Window = {
  /** 288 glucose values, mg/dL, on a 5-minute grid starting at `startT` */
  values: Float32Array;
  /** 288 mask values, 1.0 = observed, 0.0 = unobserved. Never interpolated. */
  mask: Float32Array;
  /** milliseconds since epoch — start of the 24-hour window */
  startT: number;
  /** position of the window start in a 24-hour cycle (0..287) — for the encoder's
   * circadian time embedding. For a midnight-aligned window, this is 0. */
  circadianStart: number;
};

export type HeadMeta = {
  task: string;
  dataset: string;
  source: string;
  nClasses: number;
  reliability: {
    rocAuc: number | null;
    rocAucSd: number | null;
    rocAucSubject: number | null;
    nSubjects: number;
    nWindows: number;
    nFolds: number;
    hasSignal: boolean;
  };
  applicability: {
    coverageP05: number;
    coverageP95: number;
    coverageMedian: number;
  };
  classBalance: number[];
  /** confidence weight 0..1 derived from `reliability.rocAuc` */
  confidence: number;
};

export type Head = {
  meta: HeadMeta;
  /** StandardScaler: 128 means, 128 scales */
  scale: { mean: Float32Array; scale: Float32Array };
  /** LogisticRegression: [K, 128] coef, [K] intercept, [K] classes */
  classifier: { coef: Float32Array[]; intercept: Float32Array; classes: number[] };
};

export type HeadsBundle = {
  encoder: {
    checkpoint: string;
    epoch: number;
    seed: number;
    weightsSha256: string;
    architecture: Record<string, unknown>;
  };
  signalFloor: number;
  heads: Record<string, Head>;
};

/** A single phenotype score produced by applying one head to one embedding */
export type PhenotypeScore = {
  key: string;
  task: string;
  meta: HeadMeta;
  /** probability in [0, 1] for binary heads; for multiclass, the max-class probability */
  probability: number;
  /** predicted class label (binary: 0 or 1; multiclass: argmax of classes) */
  predictedClass: number;
  /** the applicability coverage of the input window, vs the head's training coverage band */
  coverage: number;
  applicable: boolean;
  /** plain-English phrasing for the UI */
  phrasing: string;
};

export type EmbeddingResult = {
  /** the 128-d mean-pooled daily embedding */
  embedding: Float32Array;
  /** coverage of the input window (fraction of 288 positions observed) */
  coverage: number;
  /** the window that was embedded */
  window: Window;
};

export type EncodedDay = EmbeddingResult & {
  scores: PhenotypeScore[];
};
