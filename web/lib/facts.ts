// Every number the site states about the model, in one place.
//
// These are not decorative. Each one is checked against the repository by
// `tests/unit/test_documented_facts.py` and, for the model artefact, against
// `artifacts/glucofm_encoder.onnx.meta.json`. If you change a figure here and nowhere
// else, you have introduced exactly the class of error this file exists to prevent:
// the project once published the *comparator's* parameter count as its own, in five
// separate documents.
//
// Sources:
//   params        -- OpenCGMStateEvent, constructed and counted
//   corpus        -- data/canonical/windows/strict_seed17.values.npy, shape (353127, 288)
//   hours         -- manifests/sources/registry.yaml, Lane A
//   results       -- reports/eval/head_to_head_5seed.csv, MACRO[per_entry]
//   artifact      -- artifacts/glucofm_encoder.onnx.meta.json

export const MODEL = {
  /** Trainable parameters in the full pretraining model: encoder + predictor + transitions. */
  trainableTotal: 732_593,
  /** The encoder alone -- what ships as encoder.onnx and runs in your browser. */
  encoderParams: 435_633,
  predictorParams: 132_480,
  transitionParams: 164_480,
  /** The figure the paper reports, which the total is within 1.7% of. */
  paperParams: 720_000,
  /** The CGM-JEPA comparator. Recorded here so it is never mistaken for ours again. */
  comparatorParams: 521_584,
} as const;

export const CORPUS = {
  windows: 353_127,
  subjects: 240,
  hours: 33_736,
  paperHours: 109_066,
  /** 33,736 / 109,066 */
  fractionOfPaper: 0.309,
  cohorts: ["BIG IDEAs", "Shanghai T2DM", "Stanford", "Colas"],
} as const;

export const ARTIFACT = {
  checkpoint: "runs_5090/rawstats120/ckpt_ep040.pt",
  seed: 17,
  epoch: 40,
  opset: 17,
  bytes: 1_991_782,
  sha256: "b1349deffd15ab62a5a98d7c7c4a7e143bcf1dee7ad745b1e05533ff54f34768",
  zeroEmptyPatches: false,
} as const;

export const PROTOCOL = {
  seeds: [17, 29, 43, 71, 101] as const,
  epochs: 120,
  folds: 5,
  repeats: 10,
  /** 14 dataset-task probes; 18 task-source combinations (two cohorts, two sensors each). */
  probes: 14,
  taskSourceCombinations: 18,
} as const;

/** Macro over all 18 task-source rows, five seeds, epoch 120.
 *
 * Recomputed from `reports/eval/seed*_ep120_full/summary.csv` and `cgmjepa_seed*_full/`.
 *
 * These levels were previously published as 0.679 / 0.652 / 0.617. Those came from a 16-task
 * subset — an early snapshot taken before `shanghai_t2dm:hyperlipidemia` and
 * `stanford:insulin_resistance` were added — and averaging the other 16 reproduces them to
 * three decimals. The deltas were computed over all 18 throughout and were always correct,
 * which is why the discrepancy survived: every difference reconciled, only the levels did not.
 */
export const RESULTS = {
  rocAuc: { model: 0.6701, clinical: 0.6432, comparator: 0.6432, rawMasked: 0.6073, sd: 0.0034 },
  prAuc: { model: 0.5876, clinical: 0.5788, comparator: 0.5655, rawMasked: 0.5279, sd: 0.0034 },
  vsClinical: { roc: 0.0269, ciLow: 0.0222, ciHigh: 0.0316 },
  vsRawMasked: { roc: 0.0628, ciLow: 0.0581, ciHigh: 0.0675 },
  comparatorVsClinical: { roc: 0.0, ciLow: -0.0049, ciHigh: 0.005 },
  comparatorVsRawMasked: { roc: 0.0359, ciLow: 0.031, ciHigh: 0.0409 },
  /**
   * Per-task significance. The site claimed 16 of 18 after Holm correction; the correct
   * number is zero. No single task-cohort row reaches significance against the clinical
   * baseline in any seed, before or after correction. The macro advantage is real and its
   * interval excludes zero; it is a small consistent lift across tasks, not a set of
   * per-task wins. See findings/per_task.md.
   */
  perTaskSignificant: { model: 0, comparator: 0, total: 18 },
  /** Rows where this model's point estimate is ahead of the clinical baseline. */
  perTaskAhead: { model: 13, total: 18 },
} as const;

/** Tier-1 ablations: 3 seeds x 10 conditions x 40 epochs. Delta vs the full model. */
export const ABLATIONS = [
  { key: "abl_event", label: "Event stream only", mean: 0.6229, sd: 0.0031, delta: -0.0504 },
  { key: "abl_raw", label: "No decomposition (raw signal)", mean: 0.65, sd: 0.0103, delta: -0.0233 },
  { key: "abl_nocirc", label: "No time-of-day signal", mean: 0.658, sd: 0.0158, delta: -0.0152 },
  { key: "abl_state", label: "State stream only", mean: 0.6592, sd: 0.0148, delta: -0.0141 },
  { key: "abl_loo_shanghai", label: "Trained without Shanghai", mean: 0.6649, sd: 0.0064, delta: -0.0084 },
  { key: "abl_loo_stanford", label: "Trained without Stanford", mean: 0.6669, sd: 0.0069, delta: -0.0064 },
  { key: "abl_notd", label: "No temporal-dynamics loss", mean: 0.671, sd: 0.0046, delta: -0.0023 },
  { key: "abl_dense", label: "Gaps filled in by interpolation", mean: 0.6738, sd: 0.0077, delta: 0.0005 },
  { key: "abl_noaug", label: "No data augmentation", mean: 0.6757, sd: 0.0019, delta: 0.0024 },
  { key: "abl_fixedsigma", label: "Fixed (not learned) filter width", mean: 0.6765, sd: 0.0046, delta: 0.0032 },
] as const;

/** The seed-level standard deviation any ablation delta must be read against. */
export const ABLATION_SD = 0.0084;

/** Points in the paper that had to be resolved by judgement rather than by reading.
 *
 * `inferredForks` is machine-derived: `opencgm config-check` refuses to load the reference
 * config unless every ambiguous option carries an evidence tag, and reports the count. It is
 * not the same as the number of entries in DECISIONS.md, which also records decisions forced
 * by the data rather than by the paper.
 */
export const AMBIGUITY = {
  inferredForks: 19,
  decisionEntries: 25,
} as const;

export const LINKS = {
  paper: "https://arxiv.org/abs/2605.30865v2",
  comparator: "https://github.com/cruiseresearchgroup/CGM-JEPA",
  repo: "https://github.com/sfourdrinier/opencgm",
  author: "https://www.linkedin.com/in/stephanefourdrinier",
  company: "https://trackourhearts.com",
  // Narrated 75s explainer. Served from the generator's CDN rather than committed to the
  // repo: it is 4.8 MB, and Vercel would otherwise ship it in every deployment bundle.
  explainer:
    "https://assets.imagibooks.com/cdn/explainer/01a05521-8d9f-70b5-8bab-a69aa389a1d3/video.mp4?v=1",
  decisions: "https://github.com/sfourdrinier/opencgm/blob/main/DECISIONS.md",
  headToHead:
    "https://github.com/sfourdrinier/opencgm/blob/main/findings/head_to_head.md",
  ablations:
    "https://github.com/sfourdrinier/opencgm/blob/main/findings/tier1_ablations.md",
  reproduce:
    "https://github.com/sfourdrinier/opencgm/blob/main/REPRODUCE.md",
  modelCard:
    "https://github.com/sfourdrinier/opencgm/blob/main/model_cards/glucofm_encoder.md",
  comparatorPort:
    "https://github.com/sfourdrinier/opencgm/blob/main/src/opencgm_stateevent/baselines/cgm_jepa.py",
} as const;


/**
 * The seven questions the classifiers were trained to answer, in the words a reader uses.
 *
 * The model returns 128 numbers and knows nothing about disease. These are what small
 * classifiers fitted on top of those numbers were trained to separate, with the accuracy of
 * the best classifier for each. `working` is how many of that question's classifiers beat
 * chance in cross-validation -- hyperlipidemia is here because three of its four did not,
 * which is worth seeing rather than hiding.
 */
export const QUESTIONS = [
  {
    label: "How steady the day is",
    plain:
      "Which of the glucose-variability patterns this day resembles — the smooth kind, or the kind that swings.",
    bestAuc: 0.879,
    working: 1,
    total: 1,
  },
  {
    label: "Signs of insulin resistance",
    plain:
      "Whether the day looks like someone whose body has stopped responding well to its own insulin.",
    bestAuc: 0.872,
    working: 4,
    total: 5,
  },
  {
    label: "Raised diabetes risk",
    plain:
      "Whether the day looks like someone already diabetic or heading that way, rather than someone who is not.",
    bestAuc: 0.779,
    working: 4,
    total: 4,
  },
  {
    label: "Body-mass category",
    plain: "Which BMI band the person falls in — inferred from the shape of the day alone.",
    bestAuc: 0.758,
    working: 2,
    total: 2,
  },
  {
    label: "Reduced insulin production",
    plain:
      "Whether the pancreas looks like it is making less insulin than it should, which is a different problem from resisting it.",
    bestAuc: 0.694,
    working: 1,
    total: 1,
  },
  {
    label: "Raised blood fats",
    plain:
      "Whether cholesterol and triglycerides are elevated. Three of the four classifiers for this never beat chance; one did, barely.",
    bestAuc: 0.651,
    working: 1,
    total: 4,
  },
  {
    label: "Dips below the safe range",
    plain:
      "Whether the person has episodes of low glucose. The weakest of the seven, and only just above chance.",
    bestAuc: 0.566,
    working: 1,
    total: 1,
  },
] as const;


/**
 * Plain-English definitions for the terms the front page cannot avoid.
 *
 * One source, used both by the inline markers and by the glossary they point at, so a term
 * cannot be explained one way in the text and another way underneath it.
 *
 * The test is whether someone who has never read a paper could use the sentence after
 * reading the definition. "The area under the receiver operating characteristic curve" fails
 * that test; "how often it gets the order right" passes it.
 */
export const GLOSSARY = [
  {
    term: "ROC-AUC",
    short: "How often it gets two people in the right order.",
    long:
      "Show the classifier one person who has the condition and one who does not. ROC-AUC is " +
      "how often it puts them the right way round. 0.5 means it is guessing; 1.0 means it is " +
      "never wrong. 0.75 means it gets three pairs in four right, which is useful for research " +
      "and nowhere near a clinical test.",
  },
  {
    term: "point estimate",
    short: "The single best guess, before asking how sure we are.",
    long:
      "The number the measurement actually produced. On its own it says nothing about how " +
      "much it might move if the experiment were repeated — that is what the confidence " +
      "interval is for.",
  },
  {
    term: "statistically significant",
    short: "Large enough that luck is an unlikely explanation.",
    long:
      "A result is called significant when a difference this large would rarely appear by " +
      "chance alone. It says nothing about whether the difference is big enough to matter. A " +
      "tiny, useless difference can be significant, and a large, useful one can fail to be " +
      "if there were not enough people to tell.",
  },
  {
    term: "confidence interval",
    short: "The range the true value is probably inside.",
    long:
      "Written [low, high]. If the range does not include zero, the effect is unlikely to be " +
      "nothing. The narrower it is, the more the data pinned the answer down.",
  },
  {
    term: "seed",
    short: "One complete training run, from a different random start.",
    long:
      "Training involves randomness, so the same recipe run twice gives slightly different " +
      "models. Training five times from five starting points and reporting the spread shows " +
      "how much of a result is the method and how much was luck of the draw.",
  },
  {
    term: "fold",
    short: "One split of the people into a training half and a testing half.",
    long:
      "The people are divided into five groups; each takes a turn as the test group while the " +
      "other four train. Splitting by person rather than by day matters: testing on a " +
      "different day from someone the model already saw measures whether it recognises them, " +
      "not whether it learned anything.",
  },
  {
    term: "baseline",
    short: "The simpler method the model has to beat to be worth using.",
    long:
      "Here it is the summary a clinic already computes — average glucose, variability, time " +
      "in range, and a dozen similar figures. If a learned model cannot beat that, it has not " +
      "earned its complexity.",
  },
  {
    term: "classifier",
    short: "A small piece of maths that sorts things into groups.",
    long:
      "Each one here reads the model's 128 numbers and was trained to separate, say, people " +
      "with insulin resistance from people without. The classifiers are separate from the " +
      "model and much simpler than it.",
  },
  {
    term: "time in range",
    short: "The share of the day spent between 70 and 180 mg/dL.",
    long:
      "The band clinicians usually target. It is the headline number in most CGM apps, and " +
      "two days can have identical time in range and look nothing alike.",
  },
  {
    term: "ablation",
    short: "Removing one part and retraining, to see if it mattered.",
    long:
      "The only way to find out whether a design choice earns its place. Here ten parts were " +
      "removed one at a time; only the slow/fast split changed the result by more than the " +
      "noise between training runs.",
  },
  {
    term: "corpus",
    short: "The whole collection of data the model learned from.",
    long: "353,127 days of glucose readings from 240 people across four public research cohorts.",
  },
] as const;

export type GlossaryEntry = (typeof GLOSSARY)[number];
