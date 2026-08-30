import { NextResponse } from "next/server";
import { headsBundle } from "@/lib/server/encoder";

export const runtime = "nodejs";

/** Every probe, with the cohort it was fitted on and how well it actually works. */
export function GET() {
  const bundle = headsBundle();

  const heads = Object.entries(bundle.heads).map(([key, h]) => ({
    key,
    task: h.task,
    cohort: h.dataset,
    source: h.source,
    n_classes: h.n_classes,
    class_balance: h.class_balance,
    reliability: {
      roc_auc: h.reliability.roc_auc,
      roc_auc_sd: h.reliability.roc_auc_sd,
      roc_auc_subject: h.reliability.roc_auc_subject,
      n_subjects: h.reliability.n_subjects,
      n_windows: h.reliability.n_windows,
      n_folds: h.reliability.n_folds,
      has_signal: h.reliability.has_signal,
      interpretation:
        h.reliability.roc_auc == null
          ? "not measured"
          : h.reliability.roc_auc >= 0.75
            ? "moderate discrimination on held-out people"
            : h.reliability.roc_auc >= 0.65
              ? "weak discrimination on held-out people"
              : "very weak — close to chance",
    },
    applicability: {
      coverage_p05: h.applicability.coverage_p05,
      coverage_p95: h.applicability.coverage_p95,
      coverage_median: h.applicability.coverage_median,
      note:
        "The fraction-of-day-observed band this head was fitted within. A window outside " +
        "the band (±0.15) is scored as not applicable rather than extrapolated.",
    },
  }));

  return NextResponse.json({
    license: (bundle as { license?: string }).license ?? "CC-BY-NC-SA-4.0",
    license_note:
      "Share-alike, inherited from CGMacros. Redistribute adaptations of these heads under " +
      "CC-BY-NC-SA-4.0. The encoder is a separate artefact under CC-BY-NC-4.0.",
    signal_floor: bundle.signal_floor,
    signal_floor_note:
      "Heads below this cross-validated ROC-AUC are marked has_signal=false and should be " +
      "treated as uninformative.",
    calibration_warning:
      "These are unregularised logistic heads fitted in 128 dimensions on a few hundred " +
      "days. Their probabilities saturate: most inputs return values near 0 or 1, including " +
      "inputs from their own training cohort. Use them for ranking, not as probabilities. " +
      "The usable measure of each head is its held-out roc_auc.",
    published: heads,
    withheld: bundle.withheld ?? {},
    withheld_note: bundle.withheld_note ?? null,
  });
}
