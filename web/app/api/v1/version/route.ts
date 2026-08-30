import { NextResponse } from "next/server";
import { encoderMeta, headsBundle } from "@/lib/server/encoder";
import { CORPUS, MODEL, PROTOCOL } from "@/lib/facts";

export const runtime = "nodejs";

/** Which weights are running, and what they were trained on. Every result traces back here. */
export function GET() {
  const meta = encoderMeta();
  const bundle = headsBundle();
  return NextResponse.json({
    model: "opencgm-stateevent",
    description:
      "Independent public-data reconstruction of the GlucoFM dual-stream CGM encoder. " +
      "Not Google's implementation or weights.",
    encoder: {
      parameters: MODEL.encoderParams,
      parameters_note:
        "435,633 in the released encoder. The full pretraining model, including the " +
        "predictor and transition heads that are discarded after training, is 732,593.",
      checkpoint: meta.checkpoint,
      epoch: meta.epoch,
      epoch_note:
        "Epoch 40, not 120. Transfer peaks at 40 on a corpus this size; 120 is about one " +
        "seed-standard-deviation worse. See decision D024.",
      seed: meta.seed,
      onnx_sha256: meta.sha256,
      onnx_bytes: meta.size_bytes,
      opset: meta.opset,
      architecture: meta.architecture,
      input_units: meta.units,
      mask_convention: meta.mask_convention,
      embedding_dim: 128,
      embedding_note:
        "Mean pool over the 24 hourly patch tokens of the context encoder ('opencgm_mean').",
    },
    training_corpus: {
      windows: CORPUS.windows,
      subjects: CORPUS.subjects,
      hours: CORPUS.hours,
      cohorts: CORPUS.cohorts,
      fraction_of_paper_hours: CORPUS.fractionOfPaper,
      note:
        "The GlucoFM paper trains on 109,066 h, of which ~69% is a private dataset. This " +
        "model saw only the public remainder, so its scores are expected to be lower.",
    },
    evaluation_protocol: {
      seeds: PROTOCOL.seeds,
      epochs: PROTOCOL.epochs,
      folds: PROTOCOL.folds,
      repeats: PROTOCOL.repeats,
      probes: PROTOCOL.probes,
      task_source_combinations: PROTOCOL.taskSourceCombinations,
      splitting: "subject-disjoint — no person appears in both train and test",
    },
    heads_published: Object.keys(bundle.heads).length,
    heads_withheld: Object.keys(bundle.withheld ?? {}).length,
    heads_withheld_note: bundle.withheld_note ?? null,
    license: {
      code: "Apache-2.0",
      encoder_weights: "CC-BY-NC-4.0",
      probe_heads: "CC-BY-NC-SA-4.0",
      note:
        "Research and non-commercial use. The heads bundle is share-alike because eight of " +
        "its heads are fitted on CGMacros; the encoder never saw CGMacros and is not " +
        "share-alike. See DECISIONS.md D025. Commercial use of either requires a licence.",
    },
    privacy: {
      stores_readings: false,
      logs_readings: false,
      note:
        "Readings are held in memory for the duration of the request and are not written to " +
        "disk or logged. The browser demo at /try does not send them anywhere at all.",
    },
    not_a_medical_device: true,
  });
}
