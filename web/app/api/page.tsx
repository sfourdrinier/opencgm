import { ARTIFACT, LINKS, MODEL } from "@/lib/facts";

function Code({ children }: { children: React.ReactNode }) {
  return (
    <pre className="mt-3 overflow-x-auto rounded-lg border border-rule bg-ink/[0.03] p-4 text-xs leading-relaxed text-ink-soft">
      <code>{children}</code>
    </pre>
  );
}

function Endpoint({
  method,
  path,
  summary,
  children,
}: {
  method: string;
  path: string;
  summary: string;
  children?: React.ReactNode;
}) {
  return (
    <div className="border-t border-rule py-7">
      <div className="flex flex-wrap items-baseline gap-3">
        <span className="rounded bg-accent-soft px-2 py-0.5 font-mono text-xs font-semibold text-accent-ink">
          {method}
        </span>
        <code className="font-mono text-sm text-ink">{path}</code>
        <span className="text-sm text-ink-faint">{summary}</span>
      </div>
      {children ? <div className="measure mt-3 text-sm text-ink-soft">{children}</div> : null}
    </div>
  );
}

export default function ApiPage() {
  return (
    <div className="mx-auto max-w-6xl px-6 py-12">
      <h1 className="text-3xl font-semibold tracking-tight text-ink md:text-4xl">API</h1>
      <p className="lede measure mt-5">
        The same encoder this site runs, over HTTP. No key, no account, nothing to install.
      </p>

      <section className="mt-10 rounded-xl border border-rule bg-paper-raised p-6">
        <h2 className="text-lg font-semibold text-ink">One request</h2>
        <Code>{`curl -s https://<this-host>/api/v1/analyse \\
  -H 'content-type: application/json' \\
  -d '{
        "readings": [
          {"t": "2026-08-29T18:00:00Z", "mgdl": 96},
          {"t": "2026-08-29T18:05:00Z", "mgdl": 98},
          {"t": "2026-08-29T18:10:00Z", "mgdl": 103}
        ]
      }'`}</Code>
        <p className="measure mt-3 text-sm text-ink-soft">
          Send up to 20,000 readings. Timestamps are ISO-8601 or epoch milliseconds; glucose is{" "}
          <code className="font-mono text-xs">mgdl</code> or{" "}
          <code className="font-mono text-xs">mmol</code>. Without{" "}
          <code className="font-mono text-xs">window_start</code>, the 24 hours ending at the
          last reading are used.
        </p>
      </section>

      <section className="mt-10 rounded-lg border border-accent/25 bg-accent-soft px-5 py-4">
        <p className="text-sm font-semibold text-ink">What comes back</p>
        <p className="measure mt-2 text-sm text-ink-soft">
          The response carries the gridded 288-slot window as the encoder received it, with{" "}
          <code className="font-mono text-xs">null</code> for every unobserved slot and the mask
          alongside it, plus the clinical metrics for the same window, the 128-dimensional
          embedding, and every probe with its held-out ROC-AUC, the cohort and subject count it
          was fitted on, and its coverage band. Skipped probes say
          why they were skipped. Provenance names the checkpoint, epoch, seed, architecture
          flags and the SHA-256 of the ONNX file that produced the numbers.
        </p>
      </section>

      <section className="mt-12">
        <h2 className="text-lg font-semibold text-ink">Endpoints</h2>

        <Endpoint method="POST" path="/api/v1/analyse" summary="encode a day, score the probes">
          <p>
            Returns <code className="font-mono text-xs">window</code>,{" "}
            <code className="font-mono text-xs">clinical_metrics</code>,{" "}
            <code className="font-mono text-xs">embedding</code>,{" "}
            <code className="font-mono text-xs">probes</code> and{" "}
            <code className="font-mono text-xs">provenance</code>. Readings are held in memory
            for the request and are not written to disk or logged.
          </p>
          <Code>{`{
  "window": {
    "start_utc": "2026-08-29T17:53:08.897Z",
    "coverage": 0.9757,
    "values_mg_dl": [96.0, 98.0, null, 103.0, ...],   // null = not recorded
    "mask": [1, 1, 0, 1, ...],
    "circadian_start_index": 214
  },
  "clinical_metrics": {
    "mean_mg_dl": 102.9, "coefficient_of_variation": 0.1154,
    "time_in_range_70_180": 1.0, "glucose_management_indicator_percent": 5.77
  },
  "embedding": { "dim": 128, "vector": [-0.262430, -0.670614, ...] },
  "probes": [
    { "task": "hall:glucotype", "cohort": "hall", "scored": true,
      "predicted_class": 0, "raw_scores": [0.998, 0.002],
      "held_out_roc_auc": 0.8791, "n_subjects_fitted_on": 57 }
  ]
}`}</Code>
        </Endpoint>

        <Endpoint
          method="GET"
          path="/api/v1/version"
          summary="weights, corpus, protocol, licence"
        >
          <p>
            Checkpoint <code className="font-mono text-xs">{ARTIFACT.checkpoint}</code>, epoch{" "}
            {ARTIFACT.epoch}, seed {ARTIFACT.seed}, {MODEL.encoderParams.toLocaleString()}{" "}
            parameters, ONNX SHA-256{" "}
            <code className="font-mono text-xs">{ARTIFACT.sha256.slice(0, 16)}…</code>. Also
            reports the training corpus and the evaluation protocol, so a result can be tied to
            the run that produced it.
          </p>
        </Endpoint>

        <Endpoint method="GET" path="/api/v1/heads" summary="every probe and how well it works">
          <p>
            Per probe: task, cohort, class balance, held-out ROC-AUC with its standard
            deviation, subject-level ROC-AUC, number of subjects, folds, and the coverage band
            it is valid within. Probes below the signal floor are marked{" "}
            <code className="font-mono text-xs">has_signal: false</code>. The response also
            lists the heads withheld from publication and the licence question blocking each.
          </p>
        </Endpoint>
      </section>

      <section className="mt-10 grid gap-6 md:grid-cols-2">
        <div className="rounded-lg border border-warn/30 bg-warn-soft px-5 py-4">
          <p className="text-sm font-semibold text-ink">The scores are not probabilities</p>
          <p className="measure mt-2 text-sm text-ink-soft">
            <code className="font-mono text-xs">raw_scores</code> comes from unregularised
            logistic heads fitted in 128 dimensions on a few hundred days. They saturate: most
            inputs return values near 0 or 1, including inputs drawn from the cohort a head was
            fitted on. Rank with them. Do not report them as confidence. The meaningful number
            is <code className="font-mono text-xs">held_out_roc_auc</code>, which ranges from
            0.64 to 0.88.
          </p>
        </div>
        <div className="rounded-lg border border-rule bg-paper-raised px-5 py-4">
          <p className="text-sm font-semibold text-ink">Licence</p>
          <p className="measure mt-2 text-sm text-ink-soft">
            Code Apache-2.0. Encoder weights CC-BY-NC-4.0. The probe-head bundle is
            CC-BY-NC-SA-4.0: eight of its eighteen heads are fitted on CGMacros, whose terms are
            share-alike (D025). Commercial use of either needs a separate licence. Source and
            reproduction instructions are{" "}
            <a href={LINKS.repo} className="text-accent hover:underline">
              on GitHub
            </a>
            .
          </p>
        </div>
      </section>

      <section className="mt-10 rounded-lg border border-rule bg-paper-raised px-5 py-4">
        <p className="text-sm font-semibold text-ink">Running it yourself</p>
        <p className="measure mt-2 text-sm text-ink-soft">
          The API is part of this Next.js application and loads the same{" "}
          <code className="font-mono text-xs">encoder.onnx</code> the browser demo does, through
          onnxruntime-node. To run a private instance:{" "}
          <code className="font-mono text-xs">just web</code>. For research use at scale, prefer
          your own instance: the public one allows 30 requests a minute per IP and four
          concurrent inferences.
        </p>
      </section>
    </div>
  );
}
