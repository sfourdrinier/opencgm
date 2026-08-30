import { Callout, Reproduce, ScoreAxis, ScoreBar } from "@/components/Figures";
import {
  ABLATION_SD,
  ABLATIONS,
  AMBIGUITY,
  ARTIFACT,
  CORPUS,
  LINKS,
  PROTOCOL,
  RESULTS,
} from "@/lib/facts";

export default function Results() {
  return (
    <div className="mx-auto max-w-6xl px-6 py-12">
      <h1 className="text-3xl font-semibold tracking-tight text-ink md:text-4xl">Results</h1>
      <p className="lede measure mt-5">
        Five independently trained models, {PROTOCOL.epochs} epochs each, evaluated on{" "}
        {PROTOCOL.probes} clinical questions with subject-disjoint splits. Negative results
        included.
      </p>

      {/* ------------------------------------------------------------- headline */}
      <section className="mt-10">
        <h2 className="text-xl font-semibold text-ink">Headline</h2>
        <p className="measure mt-4 text-ink-soft">
          ROC-AUC is the probability that a randomly chosen positive case is ranked above a
          randomly chosen negative one. 0.5 is chance.
        </p>
        <div className="mt-6 max-w-3xl">
          <ScoreBar label="Chance" value={0.5} baseline />
          <ScoreBar label="Raw glucose readings" value={RESULTS.rocAuc.rawMasked} />
          <ScoreBar label="Clinical metrics" value={RESULTS.rocAuc.clinical} caption="time-in-range, mean, variability" />
          <ScoreBar label="CGM-JEPA" value={RESULTS.rocAuc.comparator} caption="the nearest published model, identical folds" />
          <ScoreBar label="OpenCGM-StateEvent" value={RESULTS.rocAuc.model} emphasis />
          <ScoreAxis />
        </div>

        <div className="mt-8 overflow-x-auto">
          <table className="w-full min-w-[42rem] border-collapse text-sm">
            <thead>
              <tr className="border-b border-rule-strong text-left text-ink-soft">
                <th className="py-2 pr-4 font-medium">Improvement over…</th>
                <th className="py-2 pr-4 font-medium">This model</th>
                <th className="py-2 pr-4 font-medium">CGM-JEPA</th>
                <th className="py-2 font-medium">Reading</th>
              </tr>
            </thead>
            <tbody className="tnum">
              <tr className="border-b border-rule">
                <td className="py-3 pr-4 text-ink-soft">Clinical metrics</td>
                <td className="py-3 pr-4 font-semibold text-ink">
                  +{RESULTS.vsClinical.roc.toFixed(4)}
                  <span className="block text-xs font-normal text-ink-faint">
                    [{RESULTS.vsClinical.ciLow.toFixed(4)}, {RESULTS.vsClinical.ciHigh.toFixed(4)}]
                  </span>
                </td>
                <td className="py-3 pr-4 text-ink-soft">
                  {RESULTS.comparatorVsClinical.roc.toFixed(4)}
                  <span className="block text-xs text-ink-faint">
                    [{RESULTS.comparatorVsClinical.ciLow.toFixed(4)},{" "}
                    {RESULTS.comparatorVsClinical.ciHigh.toFixed(4)}]
                  </span>
                </td>
                <td className="py-3 text-ink-faint">
                  This interval excludes zero; the comparator&rsquo;s does not.
                </td>
              </tr>
              <tr>
                <td className="py-3 pr-4 text-ink-soft">Raw readings</td>
                <td className="py-3 pr-4 font-semibold text-ink">
                  +{RESULTS.vsRawMasked.roc.toFixed(4)}
                </td>
                <td className="py-3 pr-4 text-ink-soft">
                  +{RESULTS.comparatorVsRawMasked.roc.toFixed(4)}
                </td>
                <td className="py-3 text-ink-faint">Both improve on raw input.</td>
              </tr>
            </tbody>
          </table>
        </div>

        <p className="measure mt-6 text-ink-soft">
          Brackets are 95% confidence intervals over seeds. The macro advantage is real and
          its interval excludes zero, but it does not survive being broken into single
          questions: this model is ahead of the clinical baseline on{" "}
          {RESULTS.perTaskAhead.model} of {RESULTS.perTaskAhead.total} task-cohort rows, and on{" "}
          <strong>none of them</strong> does the difference reach significance after Holm
          correction — in any seed. Each row has 29 to 100 people in it. What the evidence
          supports is a small consistent lift across tasks, not a set of per-task wins.
        </p>
        <Reproduce
          command="just eval-headline && just head-to-head"
          file="reports/eval/head_to_head_5seed.csv"
          note="~80 min on an RTX 3090"
        />
      </section>

      {/* ------------------------------------------------------------ ablations */}
      <section className="mt-12 border-t border-rule pt-10">
        <h2 className="text-xl font-semibold text-ink">Ablations</h2>
        <p className="measure mt-4 text-ink-soft">
          Each design choice was removed and the model retrained: ten conditions, three seeds
          each, 40 epochs. The seed-level standard deviation is{" "}
          <span className="tnum">{ABLATION_SD.toFixed(4)}</span>, so differences smaller than
          that are noise.
        </p>

        <div className="mt-6 overflow-x-auto">
          <table className="w-full min-w-[38rem] border-collapse text-sm">
            <thead>
              <tr className="border-b border-rule-strong text-left text-ink-soft">
                <th className="py-2 pr-4 font-medium">Remove this…</th>
                <th className="py-2 pr-4 font-medium">Score</th>
                <th className="py-2 pr-4 font-medium">Change</th>
                <th className="py-2 font-medium">Verdict</th>
              </tr>
            </thead>
            <tbody className="tnum">
              {ABLATIONS.map((a) => {
                const real = Math.abs(a.delta) > ABLATION_SD;
                return (
                  <tr key={a.key} className="border-b border-rule">
                    <td className="py-2.5 pr-4 text-ink-soft">{a.label}</td>
                    <td className="py-2.5 pr-4 text-ink">{a.mean.toFixed(4)}</td>
                    <td
                      className={`py-2.5 pr-4 ${
                        a.delta < -ABLATION_SD ? "font-semibold text-low" : "text-ink-faint"
                      }`}
                    >
                      {a.delta > 0 ? "+" : ""}
                      {a.delta.toFixed(4)}
                    </td>
                    <td className="py-2.5 text-xs text-ink-faint">
                      {real
                        ? a.delta < 0
                          ? "load-bearing"
                          : "removing it helped slightly"
                        : "within noise"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <div className="measure mt-6 space-y-3 text-ink-soft">
          <p>
            The event stream alone loses <span className="tnum">0.0504</span>, six times the
            noise level. Removing the decomposition entirely loses{" "}
            <span className="tnum">0.0233</span>. The state/event split is where most of the
            benefit comes from.
          </p>
          <Reproduce
            command="just ablation-sweep && just ablation-aggregate"
            file="reports/eval/tier1_ablations_3seed.csv"
            note="~8 h on an RTX 5090"
          />
          <p>
          </p>
          <p>
            Two results went against expectation. Augmentation contributes nothing measurable.
            And interpolating the sensor gaps, which the design refuses to do, scores{" "}
            <span className="tnum">+0.0005</span> — level with the masked version at this
            evaluation depth. The never-interpolate rule is defensible on other grounds, but at
            40 epochs it is not supported by a measured gain.
          </p>
        </div>
      </section>

      {/* ------------------------------------------------------- artefacts */}
      <section className="mt-12 border-t border-rule pt-10">
        <h2 className="text-xl font-semibold text-ink">Artefacts</h2>
        <p className="measure mt-4 text-ink-soft">
          Everything the numbers above were computed from, and the files this site serves.
        </p>

        <div className="mt-6 overflow-x-auto">
          <table className="w-full min-w-[40rem] border-collapse text-sm">
            <tbody className="tnum">
              <tr className="border-b border-rule">
                <td className="py-2.5 pr-4">
                  <a href="/models/encoder.onnx" className="text-accent hover:underline">
                    encoder.onnx
                  </a>
                </td>
                <td className="py-2.5 pr-4 text-ink-soft">
                  {(ARTIFACT.bytes / 1e6).toFixed(2)} MB
                </td>
                <td className="py-2.5 pr-4 font-mono text-xs text-ink-faint">
                  sha256 {ARTIFACT.sha256.slice(0, 16)}…
                </td>
                <td className="py-2.5 text-xs text-ink-faint">CC-BY-NC-4.0</td>
              </tr>
              <tr className="border-b border-rule">
                <td className="py-2.5 pr-4">
                  <a href="/models/heads.json" className="text-accent hover:underline">
                    heads.json
                  </a>
                </td>
                <td className="py-2.5 pr-4 text-ink-soft">18 probes</td>
                <td className="py-2.5 pr-4 text-xs text-ink-faint">
                  14 above the signal floor
                </td>
                <td className="py-2.5 text-xs text-ink-faint">CC-BY-NC-SA-4.0</td>
              </tr>
              <tr className="border-b border-rule">
                <td className="py-2.5 pr-4">
                  <a href="/models/encoder.meta.json" className="text-accent hover:underline">
                    encoder.meta.json
                  </a>
                </td>
                <td className="py-2.5 pr-4 text-ink-soft">provenance</td>
                <td className="py-2.5 pr-4 text-xs text-ink-faint">
                  checkpoint, epoch, seed, architecture flags
                </td>
                <td className="py-2.5 text-xs text-ink-faint">Apache-2.0</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className="mt-6 grid gap-x-8 gap-y-2 text-sm sm:grid-cols-2">
          <a href={LINKS.headToHead} className="text-accent hover:underline">
            findings/head_to_head.md — per-seed and per-task tables →
          </a>
          <a href={LINKS.ablations} className="text-accent hover:underline">
            findings/tier1_ablations.md — the full ablation matrix →
          </a>
          <a href={LINKS.decisions} className="text-accent hover:underline">
            DECISIONS.md — {AMBIGUITY.decisionEntries} dated decisions →
          </a>
          <a href={LINKS.reproduce} className="text-accent hover:underline">
            REPRODUCE.md — measured wall-clocks, copy-paste commands →
          </a>
          <a href={LINKS.comparatorPort} className="text-accent hover:underline">
            The CGM-JEPA port, so the comparison is checkable →
          </a>
          <a href={LINKS.modelCard} className="text-accent hover:underline">
            Model card — intended use, limits, withheld heads →
          </a>
        </div>

        <div className="mt-8">
          <p className="text-sm font-medium text-ink-soft">Cite</p>
          <pre className="mt-2 overflow-x-auto border border-rule bg-ink/[0.03] p-4 text-xs leading-relaxed text-ink-soft">
            <code>{`@software{fourdrinier_opencgm_stateevent,
  author  = {Fourdrinier, Stephane},
  title   = {OpenCGM-StateEvent: an independent public-data reconstruction
             of the GlucoFM dual-stream CGM foundation model},
  year    = {2026},
  url     = {${LINKS.repo}},
  note    = {Code Apache-2.0; encoder CC-BY-NC-4.0; probe heads CC-BY-NC-SA-4.0}
}`}</code>
          </pre>
        </div>
      </section>

      {/* ------------------------------------------------------------- caveats */}
      <section className="mt-16 grid gap-6 border-t border-rule pt-12 md:grid-cols-2">
        <Callout title="Before quoting these numbers" tone="warn">
          <p>
            Training used {(CORPUS.fractionOfPaper * 100).toFixed(1)}% of GlucoFM&rsquo;s
            pretraining hours; the remainder is not public. These scores are lower than the
            paper&rsquo;s and are expected to be. The comparison that holds is against the
            baselines on this page, all measured on the same folds.
          </p>
        </Callout>
        <Callout title="Longer training was worse">
          <p>
            Transfer peaks near epoch 40 and declines by epoch 120, a drop of about one seed
            standard deviation. On a corpus this size the model has more capacity than the data
            supports. The headline above reports epoch 120, the pre-registered endpoint, rather
            than the epoch that scores best. The encoder this site and its API serve is seed{" "}
            {ARTIFACT.seed} at epoch {ARTIFACT.epoch}, the transfer peak — a different
            checkpoint from the one the headline is computed over (D024).
          </p>
        </Callout>
      </section>
    </div>
  );
}
