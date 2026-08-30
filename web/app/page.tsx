import Link from "next/link";
import { Callout, ScoreAxis, ScoreBar, Stat } from "@/components/Figures";
import { StreamsThumb, WeekAtAGlance } from "@/components/ExampleCharts";
import { GlossaryList, Term } from "@/components/Glossary";
import { loadExample } from "@/lib/example-data";
import { AMBIGUITY, CORPUS, LINKS, MODEL, QUESTIONS, RESULTS } from "@/lib/facts";

export default function Home() {
  const example = loadExample();
  const complete = example.days.filter((d) => d.coverage >= 0.9);
  // The most-alike pair and how far apart their peaks are, computed rather than typed: the
  // example is regenerated from whatever recording is current, and prose that names a
  // number the data no longer supports is the failure this site keeps having to fix.
  const pair = (() => {
    const pairs = example.similarity.flatMap((row, i) =>
      row.flatMap((s, j) => (j > i ? [{ a: i, b: j, s }] : [])),
    );
    if (pairs.length === 0) return null;
    const best = pairs.reduce((x, y) => (y.s > x.s ? y : x));
    const a = example.days[best.a]!;
    const b = example.days[best.b]!;
    return {
      similarity: best.s,
      peakGap: Math.abs((a.clinical?.max_mg_dl ?? 0) - (b.clinical?.max_mg_dl ?? 0)),
    };
  })();
  // The day with the largest excursion shows the decomposition most clearly.
  const busiest = complete.reduce<(typeof complete)[number] | undefined>(
    (a, b) => ((a?.clinical?.max_mg_dl ?? 0) >= (b.clinical?.max_mg_dl ?? 0) ? a : b),
    complete[0],
  );

  return (
    <div className="mx-auto max-w-6xl px-6">
      <section className="border-b border-rule py-12 md:py-16">
        <p className="text-sm uppercase tracking-widest text-accent">
          Open reconstruction · public data only
        </p>
        <h1 className="mt-4 max-w-4xl text-4xl font-semibold leading-tight tracking-tight text-ink md:text-5xl">
          Two glucose days can share every summary number and be nothing alike.
        </h1>
        <p className="lede mt-6 max-w-3xl">
          Your CGM app reduces each day to an average and a time-in-range. This model reads the
          shape of the whole day instead — 2 MB, running in your browser tab, with your readings
          never uploaded. It is an open reconstruction of Google&rsquo;s GlucoFM, a foundation
          model published as a paper with no code and no weights.
        </p>
        <div className="mt-8 flex flex-wrap gap-3">
          <Link
            href="/try"
            className="bg-accent px-5 py-2.5 text-sm font-medium text-white hover:bg-accent-ink"
          >
            Try it on a sample day
          </Link>
          <Link
            href="/api"
            className="border border-rule-strong px-5 py-2.5 text-sm font-medium text-ink hover:border-accent hover:text-accent"
          >
            Use the API
          </Link>
          <p className="mt-1 w-full text-sm text-ink-soft">
            No file needed — sample days are built in. Or drop in your own Dexcom or Libre
            export; it never leaves your browser.
          </p>
        </div>
      </section>

      {/* The author's own words. Kept short so it does not compete with the result. */}
      <section className="border-b border-rule py-10">
        <div className="border-l-2 border-accent pl-6">
          <div className="space-y-4 text-ink-soft">
            <p>
              Cardiovascular health and blood glucose management are critical to health, and I
              follow the field closely. GlucoFM was the most interesting recent model in this
              space, and Google published no code, so I rebuilt it on public data and published
              everything &mdash; the weights, the failures, and the decisions the paper left
              open.
            </p>
            <p>
              These are the first findings I&rsquo;ve published openly. They won&rsquo;t be the
              last. A PPG-to-glucose pilot is already in{" "}
              <Link href="/paper" className="text-accent hover:underline">
                the paper
              </Link>
              , and a research studio app for that work is next.
            </p>
          </div>
          <p className="mt-4 text-sm text-ink-faint">
            <a href={LINKS.author} className="font-medium text-ink-soft hover:text-accent">
              Stephane Fourdrinier
            </a>{" "}
            &mdash; independent researcher
          </p>
        </div>
      </section>

      <section className="grid gap-12 border-b border-rule py-12 md:grid-cols-[1.1fr_1fr]">
        <div>
          <h2 className="text-2xl font-semibold tracking-tight text-ink">What it does</h2>
          <div className="measure mt-5 space-y-4 text-ink-soft">
            <p>
              A CGM sensor reports every five minutes, so one day is 288 readings. The model
              compresses that day into 128 numbers, and gives days that behaved alike nearly the
              same numbers — even when the raw traces look different.
            </p>
            <p>
              Training used no diagnoses. Between half and 60% of each day was hidden and the
              model was trained to predict the hidden hours from what remained. Labels came later, to
              test whether the resulting representation carried clinical information.
            </p>
            <p>
              It does, modestly. Averaged over {RESULTS.perTaskSignificant.total} clinical
              questions, a simple <Term name="classifier" /> reading the model&rsquo;s numbers
              beats one reading the standard summary metrics — <Term name="time in range" />,
              mean, variability — by 0.027 <Term name="ROC-AUC" />. The{" "}
              <Term name="point estimate" /> is ahead on {RESULTS.perTaskAhead.model} of the{" "}
              {RESULTS.perTaskAhead.total}, and no single question shows a gap large enough to
              be <Term name="statistically significant">significant</Term> on its own. The
              advantage is an average, and this site treats it as one.
            </p>
          </div>
        </div>

        <div className="space-y-6 self-start rounded-xl border border-rule bg-paper-raised p-6">
          <Stat
            value={MODEL.encoderParams.toLocaleString()}
            label="parameters"
            note="About 2 MB. It runs in a browser tab in under a second."
          />
          <Stat
            value={`${CORPUS.hours.toLocaleString()} h`}
            label="of public CGM data"
            note={`${CORPUS.subjects} people, ${CORPUS.cohorts.length} cohorts, ${CORPUS.windows.toLocaleString()} 24-hour windows.`}
          />
          <Stat
            value="0"
            label="private datasets"
            note="Anyone can download the training data and check the result."
          />
        </div>
      </section>

      <section className="border-b border-rule py-12">
        <h2 className="text-2xl font-semibold tracking-tight text-ink">
          What you actually get out of it
        </h2>
        <p className="mt-5 max-w-3xl text-ink-soft">
          The model returns 128 numbers for a day. On their own they mean nothing; what matters
          is what can be read off them. Small classifiers fitted on top were trained to answer
          seven questions, and how well each one works was measured on people it had never
          seen:
        </p>

        <div className="mt-6 overflow-x-auto">
          <table className="w-full min-w-[38rem] border-collapse text-sm">
            <thead>
              <tr className="border-b border-rule-strong text-left text-ink-soft">
                <th className="py-2 pr-4 font-medium">Question</th>
                <th className="py-2 pr-4 font-medium">What it means</th>
                <th className="py-2 font-medium">Best accuracy</th>
              </tr>
            </thead>
            <tbody>
              {QUESTIONS.map((q) => (
                <tr key={q.label} className="border-b border-rule align-top">
                  <td className="py-2.5 pr-4 text-ink">{q.label}</td>
                  <td className="py-2.5 pr-4 text-ink-soft">{q.plain}</td>
                  <td className="tnum py-2.5 whitespace-nowrap text-ink-soft">
                    {q.bestAuc.toFixed(2)}
                    <span className="ml-1 text-xs text-ink-faint">
                      {q.bestAuc >= 0.75 ? "moderate" : q.bestAuc >= 0.65 ? "weak" : "very weak"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-3 max-w-3xl text-xs text-ink-faint">
          Accuracy is <Term name="ROC-AUC" /> on held-out people: 0.5 is a coin flip, 1.0 is
          perfect. Nothing
          here reaches the accuracy of a clinical test, and the site never shows these as
          percentages — see below for why. Alongside the seven, the model gives you three things
          that need no classifier at all:
        </p>

        <div className="mt-6 grid gap-x-10 gap-y-6 md:grid-cols-2">
          <div className="border-l-2 border-accent pl-5">
            <h3 className="text-base font-semibold text-ink">The day split into slow and fast</h3>
            <p className="mt-2 text-sm text-ink-soft">
              A backward-looking filter separates the slow drift (green) from the spikes on top
              of it (rust). Meals and exercise land in the fast part. Useful on its own, needs
              no classifier, and is the only part of the design the ablations show to matter.
            </p>
            {busiest ? (
              <div className="mt-3 border border-rule bg-paper-raised p-2">
                <StreamsThumb day={busiest} />
              </div>
            ) : null}
          </div>
          <div className="border-l-2 border-accent pl-5">
            <h3 className="text-base font-semibold text-ink">Your days compared to each other</h3>
            <p className="mt-2 text-sm text-ink-soft">
              Days that behaved alike land close together in the 128 numbers. With a couple of
              weeks of data you can see which days were unlike the rest — again with no
              classifier involved.
            </p>
          </div>
          <div className="border-l-2 border-accent pl-5">
            <h3 className="text-base font-semibold text-ink">A ranking against 20,000 days</h3>
            <p className="mt-2 text-sm text-ink-soft">
              For each question the classifiers can answer for a given day — at most seven,
              fewer when readings are missing — where it sits among days from the training
              cohorts, with that classifier&rsquo;s measured accuracy printed next to it, so you
              can weigh how much the answer is worth.
            </p>
          </div>
          <div className="border-l-2 border-warn pl-5">
            <h3 className="text-base font-semibold text-ink">And a refusal, when that is right</h3>
            <p className="mt-2 text-sm text-ink-soft">
              If a day has too many missing readings, the classifiers decline to score it rather
              than extrapolate. And no result is ever shown as a probability: the raw scores pile
              up near 0 or 1 whatever you feed them — 0.996 can describe a thoroughly ordinary
              day — so every answer is a rank among 20,000 real days. No forecast, no diagnosis.
              A research tool, not a medical device.
            </p>
          </div>
        </div>

        <div className="mt-6 flex flex-wrap items-center gap-3">
          <Link
            href="/try"
            className="bg-accent px-5 py-2.5 text-sm font-medium text-white hover:bg-accent-ink"
          >
            Run it on your own export
          </Link>
          <Link
            href="/example"
            className="border border-rule-strong px-5 py-2.5 text-sm font-medium text-ink hover:border-accent hover:text-accent"
          >
            See a worked example first
          </Link>
          <p className="mt-1 w-full text-sm text-ink-soft">
            The analysis runs entirely in your browser; your readings are not sent to any
            server. Dexcom, Libre, or any CSV with a time column and a glucose column.
          </p>
        </div>
      </section>

      <section className="border-b border-rule py-12">
        <h2 className="text-2xl font-semibold tracking-tight text-ink">
          All of it, on one real week
        </h2>
        <p className="mt-5 max-w-3xl text-ink-soft">
          {example.n_days} days from one volunteer&rsquo;s sensor, on one shared scale. Every
          complete day is at least{" "}
          {(Math.min(...complete.map((d) => d.clinical?.time_in_range_70_180 ?? 1)) * 100).toFixed(0)}%
          inside the 70&ndash;180 mg/dL range a clinician would target, with averages between{" "}
          {Math.min(...complete.map((d) => d.clinical?.mean_mg_dl ?? 0)).toFixed(0)} and{" "}
          {Math.max(...complete.map((d) => d.clinical?.mean_mg_dl ?? 0)).toFixed(0)}. By the
          numbers a CGM app reports, these days are interchangeable.
        </p>

        <div className="mt-6">
          <WeekAtAGlance days={example.days} secondStat="tir" href="/example" />
        </div>

        <p className="mt-5 max-w-3xl text-ink-soft">
          The model reads them differently. It puts two of these days almost on top of each
          other{pair ? ` (${pair.similarity.toFixed(3)} similar)` : ""} despite peaks{" "}
          {pair ? pair.peakGap.toFixed(0) : "16"} mg/dL apart, and separates out the day that
          climbed to{" "}
          {Math.max(...complete.map((d) => d.clinical?.max_mg_dl ?? 0)).toFixed(0)}. Time in
          range barely distinguishes any of them. The shape of the day does.
        </p>
        <div className="mt-7 border-t border-rule pt-6">
          <div className="grid gap-6 md:grid-cols-[1fr_auto] md:items-end">
            <div>
              <p className="text-sm font-medium text-ink">
                The worked example takes one of these days all the way through
              </p>
              <p className="mt-2 max-w-2xl text-sm text-ink-soft">
                The day split into slow and fast, the 128 numbers it becomes, which days the
                model reads as alike, every classifier that would answer for it and every one
                that refuses — and, using two of these days, why a raw score of 0.996 can mean
                almost nothing.
              </p>
            </div>
            <Link
              href="/example"
              className="whitespace-nowrap bg-accent px-5 py-2.5 text-sm font-medium text-white hover:bg-accent-ink"
            >
              Walk through the week →
            </Link>
          </div>
        </div>
      </section>

      <section className="border-b border-rule py-12">
        <h2 className="text-2xl font-semibold tracking-tight text-ink">Results</h2>
        <p className="measure mt-5 text-ink-soft">
          <Term name="ROC-AUC" /> across {RESULTS.perTaskSignificant.total} task-cohort
          combinations, five <Term name="seed">seeds</Term>, split by person into{" "}
          <Term name="fold">folds</Term> so nobody appears in both halves. 0.5 is chance; 1.0
          would be perfect.
        </p>

        <div className="mt-8 max-w-3xl">
          <ScoreBar label="Chance" value={0.5} baseline />
          <ScoreBar
            label="Raw readings"
            value={RESULTS.rocAuc.rawMasked}
            caption="the 288 values, fed to the same classifier"
          />
          <ScoreBar
            label="Clinical metrics"
            value={RESULTS.rocAuc.clinical}
            caption="time-in-range, mean, variability"
          />
          <ScoreBar
            label="CGM-JEPA"
            value={RESULTS.rocAuc.comparator}
            caption="the nearest published model, on identical folds"
          />
          <ScoreBar label="OpenCGM-StateEvent" value={RESULTS.rocAuc.model} emphasis />
          <ScoreAxis />
        </div>

        <div className="measure mt-8 space-y-4 text-ink-soft">
          <p>
            Against clinical metrics the margin is{" "}
            <span className="tnum">+{RESULTS.vsClinical.roc.toFixed(4)}</span> ROC-AUC, 95% CI{" "}
            <span className="tnum">
              [{RESULTS.vsClinical.ciLow.toFixed(4)}, {RESULTS.vsClinical.ciHigh.toFixed(4)}]
            </span>
            . All five seeds are positive. CGM-JEPA, measured the same way on the same folds,
            comes in at <span className="tnum">{RESULTS.comparatorVsClinical.roc.toFixed(4)}</span>{" "}
            with a <Term name="confidence interval" /> spanning zero.
          </p>
          <p>
            A margin of 0.027 is small. All five training runs land positive and the interval
            excludes zero, so the average advantage is real. It does not concentrate anywhere:
            ahead on {RESULTS.perTaskAhead.model} of {RESULTS.perTaskAhead.total} questions,
            significant on none of them individually.
          </p>
        </div>

        <Link
          href="/results"
          className="mt-6 inline-block text-sm font-medium text-accent hover:underline"
        >
          Full results, ablations, and what failed →
        </Link>
      </section>

      <section className="border-b border-rule py-10">
        <GlossaryList />
      </section>

      <section className="grid gap-6 py-12 md:grid-cols-2">
        <Callout title="Not a replication" tone="warn">
          <p>
            About 69% of GlucoFM&rsquo;s {CORPUS.paperHours.toLocaleString()} pretraining hours
            come from a dataset that is not public. This model saw the{" "}
            {(CORPUS.fractionOfPaper * 100).toFixed(1)}% that is. Its scores are lower than the
            paper&rsquo;s and should be. Comparing the two numbers directly compares different
            training corpora, not different methods.
          </p>
        </Callout>

        <Callout title="What is checkable">
          <p>
            Training data is public. Every reported figure is produced by a script in the
            repository, and a test suite re-derives the headline numbers from the model and the
            corpus so the documentation cannot drift from the code. The config loader refuses
            to run unless each of the {AMBIGUITY.inferredForks} under-specified options carries
            an evidence tag, and every one is a dated entry in{" "}
            <a href={LINKS.decisions} className="text-accent hover:underline">
              DECISIONS.md
            </a>
            . Runs that failed are still in the repository.
          </p>
        </Callout>
      </section>

      <section className="border-t border-rule py-10">
        <p className="text-sm text-ink-faint">
          Method from{" "}
          <a href={LINKS.paper} className="text-accent hover:underline">
            GlucoFM, arXiv:2605.30865v2
          </a>
          . Comparator ported from the{" "}
          <a href={LINKS.comparator} className="text-accent hover:underline">
            CGM-JEPA authors&rsquo; released code
          </a>
          .
        </p>
      </section>
    </div>
  );
}
