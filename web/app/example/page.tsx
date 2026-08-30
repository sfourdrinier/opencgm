import { ExampleAnalysisView } from "@/components/ExampleAnalysis";
import { loadExample } from "@/lib/example-data";

export const metadata = {
  title: "A worked example — OpenCGM-StateEvent",
  description:
    "A real week of glucose data, run through the model and explained in full — including " +
    "the days it refuses to score.",
};

export default function ExamplePage() {
  const data = loadExample();
  const percentiles = data.days
    .flatMap((d) => d.probes.map((p) => p.percentile))
    .filter((v): v is number => v != null);
  const pctLow = percentiles.length ? Math.min(...percentiles) : 0;
  const pctHigh = percentiles.length ? Math.max(...percentiles) : 100;

  return (
    <div className="mx-auto max-w-6xl px-6 py-12">
      <h1 className="text-3xl font-semibold tracking-tight text-ink md:text-4xl">
        What the model actually tells you
      </h1>
      <p className="lede mt-5 max-w-3xl">
        {data.n_days} real days from one person&rsquo;s sensor, run through the released model.
        Nothing below is illustrative — these are the outputs, including the ones that are
        unhelpful.
      </p>

      <div className="mt-6 border-l-2 border-accent bg-accent-soft/40 py-3 pl-5 text-sm text-ink-soft">
        <p>
          The recording is from a consenting adult volunteer and is de-identified: no name,
          device, sensor identifier or calendar date is carried through. The days are numbered
          in the order they were recorded.
        </p>
      </div>

      <ExampleAnalysisView data={data} />

      {/* ------------------------------------------------------- reading the result */}
      <section className="mt-12 border-t border-rule pt-10">
        <h2 className="text-xl font-semibold text-ink">What this particular reading means</h2>
        <div className="mt-4 max-w-3xl space-y-4 text-ink-soft">
          <p>
            Across every day and every classifier, this person&rsquo;s percentiles run from the{" "}
            {Math.round(pctLow)}th to the {Math.round(pctHigh)}th. There is no single verdict
            in there. The two classifiers with the highest held-out accuracy place the
            fully-recorded days low, which reads as reassuring; weaker ones place the same days
            near the middle or above it. Where they disagree, the accuracy column is the only
            tiebreaker on offer, and even the best of them is a research result fitted on a few
            dozen people.
          </p>
          <p>
            A low ranking here is also worth reading carefully. The training corpus is largely
            people with diabetes or prediabetes, so a day from someone without it is genuinely
            unusual within that corpus. The model is saying this day looks unlike its training
            data. For this corpus that happens to coincide with looking healthy. They are not
            the same statement, and only one of them is something the model can support.
          </p>
          <p>
            Day-to-day variability is not a straight line either: the coefficient of variation
            runs{" "}
            <span className="tnum">
              {data.days
                .filter((d) => d.coverage >= 0.9)
                .map((d) => d.clinical?.coefficient_of_variation.toFixed(2))
                .join(", ")}
            </span>{" "}
            across the complete days — climbing to a peak and settling again, not rising
            steadily. Clinical guidance generally treats anything under 0.36 as stable, so all
            of them sit inside that.
          </p>
        </div>
      </section>

      {/* -------------------------------------------------------------- what it isn't */}
      <section className="mt-12 grid gap-6 border-t border-rule pt-10 md:grid-cols-2">
        <div>
          <h3 className="text-base font-semibold text-ink">What you get</h3>
          <ul className="mt-3 space-y-2 text-sm text-ink-soft">
            <li>
              <strong className="text-ink">A 128-number summary of a day.</strong> Days that
              behave alike land close together, and you can measure that distance.
            </li>
            <li>
              <strong className="text-ink">The slow and fast parts of the day, separated.</strong>{" "}
              Useful on its own for seeing which excursions are meals and which are drift.
            </li>
            <li>
              <strong className="text-ink">A ranking against 20,000 corpus days</strong> for each
              question the classifiers can answer for that day — at most seven — each labelled
              with how accurate that classifier is.
            </li>
            <li>
              <strong className="text-ink">A refusal</strong> when the recording is too sparse
              for a classifier to have anything to say.
            </li>
          </ul>
        </div>
        <div>
          <h3 className="text-base font-semibold text-ink">What you do not get</h3>
          <ul className="mt-3 space-y-2 text-sm text-ink-soft">
            <li>
              <strong className="text-ink">A prediction of your future glucose.</strong> The
              model reads a day that has already happened. It forecasts nothing.
            </li>
            <li>
              <strong className="text-ink">A diagnosis, or anything close to one.</strong> The
              best classifier here is wrong about one person in four, and it was fitted on a few
              dozen people.
            </li>
            <li>
              <strong className="text-ink">A probability.</strong> The raw classifier outputs
              saturate near 0 and 1 regardless of the input, which is why this page shows
              rankings instead.
            </li>
            <li>
              <strong className="text-ink">Advice.</strong> Nothing here should change what you
              eat, what you take, or what you do. That conversation belongs with a clinician.
            </li>
          </ul>
        </div>
      </section>
    </div>
  );
}
