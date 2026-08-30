import { Callout } from "@/components/Figures";
import { CORPUS, MODEL, PROTOCOL } from "@/lib/facts";

function Step({ n, title, children }: { n: number; title: string; children: React.ReactNode }) {
  return (
    <div className="grid gap-6 border-t border-rule py-7 md:grid-cols-[3rem_1fr]">
      <div className="tnum text-2xl font-semibold text-accent">{n}</div>
      <div>
        <h3 className="text-lg font-semibold text-ink">{title}</h3>
        <div className="mt-3 space-y-3 text-ink-soft">{children}</div>
      </div>
    </div>
  );
}

export default function HowItWorks() {
  return (
    <div className="mx-auto max-w-6xl px-6 py-12">
      <h1 className="text-3xl font-semibold tracking-tight text-ink md:text-4xl">How it works</h1>

      <div className="mt-12">
        <Step n={1} title="Start with one day, exactly as it was recorded">
          <p>
            A glucose sensor reports every five minutes, so a day is a row of 288 numbers. Real
            sensors miss readings — the sensor warms up, it falls off, someone showers. Most
            software quietly fills those gaps in by drawing a straight line through them.
          </p>
          <p>
            This model does not. A model cannot distinguish an interpolated reading from a
            measured one, so interpolation teaches it the shape of the interpolation. Each of the
            288 slots carries a flag marking whether a reading is real, and every later step reads
            that flag.
          </p>
        </Step>

        <Step n={2} title="Split the day into a slow part and a fast part">
          <p>
            A glucose trace is really two things layered on top of each other. There is the slow
            drift across the day — where your body sits when nothing much is happening. And there
            are the spikes: a meal, exercise, a dose of insulin.
          </p>
          <p>
            A smoothing filter separates them. It looks only backwards in time, so no reading is
            influenced by the future. The smoothed line is the <strong>state</strong> stream; the
            remainder is the <strong>event</strong> stream. Both are kept.
          </p>
          <p>
            The filter width is not fixed in advance. It is a trained parameter, so the model
            arrives at its own boundary between slow and fast.
          </p>
        </Step>

        <Step n={3} title="Hide part of the day and make the model guess it">
          <p>
            The day is divided into 24 hourly patches and between half and 60% of them are
            hidden, redrawn for every window. One copy of the model sees what is left and
            predicts the hidden patches. A second copy,
            updated slowly from the first, sees everything and provides the target. No labels are
            involved.
          </p>
          <p>
            Doing this well requires learning how glucose behaves: how a spike decays, how
            overnight differs from afternoon. The model is also given the time of day, which the
            ablations show is load-bearing.
          </p>
          <p>
            Patches are hidden before any normalisation or statistics are computed. Hiding them
            afterwards leaks information about the hidden patches into the visible ones, and the
            resulting model scores well for the wrong reason.
          </p>
        </Step>

        <Step n={4} title="Keep the fingerprint, throw away the scaffolding">
          <p>
            After training, the prediction machinery is discarded. What remains is the encoder
            that maps 288 readings to 128 numbers:{" "}
            {MODEL.encoderParams.toLocaleString()} parameters, about 2 MB. The{" "}
            <a href="/try" className="text-accent hover:underline">demo page</a> runs it in the
            browser; the <a href="/api" className="text-accent hover:underline">API</a> runs the
            same file server-side.
          </p>
        </Step>

        <Step n={5} title="Check whether the fingerprint is any good">
          <p>
            The encoder is frozen and its 128 numbers are handed to a logistic regression. If a
            linear classifier can answer a clinical question from them, the information was
            already present in the representation.
          </p>
          <p>
            Splits are by person, not by day. Testing on a different day from someone who also
            appears in training measures how well the model recognises that person.
          </p>
          <p>
            The {PROTOCOL.folds}-fold split is repeated {PROTOCOL.repeats} times across{" "}
            {PROTOCOL.probes} clinical questions and {PROTOCOL.seeds.length} independently
            trained models. Reported numbers are the mean and spread, not the best run.
          </p>
        </Step>
      </div>

      <div className="mt-10 grid gap-6 md:grid-cols-2">
        <Callout title="Where the data comes from">
          <p>
            {CORPUS.subjects} people and {CORPUS.hours.toLocaleString()} hours from four public
            cohorts: {CORPUS.cohorts.join(", ")}. Datasets with non-commercial or no-derivatives
            licences are used for evaluation only and never for training. A cohort the model
            cannot train on carries no leakage risk, which makes it a clean generalisation test.
          </p>
        </Callout>
        <Callout title="What the paper does not specify" tone="warn">
          <p>
            Around nineteen implementation details are not stated in the paper. Each is recorded
            as a numbered decision with the alternatives considered. Where the choice looked
            consequential, both options were trained and the measured result is reported.
          </p>
        </Callout>
      </div>
    </div>
  );
}
