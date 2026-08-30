"use client";

import { useState } from "react";

// What the parser needs, said plainly.
//
// Column detection reads values, not just header names, so most vendor exports work
// untouched. But someone whose file fails deserves to know exactly what is being looked for
// rather than being told to try again.

const EXAMPLES = [
  {
    id: "dexcom",
    label: "Dexcom Clarity",
    csv: `Index,Timestamp (YYYY-MM-DDThh:mm:ss),Event Type,Glucose Value (mg/dL)
1,2026-08-28T08:00:00,EGV,96
2,2026-08-28T08:05:00,EGV,98
3,2026-08-28T08:10:00,EGV,103`,
    note:
      "Export from Clarity as CSV and load it unchanged. The Event Type column is not "
      + "required — it is only used to skip calibration and carb rows, which have no "
      + "glucose value and would be dropped anyway.",
  },
  {
    id: "libre",
    label: "FreeStyle Libre",
    csv: `Device,Serial Number,Device Timestamp,Record Type,Historic Glucose mg/dL
FreeStyle,X1,28-08-2026 08:00,0,96
FreeStyle,X1,28-08-2026 08:15,0,98
FreeStyle,X1,28-08-2026 08:30,0,103`,
    note: "Day-first dates are read correctly. mmol/L columns are converted automatically.",
  },
  {
    id: "minimal",
    label: "Anything else",
    csv: `time,glucose_mg_dl
2026-08-28 08:00:00,96
2026-08-28 08:05:00,98
2026-08-28 08:10:00,103`,
    note: "Two columns is enough: something that parses as a date-time, and a glucose value.",
  },
];

export function CsvFormat() {
  const [active, setActive] = useState(EXAMPLES[0]!.id);
  const shown = EXAMPLES.find((e) => e.id === active)!;

  return (
    <details className="border border-rule bg-paper-raised">
      <summary className="cursor-pointer px-6 py-4 text-sm font-medium text-ink hover:text-accent">
        What file does it need?
      </summary>
      <div className="border-t border-rule px-6 py-5">
        <p className="max-w-3xl text-sm text-ink-soft">
          A CSV with one row per reading. The columns can be named anything — the parser looks
          at the values, not just the headers, so most vendor exports work as downloaded. It
          needs two things it can find:
        </p>
        <ul className="mt-3 max-w-3xl space-y-1.5 text-sm text-ink-soft">
          <li>
            <strong className="text-ink">A date and time per reading.</strong> ISO
            (2026-08-28T08:00:00), with or without a timezone, or a plain
            &ldquo;28-08-2026 08:00&rdquo;. A column of bare numbers is not accepted as a
            time, even if the header says &ldquo;timestamp&rdquo; — sensors export counters
            under that name and reading one as a clock silently ruins everything downstream.
          </li>
          <li>
            <strong className="text-ink">A glucose value.</strong> mg/dL or mmol/L; values
            under 40 are treated as mmol/L and converted. Anything outside 20&ndash;600 mg/dL
            is dropped.
          </li>
        </ul>

        <div className="mt-5 flex flex-wrap gap-2">
          {EXAMPLES.map((e) => (
            <button
              key={e.id}
              type="button"
              onClick={() => setActive(e.id)}
              className={`border px-3 py-1.5 text-xs ${
                e.id === active
                  ? "border-accent bg-accent-soft text-accent-ink"
                  : "border-rule-strong text-ink-soft hover:border-accent hover:text-accent"
              }`}
            >
              {e.label}
            </button>
          ))}
        </div>
        <pre className="mt-3 overflow-x-auto border border-rule bg-ink/[0.03] p-4 text-xs leading-relaxed text-ink-soft">
          <code>{shown.csv}</code>
        </pre>
        <p className="mt-2 text-xs text-ink-faint">{shown.note}</p>

        <p className="mt-5 max-w-3xl text-sm text-ink-soft">
          Send as many days as you like; the page finds each day, shows how much of it the
          sensor recorded, and lets you pick which to run. Gaps stay gaps — nothing is filled
          in. Readings marked as sensor warm-up are dropped when the export labels them.
        </p>
      </div>
    </details>
  );
}
