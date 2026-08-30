// Small presentational pieces shared across the explanatory pages.

export function Stat({
  value,
  label,
  note,
}: {
  value: string;
  label: string;
  note?: string;
}) {
  return (
    <div className="border-l-2 border-accent pl-4">
      <div className="tnum text-2xl font-semibold text-ink">{value}</div>
      <div className="text-sm text-ink-soft">{label}</div>
      {note ? <div className="mt-1 text-xs text-ink-faint">{note}</div> : null}
    </div>
  );
}

export function Callout({
  title,
  tone = "accent",
  children,
}: {
  title: string;
  tone?: "accent" | "warn";
  children: React.ReactNode;
}) {
  const palette =
    tone === "warn"
      ? "border-warn/30 bg-warn-soft text-ink-soft"
      : "border-accent/25 bg-accent-soft text-ink-soft";
  return (
    <aside className={`rounded-lg border px-5 py-4 ${palette}`}>
      <p className="text-sm font-semibold text-ink">{title}</p>
      <div className="mt-2 text-sm leading-relaxed">{children}</div>
    </aside>
  );
}

/** A horizontal bar comparing one score against a reference line. */
export function ScoreBar({
  label,
  value,
  min = 0.5,
  max = 0.72,
  emphasis = false,
  caption,
  baseline = false,
}: {
  label: string;
  value: number;
  min?: number;
  max?: number;
  emphasis?: boolean;
  caption?: string;
  /** Draw as a reference line rather than a bar. Used for chance, which has zero length. */
  baseline?: boolean;
}) {
  const pct = Math.max(0, Math.min(1, (value - min) / (max - min))) * 100;
  return (
    <div className="grid grid-cols-[minmax(9rem,14rem)_1fr_4rem] items-center gap-4 py-1.5">
      <div className={`text-sm ${emphasis ? "font-semibold text-ink" : "text-ink-soft"}`}>
        {label}
        {caption ? <span className="block text-xs text-ink-faint">{caption}</span> : null}
      </div>
      <div className="relative h-2.5 bg-rule/70">
        {baseline ? (
          <div className="absolute inset-y-0 left-0 w-px bg-ink-faint" />
        ) : (
          <div
            className={emphasis ? "h-2.5 bg-accent" : "h-2.5 bg-ink-faint/50"}
            style={{ width: `${pct}%` }}
          />
        )}
      </div>
      <div className="tnum text-right text-sm text-ink">{value.toFixed(3)}</div>
    </div>
  );
}

/** Ticks under a group of ScoreBars, so the truncated axis is visible on the figure. */
export function ScoreAxis({ min = 0.5, max = 0.72 }: { min?: number; max?: number }) {
  const ticks = [0.5, 0.55, 0.6, 0.65, 0.7].filter((v) => v >= min && v <= max);
  return (
    <div className="grid grid-cols-[minmax(9rem,14rem)_1fr_4rem] gap-4 pt-1">
      <div />
      <div>
        <div className="relative h-4">
          {ticks.map((v) => (
            <div
              key={v}
              className="absolute top-0 flex flex-col items-center"
              style={{ left: `${((v - min) / (max - min)) * 100}%` }}
            >
              <div className="h-1.5 w-px bg-rule-strong" />
              <span className="tnum mt-0.5 -translate-x-1/2 text-[10px] text-ink-faint">
                {v.toFixed(2)}
              </span>
            </div>
          ))}
        </div>
        <p className="mt-1.5 text-[10px] text-ink-faint">
          ROC-AUC. The axis starts at {min.toFixed(2)} — chance — not at zero.
        </p>
      </div>
      <div />
    </div>
  );
}

/**
 * How to regenerate the numbers in a table, next to the table.
 *
 * Every figure on this site is produced by a script in the repository. Saying so is worth
 * little; showing the command and linking the CSV it wrote is worth something, because a
 * reader can run it.
 */
export function Reproduce({
  command,
  file,
  href,
  note,
}: {
  command: string;
  file?: string;
  href?: string;
  note?: string;
}) {
  return (
    <div className="mt-4 flex flex-wrap items-baseline gap-x-4 gap-y-1 border-l-2 border-rule-strong pl-4 text-xs text-ink-faint">
      <span>Regenerate:</span>
      <code className="font-mono text-ink-soft">{command}</code>
      {file ? (
        href ? (
          <a href={href} className="text-accent hover:underline">
            {file}
          </a>
        ) : (
          <code className="font-mono">{file}</code>
        )
      ) : null}
      {note ? <span>{note}</span> : null}
    </div>
  );
}
