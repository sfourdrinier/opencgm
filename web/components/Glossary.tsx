"use client";

import { useState } from "react";
import { GLOSSARY } from "@/lib/facts";

// Terms the page cannot avoid, defined where the reader hits them.
//
// A tooltip alone would not do: it needs a mouse, and the reader most likely to need the
// definition is the one least likely to have one. So the inline marker carries the short
// definition on hover *and* the term is listed in full below, reachable by anchor.

/** Inline marker: dotted underline, hover for the one-liner, click to jump to the full entry. */
export function Term({ name, children }: { name: string; children?: React.ReactNode }) {
  const entry = GLOSSARY.find((g) => g.term === name);
  if (!entry) return <>{children ?? name}</>;
  return (
    <a
      href={`#term-${slug(entry.term)}`}
      title={entry.short}
      className="border-b border-dotted border-rule-strong text-inherit no-underline hover:border-accent hover:text-accent"
    >
      {children ?? name}
    </a>
  );
}

function slug(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, "-");
}

/** The full list. Collapsed by default so it does not compete with the results above it. */
export function GlossaryList() {
  const [open, setOpen] = useState(false);
  return (
    <div className="border border-rule bg-paper-raised">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-baseline justify-between px-6 py-4 text-left"
      >
        <span className="text-sm font-medium text-ink">
          What the words on this page mean
        </span>
        <span className="text-xs text-ink-faint">
          {open ? "hide" : `${GLOSSARY.length} terms, in plain English`}
        </span>
      </button>

      {open ? (
        <dl className="grid gap-x-10 gap-y-5 border-t border-rule px-6 py-5 md:grid-cols-2">
          {GLOSSARY.map((g) => (
            <div key={g.term} id={`term-${slug(g.term)}`} className="scroll-mt-24">
              <dt className="text-sm font-medium text-ink">{g.term}</dt>
              <dd className="mt-1 text-sm text-ink-soft">
                <span className="text-ink-soft">{g.short}</span>{" "}
                <span className="text-ink-faint">{g.long}</span>
              </dd>
            </div>
          ))}
        </dl>
      ) : null}
    </div>
  );
}
