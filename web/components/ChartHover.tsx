"use client";

import { useCallback, useRef, useState } from "react";

// Shared pointer tracking for the 24-hour charts.
//
// The charts are plain SVG on a 288-slot grid, so the whole interaction is: turn a pointer
// position into a slot index, and put a card near it. Pointer events rather than mouse events,
// so a finger on a phone works the same as a cursor; `touch-action: none` on the surface stops
// the browser scrolling the page while someone is reading along a trace.

export type HoverState = {
  index: number;
  clientX: number;
  clientY: number;
  /** Measurements captured by the pointer event, so the card need not read a ref during render. */
  containerWidth: number;
  relativeX: number;
} | null;

export function useChartHover() {
  const ref = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<HoverState>(null);

  const onMove = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    const el = ref.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    // The plot area is inset by the y-axis gutter; those fractions match the SVG viewBoxes.
    const left = r.left + r.width * 0.044;
    const right = r.left + r.width * 0.99;
    const frac = (e.clientX - left) / (right - left);
    const index = Math.round(Math.max(0, Math.min(1, frac)) * 287);
    setHover({
      index,
      clientX: e.clientX,
      clientY: r.top,
      containerWidth: r.width,
      relativeX: e.clientX - r.left,
    });
  }, []);

  const onLeave = useCallback(() => setHover(null), []);

  return { ref, hover, onMove, onLeave };
}

/** Time of day for a slot on the 5-minute grid. */
export function slotTime(index: number): string {
  const minutes = index * 5;
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
}

/**
 * The readout card.
 *
 * Positioned against the chart container rather than the page, and flipped to the left of the
 * cursor in the right-hand third so it never runs off a narrow screen.
 */
export function HoverCard({
  hover,
  rows,
  title,
}: {
  hover: HoverState;
  title: string;
  rows: { label: string; value: string; color?: string }[];
}) {
  if (!hover) return null;
  const flip = hover.relativeX > hover.containerWidth * 0.66;

  return (
    <div
      className="pointer-events-none absolute top-2 z-10 min-w-[8.5rem] border border-rule-strong bg-paper-raised px-3 py-2 shadow-sm"
      style={
        flip
          ? { right: hover.containerWidth - hover.relativeX + 10 }
          : { left: hover.relativeX + 10 }
      }
    >
      <div className="tnum text-xs font-medium text-ink">{title}</div>
      <div className="mt-1 space-y-0.5">
        {rows.map((row) => (
          <div key={row.label} className="flex items-baseline gap-2 whitespace-nowrap text-xs">
            {row.color ? (
              <span
                className="inline-block h-1.5 w-1.5 shrink-0"
                style={{ background: row.color }}
              />
            ) : null}
            <span className="text-ink-faint">{row.label}</span>
            <span className="tnum ml-auto text-ink">{row.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

/** The vertical line under the cursor, drawn inside the SVG. */
export function Crosshair({ x, top, bottom }: { x: number; top: number; bottom: number }) {
  return (
    <line x1={x} x2={x} y1={top} y2={bottom} stroke="var(--color-ink-faint)" strokeWidth={1} opacity={0.45} />
  );
}
