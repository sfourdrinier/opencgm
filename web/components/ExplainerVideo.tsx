"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

// The hero explainer, modelled on the ImagiExplain showcase card: a poster with an overlay
// play affordance rather than browser chrome, a silent preview on hover, and no media bytes
// fetched until the visitor is near enough to plausibly want them.
//
// Pressing play opens a theater rather than playing in place. The video is portrait, and a
// 9:16 file in a desktop browser's native fullscreen is a narrow strip between two large
// black bars — so on a fine pointer it opens a dialog sized to the video's own aspect, and
// only a coarse pointer, where the OS player fills a portrait screen properly and brings its
// own gestures, gets the native handoff.

const FOCUSABLE = 'a[href], button:not([disabled]), video[controls], [tabindex]:not([tabindex="-1"])';

/** iOS Safari exposes only the non-standard, video-element-only fullscreen entry point. */
function playInNativeFullscreen(video: HTMLVideoElement): boolean {
  const enter = Reflect.get(video, "webkitEnterFullscreen");
  if (typeof enter !== "function") return false;
  video.muted = false;
  video.currentTime = 0;
  enter.call(video);
  void video.play().catch(() => undefined);
  return true;
}

interface ExplainerVideoProps {
  src: string;
  poster: string;
  /** Shown in the corner badge, so the length is known before committing to a click. */
  durationLabel: string;
  /** Used for the play button's accessible name and the dialog's label. */
  title: string;
  className?: string;
}

export function ExplainerVideo({
  src,
  poster,
  durationLabel,
  title,
  className = "",
}: ExplainerVideoProps) {
  const previewRef = useRef<HTMLVideoElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const openerRef = useRef<HTMLButtonElement>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  // Whether a silent hover preview is worth running. Read only by pointer handlers, so it is
  // a ref, not state — nothing renders from it.
  const previewAllowed = useRef(false);
  const coarsePointer = useRef(false);

  useEffect(() => {
    if (typeof window.matchMedia !== "function") return;
    const canHover = window.matchMedia("(hover: hover) and (pointer: fine)");
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");
    function sync() {
      // A coarse pointer has no hover to start or end a preview, so autoplay would cost it
      // the download for nothing. Reduced motion is an explicit request, not a preference.
      previewAllowed.current = canHover.matches && !reduced.matches;
      coarsePointer.current = !canHover.matches;
      if (previewAllowed.current) return;
      // Turning the gate off mid-hover must stop a preview already running.
      const v = previewRef.current;
      if (v && !v.paused) v.pause();
    }
    sync();
    canHover.addEventListener("change", sync);
    reduced.addEventListener("change", sync);
    return () => {
      canHover.removeEventListener("change", sync);
      reduced.removeEventListener("change", sync);
    };
  }, []);

  // Nothing is fetched on load; the element is warmed only once it nears the viewport, so a
  // visitor who never scrolls this far pays nothing for a 5 MB file.
  useEffect(() => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    const warm = () => {
      if (previewRef.current) previewRef.current.preload = "metadata";
    };
    if (typeof IntersectionObserver !== "function") {
      warm();
      return;
    }
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          warm();
          io.disconnect();
        }
      },
      { rootMargin: "200px" },
    );
    io.observe(wrap);
    return () => io.disconnect();
  }, []);

  const close = useCallback(() => {
    setOpen(false);
    // Focus goes back to the control that opened the dialog, not to the top of the document.
    openerRef.current?.focus();
  }, []);

  // While the theater is open it owns Escape, the page scroll and the tab ring.
  useEffect(() => {
    if (!open) return;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    dialogRef.current?.focus();

    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        event.preventDefault();
        close();
        return;
      }
      if (event.key !== "Tab") return;
      const dialog = dialogRef.current;
      if (!dialog) return;
      const focusable = Array.from(dialog.querySelectorAll<HTMLElement>(FOCUSABLE));
      if (focusable.length === 0) {
        // Nothing to land on: hold focus on the dialog rather than letting it escape to the
        // page behind, which aria-modal has told assistive tech to ignore.
        event.preventDefault();
        dialog.focus();
        return;
      }
      const first = focusable[0]!;
      const last = focusable[focusable.length - 1]!;
      const active = document.activeElement;
      // Focus can sit outside the dialog even while it is open — the browser puts it on the
      // document after a click on the backdrop — so wrap from either end.
      if (!dialog.contains(active)) {
        event.preventDefault();
        (event.shiftKey ? last : first).focus();
      } else if (event.shiftKey && active === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && active === last) {
        event.preventDefault();
        first.focus();
      }
    }

    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.body.style.overflow = previousOverflow;
    };
  }, [open, close]);

  function startPreview() {
    if (open || !previewAllowed.current) return;
    const v = previewRef.current;
    if (!v) return;
    // The near-viewport warm above is an optimisation, not a guarantee — if it has not
    // landed yet the element still holds preload="none" and has no data to play, and the
    // hover would silently do nothing. Asking for the bytes here makes the preview depend
    // on the hover alone.
    if (v.readyState === 0) {
      v.preload = "auto";
      v.load();
    }
    // Muted is not a stylistic choice: browsers reject unmuted playback that no click
    // preceded, so an unmuted preview would simply never start.
    v.muted = true;
    void v.play().catch(() => {
      // A refusal is a complete, correct state: the poster simply stays put.
    });
  }

  function stopPreview() {
    const v = previewRef.current;
    if (!v || v.paused) return;
    v.pause();
    v.currentTime = 0;
  }

  function handlePlay() {
    const v = previewRef.current;
    // On a phone the expected behaviour is the OS player taking over the screen, not an
    // in-page dialog — and a portrait video fills a portrait screen exactly.
    if (v && coarsePointer.current && playInNativeFullscreen(v)) return;
    stopPreview();
    setOpen(true);
  }

  return (
    <>
      <div
        ref={wrapRef}
        className={`group relative overflow-hidden border border-rule bg-paper-raised ${className}`}
        onMouseEnter={startPreview}
        onMouseLeave={(e) => {
          // A keyboard user's focus sits on the play button inside this element; the preview
          // should survive the pointer wandering off.
          if (e.currentTarget.contains(document.activeElement)) return;
          stopPreview();
        }}
      >
        <video
          ref={previewRef}
          src={src}
          poster={poster}
          className="h-full w-full object-cover"
          preload="none"
          playsInline
          muted
          loop
          tabIndex={-1}
          aria-hidden="true"
        />
        <span className="pointer-events-none absolute bottom-2 right-2 bg-ink/75 px-1.5 py-0.5 text-xs font-medium tabular-nums text-white">
          {durationLabel}
        </span>
        <button
          ref={openerRef}
          type="button"
          onClick={handlePlay}
          onFocus={startPreview}
          onBlur={stopPreview}
          aria-label={`Play ${title}`}
          className="absolute inset-0 flex items-center justify-center bg-ink/0 transition-colors duration-300 hover:bg-ink/10 focus-visible:bg-ink/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-accent"
        >
          {/* Visible by default. Hiding it until hover would leave every touch visitor
              looking at a still with nothing saying it is actionable. */}
          <span className="flex h-14 w-14 items-center justify-center border border-rule-strong bg-paper text-ink shadow-sm transition-transform duration-300 group-hover:scale-105">
            <svg viewBox="0 0 24 24" className="ml-0.5 h-6 w-6" aria-hidden="true">
              <path d="M8 5.5v13l11-6.5-11-6.5Z" fill="currentColor" />
            </svg>
          </span>
        </button>
      </div>

      {typeof document !== "undefined" &&
        open &&
        createPortal(
          <div
            className="fixed inset-0 z-50 flex items-center justify-center bg-ink/80 p-4 sm:p-8"
            onMouseDown={(e) => {
              // Only a press that both starts and ends on the backdrop closes, so a drag that
              // began on the video's scrubber does not dismiss it.
              if (e.target === e.currentTarget) close();
            }}
          >
            <div
              ref={dialogRef}
              role="dialog"
              aria-modal="true"
              aria-label={title}
              tabIndex={-1}
              className="relative flex max-h-full flex-col outline-none"
            >
              <video
                src={src}
                poster={poster}
                className="max-h-[85vh] w-auto max-w-full border border-ink/20 bg-black"
                controls
                autoPlay
                playsInline
              />
              <button
                type="button"
                onClick={close}
                className="mt-3 self-end border border-paper/30 px-3 py-1.5 text-sm font-medium text-paper hover:border-paper hover:bg-paper/10"
              >
                Close
              </button>
            </div>
          </div>,
          document.body,
        )}
    </>
  );
}
