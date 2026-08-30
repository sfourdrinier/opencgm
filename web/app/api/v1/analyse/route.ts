import { NextResponse } from "next/server";
import { clinicalMetrics } from "@/lib/clinical";
import { applyHead, encode, encoderMeta, headsBundle } from "@/lib/server/encoder";
import {
  MAX_BODY_BYTES,
  clientKey,
  limitHeaders,
  takeToken,
  withSlot,
} from "@/lib/server/ratelimit";

export const runtime = "nodejs";
export const maxDuration = 30;

const N = 288;
const STEP_MS = 5 * 60 * 1000;
const MAX_READINGS = 20_000;
const COVERAGE_TOLERANCE = 0.15;

type Body = {
  readings?: { t: string | number; mgdl?: number; mmol?: number }[];
  window_start?: string | number;
  /** IANA zone used to place naive timestamps and to compute circadian phase. */
  timezone?: string;
};

/**
 * Milliseconds since epoch for a timestamp that names no offset, read in `zone`.
 *
 * `timezone` used to be declared and never read, so a naive timestamp was resolved by
 * `Date.parse` against whatever zone the server happened to run in. The same request then
 * produced different embeddings on different hosts, because the circadian index moved -- a
 * reproducibility bug in a project whose whole claim is reproducibility.
 */
function parseInZone(value: string, zone: string): number {
  const m = value.trim().match(/^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})(?::(\d{2}))?/);
  if (!m) return Number.NaN;
  const [y, mo, d, h, mi, s] = m.slice(1).map((v) => Number(v ?? 0));
  const asUtc = Date.UTC(y!, mo! - 1, d!, h!, mi!, s ?? 0);
  const fmt = new Intl.DateTimeFormat("en-US", {
    timeZone: zone,
    hour12: false,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
  const parts = Object.fromEntries(
    fmt.formatToParts(new Date(asUtc)).map((part) => [part.type, part.value]),
  );
  const shifted = Date.UTC(
    Number(parts.year),
    Number(parts.month) - 1,
    Number(parts.day),
    Number(parts.hour) % 24,
    Number(parts.minute),
    Number(parts.second),
  );
  return asUtc - (shifted - asUtc);
}

function hasOffset(value: string): boolean {
  return /(?:Z|[+-]\d{2}:?\d{2})$/.test(value.trim());
}

/** Wall-clock hour and minute of an instant, in `zone`. */
function localHourMinute(ms: number, zone: string): { hour: number; minute: number } {
  const fmt = new Intl.DateTimeFormat("en-US", {
    timeZone: zone,
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
  });
  const parts = Object.fromEntries(
    fmt.formatToParts(new Date(ms)).map((part) => [part.type, part.value]),
  );
  return { hour: Number(parts.hour) % 24, minute: Number(parts.minute) };
}

function bad(message: string, hint?: string) {
  return NextResponse.json({ error: message, hint }, { status: 400 });
}

/** Median-collapse readings onto the 5-minute grid. Empty slots stay empty. */
function gridify(readings: { t: number; mgdl: number }[], startT: number) {
  const buckets: number[][] = Array.from({ length: N }, () => []);
  for (const r of readings) {
    const idx = Math.floor((r.t - startT) / STEP_MS);
    if (idx >= 0 && idx < N) buckets[idx]!.push(r.mgdl);
  }
  const values = new Float32Array(N);
  const mask = new Float32Array(N);
  buckets.forEach((b, i) => {
    if (!b.length) return;
    const s = [...b].sort((x, y) => x - y);
    const mid = s.length >> 1;
    values[i] = s.length % 2 ? s[mid]! : (s[mid - 1]! + s[mid]!) / 2;
    mask[i] = 1;
  });
  return { values, mask };
}

export async function POST(req: Request) {
  const gate = takeToken(clientKey(req));
  if (!gate.ok) {
    if (gate.reason === "busy") {
      return NextResponse.json(
        {
          error: "The encoder is at capacity. Retry shortly.",
          hint: "For sustained or batch use, run your own instance: `just web`.",
        },
        { status: 503, headers: { "retry-after": "2" } },
      );
    }
    return NextResponse.json(
      {
        error: "Rate limit exceeded.",
        hint: "This endpoint is free and unauthenticated, so it is rate limited per IP. " +
          "For bulk work, run your own instance: `just web`.",
        retry_after_seconds: gate.retryAfterSeconds,
      },
      { status: 429, headers: { "retry-after": String(gate.retryAfterSeconds) } },
    );
  }

  // Reject oversized payloads before parsing them, not after.
  const declared = Number(req.headers.get("content-length") ?? "0");
  if (Number.isFinite(declared) && declared > MAX_BODY_BYTES) {
    return bad(
      `Request body is too large (${declared} bytes; limit ${MAX_BODY_BYTES}).`,
      "Send a single day of readings, or run your own instance for bulk work.",
    );
  }

  let body: Body;
  try {
    const text = await req.text();
    if (text.length > MAX_BODY_BYTES) {
      return bad(`Request body is too large (limit ${MAX_BODY_BYTES} bytes).`);
    }
    body = JSON.parse(text) as Body;
  } catch {
    return bad("Request body is not valid JSON.");
  }

  const raw = body.readings;
  if (!Array.isArray(raw) || raw.length === 0) {
    return bad(
      "`readings` must be a non-empty array.",
      'Each entry: {"t": "2026-01-15T08:00:00Z", "mgdl": 96}. `mmol` is accepted instead of `mgdl`.',
    );
  }
  if (raw.length > MAX_READINGS) {
    return bad(`Too many readings (${raw.length}); the limit is ${MAX_READINGS}.`);
  }

  const zone = body.timezone ?? "UTC";
  try {
    new Intl.DateTimeFormat("en-US", { timeZone: zone });
  } catch {
    return bad(
      `Unknown timezone ${JSON.stringify(zone)}.`,
      "Use an IANA name such as Europe/Paris, or send timestamps with an explicit offset.",
    );
  }

  const parsed: { t: number; mgdl: number }[] = [];
  for (const r of raw) {
    const t =
      typeof r.t === "number"
        ? r.t
        : hasOffset(String(r.t))
          ? Date.parse(String(r.t))
          : parseInZone(String(r.t), zone);
    if (!Number.isFinite(t)) continue;
    let mgdl = r.mgdl;
    if (mgdl == null && r.mmol != null) mgdl = r.mmol * 18.0182;
    if (mgdl == null || !Number.isFinite(mgdl) || mgdl < 20 || mgdl > 600) continue;
    parsed.push({ t, mgdl });
  }
  if (!parsed.length) {
    return bad(
      "No readings survived parsing.",
      "Timestamps must be ISO-8601 or epoch milliseconds; glucose must be 20-600 mg/dL.",
    );
  }
  parsed.sort((a, b) => a.t - b.t);

  // Default to the 24 hours ending at the last reading.
  const startT =
    body.window_start != null
      ? typeof body.window_start === "number"
        ? body.window_start
        : hasOffset(String(body.window_start))
          ? Date.parse(String(body.window_start))
          : parseInZone(String(body.window_start), zone)
      : parsed[parsed.length - 1]!.t - (N - 1) * STEP_MS;
  if (!Number.isFinite(startT)) return bad("`window_start` is not a valid timestamp.");

  const { values, mask } = gridify(parsed, startT);
  let observed = 0;
  for (let i = 0; i < N; i += 1) observed += mask[i]!;
  const coverage = observed / N;

  if (observed === 0) {
    return bad(
      "No readings fall inside the 24-hour window.",
      "Pass `window_start` explicitly, or send readings from a single day.",
    );
  }

  // Circadian phase is a wall-clock quantity: 08:00 means 08:00 where the person was, not
  // 08:00 UTC. Computed in the request's zone so the result does not depend on the host.
  const { hour, minute } = localHourMinute(startT, zone);
  const circadianStart = Math.floor((hour * 60 + minute) / 5) % N;

  const embedding = await withSlot(() => encode(values, mask, circadianStart));
  const bundle = headsBundle();
  const meta = encoderMeta();

  const probes = Object.entries(bundle.heads).map(([key, h]) => {
    const applicable =
      coverage >= h.applicability.coverage_p05 - COVERAGE_TOLERANCE &&
      coverage <= h.applicability.coverage_p95 + COVERAGE_TOLERANCE;
    if (!applicable || !h.reliability.has_signal) {
      return {
        key,
        task: h.task,
        cohort: h.dataset,
        scored: false,
        reason: !h.reliability.has_signal
          ? "This probe showed no measurable signal in cross-validation."
          : `Window coverage ${(coverage * 100).toFixed(1)}% is outside the ` +
            `${(h.applicability.coverage_p05 * 100).toFixed(0)}-` +
            `${(h.applicability.coverage_p95 * 100).toFixed(0)}% band this probe was fitted on.`,
      };
    }
    const proba = applyHead(h, embedding);
    const argmax = proba.indexOf(Math.max(...proba));
    return {
      key,
      task: h.task,
      cohort: h.dataset,
      scored: true,
      classes: h.classifier.classes,
      predicted_class: h.classifier.classes[argmax] ?? argmax,
      raw_scores: proba.map((p) => +p.toFixed(6)),
      raw_scores_warning:
        "Uncalibrated. These saturate near 0 and 1 for most inputs and must not be read as " +
        "probabilities. Rank with them; do not quote them.",
      held_out_roc_auc: h.reliability.roc_auc,
      n_subjects_fitted_on: h.reliability.n_subjects,
    };
  });

  return NextResponse.json(
    {
      window: {
        start_utc: new Date(startT).toISOString(),
        end_utc: new Date(startT + (N - 1) * STEP_MS).toISOString(),
        grid_minutes: 5,
        positions: N,
        circadian_start_index: circadianStart,
        circadian_start_note:
          "Index of the window start within a 288-step day; feeds the time-of-day embedding. " +
          `Computed in ${zone}. Timestamps carrying an explicit offset are used as given; ` +
          "naive timestamps are read in that zone.",
        timezone: zone,
        coverage,
        coverage_note: "Fraction of the 288 slots with a real reading. Gaps are never filled.",
        values_mg_dl: Array.from(values).map((v, i) => (mask[i] ? +v.toFixed(1) : null)),
        values_note:
          "The gridded window exactly as the encoder received it. `null` means unobserved — " +
          "the model is told the slot is empty rather than given a guess.",
        mask: Array.from(mask, (m) => (m ? 1 : 0)),
      },
        clinical_metrics: {
        ...clinicalMetrics(values, mask),
        note:
          "Computed from observed readings only. Gaps are excluded, never interpolated, so " +
          "these are conditional on what the sensor actually recorded.",
      },
      embedding: {
        dim: 128,
        pooling: "mean over the 24 hourly patch tokens",
        vector: Array.from(embedding).map((v) => +v.toFixed(6)),
        note:
          "This is the model's entire output. Everything below is a small classifier fitted on " +
          "top of this vector; nothing else is read from the trace.",
      },
      probes,
      provenance: {
        checkpoint: meta.checkpoint,
        epoch: meta.epoch,
        seed: meta.seed,
        onnx_sha256: meta.sha256,
        architecture: meta.architecture,
        heads_withheld: bundle.withheld_note ?? null,
      },
      readings_received: raw.length,
      readings_used: parsed.length,
      not_a_medical_device:
        "Research output. Not a diagnosis and not a basis for any decision about anyone's health.",
    },
    { headers: limitHeaders(gate.remaining) },
  );
}
