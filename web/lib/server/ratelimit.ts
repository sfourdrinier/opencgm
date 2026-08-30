// Abuse protection for the public API.
//
// The endpoints are unauthenticated by design -- an API you need an account for is not much
// use to someone evaluating a paper. That means the only thing between the encoder and a
// scraper is this file, and the bill is paid by one person.
//
// Three independent limits, cheapest check first:
//   1. Request body size, rejected before JSON is parsed.
//   2. A per-IP token bucket, refilled continuously rather than in fixed windows, so a
//      client cannot burst the full quota at each window boundary.
//   3. A global concurrency gate, because ONNX inference is CPU-bound and a dozen parallel
//      requests degrade every one of them rather than queueing politely.
//
// State is per-instance and in-memory. On a single host that is exactly right. On a
// horizontally-scaled deploy each instance enforces its own share, which under-counts a
// distributed attacker -- if this ever runs behind more than one instance, move the bucket
// to the platform's KV store or put a WAF in front. It is not a substitute for one.

type Bucket = { tokens: number; last: number };

const BUCKETS = new Map<string, Bucket>();

/** Sustained rate, per IP. */
const REFILL_PER_SECOND = 0.5; // 30 requests/minute sustained
/** Burst allowance, per IP. */
const CAPACITY = 15;
/** Drop idle buckets so a long-running instance does not grow a map of every visitor. */
const IDLE_EVICT_MS = 10 * 60 * 1000;
let lastSweep = Date.now();

export const MAX_BODY_BYTES = 2 * 1024 * 1024;
const MAX_CONCURRENT = 4;
let inFlight = 0;

export function clientKey(req: Request): string {
  const h = req.headers;
  // Trust the platform's forwarding header when present; fall back to a constant so a
  // misconfigured proxy fails closed into one shared bucket rather than open into none.
  const fwd = h.get("x-forwarded-for");
  if (fwd) return fwd.split(",")[0]!.trim();
  return h.get("x-real-ip") ?? h.get("cf-connecting-ip") ?? "unknown";
}

function sweep(now: number) {
  if (now - lastSweep < IDLE_EVICT_MS) return;
  lastSweep = now;
  for (const [key, b] of BUCKETS) {
    if (now - b.last > IDLE_EVICT_MS) BUCKETS.delete(key);
  }
}

export type LimitResult =
  | { ok: true; remaining: number }
  | { ok: false; reason: "rate"; retryAfterSeconds: number }
  | { ok: false; reason: "busy" };

export function takeToken(key: string): LimitResult {
  const now = Date.now();
  sweep(now);

  const bucket = BUCKETS.get(key) ?? { tokens: CAPACITY, last: now };
  const elapsed = (now - bucket.last) / 1000;
  bucket.tokens = Math.min(CAPACITY, bucket.tokens + elapsed * REFILL_PER_SECOND);
  bucket.last = now;

  if (bucket.tokens < 1) {
    BUCKETS.set(key, bucket);
    const wait = Math.ceil((1 - bucket.tokens) / REFILL_PER_SECOND);
    return { ok: false, reason: "rate", retryAfterSeconds: Math.max(1, wait) };
  }

  bucket.tokens -= 1;
  BUCKETS.set(key, bucket);

  if (inFlight >= MAX_CONCURRENT) return { ok: false, reason: "busy" };
  return { ok: true, remaining: Math.floor(bucket.tokens) };
}

/** Wrap the expensive part so the concurrency gate is released even on failure. */
export async function withSlot<T>(fn: () => Promise<T>): Promise<T> {
  inFlight += 1;
  try {
    return await fn();
  } finally {
    inFlight -= 1;
  }
}

export function limitHeaders(remaining: number): Record<string, string> {
  return {
    "x-ratelimit-limit": String(CAPACITY),
    "x-ratelimit-remaining": String(remaining),
    "x-ratelimit-policy": `${CAPACITY} burst; ${REFILL_PER_SECOND * 60}/min sustained`,
  };
}
