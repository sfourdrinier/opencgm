# Deploying the site

The Next.js app is in `web/`, not at the repository root. Everything below follows from that.

## First deployment

    cd web
    npx vercel login
    npx vercel link          # answer: yes, link to existing project or create new

**Set the Root Directory to `web`.** In the dashboard: Project → Settings → General → Root
Directory. If this is wrong, the build fails with "No Next.js version detected" and no amount
of build-command fiddling helps, because Vercel is looking at the repository root where there
is no `package.json`.

    npx vercel           # preview deployment, a throwaway URL
    npx vercel --prod    # production

## Function limits: set them in the route, not in `vercel.json`

`vercel.json`'s `functions` key matches the legacy `api/` directory convention. Pointing it at
an App Router path fails the build with:

    The pattern "app/api/v1/analyse/route.ts" defined in `functions`
    doesn't match any Serverless Functions inside the `api` directory.

For the App Router, per-route limits are segment exports in the route file itself.
`app/api/v1/analyse/route.ts` already carries them:

    export const runtime = "nodejs";   // onnxruntime-node needs Node, not Edge
    export const maxDuration = 30;     // a cold start initialising the ONNX runtime
                                       // can exceed the 10 s default

**Memory is a project setting, not a file one.** Project → Settings → Functions → Function
CPU. The analyse route loads a 1.9 MB ONNX graph and runs inference; the default allocation
works but a higher tier also raises CPU, which is what actually makes it quick. Raise it only
if the endpoint feels slow — the browser demo does not touch it.

## Why `vercel.json` still exists

**Immutable caching for `/models/` and `/ort/`.** The encoder is 1.9 MB and the ONNX Runtime
WASM is another 14 MB. They are content-stable for the life of a deployment, and without this
header a returning visitor re-downloads them.

## The failure mode to check first

The API routes read `public/models/*` from disk at request time. Serverless bundlers trace
*imports*, not runtime `readFileSync`, so without `outputFileTracingIncludes` in
`next.config.ts` the functions deploy **without their model** and fail with ENOENT on the
first call — while every static page, which reads the same files at build time, looks
perfectly healthy. That config is already in place. After deploying, confirm it survived:

    curl -s https://<deployment>/api/v1/version | head -c 200
    curl -s -X POST https://<deployment>/api/v1/analyse \
      -H 'content-type: application/json' \
      -d '{"readings":[{"t":"2026-08-28T08:00:00Z","mgdl":96},{"t":"2026-08-28T08:05:00Z","mgdl":98}]}' \
      | head -c 200

A 500 with ENOENT means the tracing did not take.

## Rate limiting is weaker on serverless than it looks

`lib/server/ratelimit.ts` is an in-memory token bucket. On a single host that is exactly
right. On Vercel each lambda instance keeps its own bucket, so the advertised 30 requests a
minute per IP is enforced per instance rather than globally. If the endpoint sees real
traffic, move the bucket to a KV store or put a WAF in front. It is a speed bump, not a gate.

## Environment

Nothing is required. `NEXT_DEV_ORIGINS` is development-only and should not be set in Vercel.
