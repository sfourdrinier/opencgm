<!-- web/VERCEL.md -->

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

## Bundle size

The first deploy fails with:

    Error: Total bundle size (624.06 MB) exceeds the maximum function size (500 MB).

That is onnxruntime-node, which ships prebuilt binaries for every platform it supports --
503 MB of them -- and the tracer packs all of it. A Linux x64 serverless function needs two
files out of that set. The CUDA provider alone is 219 MB and there is no GPU to use it.

`next.config.ts` excludes win32, darwin, linux/arm64 and the CUDA and TensorRT providers,
which brings the function to 53 MB.

One subtlety worth knowing, because the symptom is silent: the tracer follows the `require`
of `onnxruntime_binding.node` but not the `dlopen` of `libonnxruntime.so.1` sitting beside
it. Exclude too aggressively, or forget to include that library explicitly, and the function
ships a binding with no runtime behind it -- deploying cleanly and failing on the first call.
`outputFileTracingIncludes` names it for that reason.

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

## Browser sensor import

The experimental importer is available by direct URL at
<https://opencgm.vercel.app/sensor-import>. It is intentionally not linked from
navigation, the home page, the try page, or the sitemap while hardware proof is
being completed. For local development, use
`http://localhost:3000/sensor-import` (or the port printed by Next).

The production URL is a secure context. The localhost HTTP exception is only for
development; other HTTP origins must not be documented as supported. The
connect action must begin from the user's button click, and browser capability
detection—not a user-agent string—decides whether Web Bluetooth is available.
The current support copy shown before connecting is exactly: “Requires Google
Chrome, Bluetooth, and a Dexcom G7 sensor.” Keep origin and secure-context
details here and in tests rather than expanding the primary UI copy.

The page connects directly through Web Bluetooth, runs the opaque engine and
analysis in the browser, and never sends sensor readings, credentials, or
exports to a server. It has no background connection. Static first-party assets
still have to be downloaded to render the page; that is not a sensor-data upload.
The route has no relay, proxy, or private-origin exception.

The only browser engine module is
`/sensor/sensor-engine.abi1.wasm`, accompanied by its manifest. The loader
verifies the manifest digest before instantiation. Keep the stable WASM filename
on `Cache-Control: public, max-age=0, must-revalidate`; artifact updates must be
revalidated rather than treated as immutable. `web/vercel.json` and
`next.config.ts` carry the corresponding header and same-origin CSP policy.
Before a deploy, run:

```bash
cd web
pnpm run verify:sensor-artifact
pnpm run test:focused -- lib/sensor/artifact.test.mts
pnpm run type-check
pnpm run build
```

If the chooser shows no sensor, another application may have the sensor's radio
session or the sensor may be out of range/asleep; stop the competing session,
bring the sensor nearby, and retry. A cancelled chooser, denied permission,
missing pairing material, or link loss is surfaced as a local retryable state.
The importer only returns history the sensor makes available. It can therefore
complete with partial history and warnings; a partial import is not evidence
that the unavailable interval contained no readings.
