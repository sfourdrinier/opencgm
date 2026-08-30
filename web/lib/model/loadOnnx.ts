// Singleton ONNX Runtime Web session for the released encoder.
//
// The session is lazy-initialised on first call to `embed()` and cached for the lifetime
// of the page. Single-thread WASM (no COOP/COEP headers needed). The `.wasm` artefacts
// are copied to `/public/ort/` by the postinstall script (`scripts/copy-ort-assets.ts`).

import * as ort from "onnxruntime-web";
import type { Window } from "../types";

// Configure WASM paths and execution providers exactly once, at module load.
// Next.js will inline this in the client bundle; ORT will lazy-fetch the .wasm files
// from /public/ort/ when an InferenceSession is first created.
let configured = false;
function configureOrt() {
  if (configured) return;
  // Path to /public/ort/* — relative to the page origin, served as static assets by
  // Next.js / Vercel. Single-thread WASM avoids the SharedArrayBuffer / COOP+COEP dance.
  ort.env.wasm.wasmPaths = "/ort/";
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;
  configured = true;
}

let sessionPromise: Promise<ort.InferenceSession> | null = null;

async function getSession(): Promise<ort.InferenceSession> {
  configureOrt();
  if (!sessionPromise) {
    sessionPromise = ort.InferenceSession.create("/models/encoder.onnx", {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
  }
  return sessionPromise;
}

/** Run the encoder on a single 24-hour window. Returns the 128-d §19.1 embedding. */
export async function embed(window: Window): Promise<Float32Array> {
  const session = await getSession();

  // ONNX Runtime Web expects typed-array inputs with the right shape.
  const values = new ort.Tensor("float32", window.values, [1, 288]);
  const mask = new ort.Tensor("float32", window.mask, [1, 288]);
  // The exported graph declares `circadian_start` with rank 1 -- shape [B], not [B, 1].
  // See artifacts/glucofm_encoder.onnx.meta.json: input_shapes ["[B, 288]", "[B, 288]", "[B]"].
  const circ = new ort.Tensor("int64", BigInt64Array.from([BigInt(window.circadianStart)]), [1]);

  const out = await session.run({ values, mask, circadian_start: circ });
  const tensor = out["embedding"];
  if (!tensor) throw new Error("encoder.onnx did not return 'embedding'");
  const arr = tensor.data as Float32Array;
  return new Float32Array(arr.buffer, arr.byteOffset, 128);
}

/** Embed a batch of windows. Useful for the "scrolling" multi-day view. */
export async function embedBatch(windows: Window[]): Promise<Float32Array[]> {
  const session = await getSession();
  const B = windows.length;
  const valuesArr = new Float32Array(B * 288);
  const maskArr = new Float32Array(B * 288);
  const circArr = new BigInt64Array(B);
  for (let i = 0; i < B; i += 1) {
    valuesArr.set(windows[i]!.values, i * 288);
    maskArr.set(windows[i]!.mask, i * 288);
    circArr[i] = BigInt(windows[i]!.circadianStart);
  }
  const values = new ort.Tensor("float32", valuesArr, [B, 288]);
  const mask = new ort.Tensor("float32", maskArr, [B, 288]);
  const circ = new ort.Tensor("int64", circArr, [B]);
  const out = await session.run({ values, mask, circadian_start: circ });
  const tensor = out["embedding"];
  if (!tensor) throw new Error("encoder.onnx did not return 'embedding'");
  const data = tensor.data as Float32Array;
  const out_: Float32Array[] = [];
  for (let i = 0; i < B; i += 1) {
    out_.push(new Float32Array(data.buffer, data.byteOffset + i * 128 * 4, 128));
  }
  return out_;
}


// ---------------------------------------------------------------------------------------
// The state/event decomposition.
//
// A second graph, exported from the same checkpoint, returns the two streams alongside the
// embedding. It is a separate file so the 2 MB encoder stays the only thing a visitor must
// download to get a result; the streams build is fetched on demand.
//
// Units: both streams are in per-window z-scores (masked instance norm over observed
// positions), not mg/dL. At an observed position state + event reconstructs the normalized
// input. At an unobserved position event is exactly 0 and state is the causal filter's
// estimate from past observed samples.

let streamsSession: Promise<ort.InferenceSession> | null = null;

export type Streams = { state: Float32Array; event: Float32Array };

export async function decompose(window: Window): Promise<Streams> {
  configureOrt();
  if (!streamsSession) {
    streamsSession = ort.InferenceSession.create("/models/encoder_streams.onnx", {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
  }
  const session = await streamsSession;
  const out = await session.run({
    values: new ort.Tensor("float32", window.values, [1, 288]),
    mask: new ort.Tensor("float32", window.mask, [1, 288]),
    circadian_start: new ort.Tensor(
      "int64",
      BigInt64Array.from([BigInt(window.circadianStart)]),
      [1],
    ),
  });
  const state = out["state_signal"];
  const event = out["event_signal"];
  if (!state || !event) throw new Error("streams model returned no state_signal/event_signal");
  return {
    state: Float32Array.from(state.data as Float32Array),
    event: Float32Array.from(event.data as Float32Array),
  };
}
