"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ExampleAnalysisView } from "@/components/ExampleAnalysis";
import { runReadingsAnalysis } from "@/lib/analysis-runner";
import { exportReadingsCsv, exportReadingsNdjson } from "@/lib/exports/readings";
import {
  createDefaultSensorController,
  getWebSensorSupport,
  type SensorController,
  type WebSensorSupport,
} from "@/lib/sensor/ubm-controller";
import { CredentialVault } from "@/lib/sensor/credential-vault";
import type { SensorImportResult } from "@/lib/sensor/contracts";
import { SensorImportCoverage, sensorImportCoverage } from "@/components/SensorImportCoverage";
import { SensorImportTimeline } from "@/components/SensorImportTimeline";

export { sensorImportCoverage } from "@/components/SensorImportCoverage";

export type BrowserCapabilityInput = {
  readonly secureContext: boolean;
  readonly bluetooth: boolean;
  readonly origin?: string;
};

export type SensorImportClientProps = {
  readonly createController?: () => SensorController | Promise<SensorController>;
  readonly createVault?: () => Promise<CredentialVault>;
};

type ImportStage = "idle" | "connecting" | "backfill" | "analysis" | "done" | "error" | "cancelled";

export const privacyCopy = "Your data stays in this browser. Sensor readings and credentials are processed locally and are never uploaded or sent to a backend.";

export function sensorImportDiagnosticsVisible(search: string): boolean {
  return search === "?debug=yesplease";
}

const DEFAULT_SENSOR_OPTIONS = {
  serviceUuid: "0000f808-0000-1000-8000-00805f9b34fb",
  channels: {
    authentication: "0000f809-0000-1000-8000-00805f9b34fb",
    control: "0000f80a-0000-1000-8000-00805f9b34fb",
    backfill: "0000f80b-0000-1000-8000-00805f9b34fb",
    "extra-data": "0000f80c-0000-1000-8000-00805f9b34fb",
  },
};

const defaultCreateController = () => createDefaultSensorController(DEFAULT_SENSOR_OPTIONS);
const defaultCreateVault = () => CredentialVault.createPersistentAsync();

function initialBrowserCapability(): BrowserCapabilityInput | null {
  if (typeof navigator === "undefined") return null;
  return {
    secureContext: globalThis.isSecureContext,
    bluetooth: "bluetooth" in navigator,
    origin: typeof globalThis.location === "undefined" ? undefined : globalThis.location.origin,
  };
}

export function browserSupportMessage(input: BrowserCapabilityInput): string {
  const localhost = input.origin !== undefined && /^http:\/\/localhost(?::\d+)?$/u.test(input.origin);
  if (!input.secureContext && !localhost) {
    return "Use Google Chrome with Bluetooth enabled and a Dexcom G7 sensor nearby.";
  }
  if (!input.bluetooth) {
    return "Use Google Chrome with a Dexcom G7 and Bluetooth enabled. This browser does not report Web Bluetooth support.";
  }
  return "Browser support is ready. Have your Dexcom G7 nearby with Bluetooth enabled.";
}

export function importProgressLabel(stage: ImportStage | "loading"): string {
  switch (stage) {
    case "loading": return "Preparing a local connection…";
    case "connecting": return "Connect your Dexcom G7 when prompted…";
    case "backfill": return "Reading available history…";
    case "analysis": return "Running the full analysis…";
    case "done": return "Import ready";
    case "cancelled": return "Import cancelled";
    case "error": return "Import could not be completed";
    case "idle": return "Ready to connect";
  }
}

export function sensorImportError(error: unknown): string {
  const message = error instanceof Error ? error.message.toLowerCase() : "";
  if (message.includes("activation")) return "Connect must begin from the Connect button.";
  if (message.includes("cancel")) return "The connection was cancelled before history was read.";
  if (message.includes("pair")) return "The sensor needs pairing information.";
  return "The local sensor connection did not complete. Check that the sensor is nearby, then try again.";
}

function download(name: string, body: string, type: string): void {
  const blob = new Blob([body], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  URL.revokeObjectURL(url);
}

export function SensorImportClient({
  createController = defaultCreateController,
  createVault = defaultCreateVault,
}: SensorImportClientProps = {}) {
  const diagnosticsVisible = typeof window !== "undefined" && sensorImportDiagnosticsVisible(window.location.search);
  const [capability] = useState<BrowserCapabilityInput | null>(initialBrowserCapability);
  const [support] = useState<WebSensorSupport | null>(() => capability ? getWebSensorSupport(capability) : null);
  const [stage, setStage] = useState<ImportStage>("idle");
  const [sensorName, setSensorName] = useState("Dexcom G7");
  const [pairingCode, setPairingCode] = useState("");
  const [credentialBytes, setCredentialBytes] = useState<Uint8Array | null>(null);
  const [certificateBytes, setCertificateBytes] = useState<Uint8Array | null>(null);
  const [remember, setRemember] = useState(false);
  const [result, setResult] = useState<SensorImportResult | null>(null);
  const [analysis, setAnalysis] = useState<Awaited<ReturnType<typeof runReadingsAnalysis>>["analysis"] | null>(null);
  const [progress, setProgress] = useState("");
  const [error, setError] = useState<string | null>(null);
  const controllerRef = useRef<SensorController | null>(null);
  const vaultRef = useRef<CredentialVault | null>(null);
  const rememberedCredentialRef = useRef<Uint8Array | null>(null);
  const cancelledRef = useRef(false);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    void createVault().then((vault) => {
      if (mountedRef.current) vaultRef.current = vault;
      void vault.load(sensorName).then((credential) => {
        if (mountedRef.current) rememberedCredentialRef.current = credential;
      }).catch(() => undefined);
    }).catch(() => undefined);
    return () => {
      mountedRef.current = false;
      const controller = controllerRef.current;
      controllerRef.current = null;
      if (controller) void controller.stop();
    };
  }, [createVault]);

  useEffect(() => {
    const vault = vaultRef.current;
    if (!vault) return;
    rememberedCredentialRef.current = null;
    void vault.load(sensorName).then((credential) => {
      if (mountedRef.current) rememberedCredentialRef.current = credential;
    }).catch(() => undefined);
  }, [sensorName]);

  const stop = useCallback(async () => {
    const controller = controllerRef.current;
    controllerRef.current = null;
    if (controller) await controller.stop();
  }, []);

  const connect = useCallback(async () => {
    if (support?.state !== "supported" || stage === "connecting" || stage === "backfill" || stage === "analysis") return;
    setError(null);
    cancelledRef.current = false;
    setAnalysis(null);
    setResult(null);
    setProgress(importProgressLabel("connecting"));
    setStage("connecting");
    try {
      const created = createController();
      const controller = created instanceof Promise ? await created : created;
      controllerRef.current = controller;
      const credential = credentialBytes;
      const certificateBundle = certificateBytes;
      const remembered = rememberedCredentialRef.current;
      setStage("backfill");
      setProgress(importProgressLabel("backfill"));
      const imported = await controller.importSensor({
        sensorName,
        userActivation: true,
        credential: credential ?? remembered,
        pairingCode: pairingCode.trim() || null,
        certificateBundle,
      });
      const vault = vaultRef.current;
      if (remember && credential && vault) await vault.save(imported.metadata.sensorId, credential);
      if (!mountedRef.current || cancelledRef.current) return;
      setResult(imported);
      setStage("analysis");
      setProgress(importProgressLabel("analysis"));
      try {
        const analyzed = await runReadingsAnalysis(imported.analysisReadings, setProgress);
        if (mountedRef.current) setAnalysis(analyzed.analysis);
      } catch {
        if (mountedRef.current) setError("The history was imported, but it was too sparse to run a full analysis.");
      }
      if (mountedRef.current && !cancelledRef.current) {
        setStage("done");
        setProgress(importProgressLabel("done"));
      }
    } catch (caught) {
      if (!mountedRef.current || cancelledRef.current) return;
      setStage("error");
      setError(sensorImportError(caught));
      setProgress(importProgressLabel("error"));
    }
  }, [certificateBytes, createController, credentialBytes, pairingCode, remember, stage, support, sensorName]);

  const disconnect = useCallback(async () => {
    await stop();
    if (mountedRef.current) {
      setStage("idle");
      setProgress("");
    }
  }, [stop]);

  const forget = useCallback(async () => {
    const id = result?.metadata.sensorId;
    const vault = vaultRef.current;
    if (id && vault) await vault.forget(id);
    setCredentialBytes(null);
    setPairingCode("");
    setRemember(false);
  }, [result]);

  const busy = stage === "connecting" || stage === "backfill" || stage === "analysis";
  const coverage = result ? sensorImportCoverage(result) : null;
  const exportName = result?.metadata.sensorId || "sensor-import";

  return (
    <div className="mt-8 space-y-6">
      <p className="border-2 border-accent bg-accent-soft/40 p-6 text-base font-semibold leading-7 text-ink">{privacyCopy}</p>
      <section className="border-2 border-accent bg-accent-soft/40 p-6" aria-labelledby="sensor-prerequisites-title">
        <h2 id="sensor-prerequisites-title" className="text-lg font-semibold text-ink">Before you connect</h2>
        <p className="mt-3 max-w-3xl text-sm leading-6 text-ink-soft">Requires Google Chrome, Bluetooth, and a Dexcom G7 sensor.</p>
        <p className="mt-3 text-sm text-ink-soft" role="status">{support && capability ? browserSupportMessage(capability) : "Checking browser support…"}</p>
      </section>

      <section className="border border-rule bg-paper-raised p-6" aria-labelledby="sensor-connect-title">
        <div className="flex flex-wrap items-baseline justify-between gap-3">
          <h2 id="sensor-connect-title" className="text-lg font-semibold text-ink">Connect locally</h2>
          <span className="text-xs text-ink-faint">Nothing is uploaded. No background connection.</span>
        </div>
        <div className="mt-5 grid gap-4 sm:grid-cols-2">
          <label className="text-sm text-ink-soft">Sensor name<input value={sensorName} onChange={(event) => setSensorName(event.target.value)} className="mt-1 block w-full border border-rule-strong bg-paper px-3 py-2 text-ink" /></label>
          <label className="text-sm text-ink-soft">Pairing code (optional)<input value={pairingCode} onChange={(event) => setPairingCode(event.target.value)} inputMode="numeric" autoComplete="off" className="mt-1 block w-full border border-rule-strong bg-paper px-3 py-2 text-ink" /></label>
          {diagnosticsVisible ? <>
            <label className="text-sm text-ink-soft">Remembered key (optional)<input type="file" onChange={(event) => { const file = event.target.files?.[0] ?? null; if (file) void file.arrayBuffer().then((bytes) => { if (mountedRef.current) setCredentialBytes(new Uint8Array(bytes)); }); else setCredentialBytes(null); }} className="mt-1 block w-full text-sm text-ink-soft file:mr-3 file:border file:border-rule-strong file:bg-paper file:px-3 file:py-1.5 file:text-xs" /></label>
            <label className="text-sm text-ink-soft">Opaque certificate bundle (optional)<input type="file" onChange={(event) => { const file = event.target.files?.[0] ?? null; if (file) void file.arrayBuffer().then((bytes) => { if (mountedRef.current) setCertificateBytes(new Uint8Array(bytes)); }); else setCertificateBytes(null); }} className="mt-1 block w-full text-sm text-ink-soft file:mr-3 file:border file:border-rule-strong file:bg-paper file:px-3 file:py-1.5 file:text-xs" /></label>
          </> : null}
        </div>
        <label className="mt-4 flex items-center gap-2 text-sm text-ink-soft"><input type="checkbox" checked={remember} onChange={(event) => setRemember(event.target.checked)} />Remember this sensor on this browser</label>
        <div className="mt-5 flex flex-wrap items-center gap-3">
          <button type="button" onClick={() => void connect()} disabled={support?.state !== "supported" || busy} className="bg-accent px-5 py-2.5 text-sm font-medium text-white hover:bg-accent-ink disabled:cursor-not-allowed disabled:opacity-50">{busy ? progress || "Working…" : stage === "error" ? "Retry connection" : "Connect Dexcom G7"}</button>
          {busy ? <button type="button" onClick={() => { cancelledRef.current = true; void stop(); setStage("cancelled"); setProgress(importProgressLabel("cancelled")); }} className="border border-rule-strong px-4 py-2.5 text-sm text-ink-soft hover:border-accent hover:text-accent">Cancel</button> : null}
          {stage === "done" || result ? <button type="button" onClick={() => void disconnect()} className="border border-rule-strong px-4 py-2.5 text-sm text-ink-soft hover:border-accent hover:text-accent">Disconnect</button> : null}
          {result ? <button type="button" onClick={() => void forget()} className="text-sm text-ink-faint underline decoration-dotted underline-offset-4 hover:text-accent">Forget local key</button> : null}
        </div>
        <p className="mt-4 text-sm text-ink-soft" role="status" aria-live="polite">{progress || importProgressLabel(stage)}</p>
        {error ? <div className="mt-3 border border-low/40 bg-low/5 px-4 py-3 text-sm text-ink-soft" role="alert">{error}</div> : null}
      </section>

      {coverage && result ? <SensorImportCoverage summary={coverage} /> : null}
      {result ? <SensorImportTimeline records={result.records} /> : null}

      {result ? (
        <section className="border border-rule bg-paper-raised p-5" aria-labelledby="sensor-export-title">
          <div className="flex flex-wrap items-baseline justify-between gap-3"><h2 id="sensor-export-title" className="text-lg font-semibold text-ink">Keep a local copy</h2><span className="text-xs text-ink-faint">Downloads stay in your browser.</span></div>
          <div className="mt-4 flex flex-wrap gap-3">
            <button type="button" onClick={() => download(`${exportName}-readings.csv`, exportReadingsCsv(result.records), "text/csv") } className="border border-rule-strong px-4 py-2 text-sm text-ink-soft hover:border-accent hover:text-accent">Download CSV</button>
            <button type="button" onClick={() => download(`${exportName}-readings.ndjson`, exportReadingsNdjson(result.records, { ...result.metadata, completeness: result.completeness, warnings: result.warnings }), "application/x-ndjson") } className="border border-rule-strong px-4 py-2 text-sm text-ink-soft hover:border-accent hover:text-accent">Download NDJSON</button>
          </div>
        </section>
      ) : null}

      {analysis ? <section aria-labelledby="sensor-analysis-title"><div className="border-b border-rule pb-3"><h2 id="sensor-analysis-title" className="text-lg font-semibold text-ink">Your analysis</h2><p className="mt-1 text-xs text-ink-faint">The full local analysis used by the worked example.</p></div><ExampleAnalysisView data={analysis} owner="own" /></section> : null}

      <p className="measure text-xs leading-5 text-ink-faint">This is a research instrument, not medical advice or a medical device. The page reads only what the connected sensor makes available, keeps data in this tab, and does not maintain a background connection.</p>
    </div>
  );
}
