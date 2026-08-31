"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ExampleAnalysisView } from "@/components/ExampleAnalysis";
import { runReadingsAnalysis } from "@/lib/analysis-runner";
import { exportReadingsCsv, exportReadingsNdjson } from "@/lib/exports/readings";
import {
  createDefaultSensorController,
  getWebSensorSupport,
  type SensorController,
  type SensorCredentialCallbacks,
} from "@/lib/sensor/ubm-controller";
import { CredentialVault } from "@/lib/sensor/credential-vault";
import { createReadingArchive, type ReadingArchive } from "@/lib/sensor/reading-archive";
import type { ImportedSensorReading, SensorImportResult } from "@/lib/sensor/contracts";
import { SensorImportCoverage, sensorImportCoverage } from "@/components/SensorImportCoverage";
import { SensorImportTimeline } from "@/components/SensorImportTimeline";

export { sensorImportCoverage } from "@/components/SensorImportCoverage";

export type BrowserCapabilityInput = {
  readonly secureContext: boolean;
  readonly bluetooth: boolean;
  readonly origin?: string;
};

export type SensorImportClientProps = {
  readonly createController?: (credentials?: SensorCredentialCallbacks) => SensorController | Promise<SensorController>;
  readonly createVault?: () => Promise<CredentialVault>;
  readonly createArchive?: () => Promise<ReadingArchive>;
};

type ImportStage = "idle" | "connecting" | "backfill" | "analysis" | "done" | "error" | "cancelled";

export const privacyCopy = "Your data stays in this browser. Sensor readings and credentials are processed locally and are never uploaded or sent to a backend.";
export const pairingCodeGuidance = "Enter the four ASCII digits from the sensor applicator or pairing material.";
export const pairingCodeRequiredCopy = "— Need pairing code";
export const sensorImportWaitingCopy = "A sensor may become available briefly about every five minutes. Keep this page open; discovery can take several minutes. A sleeping or unavailable remembered sensor remains in a waiting state.";
export const sensorSelectedCopy = "Sensor selected. Connecting, authenticating, and reading available history…";

export function isValidPairingCode(value: string): boolean {
  return /^[0-9]{4}$/u.test(value);
}

export function sensorImportDiagnosticsVisible(search: string): boolean {
  return search === "?debug=yesplease";
}

export const sensorImportBluetoothOptions = {
  chooserServiceUuid: "0000febc-0000-1000-8000-00805f9b34fb",
  serviceUuid: "f8083532-849e-531c-c594-30f1f86a4ea5",
  channels: {
    authentication: "f8083535-849e-531c-c594-30f1f86a4ea5",
    control: "f8083534-849e-531c-c594-30f1f86a4ea5",
    backfill: "f8083536-849e-531c-c594-30f1f86a4ea5",
    "extra-data": "f8083538-849e-531c-c594-30f1f86a4ea5",
  },
};

const defaultCreateController = (credentials?: SensorCredentialCallbacks) => createDefaultSensorController(sensorImportBluetoothOptions, credentials);
const defaultCreateVault = () => CredentialVault.createPersistentAsync();
const defaultCreateArchive = () => createReadingArchive();

export function sensorImportCredentialCallbacks(
  vault: Pick<CredentialVault, "load" | "save"> | null,
  remember: boolean,
  onPeerSelected: (peerId: string) => void,
  onReading?: (reading: ImportedSensorReading) => Promise<void> | void,
): SensorCredentialCallbacks {
  return {
    loadCredential: vault ? (peerId) => vault.load(peerId) : undefined,
    saveCredential: remember && vault ? (peerId, credential) => vault.save(peerId, credential) : undefined,
    onPeerSelected,
    onReading,
  };
}

function initialBrowserCapability(): BrowserCapabilityInput | null {
  if (typeof window === "undefined") return null;
  return browserCapabilitySnapshot({
    secureContext: window.isSecureContext,
    navigator: window.navigator,
    origin: window.location.origin,
  });
}

export function browserCapabilitySnapshot(input: {
  readonly secureContext: boolean;
  readonly navigator: object;
  readonly origin?: string;
}): BrowserCapabilityInput {
  return { secureContext: input.secureContext, bluetooth: "bluetooth" in input.navigator, origin: input.origin };
}

export function browserSupportMessage(input: BrowserCapabilityInput & { readonly userAgent?: string }): string {
  const localhost = input.origin !== undefined && /^http:\/\/localhost(?::\d+)?$/u.test(input.origin);
  if (!input.secureContext && !localhost) {
    return "Use Google Chrome with Bluetooth enabled and a Dexcom G7 sensor nearby.";
  }
  if (!input.bluetooth) {
    const userAgent = input.userAgent ?? "";
    if (/Linux/u.test(userAgent) && !/Android/u.test(userAgent)) {
      return "On Linux, confirm a Bluetooth adapter named hci0 is present (for example with bluetoothctl list), fully quit and relaunch Chrome with --enable-experimental-web-platform-features, then reload this page.";
    }
    return "Use Google Chrome with a Dexcom G7 and Bluetooth enabled. Fully quit and relaunch Chrome, then reload this page; this browser does not report Web Bluetooth support.";
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
  if (message.includes("choose sensor")) return "Choose a sensor to reconnect.";
  if (message.includes("os pairing")) return "Chrome cannot finish a new operating-system Bluetooth bond here. Use a browser-authorized sensor with prepared pairing information, then retry.";
  if (message.includes("timed") || message.includes("timeout")) return "The sensor connection timed out. Keep the sensor nearby, stop any competing phone connection, and retry during its next Bluetooth window.";
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
  createArchive = defaultCreateArchive,
}: SensorImportClientProps = {}) {
  const diagnosticsVisible = typeof window !== "undefined" && sensorImportDiagnosticsVisible(window.location.search);
  const [capability, setCapability] = useState<BrowserCapabilityInput | null>(null);
  const support = capability ? getWebSensorSupport(capability) : null;
  const [stage, setStage] = useState<ImportStage>("idle");
  const [sensorName, setSensorName] = useState("Dexcom G7");
  const [pairingCode, setPairingCode] = useState("");
  const [vaultReady, setVaultReady] = useState(false);
  const [archiveReady, setArchiveReady] = useState(false);
  const [archivedReadings, setArchivedReadings] = useState<ImportedSensorReading[]>([]);
  const [authorizedPeerFound, setAuthorizedPeerFound] = useState(false);
  const [remember, setRemember] = useState(false);
  const [result, setResult] = useState<SensorImportResult | null>(null);
  const [analysis, setAnalysis] = useState<Awaited<ReturnType<typeof runReadingsAnalysis>>["analysis"] | null>(null);
  const [progress, setProgress] = useState("");
  const [error, setError] = useState<string | null>(null);
  const controllerRef = useRef<SensorController | null>(null);
  const vaultRef = useRef<CredentialVault | null>(null);
  const archiveRef = useRef<ReadingArchive | null>(null);
  const credentialRef = useRef<Uint8Array | null>(null);
  const certificateRef = useRef<Uint8Array | null>(null);
  const selectedPeerIdRef = useRef<string | null>(null);
  const autoReconnectStartedRef = useRef(false);
  const cancelledRef = useRef(false);
  const mountedRef = useRef(true);

  useEffect(() => {
    const frame = window.requestAnimationFrame(() => setCapability(initialBrowserCapability()));
    return () => window.cancelAnimationFrame(frame);
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    void createVault().then((vault) => {
      if (mountedRef.current) {
        vaultRef.current = vault;
        setVaultReady(true);
      }
    }).catch(() => { if (mountedRef.current) setVaultReady(true); });
    void createArchive().then(async (archive) => {
      const readings = await archive.list();
      if (mountedRef.current) {
        archiveRef.current = archive;
        setArchivedReadings(readings);
        setArchiveReady(true);
      }
    }).catch(() => { if (mountedRef.current) setArchiveReady(true); });
    return () => {
      mountedRef.current = false;
      const controller = controllerRef.current;
      controllerRef.current = null;
      if (controller) void controller.stop();
    };
  }, [createArchive, createVault]);

  const stop = useCallback(async () => {
    const controller = controllerRef.current;
    controllerRef.current = null;
    if (controller) await controller.stop();
  }, []);

  const connect = useCallback(async (selection: "chooser" | "authorized" = "chooser") => {
    if (support?.state !== "supported" || stage === "connecting" || stage === "backfill" || stage === "analysis") return;
    if (selection === "chooser" && !isValidPairingCode(pairingCode)) {
      setError(pairingCodeGuidance);
      return;
    }
    setError(null);
    cancelledRef.current = false;
    setAnalysis(null);
    setResult(null);
    setProgress(selection === "authorized" ? sensorImportWaitingCopy : importProgressLabel("connecting"));
    setStage("connecting");
    try {
      const vault = vaultRef.current;
      const archive = archiveRef.current;
      const credentials = sensorImportCredentialCallbacks(vault, remember, (peerId) => {
        selectedPeerIdRef.current = peerId;
        setProgress(sensorSelectedCopy);
        if (selection === "authorized") setAuthorizedPeerFound(true);
      }, archive ? reading => archive.save(reading) : undefined);
      const existingReadings = archive ? await archive.list() : [];
      const backfillSeconds = existingReadings.filter(reading => reading.source === "backfill").map(reading => reading.sensorSeconds);
      const historyStartSeconds = backfillSeconds.length > 0 ? Math.max(...backfillSeconds) : null;
      const created = createController(credentials);
      const controller = created instanceof Promise ? await created : created;
      controllerRef.current = controller;
      const imported = await controller.importSensor({
        sensorName,
        userActivation: selection === "chooser",
        credential: credentialRef.current,
        pairingCode: selection === "authorized" ? null : pairingCode,
        certificateBundle: certificateRef.current,
        selection,
        historyStartSeconds,
      });
      if (!mountedRef.current || cancelledRef.current) return;
      if (archive) {
        await archive.ingest(imported.records);
        if (mountedRef.current) setArchivedReadings(await archive.list());
      }
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
      const archive = archiveRef.current;
      if (archive && mountedRef.current) setArchivedReadings(await archive.list());
      if (selection === "authorized") {
        if (mountedRef.current && !cancelledRef.current) {
          setStage("idle");
          setProgress(sensorImportWaitingCopy);
        }
        return;
      }
      if (!mountedRef.current || cancelledRef.current) return;
      setStage("error");
      setError(sensorImportError(caught));
      setProgress(importProgressLabel("error"));
    }
  }, [createController, pairingCode, remember, stage, support, sensorName]);

  useEffect(() => {
    if (!vaultReady || !archiveReady || support?.state !== "supported" || stage !== "idle" || autoReconnectStartedRef.current) return;
    autoReconnectStartedRef.current = true;
    void connect("authorized");
  }, [archiveReady, connect, stage, support, vaultReady]);

  const disconnect = useCallback(async () => {
    await stop();
    if (mountedRef.current) {
      setStage("idle");
      setProgress("");
    }
  }, [stop]);

  const forget = useCallback(async () => {
    const id = selectedPeerIdRef.current;
    const vault = vaultRef.current;
    if (id && vault) await vault.forget(id);
    credentialRef.current = null;
    setPairingCode("");
    setRemember(false);
  }, []);

  const busy = stage === "connecting" || stage === "backfill" || stage === "analysis";
  const coverage = result ? sensorImportCoverage(result) : null;
  const exportName = archivedReadings.at(-1)?.sensorId || result?.metadata.sensorId || "dexcom-import";
  const latestArchived = [...archivedReadings].reverse().find(reading => reading.reliable && reading.mgdl !== null) ?? archivedReadings.at(-1);

  return (
    <div className="mt-8 space-y-6">
      <p className="border-2 border-accent bg-accent-soft/40 p-6 text-base font-semibold leading-7 text-ink">{privacyCopy}</p>
      <section className="border-2 border-accent bg-accent-soft/40 p-6" aria-labelledby="sensor-prerequisites-title">
        <h2 id="sensor-prerequisites-title" className="text-lg font-semibold text-ink">Before you connect</h2>
        <p className="mt-3 max-w-3xl text-sm leading-6 text-ink-soft">Requires Google Chrome, Bluetooth, and a Dexcom G7 sensor.</p>
        <p className="mt-3 text-sm text-ink-soft" role="status">{support && capability ? browserSupportMessage({ ...capability, userAgent: typeof navigator === "undefined" ? undefined : navigator.userAgent }) : "Checking browser support…"}</p>
        <p className="mt-3 text-sm leading-6 text-ink-soft">{sensorImportWaitingCopy}</p>
      </section>

      <section className="border border-rule bg-paper-raised p-6" aria-labelledby="sensor-connect-title">
        <div className="flex flex-wrap items-baseline justify-between gap-3">
          <h2 id="sensor-connect-title" className="text-lg font-semibold text-ink">Connect locally</h2>
          <span className="text-xs text-ink-faint">Nothing is uploaded. The page remains open while it waits for the sensor.</span>
        </div>
        <div className="mt-5 grid gap-4 sm:grid-cols-2">
          <label className="text-sm text-ink-soft">Sensor name<input value={sensorName} onChange={(event) => setSensorName(event.target.value)} className="mt-1 block w-full border border-rule-strong bg-paper px-3 py-2 text-ink" /></label>
          <label className="text-sm text-ink-soft">Pairing code<input required value={pairingCode} onChange={(event) => setPairingCode(event.target.value)} inputMode="numeric" autoComplete="off" aria-describedby="pairing-code-guidance" className="mt-1 block w-full border border-rule-strong bg-paper px-3 py-2 text-ink" /></label>
          {diagnosticsVisible ? <>
            <label className="text-sm text-ink-soft">Remembered key (optional)<input type="file" onChange={(event) => { const file = event.target.files?.[0] ?? null; if (file) void file.arrayBuffer().then((bytes) => { if (mountedRef.current) credentialRef.current = new Uint8Array(bytes); }); else credentialRef.current = null; }} className="mt-1 block w-full text-sm text-ink-soft file:mr-3 file:border file:border-rule-strong file:bg-paper file:px-3 file:py-1.5 file:text-xs" /></label>
            <label className="text-sm text-ink-soft">Opaque certificate bundle (optional)<input type="file" onChange={(event) => { const file = event.target.files?.[0] ?? null; if (file) void file.arrayBuffer().then((bytes) => { if (mountedRef.current) certificateRef.current = new Uint8Array(bytes); }); else certificateRef.current = null; }} className="mt-1 block w-full text-sm text-ink-soft file:mr-3 file:border file:border-rule-strong file:bg-paper file:px-3 file:py-1.5 file:text-xs" /></label>
          </> : null}
        </div>
        <p id="pairing-code-guidance" className="mt-2 text-xs text-ink-faint">{pairingCodeGuidance}</p>
        <label className="mt-4 flex items-center gap-2 text-sm text-ink-soft"><input type="checkbox" checked={remember} onChange={(event) => setRemember(event.target.checked)} />Remember this sensor on this browser</label>
        <div className="mt-5 flex flex-wrap items-center gap-3">
          <button type="button" onClick={() => void connect()} disabled={support?.state !== "supported" || busy || !isValidPairingCode(pairingCode)} className="bg-accent px-5 py-2.5 text-sm font-medium text-white hover:bg-accent-ink disabled:cursor-not-allowed disabled:opacity-50">{stage === "analysis" ? "Analyzing…" : busy ? "Connecting…" : authorizedPeerFound ? "Choose another sensor" : stage === "error" ? "Retry connection" : "Connect Dexcom G7"}</button>
          {!busy && !isValidPairingCode(pairingCode) ? <span className="text-sm font-semibold text-low" role="status">{pairingCodeRequiredCopy}</span> : null}
          {busy ? <button type="button" onClick={() => { cancelledRef.current = true; void stop(); setStage("cancelled"); setProgress(importProgressLabel("cancelled")); }} className="border border-rule-strong px-4 py-2.5 text-sm text-ink-soft hover:border-accent hover:text-accent">Cancel</button> : null}
          {stage === "done" || result ? <button type="button" onClick={() => void disconnect()} className="border border-rule-strong px-4 py-2.5 text-sm text-ink-soft hover:border-accent hover:text-accent">Disconnect</button> : null}
          {result ? <button type="button" onClick={() => void forget()} className="text-sm text-ink-faint underline decoration-dotted underline-offset-4 hover:text-accent">Forget local key</button> : null}
        </div>
        <p className="mt-4 text-sm text-ink-soft" role="status" aria-live="polite">{progress || importProgressLabel(stage)}</p>
        {error ? <div className="mt-3 border border-low/40 bg-low/5 px-4 py-3 text-sm text-ink-soft" role="alert">{error}</div> : null}
      </section>

      {coverage && result ? <SensorImportCoverage summary={coverage} /> : null}
      {archivedReadings.length > 0 ? <SensorImportTimeline records={archivedReadings} /> : result ? <SensorImportTimeline records={result.records} /> : null}

      {archivedReadings.length > 0 ? (
        <section className="border border-rule bg-paper-raised p-5" aria-labelledby="sensor-export-title">
          <div className="flex flex-wrap items-baseline justify-between gap-3"><h2 id="sensor-export-title" className="text-lg font-semibold text-ink">Saved in this browser</h2><span className="text-xs text-ink-faint">{archivedReadings.length} readings · downloads stay local.</span></div>
          {latestArchived ? <p className="mt-3 text-sm text-ink-soft">Latest saved value: <strong className="text-ink">{latestArchived.mgdl ?? "—"} mg/dL</strong> · {new Date(latestArchived.atMs).toLocaleString()}</p> : null}
          <div className="mt-4 flex flex-wrap gap-3">
            <button type="button" onClick={() => download(`${exportName}-readings.csv`, exportReadingsCsv(archivedReadings), "text/csv") } className="border border-rule-strong px-4 py-2 text-sm text-ink-soft hover:border-accent hover:text-accent">Download CSV</button>
            <button type="button" onClick={() => download(`${exportName}-readings.ndjson`, exportReadingsNdjson(archivedReadings, { sensorId: exportName, readingCount: archivedReadings.length, completeness: result?.completeness ?? "partial", warnings: result?.warnings ?? [] }), "application/x-ndjson") } className="border border-rule-strong px-4 py-2 text-sm text-ink-soft hover:border-accent hover:text-accent">Download NDJSON</button>
          </div>
        </section>
      ) : null}

      {analysis ? <section aria-labelledby="sensor-analysis-title"><div className="border-b border-rule pb-3"><h2 id="sensor-analysis-title" className="text-lg font-semibold text-ink">Your analysis</h2><p className="mt-1 text-xs text-ink-faint">The full local analysis used by the worked example.</p></div><ExampleAnalysisView data={analysis} owner="own" /></section> : null}

      <p className="measure text-xs leading-5 text-ink-faint">This is a research instrument, not medical advice or a medical device. The page reads only what the connected sensor makes available, keeps data in this tab, and does not connect when the page is closed.</p>
    </div>
  );
}
