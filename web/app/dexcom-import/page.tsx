import { SensorImportClient } from "@/components/SensorImportClient";

export const metadata = {
  title: "Dexcom import — OpenCGM-StateEvent",
  description: "Read available Dexcom G7 history locally in your browser. Nothing is uploaded.",
};

export default function DexcomImportPage() {
  return (
    <div className="mx-auto max-w-6xl px-6 py-12">
      <h1 className="text-3xl font-semibold tracking-tight text-ink md:text-4xl">Import from Dexcom G7</h1>
      <p className="lede mt-5 max-w-3xl">Read the history your sensor makes available, inspect its coverage, and run the same analysis used by the rest of this site.</p>
      <SensorImportClient />
    </div>
  );
}
