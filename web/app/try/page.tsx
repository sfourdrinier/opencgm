import { TryClient } from "@/components/TryClient";

export const metadata = {
  title: "Try it — OpenCGM-StateEvent",
  description:
    "Run the encoder on a day of glucose data, in your browser. Nothing is uploaded.",
};

export default function TryPage() {
  return (
    <div className="mx-auto max-w-6xl px-6 py-12">
      <h1 className="text-3xl font-semibold tracking-tight text-ink md:text-4xl">Try it</h1>
      <p className="lede mt-5 max-w-3xl">
        Drop in a CGM export and see what the model makes of it. Nothing is uploaded — the file
        is parsed and the model runs in this tab.
      </p>
      <TryClient />
    </div>
  );
}
