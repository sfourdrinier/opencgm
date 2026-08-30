import type { Metadata } from "next";
import { Inter } from "next/font/google";
import Link from "next/link";
import { Nav } from "@/components/Nav";
import { LINKS } from "@/lib/facts";
import "./globals.css";

// globals.css names Inter in --font-sans but nothing ever fetched it, so every visitor was
// served the system fallback. Load it properly, and hand the variable to the token.
const inter = Inter({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "OpenCGM-StateEvent — an open reconstruction of GlucoFM",
  description:
    "A small, open, independently rebuilt foundation model for continuous glucose monitoring. " +
    "Trained only on public data. Every number checkable, every decision written down.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={inter.variable}>
      <body>
        <header className="border-b border-rule bg-paper-raised/80 backdrop-blur sticky top-0 z-40">
          <div className="mx-auto flex max-w-6xl flex-wrap items-baseline gap-x-6 gap-y-2 px-6 py-4">
            <Link href="/" className="font-semibold tracking-tight text-ink">
              OpenCGM<span className="text-accent">-StateEvent</span>
            </Link>
            <Nav />
            <a
              href="https://github.com/sfourdrinier/opencgm"
              className="ml-auto text-sm text-ink-faint hover:text-accent"
            >
              Source ↗
            </a>
          </div>
        </header>

        <main>{children}</main>

        <footer className="mt-16 border-t border-rule bg-paper-raised">
          <div className="mx-auto max-w-6xl px-6 py-10">
            <div className="grid gap-x-10 gap-y-6 text-sm text-ink-faint md:grid-cols-3">
              <div>
                <p className="font-medium text-ink-soft">OpenCGM-StateEvent</p>
                <p className="mt-2">
                  An independent reconstruction from a published description. Not Google&rsquo;s
                  implementation, not Google&rsquo;s weights, and not affiliated with or endorsed
                  by Google.
                </p>
                <p className="mt-2">
                  Built by{" "}
                  <a
                    href="https://www.linkedin.com/in/stephanefourdrinier"
                    className="text-accent hover:underline"
                  >
                    Stephane Fourdrinier
                  </a>
                  . Independent research, no institutional funding.
                </p>
              </div>

              <div>
                <p className="font-medium text-ink-soft">Not a medical device</p>
                <p className="mt-2">
                  Nothing here diagnoses, treats, or should inform a decision about
                  anyone&rsquo;s health. The probes are research instruments fitted on a few
                  hundred people and are wrong often. Talk to a clinician.
                </p>
              </div>

              <div>
                <p className="font-medium text-ink-soft">Licence &amp; source</p>
                <p className="mt-2">
                  Code Apache-2.0. Encoder weights CC-BY-NC-4.0. Probe heads CC-BY-NC-SA-4.0,
                  because eight of them are fitted on share-alike material.
                </p>
                <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1">
                  <a href={LINKS.repo} className="text-accent hover:underline">
                    GitHub
                  </a>
                  <a href="/paper" className="text-accent hover:underline">
                    Paper
                  </a>
                  <a href={LINKS.decisions} className="text-accent hover:underline">
                    Decisions
                  </a>
                  <a href={LINKS.reproduce} className="text-accent hover:underline">
                    Reproduce
                  </a>
                </div>
              </div>
            </div>
          </div>
        </footer>
      </body>
    </html>
  );
}
