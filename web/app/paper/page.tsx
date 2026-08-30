import { readFileSync } from "node:fs";
import { join } from "node:path";
import { marked } from "marked";
import { LINKS } from "@/lib/facts";

export const metadata = {
  title: "Paper — OpenCGM-StateEvent",
  description:
    "The full write-up: method, protocol, results, ablations, and limitations.",
};

/** Rendered from the repository's paper.md, so the page cannot drift from the source. */
function paperHtml(): { html: string; toc: { depth: number; text: string; id: string }[] } {
  const md = readFileSync(join(process.cwd(), "..", "paper.md"), "utf8");

  const toc: { depth: number; text: string; id: string }[] = [];
  const slug = (s: string) =>
    s
      .toLowerCase()
      .replace(/[^\w\s-]/g, "")
      .trim()
      .replace(/\s+/g, "-");

  const renderer = new marked.Renderer();
  renderer.heading = function ({ tokens, depth }) {
    // Render inline markdown inside the heading, and keep a plain-text copy for the
    // contents list -- otherwise the sidebar shows literal `**` and backticks.
    const html = this.parser.parseInline(tokens);
    const text = html.replace(/<[^>]+>/g, "");
    const id = slug(text);
    if (depth === 2 || depth === 3) toc.push({ depth, text, id });
    const size =
      depth === 1
        ? "mt-0 text-3xl font-semibold tracking-tight"
        : depth === 2
          ? "mt-10 border-t border-rule pt-8 text-2xl font-semibold tracking-tight"
          : "mt-9 text-lg font-semibold";
    return `<h${depth} id="${id}" class="${size} text-ink scroll-mt-24">${html}</h${depth}>`;
  };

  const html = marked.parse(md, { renderer, gfm: true, async: false }) as string;
  return { html, toc };
}

export default function PaperPage() {
  const { html, toc } = paperHtml();

  return (
    <div className="mx-auto max-w-6xl px-6 py-12">
      <div className="grid gap-10 lg:grid-cols-[16rem_1fr]">
        <aside className="lg:sticky lg:top-24 lg:self-start">
          <p className="text-xs font-semibold uppercase tracking-widest text-ink-faint">
            Contents
          </p>
          <nav className="mt-3 space-y-1.5 text-sm">
            {toc.map((h) => (
              <a
                key={h.id}
                href={`#${h.id}`}
                className={`block hover:text-accent ${
                  h.depth === 2 ? "text-ink-soft" : "pl-3 text-ink-faint"
                }`}
              >
                {h.text}
              </a>
            ))}
          </nav>
          <div className="mt-8 border-t border-rule pt-4 text-xs text-ink-faint">
            <p>
              Rendered from{" "}
              <code className="font-mono">paper.md</code> in the repository, so this page and
              the source cannot disagree.
            </p>
            <a
              href={`${LINKS.repo}/blob/main/paper.md`}
              className="mt-2 inline-block text-accent hover:underline"
            >
              View the source →
            </a>
          </div>
        </aside>

        <article
          className="paper-prose min-w-0"
          dangerouslySetInnerHTML={{ __html: html }}
        />
      </div>
    </div>
  );
}
