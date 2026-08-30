"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV = [
  { href: "/how-it-works", label: "How it works" },
  { href: "/example", label: "Example" },
  { href: "/results", label: "Results" },
  { href: "/paper", label: "Paper" },
  { href: "/try", label: "Try it" },
  { href: "/api", label: "API" },
];

export function Nav() {
  const pathname = usePathname();
  return (
    <nav className="flex flex-wrap gap-x-5 gap-y-1 text-sm">
      {NAV.map((item) => {
        const active = pathname === item.href;
        return (
          <Link
            key={item.href}
            href={item.href}
            aria-current={active ? "page" : undefined}
            className={
              active
                ? "border-b-2 border-accent pb-0.5 font-medium text-ink"
                : "border-b-2 border-transparent pb-0.5 text-ink-soft hover:text-accent"
            }
          >
            {item.label}
          </Link>
        );
      })}
    </nav>
  );
}
