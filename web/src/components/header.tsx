"use client";

import Link from "next/link";
import { useEffect, useRef, useState, type ReactNode } from "react";

import { Button, cx, Logo, REPO_URL } from "./ui";

type MenuItem = { href: string; title: string; body: string; external?: boolean };
type NavEntry = { label: string; href?: string; external?: boolean; items?: MenuItem[] };

const NAV: NavEntry[] = [
  {
    label: "Platform",
    items: [
      { href: "/#platform", title: "How it works", body: "Classify, three signals, fusion, rerank, a bounded agent." },
      { href: "/#results", title: "Results", body: "What was measured, and what limits it." },
      { href: "/#versions", title: "Versions", body: "Content-addressed indexes and snippet families." },
      { href: "/search", title: "Console", body: "Ask the codebase and see every signal's evidence." },
    ],
  },
  {
    label: "Resources",
    items: [
      {
        href: `${REPO_URL}#readme`,
        title: "README",
        body: "Quickstart, results and the repository map.",
        external: true,
      },
      {
        href: `${REPO_URL}/blob/main/docs/Setup.md`,
        title: "Setup",
        body: "Install paths and the model export.",
        external: true,
      },
      {
        href: `${REPO_URL}/blob/main/docs/API.md`,
        title: "API reference",
        body: "Five endpoints and their contracts.",
        external: true,
      },
      {
        href: `${REPO_URL}/blob/main/docs/BuildLog.md`,
        title: "Build log",
        body: "What was built, measured and corrected.",
        external: true,
      },
    ],
  },
  { label: "Families", href: "/search?tab=families" },
  { label: "GitHub", href: REPO_URL, external: true },
];

function MenuLink({ item, onNavigate }: { item: MenuItem; onNavigate: () => void }) {
  const inner = (
    <>
      <span className="text-body-14 block font-medium">{item.title}</span>
      <span className="text-body-14 mt-0.5 block opacity-60">{item.body}</span>
    </>
  );
  const className = "block rounded-sm px-4 py-3 transition-colors hover:bg-white/[0.06]";
  return item.external ? (
    <a href={item.href} target="_blank" rel="noreferrer" className={className} onClick={onNavigate}>
      {inner}
    </a>
  ) : (
    <Link href={item.href} className={className} onClick={onNavigate}>
      {inner}
    </Link>
  );
}

/**
 * Sticky header. Transparent over the hero, solid and a little shorter once the
 * page scrolls, with click-or-hover dropdowns on desktop and a sheet on mobile.
 */
export function Header({ right }: { right?: ReactNode }) {
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState<string | null>(null);
  const [mobileOpen, setMobileOpen] = useState(false);
  const navRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 24);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    if (!open && !mobileOpen) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setOpen(null);
        setMobileOpen(false);
      }
    };
    const onClick = (event: MouseEvent) => {
      if (navRef.current && !navRef.current.contains(event.target as Node)) setOpen(null);
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener("mousedown", onClick);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("mousedown", onClick);
    };
  }, [open, mobileOpen]);

  const close = () => {
    setOpen(null);
    setMobileOpen(false);
  };

  return (
    <header
      className={cx(
        "sticky inset-x-0 top-0 z-50 text-white transition-[background-color,border-color,height] duration-300",
        scrolled || mobileOpen
          ? "h-16 border-b border-white/[0.07] bg-black/90 backdrop-blur-md"
          : "h-16 border-b border-transparent bg-transparent lg:h-[5.375rem]",
      )}
    >
      <div className="container-page flex h-full items-center justify-between gap-6">
        <Link href="/" aria-label="Axiom home" onClick={close}>
          <Logo />
        </Link>

        <nav ref={navRef} className="hidden items-center gap-1 md:flex" aria-label="Primary">
          {NAV.map((entry) =>
            entry.items ? (
              <div key={entry.label} className="relative" onMouseLeave={() => setOpen(null)}>
                <button
                  type="button"
                  aria-expanded={open === entry.label}
                  aria-haspopup="true"
                  onClick={() => setOpen(open === entry.label ? null : entry.label)}
                  onMouseEnter={() => setOpen(entry.label)}
                  className="text-body-14 flex cursor-pointer items-center gap-1.5 rounded-full px-3.5 py-2 opacity-85 transition-opacity hover:opacity-100"
                >
                  {entry.label}
                  <svg
                    viewBox="0 0 12 12"
                    className={cx("size-2.5 transition-transform", open === entry.label && "rotate-180")}
                    aria-hidden="true"
                  >
                    <path d="M2.5 4.5 6 8l3.5-3.5" fill="none" stroke="currentColor" strokeWidth="1.4" />
                  </svg>
                </button>
                <div
                  className={cx(
                    "absolute top-full left-1/2 w-[22rem] -translate-x-1/2 pt-3 transition-all duration-200",
                    open === entry.label ? "visible translate-y-0 opacity-100" : "invisible -translate-y-1 opacity-0",
                  )}
                >
                  <div className="frame-dots border border-white/10 bg-[#110d0b]/97 p-2 shadow-2xl backdrop-blur-md [--frame-dot:var(--color-stroke-3)]">
                    {entry.items.map((item) => (
                      <MenuLink key={item.title} item={item} onNavigate={close} />
                    ))}
                  </div>
                </div>
              </div>
            ) : entry.external ? (
              <a
                key={entry.label}
                href={entry.href}
                target="_blank"
                rel="noreferrer"
                className="text-body-14 rounded-full px-3.5 py-2 opacity-85 transition-opacity hover:opacity-100"
              >
                {entry.label}
              </a>
            ) : (
              <Link
                key={entry.label}
                href={entry.href!}
                className="text-body-14 rounded-full px-3.5 py-2 opacity-85 transition-opacity hover:opacity-100"
              >
                {entry.label}
              </Link>
            ),
          )}
        </nav>

        <div className="flex items-center gap-3">
          {right ?? (
            <Button href="/search" className="max-sm:hidden">
              Open the console
            </Button>
          )}
          <button
            type="button"
            aria-expanded={mobileOpen}
            aria-controls="mobile-nav"
            onClick={() => setMobileOpen(!mobileOpen)}
            className="flex size-9 cursor-pointer items-center justify-center rounded-full border border-white/15 md:hidden"
          >
            <span className="sr-only">{mobileOpen ? "Close menu" : "Open menu"}</span>
            <svg viewBox="0 0 16 16" className="size-4" aria-hidden="true">
              {mobileOpen ? (
                <path d="m3.5 3.5 9 9m0-9-9 9" stroke="currentColor" strokeWidth="1.5" />
              ) : (
                <path d="M2 5h12M2 11h12" stroke="currentColor" strokeWidth="1.5" />
              )}
            </svg>
          </button>
        </div>
      </div>

      <div
        id="mobile-nav"
        className={cx(
          "fixed inset-x-0 top-16 bottom-0 overflow-y-auto bg-black/97 px-5 pb-10 backdrop-blur-md transition-opacity md:hidden",
          mobileOpen ? "visible opacity-100" : "invisible opacity-0",
        )}
      >
        <Link href="/search" onClick={close} className="text-heading-32 block border-b border-white/10 py-5">
          Open the console
        </Link>
        {NAV.map((entry) => (
          <div key={entry.label} className="border-b border-white/10 py-5">
            {entry.items ? (
              <>
                <p className="text-mono-s uppercase opacity-60">{entry.label}</p>
                <div className="-mx-4 mt-2">
                  {entry.items.map((item) => (
                    <MenuLink key={item.title} item={item} onNavigate={close} />
                  ))}
                </div>
              </>
            ) : (
              <a
                href={entry.href}
                onClick={close}
                {...(entry.external ? { target: "_blank", rel: "noreferrer" } : {})}
                className="text-body-20 block"
              >
                {entry.label}
              </a>
            )}
          </div>
        ))}
      </div>
    </header>
  );
}
