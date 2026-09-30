import Link from "next/link";
import { useId, type ComponentProps, type ReactNode } from "react";

export const REPO_URL = "https://github.com/HarshdeepAthawale/AXIOM";

export function cx(...parts: Array<string | false | null | undefined>) {
  return parts.filter(Boolean).join(" ");
}

/** Uppercase mono label led by a small sun-coloured dot. */
export function Eyebrow({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <p className={cx("text-mono-s flex items-center gap-2.5 uppercase opacity-80", className)}>
      <span className="bg-sun size-1.5 shrink-0 rounded-full" aria-hidden="true" />
      {children}
    </p>
  );
}

export function Logo({ className }: { className?: string }) {
  return (
    <span className={cx("inline-flex items-center gap-2.5", className)}>
      <svg viewBox="0 0 24 24" className="size-6" aria-hidden="true">
        <circle cx="12" cy="12" r="11" fill="none" stroke="currentColor" strokeOpacity="0.35" />
        <path d="M5.5 16.5 12 5l6.5 11.5" fill="none" stroke="currentColor" strokeWidth="1.4" />
        <circle cx="12" cy="13.2" r="2.1" fill="var(--color-sun)" />
      </svg>
      <span className="font-heading text-[1.375rem] leading-none font-normal tracking-tight">Axiom</span>
    </span>
  );
}

type ButtonProps = {
  href: string;
  children: ReactNode;
  variant?: "light" | "ghost" | "dark";
  external?: boolean;
  className?: string;
};

export function Button({ href, children, variant = "light", external, className }: ButtonProps) {
  const styles = {
    light: "bg-bone text-night hover:bg-white",
    dark: "bg-night text-bone hover:bg-midnight",
    ghost: "border border-current/25 hover:border-current/60",
  }[variant];
  const content = (
    <>
      {children}
      <svg viewBox="0 0 16 16" className="size-3.5 transition-transform group-hover:translate-x-0.5" aria-hidden="true">
        <path d="M3 8h9M8.5 4.5 12 8l-3.5 3.5" fill="none" stroke="currentColor" strokeWidth="1.5" />
      </svg>
    </>
  );
  const classes = cx(
    "group text-body-14 inline-flex items-center gap-2 rounded-full px-5 py-2.5 font-medium transition-colors",
    styles,
    className,
  );
  return external ? (
    <a href={href} target="_blank" rel="noreferrer" className={classes}>
      {content}
    </a>
  ) : (
    <Link href={href} className={classes}>
      {content}
    </Link>
  );
}

type DuneTone = "night" | "dusk";

// Crest lines only; each dune fills down to the bottom edge. Ordered far to near.
const DUNE_CRESTS = [
  "M0 236C150 224 262 204 384 212C470 218 522 234 612 231C760 226 884 194 1012 203C1132 211 1222 233 1322 226C1382 222 1416 215 1440 213",
  "M0 280C112 268 212 249 334 260C432 270 484 293 594 289C706 285 804 249 934 255C1062 261 1124 292 1244 291C1334 290 1392 277 1440 271",
  "M0 326C160 318 302 294 452 310C582 324 662 350 802 345C952 340 1052 311 1192 317C1312 323 1382 340 1440 338",
  "M0 370C222 360 424 354 644 368C864 382 1044 392 1264 379C1352 374 1412 366 1440 364",
];

const DUNE_PALETTE: Record<
  DuneTone,
  { sky: string; sun: number; bloom: number; crest: number; layers: Array<[string, string, number]> }
> = {
  // [top colour, trough colour, opacity], far to near. Far layers are paler: haze.
  night: {
    sky: "#2a170b",
    sun: 0.55,
    bloom: 0.32,
    crest: 0.11,
    layers: [
      ["#3b2a1f", "#241911", 0.9],
      ["#2e2019", "#19120d", 1],
      ["#211812", "#110c09", 1],
      ["#0f0b09", "#0b0907", 1],
    ],
  },
  dusk: {
    sky: "#4a260f",
    sun: 0.95,
    bloom: 0.5,
    crest: 0.45,
    layers: [
      ["#86583a", "#553621", 0.85],
      ["#734a2f", "#3f2819", 1],
      ["#4b3021", "#24170f", 1],
      ["#1d130d", "#070504", 1],
    ],
  },
};

/**
 * A desert at the golden hour: a low sun on the horizon, dunes receding into
 * haze, crests lit from the sun's side, and a little film grain. Drawn in SVG
 * so nothing is borrowed.
 */
export function Dunes({ className, tone = "night" }: { className?: string; tone?: DuneTone }) {
  const id = useId().replace(/:/g, "");
  const palette = DUNE_PALETTE[tone];
  const sunX = 1060;
  const horizon = 212;
  return (
    <svg
      viewBox="0 0 1440 420"
      preserveAspectRatio="xMidYMax slice"
      className={cx(
        "pointer-events-none w-full [mask-image:linear-gradient(to_bottom,transparent,#000_42%)]",
        className,
      )}
      aria-hidden="true"
    >
      <defs>
        <linearGradient id={`${id}-sky`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={palette.sky} stopOpacity="0" />
          <stop offset="55%" stopColor={palette.sky} stopOpacity="0.55" />
          <stop offset="100%" stopColor={palette.sky} stopOpacity="0.9" />
        </linearGradient>
        <radialGradient id={`${id}-bloom`} cx={sunX} cy={horizon} r="520" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="#ffb676" stopOpacity={palette.bloom} />
          <stop offset="35%" stopColor="#ff8b3e" stopOpacity={palette.bloom * 0.4} />
          <stop offset="100%" stopColor="#ff8b3e" stopOpacity="0" />
        </radialGradient>
        <radialGradient id={`${id}-sun`} cx={sunX} cy={horizon} r="46" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="#fff1dc" stopOpacity={palette.sun} />
          <stop offset="55%" stopColor="#ffc07e" stopOpacity={palette.sun * 0.85} />
          <stop offset="100%" stopColor="#ff8b3e" stopOpacity="0" />
        </radialGradient>
        <linearGradient id={`${id}-crest`} x1="0" y1="0" x2="1440" y2="0" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="#ffc58a" stopOpacity="0" />
          <stop offset={`${(sunX / 1440) * 100}%`} stopColor="#ffd3a4" stopOpacity={palette.crest} />
          <stop offset="100%" stopColor="#ffc58a" stopOpacity="0.04" />
        </linearGradient>
        {palette.layers.map(([top, bottom], index) => (
          <linearGradient key={index} id={`${id}-dune-${index}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={top} />
            <stop offset="100%" stopColor={bottom} />
          </linearGradient>
        ))}
        <filter id={`${id}-grain`} x="0" y="0" width="100%" height="100%">
          <feTurbulence type="fractalNoise" baseFrequency="0.85" numOctaves="2" stitchTiles="stitch" />
          <feColorMatrix values="0 0 0 0 1  0 0 0 0 0.9  0 0 0 0 0.8  0 0 0 0.06 0" />
        </filter>
      </defs>

      <rect width="1440" height="420" fill={`url(#${id}-sky)`} />
      <rect width="1440" height="420" fill={`url(#${id}-bloom)`} />
      <circle cx={sunX} cy={horizon} r="46" fill={`url(#${id}-sun)`} />

      {DUNE_CRESTS.map((crest, index) => (
        <g key={index} opacity={palette.layers[index][2]}>
          <path d={`${crest}V420H0Z`} fill={`url(#${id}-dune-${index})`} />
          <path d={crest} fill="none" stroke={`url(#${id}-crest)`} strokeWidth={index === 0 ? 1 : 1.4} />
        </g>
      ))}

      <rect width="1440" height="420" filter={`url(#${id}-grain)`} />
    </svg>
  );
}

export function Marquee({ items, className }: { items: string[]; className?: string }) {
  const row = [...items, ...items];
  return (
    <div
      className={cx(
        "relative overflow-hidden [mask-image:linear-gradient(90deg,transparent,#000_12%,#000_88%,transparent)]",
        className,
      )}
    >
      <div className="animate-marquee flex w-max gap-14">
        {row.map((item, index) => (
          <span
            key={`${item}-${index}`}
            className="text-mono-m flex items-center gap-3 whitespace-nowrap uppercase opacity-60"
          >
            <span className="bg-stroke-2 size-1 rounded-full" aria-hidden="true" />
            {item}
          </span>
        ))}
      </div>
    </div>
  );
}

export function Footer() {
  const columns: Array<{ title: string; links: Array<{ label: string; href: string }> }> = [
    {
      title: "Product",
      links: [
        { label: "Console", href: "/search" },
        { label: "Snippet families", href: "/search?tab=families" },
        { label: "API reference", href: `${REPO_URL}/blob/main/docs/API.md` },
      ],
    },
    {
      title: "Project",
      links: [
        { label: "README", href: `${REPO_URL}#readme` },
        { label: "Setup", href: `${REPO_URL}/blob/main/docs/Setup.md` },
        { label: "Build log", href: `${REPO_URL}/blob/main/docs/BuildLog.md` },
      ],
    },
    {
      title: "Team Incognito",
      links: [
        { label: "Prabinder Singh", href: REPO_URL },
        { label: "Anish Grover", href: REPO_URL },
        { label: "Harshdeep Athawale", href: REPO_URL },
        { label: "Parth Deshmukh", href: REPO_URL },
      ],
    },
  ];
  return (
    <footer className="relative overflow-hidden bg-black pt-24 pb-64 text-white lg:pt-36 lg:pb-80">
      <div
        className="absolute inset-x-0 top-0 h-[55%] bg-[radial-gradient(ellipse_at_30%_20%,#3a2414_0%,transparent_60%)]"
        aria-hidden="true"
      />
      <Dunes tone="dusk" className="absolute inset-x-0 bottom-0 h-64 lg:h-80" />
      <div className="container-page relative">
        <h2 className="text-heading-56 max-w-[40rem]">Ask your codebase where it already happens</h2>
        <div className="mt-8 flex flex-wrap gap-3">
          <Button href="/search">Open the console</Button>
          <Button href={REPO_URL} external variant="ghost">
            View on GitHub
          </Button>
        </div>
        <div className="mt-24 grid gap-10 border-t border-white/10 pt-10 sm:grid-cols-2 lg:mt-36 lg:grid-cols-4">
          <div>
            <Logo />
            <p className="text-body-14 mt-4 max-w-60 opacity-60">
              Samsung PRISM GenAI Hackathon 2026 · Theme 01. Thapar Institute of Engineering &amp; Technology.
            </p>
          </div>
          {columns.map((column) => (
            <div key={column.title}>
              <h3 className="text-mono-s uppercase opacity-60">{column.title}</h3>
              <ul className="mt-4 space-y-2.5">
                {column.links.map((link) => (
                  <li key={link.label}>
                    <a href={link.href} className="text-body-14 opacity-80 transition-opacity hover:opacity-100">
                      {link.label}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>
    </footer>
  );
}

export function Card({ className, ...props }: ComponentProps<"div">) {
  return <div className={cx("frame-dots border", className)} {...props} />;
}
