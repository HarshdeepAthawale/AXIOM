"use client";

import { useEffect, useState } from "react";

import { Card, cx } from "./ui";

type Row = { loc: string; symbol: string; signals: Array<[string, string]>; why: string };
type Scene = { type: string; dot: string; query: string; highlight: string[]; rows: Row[] };

// Real output from `axiom query --top-k 3` on the bundled test repository
// (tests/fixtures/repo_v1) with the MiniLM embedder and ms-marco reranker.
const SCENES: Scene[] = [
  {
    type: "structural",
    dot: "bg-oasis",
    query: "Which files call preprocessInput before resolveTool?",
    highlight: ["preprocessInput", "resolveTool"],
    rows: [
      {
        loc: "src/tools/registry.js:8-15",
        symbol: "dispatch",
        signals: [
          ["Dense", "#1"],
          ["BM25", "#1"],
          ["Call graph", "#2"],
        ],
        why: "calls preprocessInput (ordinal 0) before resolveTool (ordinal 1)",
      },
      {
        loc: "src/broken/syntax_error.js:1-11",
        symbol: "handleBroken",
        signals: [
          ["Dense", "#4"],
          ["BM25", "#2"],
          ["Call graph", "#1"],
        ],
        why: "calls preprocessInput (ordinal 0) before resolveTool (ordinal 1)",
      },
      {
        loc: "src/agents/bluetooth.js:8-53",
        symbol: "BluetoothAgent",
        signals: [
          ["Dense", "#9"],
          ["BM25", "#13"],
          ["Call graph", "#4"],
        ],
        why: "calls preprocessInput (rank 4 in the call graph)",
      },
    ],
  },
  {
    type: "usage",
    dot: "bg-sun",
    query: "Where is the Bluetooth-settings deeplink used?",
    highlight: ["Bluetooth-settings"],
    rows: [
      {
        loc: "src/agents/bluetooth.js:28-35",
        symbol: "openSettings",
        signals: [
          ["Dense", "#2"],
          ["BM25", "#2"],
        ],
        why: "matches Bluetooth-settings lexically (BM25 rank 2)",
      },
      {
        loc: "src/agents/bluetooth.js:1-3",
        symbol: "module block",
        signals: [
          ["Dense", "#4"],
          ["BM25", "#1"],
        ],
        why: "matches Bluetooth-settings lexically (BM25 rank 1)",
      },
      {
        loc: "src/i18n/labels.js:11-13",
        symbol: "déeplinkFrançais",
        signals: [
          ["Dense", "#1"],
          ["BM25", "#3"],
        ],
        why: "matches Bluetooth-settings lexically (BM25 rank 3)",
      },
    ],
  },
  {
    type: "semantic",
    dot: "bg-dawn",
    query: "How is user input normalized before dispatch?",
    highlight: ["normalized"],
    rows: [
      {
        loc: "src/main.js:1-2",
        symbol: "module block",
        signals: [
          ["Dense", "#2"],
          ["BM25", "#7"],
        ],
        why: "nearest neighbour of the query embedding (rank 2)",
      },
      {
        loc: "src/utils/normalize.js:33-40",
        symbol: "preprocessInput",
        signals: [
          ["Dense", "#4"],
          ["BM25", "#2"],
        ],
        why: "nearest neighbour of the query embedding (rank 4)",
      },
      {
        loc: "src/utils/normalize.js:24-28",
        symbol: "normalizeInput",
        signals: [
          ["Dense", "#1"],
          ["BM25", "#11"],
        ],
        why: "nearest neighbour of the query embedding (rank 1)",
      },
    ],
  },
];

const TYPE_MS = 34;
const HOLD_MS = 5200;

function Highlighted({ text, words }: { text: string; words: string[] }) {
  const pattern = new RegExp(`(${words.map((word) => word.replace(/[-]/g, "\\-")).join("|")})`);
  return (
    <>
      {text.split(pattern).map((part, index) =>
        words.includes(part) ? (
          <span key={index} className="text-sun italic">
            {part}
          </span>
        ) : (
          <span key={index}>{part}</span>
        ),
      )}
    </>
  );
}

/** The hero's "video": the console answering the three archetype questions in turn. */
export function HeroPreview() {
  const [scene, setScene] = useState(0);
  const [typed, setTyped] = useState(SCENES[0].query.length);
  const [animate, setAnimate] = useState(false);

  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const start = setTimeout(() => setAnimate(true), 0);
    return () => clearTimeout(start);
  }, []);

  useEffect(() => {
    if (!animate) return;
    const query = SCENES[scene].query;
    if (typed < query.length) {
      const timer = setTimeout(() => setTyped(typed + 1), TYPE_MS);
      return () => clearTimeout(timer);
    }
    const timer = setTimeout(() => {
      setScene((scene + 1) % SCENES.length);
      setTyped(0);
    }, HOLD_MS);
    return () => clearTimeout(timer);
  }, [animate, scene, typed]);

  const current = SCENES[scene];
  const done = typed >= current.query.length;
  const visible = current.query.slice(0, typed);

  return (
    <Card
      className="animate-rise relative mt-14 border-white/12 bg-[#120e0b]/90 [--frame-dot:var(--color-stroke-3)] backdrop-blur-sm [animation-delay:0.15s] lg:mt-20"
      aria-label="The console answering an example query"
    >
      <div className="flex items-center justify-between gap-4 border-b border-white/10 px-5 py-3.5">
        <span className="text-mono-s uppercase opacity-70">axiom query</span>
        <div className="flex items-center gap-4">
          <div className="hidden gap-1.5 sm:flex" aria-hidden="true">
            {SCENES.map((item, index) => (
              <span
                key={item.type}
                className={cx(
                  "h-1 rounded-full transition-all duration-500",
                  index === scene ? "bg-sun w-6" : "w-2 bg-white/20",
                )}
              />
            ))}
          </div>
          <span className="text-mono-s flex items-center gap-2 uppercase opacity-70">
            <span className={cx("size-1.5 rounded-full", current.dot)} /> {current.type} · 1 pass
          </span>
        </div>
      </div>
      <div className="px-5 py-6 md:px-8">
        <p className="font-heading min-h-[2.4em] text-[clamp(1.25rem,2.4vw,1.75rem)] font-light" aria-live="polite">
          <Highlighted text={visible} words={current.highlight} />
          <span className="animate-blink ml-0.5 inline-block h-[0.9em] w-px translate-y-[0.1em] bg-current" />
        </p>
        <ol className="mt-5 min-h-[15.5rem] divide-y divide-white/8 border-t border-white/8 md:min-h-[13.5rem]">
          {done &&
            current.rows.map((row, index) => (
              <li
                key={`${scene}-${row.loc}`}
                className="animate-rise grid gap-2 py-4 md:grid-cols-[2.5rem_1fr_auto] md:items-center md:gap-6"
                style={{ animationDelay: `${index * 120}ms` }}
              >
                <span className="text-mono-m text-sun">0{index + 1}</span>
                <div className="min-w-0">
                  <p className="text-mono-m truncate">
                    {row.loc} <span className="opacity-60">·</span> <span className="text-desert">{row.symbol}</span>
                  </p>
                  <p className="text-body-14 mt-1 truncate opacity-70">why: {row.why}</p>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {row.signals.map(([name, rank]) => (
                    <span key={name} className="text-mono-s rounded-full border border-white/12 px-2.5 py-1 opacity-85">
                      {name} {rank}
                    </span>
                  ))}
                </div>
              </li>
            ))}
        </ol>
      </div>
    </Card>
  );
}
