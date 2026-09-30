import { Header } from "@/components/header";
import { HeroPreview } from "@/components/hero-preview";
import { Reveal } from "@/components/reveal";
import { Button, Card, Dunes, Eyebrow, Footer, Marquee, REPO_URL } from "@/components/ui";

const STACK = [
  "tree-sitter",
  "FAISS",
  "bm25s",
  "ONNX Runtime INT8",
  "SQLite call graph",
  "Reciprocal Rank Fusion",
  "Cross-encoder rerank",
  "FastAPI",
  "MTEB · CoIR",
];

export default function Home() {
  return (
    <>
      <Header />
      <main>
        <Hero />
        <Problem />
        <Platform />
        <QueryTypes />
        <Results />
        <Versions />
        <Laptop />
      </main>
      <Footer />
    </>
  );
}

function Hero() {
  return (
    <section className="bg-void relative overflow-clip pt-16 pb-12 text-white md:pt-24 lg:pt-28 lg:pb-20">
      <Dunes className="absolute inset-x-0 bottom-0 h-[26rem] opacity-80" />
      <div className="container-page relative">
        <div className="animate-rise max-w-[46rem]">
          <Eyebrow>Samsung PRISM GenAI Hackathon · Theme 01</Eyebrow>
          <h1 className="text-heading-64 mt-6 text-pretty">Code retrieval for codebases no context window can hold</h1>
          <p className="text-body-18 mt-5 max-w-[36rem] opacity-80 lg:mt-7">
            Engineers rarely ask for new code. They ask where something already happens. Axiom answers with a ranked
            list of snippets and exact <span className="font-mono text-[0.92em]">file:line</span> locations, fusing
            three retrieval signals under a bounded agent, entirely on a CPU.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Button href="/search">Open the console</Button>
            <Button href={REPO_URL} external variant="ghost">
              View on GitHub
            </Button>
          </div>
        </div>

        <HeroPreview />

        <div className="mt-14 lg:mt-20">
          <p className="text-mono-s mb-5 text-center uppercase opacity-60">Built on</p>
          <Marquee items={STACK} />
        </div>
      </div>
    </section>
  );
}

function Problem() {
  return (
    <section className="relative overflow-clip bg-[linear-gradient(180deg,#0b0907_0%,#110e0c_45%,var(--color-twilight)_100%)] pt-20 pb-28 text-white md:pt-28 lg:pt-36 lg:pb-44">
      <div className="container-page">
        <Eyebrow>The problem</Eyebrow>
        <Reveal>
          <p className="text-heading-48 mt-6 max-w-[58rem] text-pretty">
            A ten-thousand-file codebase is tens of millions of tokens.{" "}
            <span className="opacity-55">
              No LLM context window holds it, so the model never sees the code it needs.
            </span>
          </p>
        </Reveal>
        <div className="mt-16 grid gap-px overflow-hidden border border-white/10 bg-white/10 md:grid-cols-3 lg:mt-24">
          {[
            [
              "Keyword search misses meaning",
              "“Preprocessing” never matches normalize(). The words in the question are not the words in the code.",
            ],
            [
              "Embeddings miss order",
              "“Which files call X before Y” is about the call graph. No snippet’s text contains the answer.",
            ],
            [
              "Indexes go stale",
              "Code changes every commit. Re-embedding the whole repository for each version does not scale.",
            ],
          ].map(([title, body], index) => (
            <Reveal key={title} delay={index * 120} className="bg-[#100d0c]/95 p-7 lg:p-9">
              <h3 className="text-heading-24">{title}</h3>
              <p className="text-body-16 mt-3 opacity-75">{body}</p>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  );
}

function Platform() {
  const steps = [
    {
      n: "01",
      title: "Classify and plan",
      body: "Semantic, structural, usage or hybrid. Identifiers extracted, terms expanded.",
    },
    {
      n: "02",
      title: "Three signals in parallel",
      body: "Dense embeddings, code-aware BM25, and an order-aware tree-sitter call graph.",
    },
    {
      n: "03",
      title: "Weighted fusion",
      body: "Reciprocal Rank Fusion with weights chosen by query type. An empty signal hands its weight on.",
    },
    {
      n: "04",
      title: "Cross-encoder rerank",
      body: "The top candidates are rescored by a model that reads query and code together.",
    },
    {
      n: "05",
      title: "Bounded agent",
      body: "If scores are weak it rewrites and retrieves again: at most two passes, a 5 s deadline.",
    },
  ];
  return (
    <section
      id="platform"
      className="to-dust relative overflow-clip bg-gradient-to-b from-white pt-20 pb-20 text-black md:pt-28 lg:pt-36 lg:pb-32"
    >
      <div className="container-page">
        <Reveal className="grid gap-10 lg:grid-cols-[1.1fr_1fr] lg:gap-20">
          <div>
            <Eyebrow className="opacity-70">Platform</Eyebrow>
            <h2 className="text-heading-48 mt-6 text-pretty">Axiom makes a codebase answerable</h2>
          </div>
          <div className="lg:pt-12">
            <p className="text-body-18 text-black/75">
              Retrieval only: Axiom never generates code. It finds, ranks and locates what already exists, and shows its
              working: every result carries the rank each signal gave it and the reason it ranked.
            </p>
            <p className="text-body-18 mt-4 text-black/75">
              The optional local LLM classifies and rewrites queries. It never reads your code, and the module boundary
              enforces that.
            </p>
          </div>
        </Reveal>
        <Reveal>
          <Card className="border-stroke-1 mt-16 bg-white/70 [--frame-dot:var(--color-stroke-2)] lg:mt-24">
            <ol className="grid divide-y divide-[var(--color-stroke-1)] md:grid-cols-5 md:divide-x md:divide-y-0">
              {steps.map((step) => (
                <li key={step.n} className="p-6 lg:p-7">
                  <span className="text-mono-s text-ember">{step.n}</span>
                  <h3 className="text-heading-24 mt-4">{step.title}</h3>
                  <p className="text-body-14 mt-2.5 text-black/65">{step.body}</p>
                </li>
              ))}
            </ol>
          </Card>
        </Reveal>
      </div>
    </section>
  );
}

function QueryTypes() {
  const types = [
    {
      tag: "Semantic",
      color: "bg-dawn",
      q: "How is the input preprocessed before going to the main function?",
      a: "Dense embeddings match by meaning, so code that never says “preprocess” can still rank.",
    },
    {
      tag: "Structural",
      color: "bg-oasis",
      q: "Which files call preprocessInput before resolveTool?",
      a: "Answered from the call graph, with the ordinal of each call as evidence.",
    },
    {
      tag: "Usage",
      color: "bg-sun",
      q: "Where is the Bluetooth-settings deeplink used?",
      a: "An exact literal: BM25 leads, and fusion weights let it win.",
    },
  ];
  return (
    <section className="bg-dust relative overflow-clip pt-6 pb-16 text-black lg:pb-20">
      <div className="container-page">
        <p className="text-body-18 mx-auto max-w-[36rem] text-center text-black/70">
          Three questions from the problem statement, three different kinds of evidence. Axiom routes each one to the
          signal that can actually answer it.
        </p>
        <div className="mt-12 grid gap-5 md:grid-cols-3">
          {types.map((type, index) => (
            <Reveal key={type.tag} delay={index * 120} className="flex">
              <Card className="border-stroke-1 flex w-full flex-col bg-white p-7 [--frame-dot:var(--color-stroke-2)]">
                <span className="text-mono-s flex items-center gap-2 uppercase text-black/65">
                  <span className={`${type.color} size-1.5 rounded-full`} /> {type.tag}
                </span>
                <p className="text-heading-24 mt-5 italic">“{type.q}”</p>
                <p className="text-body-14 mt-auto pt-6 text-black/65">{type.a}</p>
              </Card>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  );
}

function Results() {
  const stats = [
    { value: "3", label: "Retrieval signals fused on every query" },
    { value: "0.9 s", label: "Warm query on a laptop CPU, cross-encoder included" },
    { value: "50 of 101", label: "Chunks re-embedded when 50 files change" },
    { value: "0", label: "Embedding calls for renamed, moved or reverted code" },
    { value: "109", label: "Snippet families tracked across three versions" },
    { value: "659", label: "Tests passing on a fresh clone" },
  ];
  return (
    <section
      id="results"
      className="from-dust relative overflow-clip bg-gradient-to-b to-white pt-16 pb-16 text-black md:pt-20 lg:pt-24 lg:pb-20"
    >
      <div className="container-page">
        <Eyebrow className="opacity-70">Results</Eyebrow>
        <h2 className="text-heading-40 mt-6 max-w-[30rem]">Measured, not projected</h2>
        <dl className="mt-14 grid gap-x-10 border-t border-[var(--color-stroke-1)] sm:grid-cols-2 lg:mt-20 lg:grid-cols-3">
          {stats.map((stat, index) => (
            <Reveal
              key={stat.label}
              delay={(index % 3) * 120}
              className="flex flex-col border-b border-[var(--color-stroke-1)] py-8 lg:py-10"
            >
              <dt className="text-body-16 order-2 mt-3 max-w-[18rem] text-black/65">{stat.label}</dt>
              <dd className="font-heading order-1 text-[clamp(2.75rem,5vw,4rem)] leading-none font-light tracking-tight">
                {stat.value}
              </dd>
            </Reveal>
          ))}
        </dl>
        <div className="mt-12 grid gap-6 lg:grid-cols-[1fr_1.2fr] lg:items-end">
          <p className="text-body-16 max-w-[34rem] text-black/65">
            Screening benchmark, CoIR <span className="font-mono text-[0.9em]">AppsRetrieval</span>, full test split:
            3,765 queries over 8,765 documents, with the 22M-parameter fallback embedder. Recall@100 is the ceiling, and
            a stronger first-stage embedder is the lever.
          </p>
          <table className="text-body-14 w-full border-collapse">
            <thead>
              <tr className="text-mono-s text-left uppercase text-black/60">
                <th className="py-2 font-normal">Arm</th>
                <th className="py-2 text-right font-normal">NDCG@10</th>
                <th className="py-2 text-right font-normal">MRR@10</th>
                <th className="py-2 text-right font-normal">Recall@100</th>
              </tr>
            </thead>
            <tbody className="font-mono">
              {[
                ["BM25 only", "0.91", "—", "—"],
                ["Dense only", "7.59", "6.39", "27.22"],
                ["Hybrid RRF", "7.78", "6.60", "27.17"],
              ].map((row, index) => (
                <tr
                  key={row[0]}
                  className={`border-t border-[var(--color-stroke-1)] ${index === 2 ? "font-semibold" : ""}`}
                >
                  <td className="py-2.5 font-sans">{row[0]}</td>
                  {row.slice(1).map((cell, cellIndex) => (
                    <td key={cellIndex} className="py-2.5 text-right">
                      {cell}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

const DIFF = `--- dispatch-payload-8.js@v1.0.0
+++ dispatch-payload-8.js@v2.0.0
@@ -1,5 +1,7 @@
 export async function dispatchPayload8(input, session = {}) {
+  if (input === null || input === undefined) return null;
   const normalized = String(input || '').trim();
+  session.span = session.span || 'dispatchPayload8';
   const resolveUtterance4Result = await resolveUtterance4(normalized, session);`;

function Versions() {
  return (
    <section id="versions" className="relative overflow-clip bg-white pt-12 pb-20 text-black lg:pt-16 lg:pb-28">
      <div className="container-page">
        <div className="mx-auto max-w-[40rem] text-center">
          <Eyebrow className="justify-center opacity-70">Versions</Eyebrow>
          <h2 className="text-heading-40 mt-6">Every version, one index</h2>
          <p className="text-body-18 mt-5 text-black/70">
            Chunks are content-addressed, so a new commit re-embeds only what changed. Across versions, near-identical
            snippets collapse into one family with its diffs, instead of crowding the results with copies.
          </p>
        </div>
        <Reveal>
          <Card className="border-stroke-1 mx-auto mt-14 max-w-[56rem] overflow-hidden bg-[#fdfaf4] [--frame-dot:var(--color-stroke-2)]">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[var(--color-stroke-1)] px-5 py-3.5">
              <span className="text-mono-m">
                dispatchPayload8 <span className="text-black/55">· src/tools/adapters</span>
              </span>
              <span className="flex items-center gap-2">
                {["v1.0.0", "v2.0.0", "v3.0.0"].map((version, index) => (
                  <span key={version} className="text-mono-s flex items-center gap-2 text-black/65">
                    {index > 0 && <span className="bg-stroke-2 h-px w-5" />}
                    <span className={`size-1.5 rounded-full ${index === 1 ? "bg-sun" : "bg-stroke-2"}`} />
                    {version}
                  </span>
                ))}
              </span>
            </div>
            <pre className="text-mono-m overflow-x-auto p-5 leading-6">
              {DIFF.split("\n").map((line, index) => (
                <div
                  key={index}
                  className={
                    line.startsWith("+") && !line.startsWith("+++")
                      ? "bg-[#e8f3ea] text-[#1f5a2c]"
                      : line.startsWith("@@") || line.startsWith("---") || line.startsWith("+++")
                        ? "text-black/55"
                        : "text-black/80"
                  }
                >
                  {line || " "}
                </div>
              ))}
            </pre>
          </Card>
        </Reveal>
      </div>
    </section>
  );
}

function Laptop() {
  const features = [
    ["Runs on a CPU", "INT8 ONNX models, FAISS and BM25. No GPU, and a warm query answers in about a second."],
    [
      "Degrades, never fails",
      "No models, no faiss, no tree-sitter: still ranked results, and it names the rung that loaded.",
    ],
    ["The LLM never reads code", "A local model may classify and rewrite queries. Your source never enters a prompt."],
    ["Local by default", "The API and the UI bind to 127.0.0.1. Nothing is sent to a third-party service."],
  ];
  return (
    <section className="relative overflow-clip bg-white pb-24 text-black lg:pb-40">
      <div className="container-page">
        <div className="grid gap-12 border-t border-[var(--color-stroke-1)] pt-16 lg:grid-cols-[22rem_1fr] lg:gap-20 lg:pt-24">
          <div>
            <p className="text-body-16 text-black/65">Built for the evaluator’s machine</p>
            <h2 className="text-heading-32 mt-3">Ships as a laptop tool, not a cluster</h2>
            <div className="mt-8">
              <Button href={`${REPO_URL}#quickstart`} external variant="dark">
                Five-minute quickstart
              </Button>
            </div>
          </div>
          <ul className="grid gap-x-12 gap-y-10 sm:grid-cols-2">
            {features.map(([title, body], index) => (
              <Reveal as="li" key={title} delay={(index % 2) * 120}>
                <h3 className="text-body-20 flex items-center gap-2.5">
                  <span className="bg-sun size-1.5 rounded-full" aria-hidden="true" />
                  {title}
                </h3>
                <p className="text-body-16 mt-2 max-w-[22rem] text-black/65">{body}</p>
              </Reveal>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}
