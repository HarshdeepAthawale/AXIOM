# Axiom web

The web frontend: a landing page (`/`) and the search console (`/search`, with a snippet-families tab).
Next.js 16, React 19 and Tailwind CSS 4. It talks to `axiom serve` through a same-origin proxy, so the
API needs no CORS and stays bound to localhost.

## Run it

```bash
# 1. The backend, from the repository root
axiom serve --profile demo --index-root <index dir>        # http://127.0.0.1:8000

# 2. The frontend
cd web
npm install
npm run dev                                                 # http://127.0.0.1:3000
```

`AXIOM_API_URL` points the proxy at a different backend (default `http://127.0.0.1:8000`). For a
production build: `npm run build && npm start`. Checks: `npm test` (Vitest), `npm run lint`,
`npm run format`.

## Layout

| Path | What |
|---|---|
| `src/app/page.tsx` | Landing page |
| `src/app/search/page.tsx` | Console route |
| `src/app/chunk/[id]/page.tsx` | Chunk detail: full source, calls in source order, imports, provenance |
| `src/components/console.tsx` | Search, result cards, fusion weights, plan, timings, families and diffs |
| `src/components/header.tsx` | Scroll-aware header with dropdown menus and a mobile sheet |
| `src/components/hero-preview.tsx` | The hero's animated console, replaying real query output |
| `src/components/reveal.tsx` | Scroll-triggered fade-in, off under reduced motion |
| `src/lib/format.ts` | Pure formatting helpers, unit tested in `format.test.ts` |
| `src/components/ui.tsx` | Footer, buttons, eyebrow labels, dunes, marquee, framed cards |
| `src/lib/api.ts` | Typed client for the REST API ([docs/API.md](../docs/API.md)) |
| `src/app/globals.css` | Design tokens, fluid type scale, corner-dot frame |
| `next.config.ts` | `/api/axiom/*` → `axiom serve` `/v1/*` proxy |

## Design

Dark "desert at night" palette (`void`, `night`, `midnight`, `dusk`) with cream surfaces for the
explanatory sections, one sun-orange accent, and the three retrieval signals colour-coded throughout:
dense in dawn blue, BM25 in sun orange, the call graph in oasis green. Headings are Newsreader at a
light weight, body text Inter, labels Geist Mono. The dunes are drawn in SVG.
