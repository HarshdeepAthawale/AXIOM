# AI disclosure

Team Incognito used AI tools while building this submission. This file says which, and for what.

| Tool | Used for |
|---|---|
| Claude (Anthropic), through Claude Code | Writing and reviewing code across the backend (`src/axiom/`), tests, scripts and the web frontend (`web/`); debugging; writing and editing documentation, the README and the presentation; the deployment configuration; producing the demo video's screen recording and narration script |
| ElevenLabs | The demo video's voiceover, generated from `submission/voiceover.txt` |

## What the team owns

The problem framing, the architecture and its decisions (`docs/Decisions.md`), and the choice of what to
build, measure and claim are the team's. AI output was reviewed, run and tested before it was kept:
the suite (`pytest`, `npm test`) passes on a fresh clone, and every number in the README, deck and
video comes from a run recorded in this repository (`artifacts/experiments.csv`, `docs/BuildLog.md`).

## AI inside the product

Axiom itself can optionally use a small local LLM (Qwen2.5-1.5B, GGUF) to classify and rewrite
queries. It is off by default, never reads source code, and never calls a third-party API. The
embedding and reranking models are open models run locally on CPU: `all-MiniLM-L6-v2` and
`ms-marco-MiniLM-L-6-v2`.
