# Axiom demo video: script and shot list

About 3 minutes, roughly 400 words of narration at a calm 140–150 words per minute. The screen
recording (`submission/axiom_demo.mp4`) follows this shot list scene by scene; the timestamps are the
recording's. Paste `submission/voiceover.txt` into ElevenLabs; it is the same narration with pause
tags. Every number spoken below is measured (see README "Results").

| Time | Screen | Narration |
|---|---|---|
| 0:00–0:18 | Landing page hero; the console preview types a question and ranks three results | Engineers rarely ask an AI to write new code. They ask where something already happens. In a voice-assistant codebase with thousands of JavaScript files, that question has no easy answer. This is Axiom: agentic code intelligence, built by Team Incognito for Samsung PRISM. |
| 0:18–0:38 | Scroll to "The problem" and its three cards | A ten-thousand-file repository is tens of millions of tokens, and no language model can hold it. Keyword search misses meaning, embeddings miss the order of calls, and every commit makes an index go stale. So Axiom doesn't ask a model to read your code. It retrieves it. |
| 0:38–1:06 | "Axiom makes a codebase answerable" and the five pipeline steps | Every question is classified first: semantic, structural, usage or hybrid. Then three signals run in parallel. Dense embeddings for meaning, BM25 for exact identifiers, and a tree-sitter call graph that remembers the order functions are called in. The rankings are fused with weights chosen for the query type, rescored by a cross-encoder, and checked by a bounded agent that can rewrite the query and search again: at most twice, within five seconds. |
| 1:06–1:22 | The three query types, then "Measured, not projected" | Three questions from the problem statement, three kinds of evidence. And every number here is measured. When fifty files change, only those fifty chunks are re-embedded. Renamed code costs nothing. Six hundred and fifty-nine tests pass on a fresh clone. |
| 1:22–1:32 | "Every version, one index" and its diff | Codebases change, so Axiom indexes every version by content. The same function across releases becomes one family, with its diff. |
| 1:32–2:10 | Console: type "Which files call resolveUtterance4 before normalizeTool1?", results appear, scroll through them | Here is the console, running on a laptop CPU. Let's ask a structural question: which files call resolveUtterance4 before normalizeTool1? The classifier routes it to the call graph, which now carries sixty percent of the weight. Every result shows its exact file and lines, the rank each signal gave it, and why it ranked. Look at dispatchPayload8: it calls resolveUtterance4 at position two, before normalizeTool1 at position three. That order is the answer, and no embedding can see it. |
| 2:10–2:26 | Click the Usage example; weights shift to BM25 | A usage question is different: an exact string. Now BM25 leads the fusion, and the files that contain that deeplink come straight to the top. |
| 2:26–2:42 | Open a result's detail page: source, "Calls, in source order", provenance | Open any result for the full source, its calls in the order they happen, and exactly where it came from: commit and content hash. |
| 2:42–2:56 | Snippet families tab, a family's diff between versions | And across versions, the families view shows how each function evolved: one entry instead of three near-duplicates. |
| 2:56–3:10 | Back to the landing page footer: "Ask your codebase where it already happens" | Axiom. Retrieval you can verify, on hardware you already have. It's open source, so try it on GitHub. Thank you. |

## Notes for recording the voice

- Suggested ElevenLabs settings: a calm, clear narration voice; stability around 0.5, similarity
  around 0.75, style low. Pronounce "Axiom" as AK-see-um.
- Say identifiers as words: "resolve utterance four", "normalize tool one", "dispatch payload eight".
  `voiceover.txt` already spells them that way.
- If the generated audio runs long, trim silence rather than speeding up the voice; each scene in
  the recording has a few seconds of slack.
