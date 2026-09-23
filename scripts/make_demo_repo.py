"""Generate a multi-version JavaScript demo corpus.

`OQ-07` asks which repository anchors the P1 and Bonus demos. A real tagged
open-source repo is the honest answer for the *live* demo, but three separate
requirements cannot be met by any repo we do not control:

* `NFR-02` is measured on a **50-changed-file diff**, which needs a corpus where
  we choose exactly how many files move between two versions.
* `FR-21`'s `SnippetFamily` grouping needs the same symbol to survive several
  versions under *light* edits -- edits small enough to stay above the
  `AXIOM_DEDUPE_COSINE` threshold. Found repos rarely oblige.
* `FR-19`'s zero-cost-rename claim needs a version where a file's content is
  provably byte-identical under a new path.

So this generator produces a corpus shaped like the problem statement's
voice-assistant codebase -- intent routers, tool adapters, deeplink handlers,
preprocessing utilities, real call chains across modules, ESM and CommonJS
mixed -- across a tagged git history built to exercise exactly those paths.

It is deterministic: the same `--seed` yields a byte-identical repository, so a
latency or reindex number taken from it is reproducible from the command alone
(`NFR-08`, `NFR-09`).

This is a *test and benchmark fixture*, and the documentation must say so. It is
not evidence that Axiom works on real-world JavaScript; that claim belongs to
the live demo on a real repo.

Usage::

    python scripts/make_demo_repo.py /tmp/axiom_demo --files 60 --versions 3
    axiom index /tmp/axiom_demo --version-id v1.0.0
    axiom reindex --to v2.0.0          # the NFR-02 measurement
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

#: Module families, each a directory with its own naming and call conventions.
AREAS = ("intent", "preprocess", "tools/adapters", "deeplink", "telemetry", "util")

#: Deeplink targets. 'bluetooth-settings' is the problem statement's own Q3
#: example and must appear as a verbatim string literal -- it is the
#: highest-precision lexical signal in the corpus and what BM25 wins on.
DEEPLINKS = (
    "bluetooth-settings",
    "wifi-settings",
    "display-settings",
    "sound-settings",
    "battery-settings",
)

VERBS = (
    "resolve",
    "handle",
    "dispatch",
    "normalize",
    "sanitize",
    "transform",
    "validate",
    "parse",
    "build",
    "emit",
    "collect",
    "route",
)
NOUNS = (
    "Intent",
    "Utterance",
    "Slot",
    "Tool",
    "Deeplink",
    "Session",
    "Payload",
    "Event",
    "Command",
    "Context",
)


@dataclass(frozen=True)
class Symbol:
    """One generated function and where it lives."""

    name: str
    path: str
    area: str
    calls: tuple[str, ...]
    deeplink: str | None
    commonjs: bool


def _plan(file_count: int, rng: random.Random) -> list[Symbol]:
    """Lay out symbols across files, then wire a call graph over them.

    Call edges point from later files to earlier ones, which keeps the graph
    acyclic and gives ordered-pair queries (`FR-10`, archetype Q2) a real
    answer rather than a cycle to trip over.
    """
    symbols: list[Symbol] = []
    for index in range(file_count):
        area = AREAS[index % len(AREAS)]
        verb, noun = rng.choice(VERBS), rng.choice(NOUNS)
        name = f"{verb}{noun}{index}"
        stem = f"{verb}-{noun.lower()}-{index}"
        # Callees are drawn from already-placed symbols, so every edge resolves.
        pool = [s.name for s in symbols]
        count = min(len(pool), rng.choice((0, 1, 2, 2, 3)))
        calls = tuple(rng.sample(pool, count)) if count else ()
        symbols.append(
            Symbol(
                name=name,
                path=f"src/{area}/{stem}.js",
                area=area,
                calls=calls,
                # A fifth of the corpus carries a deeplink literal.
                deeplink=rng.choice(DEEPLINKS) if index % 5 == 0 else None,
                # A quarter uses CommonJS, so the chunker's is_exported handling
                # is exercised on both module systems (FR-04).
                commonjs=index % 4 == 3,
            )
        )
    return symbols


def _render(symbol: Symbol, symbols: list[Symbol], revision: int) -> str:
    """Emit one JavaScript module.

    ``revision`` drives the *light* edits between versions: a new guard clause
    and a changed comment. The body is otherwise stable, which is what keeps a
    symbol's successive versions similar enough to group into one
    `SnippetFamily` instead of reading as unrelated code.
    """
    by_name = {s.name: s for s in symbols}
    lines: list[str] = []

    imports = sorted({by_name[c].path for c in symbol.calls})
    for path in imports:
        callees = sorted(c for c in symbol.calls if by_name[c].path == path)
        target = by_name[callees[0]]
        rel = _relative_specifier(symbol.path, path)
        if target.commonjs:
            lines.append(f"const {{ {', '.join(callees)} }} = require('{rel}');")
        else:
            lines.append(f"import {{ {', '.join(callees)} }} from '{rel}';")
    if imports:
        lines.append("")

    lines.append("/**")
    lines.append(f" * {symbol.area} stage: {symbol.name}.")
    if revision > 1:
        lines.append(f" * Revised in v{revision}.0.0 to guard against empty input.")
    if revision > 2:
        lines.append(f" * v{revision}.0.0 also records a telemetry span.")
    if symbol.deeplink:
        lines.append(f" * Routes to the {symbol.deeplink} deeplink.")
    lines.append(" */")

    signature = f"async function {symbol.name}(input, session = {{}})"
    lines.append(f"{signature} {{" if symbol.commonjs else f"export {signature} {{")

    if revision > 1:
        lines.append("  if (input === null || input === undefined) return null;")
    lines.append("  const normalized = String(input || '').trim();")
    if revision > 2:
        # A second, still-small edit so v3 differs from v2. Each successive
        # revision must change the text or SnippetFamily has nothing to diff --
        # but the change stays light enough to stay above the dedupe cosine.
        lines.append(f"  session.span = session.span || '{symbol.name}';")

    for callee in symbol.calls:
        lines.append(f"  const {callee}Result = await {callee}(normalized, session);")
    if symbol.deeplink:
        lines.append(f"  const target = 'app://settings/{symbol.deeplink}';")
        lines.append("  session.lastDeeplink = target;")

    returned = (
        f"{{ ok: true, target, via: '{symbol.name}' }}"
        if symbol.deeplink
        else f"{{ ok: true, via: '{symbol.name}' }}"
    )
    lines.append(f"  return {returned};")
    lines.append("}")

    if symbol.commonjs:
        lines.append("")
        lines.append(f"module.exports = {{ {symbol.name} }};")
    lines.append("")
    return "\n".join(lines)


def _relative_specifier(from_path: str, to_path: str) -> str:
    """Build the './x' / '../x' specifier linking two repo-relative paths."""
    source, target = Path(from_path).parent, Path(to_path)
    up = source.parts
    down = target.parts
    common = 0
    while common < min(len(up), len(down)) and up[common] == down[common]:
        common += 1
    hops = [".."] * (len(up) - common)
    rest = list(down[common:])
    spec = "/".join(hops + rest) if hops else "./" + "/".join(rest)
    return spec


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
            "GIT_AUTHOR_NAME": "Axiom Fixture",
            "GIT_AUTHOR_EMAIL": "fixture@axiom.invalid",
            "GIT_COMMITTER_NAME": "Axiom Fixture",
            "GIT_COMMITTER_EMAIL": "fixture@axiom.invalid",
            # Fixed timestamps keep the whole repo reproducible (NFR-08).
            "GIT_AUTHOR_DATE": "2026-09-01T00:00:00Z",
            "GIT_COMMITTER_DATE": "2026-09-01T00:00:00Z",
        },
    )


def build(
    root: Path, file_count: int, versions: int, changed: int, seed: int
) -> list[tuple[str, int, int]]:
    """Materialise the repository. Returns (tag, files_written, files_changed)."""
    rng = random.Random(seed)
    symbols = _plan(file_count, rng)
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q", "-b", "main")

    summary: list[tuple[str, int, int]] = []
    for revision in range(1, versions + 1):
        # Exactly `changed` files move between versions, so NFR-02's "50 changed
        # files" is a parameter of the measurement rather than an accident of
        # whatever corpus we happened to find.
        touched = symbols if revision == 1 else symbols[:changed]
        for symbol in touched:
            path = root / symbol.path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_render(symbol, symbols, revision), encoding="utf-8")

        if revision == 2 and len(symbols) > changed:
            # A pure rename with byte-identical content: FR-19 claims this costs
            # zero embedding forward passes, and TC-075 asserts it.
            victim = symbols[-1]
            old = root / victim.path
            new = old.with_name(f"renamed-{old.name}")
            if old.exists():
                _git(root, "mv", str(old.relative_to(root)), str(new.relative_to(root)))

        tag = f"v{revision}.0.0"
        _git(root, "add", "-A")
        _git(root, "commit", "-q", "-m", f"{tag}: generated revision {revision}")
        _git(root, "tag", tag)
        summary.append((tag, len(symbols), len(touched)))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory to create.")
    parser.add_argument("--files", type=int, default=60, help="Source files per version.")
    parser.add_argument("--versions", type=int, default=3, help="Tagged versions to build.")
    parser.add_argument(
        "--changed",
        type=int,
        default=50,
        help="Files modified between consecutive versions. Default 50 is NFR-02's figure.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing root.")
    args = parser.parse_args(argv)

    if args.root.exists():
        if not args.force:
            print(f"error: {args.root} exists; pass --force to overwrite", file=sys.stderr)
            return 2
        import shutil

        shutil.rmtree(args.root)

    if args.changed > args.files:
        print("error: --changed cannot exceed --files", file=sys.stderr)
        return 2

    summary = build(args.root, args.files, args.versions, args.changed, args.seed)
    print(f"generated {args.root}")
    for tag, total, touched in summary:
        print(f"  {tag:10} {total:4} files, {touched:4} written")
    print()
    print("next:")
    print(f"  axiom index {args.root} --version-id {summary[0][0]}")
    if len(summary) > 1:
        print(f"  axiom reindex --to {summary[1][0]}     # NFR-02: {args.changed} changed files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
