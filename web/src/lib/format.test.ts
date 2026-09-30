import { describe, expect, it } from "vitest";

import { codeLines, diffLineKind, locationRef, splitReason, weightShares } from "./format";

describe("diffLineKind", () => {
  it("treats file and hunk headers as headers, not as changes", () => {
    expect(diffLineKind("--- a.js@v1.0.0")).toBe("header");
    expect(diffLineKind("+++ a.js@v2.0.0")).toBe("header");
    expect(diffLineKind("@@ -1,5 +1,7 @@")).toBe("header");
  });

  it("classifies additions, removals and context", () => {
    expect(diffLineKind("+  if (input === null) return null;")).toBe("added");
    expect(diffLineKind("-  return old;")).toBe("removed");
    expect(diffLineKind("   const normalized = String(input);")).toBe("context");
    expect(diffLineKind("")).toBe("context");
  });
});

describe("codeLines", () => {
  it("splits on Windows, old Mac and Unix line endings alike", () => {
    expect(codeLines("a\r\nb\rc\nd")).toEqual(["a", "b", "c", "d"]);
  });
});

describe("weightShares", () => {
  it("normalises weights to percentages in a fixed signal order", () => {
    const shares = weightShares({ structural: 0.6, dense: 0.2, sparse: 0.2 });
    expect(shares.map((share) => share.signal)).toEqual(["dense", "sparse", "structural"]);
    expect(shares.map((share) => Math.round(share.percent))).toEqual([20, 20, 60]);
  });

  it("treats a missing signal as zero and never divides by zero", () => {
    expect(weightShares({ dense: 0.51, sparse: 0.49 }).map((share) => Math.round(share.percent))).toEqual([51, 49, 0]);
    expect(weightShares({}).every((share) => share.percent === 0)).toBe(true);
  });
});

describe("splitReason", () => {
  it("separates the dominant signal from an explanation that contains colons", () => {
    expect(splitReason("structural: calls a (ordinal 0) before b: see graph")).toEqual({
      signal: "structural",
      detail: "calls a (ordinal 0) before b: see graph",
    });
  });

  it("keeps a reason without a signal prefix intact", () => {
    expect(splitReason("no prefix here")).toEqual({ signal: "", detail: "no prefix here" });
  });
});

describe("locationRef", () => {
  it("formats file:start-end", () => {
    expect(locationRef({ file_path: "src/tools/registry.js", start_line: 8, end_line: 15 })).toBe(
      "src/tools/registry.js:8-15",
    );
  });
});
