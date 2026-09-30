// Pure formatting helpers shared by the console views. Kept free of React so
// they can be unit tested.

import type { SignalName } from "./api";

export const SIGNALS: SignalName[] = ["dense", "sparse", "structural"];

export type DiffLineKind = "header" | "added" | "removed" | "context";

/** Classify one line of a unified diff. File and hunk headers are never +/- lines. */
export function diffLineKind(line: string): DiffLineKind {
  if (line.startsWith("---") || line.startsWith("+++") || line.startsWith("@@")) return "header";
  if (line.startsWith("+")) return "added";
  if (line.startsWith("-")) return "removed";
  return "context";
}

/** Split source into display lines, tolerating Windows line endings. */
export function codeLines(text: string): string[] {
  return text.replace(/\r\n?/g, "\n").split("\n");
}

/** Each signal's share of the fusion weights, as whole percentages in SIGNALS order. */
export function weightShares(
  weights: Partial<Record<SignalName, number>>,
): Array<{ signal: SignalName; percent: number }> {
  const total = SIGNALS.reduce((sum, signal) => sum + Math.max(0, weights[signal] ?? 0), 0);
  return SIGNALS.map((signal) => ({
    signal,
    percent: total > 0 ? (Math.max(0, weights[signal] ?? 0) / total) * 100 : 0,
  }));
}

/**
 * Split a match_reason ("structural: calls X (ordinal 0) ...") into the dominant
 * signal and the explanation. The explanation itself may contain colons.
 */
export function splitReason(reason: string): { signal: string; detail: string } {
  const at = reason.indexOf(":");
  if (at < 0) return { signal: "", detail: reason.trim() };
  return { signal: reason.slice(0, at).trim(), detail: reason.slice(at + 1).trim() };
}

/** "src/a.js:5-10" for a chunk location. */
export function locationRef(location: { file_path: string; start_line: number; end_line: number }): string {
  return `${location.file_path}:${location.start_line}-${location.end_line}`;
}
