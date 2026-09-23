/**
 * Input preprocessing utilities.
 *
 * Every request body passes through here before the agent layer sees it,
 * which is what makes this the answer to "how is the input preprocessed
 * before going to the main function?" (query archetype Q1).
 */

/**
 * Strip control characters and collapse whitespace in a raw payload.
 */
export function sanitizeInput(raw) {
  if (raw === null || raw === undefined) {
    return '';
  }
  const asText = String(raw);
  const stripped = asText.replace(/[\u0000-\u001f]/g, ' ');
  return stripped.replace(/\s+/g, ' ').trim();
}

/**
 * Lower-case and normalise unicode so token matching is stable.
 */
export function normalizeInput(raw) {
  const cleaned = sanitizeInput(raw);
  const folded = cleaned.normalize('NFKC');
  const deaccented = folded.replace(/\p{Diacritic}/gu, '');
  return deaccented.toLowerCase();
}

/**
 * The public preprocessing entry point used by the agent layer.
 */
export function preprocessInput(raw, options) {
  const normalized = normalizeInput(raw);
  const limit = options && options.limit ? options.limit : 4096;
  return {
    text: normalized.slice(0, limit),
    truncated: normalized.length > limit,
  };
}
