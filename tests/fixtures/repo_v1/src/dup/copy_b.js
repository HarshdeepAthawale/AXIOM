/**
 * Duplicated content at two paths. TC-016 and TC-031 rely on this body being
 * byte-identical in copy_a.js and copy_b.js, so content_hash collides while
 * chunk_id does not.
 */
export function duplicatedHelper(items) {
  const seen = new Set();
  const out = [];
  for (const item of items) {
    if (!seen.has(item)) {
      seen.add(item);
      out.push(item);
    }
  }
  return out;
}
