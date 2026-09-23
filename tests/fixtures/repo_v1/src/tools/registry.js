const { preprocessInput } = require('../utils/normalize.js');

/**
 * Tool registry. Dispatch preprocesses the payload before it resolves a
 * tool, which is the ordered call pair that query archetype Q2 asks about:
 * "which files call preprocessInput before resolveTool?".
 */
function dispatch(rawPayload) {
  const prepared = preprocessInput(rawPayload);
  const tool = resolveTool(prepared.text);
  if (!tool) {
    return null;
  }
  return tool.invoke(prepared.text);
}

/**
 * Look a tool up by its normalised name.
 */
function resolveTool(name) {
  const table = loadTable();
  return table[name] || null;
}

/**
 * Build the lookup table lazily, with a nested closure per entry.
 */
function loadTable() {
  const entries = ['bluetooth-settings', 'wifi-settings'];
  const table = {};
  entries.forEach(function register(entry) {
    table[entry] = {
      invoke: function invoke(payload) {
        return { tool: entry, payload: payload };
      },
    };
  });
  return table;
}

/**
 * Drop the memoised table so the next dispatch rebuilds it.
 */
function clearCache() {
  return loadTable();
}

module.exports = { dispatch, resolveTool, loadTable, clearCache };
