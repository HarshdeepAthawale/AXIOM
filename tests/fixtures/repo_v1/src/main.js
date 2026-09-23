const { dispatch } = require('./tools/registry.js');
const { preprocessInput } = require('./utils/normalize.js');

/**
 * Process one inbound request end to end.
 */
function main(request) {
  const prepared = preprocessInput(request.body);
  const outcome = dispatch(prepared.text);
  return { ok: outcome !== null, outcome: outcome };
}

module.exports = { main };
