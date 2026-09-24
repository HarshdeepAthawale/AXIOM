const { main } = require('../main.js');

/**
 * Added in v2: emit one counter per processed request.
 */
function record(request) {
  const outcome = main(request);
  return { ok: outcome.ok, at: 0 };
}

module.exports = { record };
