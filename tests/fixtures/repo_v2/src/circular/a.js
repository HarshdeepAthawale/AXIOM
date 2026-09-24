const { beta } = require('./b.js');

function alpha(value) {
  return beta(value) + 1;
}

module.exports = { alpha };
