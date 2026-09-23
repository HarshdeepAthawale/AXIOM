const { alpha } = require('./a.js');

function beta(value) {
  if (value <= 0) {
    return 0;
  }
  return alpha(value - 1);
}

module.exports = { beta };
