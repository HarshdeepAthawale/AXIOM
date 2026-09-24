/**
 * Nested closures. Not 200 deep -- deep enough that the walker must descend
 * several levels while staying inside settings.max_ast_depth.
 */
export function buildPipeline(seed) {
  return function level1(a) {
    return function level2(b) {
      return function level3(c) {
        return function level4(d) {
          return function level5(e) {
            return seed + a + b + c + d + e;
          };
        };
      };
    };
  };
}
