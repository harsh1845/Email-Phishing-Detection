export { unwrapUrl } from './unwrap.js';
export { parseUrl, coerceUrl, isDangerousScheme, isIpAddress } from './parse.js';
export { getEtldPlusOne, splitHost } from './psl.js';
export { extractFeatureMap, extractFeatureVector, FEATURE_NAMES } from './features.js';
export { scoreUrl, scoreEmailLinks, worstVerdict } from './score.js';
export { runHeuristics } from './heuristics.js';
export { detectLookalike, normalizeHomoglyphs, damerauLevenshtein } from './lookalike.js';
export { buildDefaultAllowlist, DEFAULT_ALLOWLIST } from './data/allowlist.js';
export { BRANDS, BRAND_OFFICIAL_DOMAINS } from './data/brands.js';
export { SHORTENERS, isShortener } from './data/shorteners.js';
export { BloomFilter } from './bloom.js';
export { applyEmailContext } from './context.js';
export { DEFAULT_THRESHOLDS } from './types.js';
export type {
  EmailContext,
  FeatureMap,
  LinkVerdict,
  ParsedUrl,
  ScoreOptions,
  ScoreThresholds,
  VerdictLevel,
  VerdictSource,
} from './types.js';
export type { BloomFilterJSON } from './bloom.js';
