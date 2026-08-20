import { CREDENTIAL_KEYWORDS } from './data/brands.js';
import { FEATURE_NAMES } from './data/feature-names.js';
import { isShortener } from './data/shorteners.js';
import { isIpAddress } from './psl.js';
import { parseUrl } from './parse.js';
import {
  brandInHostnameNotEtld,
  brandInPath,
  brandInSubdomain,
  shortestBrandDistance,
} from './lookalike.js';
import type { FeatureMap } from './types.js';

export { FEATURE_NAMES };

function shannonEntropy(value: string): number {
  if (!value.length) return 0;
  const freq = new Map<string, number>();
  for (const ch of value) {
    freq.set(ch, (freq.get(ch) ?? 0) + 1);
  }
  let entropy = 0;
  const n = value.length;
  for (const count of freq.values()) {
    const p = count / n;
    entropy -= p * Math.log2(p);
  }
  return entropy;
}

function countMatches(haystack: string, re: RegExp): number {
  const m = haystack.match(re);
  return m ? m.length : 0;
}

function longestDigitRun(value: string): number {
  let best = 0;
  let cur = 0;
  for (const ch of value) {
    if (ch >= '0' && ch <= '9') {
      cur += 1;
      if (cur > best) best = cur;
    } else {
      cur = 0;
    }
  }
  return best;
}

function sensitiveKeywordCount(url: string): number {
  const lower = url.toLowerCase();
  let count = 0;
  for (const kw of CREDENTIAL_KEYWORDS) {
    if (lower.includes(kw)) count += 1;
  }
  return count;
}

export function extractFeatureMap(rawUrl: string): FeatureMap {
  const parsed = parseUrl(rawUrl);
  const url = parsed?.href ?? rawUrl;
  const hostname = parsed?.hostname ?? '';
  const path = parsed?.pathname ?? '';
  const query = parsed?.search ?? '';
  const fragment = parsed?.hash ?? '';
  const letters = countMatches(url, /[a-zA-Z]/g);
  const digits = countMatches(url, /\d/g);
  const vowels = countMatches(hostname, /[aeiou]/gi);

  const map: FeatureMap = {
    url_length: url.length,
    hostname_length: hostname.length,
    path_length: path.length,
    query_length: query.length,
    fragment_length: fragment.length,
    num_dots: countMatches(url, /\./g),
    num_hyphens: countMatches(url, /-/g),
    num_underscores: countMatches(url, /_/g),
    num_slashes: countMatches(url, /\//g),
    num_question_marks: countMatches(url, /\?/g),
    num_equals: countMatches(url, /=/g),
    num_ampersands: countMatches(url, /&/g),
    num_percent: countMatches(url, /%/g),
    num_digits: digits,
    digit_ratio: url.length ? digits / url.length : 0,
    num_letters: letters,
    letter_ratio: url.length ? letters / url.length : 0,
    has_https: parsed?.protocol === 'https:' ? 1 : 0,
    has_http: parsed?.protocol === 'http:' ? 1 : 0,
    has_ip: hostname && isIpAddress(hostname) ? 1 : 0,
    has_at: url.includes('@') ? 1 : 0,
    has_double_slash_redirect: url.indexOf('//', 8) !== -1 ? 1 : 0,
    has_punycode: hostname.includes('xn--') ? 1 : 0,
    has_port: parsed?.port ? 1 : 0,
    subdomain_count: parsed?.subdomain
      ? parsed.subdomain.split('.').filter((l) => l && l !== 'www').length
      : 0,
    tld_length: parsed?.tld.length ?? 0,
    registered_domain_length: parsed?.registeredName.length ?? 0,
    path_depth: path.split('/').filter(Boolean).length,
    num_query_params: query
      ? query
          .replace(/^\?/, '')
          .split('&')
          .filter(Boolean).length
      : 0,
    sensitive_keyword_count: sensitiveKeywordCount(url),
    brand_in_subdomain: parsed && brandInSubdomain(parsed) ? 1 : 0,
    brand_in_path: parsed && brandInPath(path, parsed.etld1) ? 1 : 0,
    brand_in_hostname_not_etld: parsed && brandInHostnameNotEtld(parsed) ? 1 : 0,
    shortest_brand_distance: parsed ? Math.min(10, shortestBrandDistance(parsed.registeredName)) : 10,
    is_shortener: parsed && isShortener(parsed.etld1) ? 1 : 0,
    hostname_entropy: shannonEntropy(hostname),
    url_entropy: shannonEntropy(url),
    has_hex_escape: /%[0-9a-fA-F]{2}/.test(url) ? 1 : 0,
    vowel_ratio_hostname: hostname.length ? vowels / hostname.length : 0,
    consecutive_digits: longestDigitRun(url),
    max_label_length: parsed ? Math.max(0, ...parsed.labels.map((l) => l.length)) : 0,
    num_labels: parsed?.labels.length ?? 0,
    has_www: hostname === 'www' || hostname.startsWith('www.') ? 1 : 0,
    has_credential_in_userinfo: parsed?.username ? 1 : 0,
  };

  return map;
}

export function extractFeatureVector(rawUrl: string): Float32Array {
  const map = extractFeatureMap(rawUrl);
  const vec = new Float32Array(FEATURE_NAMES.length);
  for (let i = 0; i < FEATURE_NAMES.length; i += 1) {
    vec[i] = map[FEATURE_NAMES[i]] ?? 0;
  }
  return vec;
}

export function featureVectorToObject(vec: Float32Array): FeatureMap {
  const map: FeatureMap = {};
  for (let i = 0; i < FEATURE_NAMES.length; i += 1) {
    map[FEATURE_NAMES[i]] = vec[i] ?? 0;
  }
  return map;
}
