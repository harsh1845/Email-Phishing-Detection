import { CREDENTIAL_KEYWORDS } from './data/brands.js';
import { isShortener } from './data/shorteners.js';
import { isDangerousScheme, isIpAddress } from './parse.js';
import { detectLookalike } from './lookalike.js';
import type { LinkVerdict, ParsedUrl, VerdictLevel } from './types.js';

export interface HeuristicResult {
  level: VerdictLevel;
  score: number;
  reasons: string[];
}

function emptyResult(): HeuristicResult {
  return { level: 'safe', score: 0.05, reasons: [] };
}

export function runHeuristics(raw: string, parsed: ParsedUrl | null): HeuristicResult {
  const result = emptyResult();

  if (isDangerousScheme(raw)) {
    return {
      level: 'danger',
      score: 0.99,
      reasons: ['Link uses a dangerous scheme (javascript/data/blob) instead of http(s)'],
    };
  }

  if (!parsed) {
    return {
      level: 'suspicious',
      score: 0.55,
      reasons: ['URL could not be parsed'],
    };
  }

  if (
    parsed.username ||
    (raw.includes('@') &&
      parsed.hostname &&
      raw.indexOf('@') < raw.indexOf(parsed.hostname))
  ) {
    result.level = 'danger';
    result.score = Math.max(result.score, 0.92);
    result.reasons.push('URL contains credentials or an @ before the host (common redirect trick)');
  }

  if (isIpAddress(parsed.hostname)) {
    const lower = raw.toLowerCase();
    const hasCredKw = CREDENTIAL_KEYWORDS.some((kw) => lower.includes(kw));
    if (hasCredKw) {
      result.level = 'danger';
      result.score = Math.max(result.score, 0.95);
      result.reasons.push('IP-address host combined with a login/account keyword');
    } else {
      result.level = result.level === 'danger' ? 'danger' : 'suspicious';
      result.score = Math.max(result.score, 0.7);
      result.reasons.push('Host is a raw IP address instead of a domain name');
    }
  }

  if (parsed.hostname.includes('xn--')) {
    const lookalike = detectLookalike(parsed);
    if (lookalike) {
      result.level = 'danger';
      result.score = Math.max(result.score, 0.97);
      result.reasons.push(
        `Internationalized domain looks like ${lookalike.brand.name} (${lookalike.official}) but is ${parsed.etld1}`,
      );
    } else {
      result.level = result.level === 'danger' ? 'danger' : 'suspicious';
      result.score = Math.max(result.score, 0.6);
      result.reasons.push('Domain uses punycode (xn--), which can hide lookalike characters');
    }
  }

  const lookalike = detectLookalike(parsed);
  if (lookalike) {
    const msg =
      lookalike.kind === 'subdomain-brand'
        ? `Brand name “${lookalike.brand.name}” appears in a subdomain of ${parsed.etld1}`
        : `Looks like ${lookalike.brand.name} (${lookalike.official}) but the real domain is ${parsed.etld1}`;
    result.level = 'danger';
    result.score = Math.max(result.score, lookalike.kind === 'combosquat' ? 0.9 : 0.96);
    result.reasons.push(msg);
  }

  if (isShortener(parsed.etld1)) {
    if (result.level === 'safe') result.level = 'suspicious';
    result.score = Math.max(result.score, 0.5);
    result.reasons.push(`URL shortener (${parsed.etld1}) hides the real destination`);
  }

  if (parsed.protocol === 'http:' && CREDENTIAL_KEYWORDS.some((kw) => raw.toLowerCase().includes(kw))) {
    if (result.level === 'safe') result.level = 'suspicious';
    result.score = Math.max(result.score, 0.55);
    result.reasons.push('Insecure http:// link with a login or account keyword');
  }

  return result;
}

export function verdictShell(
  url: string,
  finalUrl: string,
  parsed: ParsedUrl | null,
): Pick<LinkVerdict, 'url' | 'finalUrl' | 'hostname' | 'etld1'> {
  return {
    url,
    finalUrl,
    hostname: parsed?.hostname ?? '',
    etld1: parsed?.etld1 ?? '',
  };
}
