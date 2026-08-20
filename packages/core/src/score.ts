import { buildDefaultAllowlist } from './data/allowlist.js';
import { applyEmailContext } from './context.js';
import { extractFeatureVector } from './features.js';
import { runHeuristics, verdictShell } from './heuristics.js';
import { parseUrl } from './parse.js';
import { unwrapUrl } from './unwrap.js';
import {
  DEFAULT_THRESHOLDS,
  type EmailContext,
  type LinkVerdict,
  type ScoreOptions,
  type VerdictLevel,
  type VerdictSource,
} from './types.js';

const DEFAULT_ALLOW = buildDefaultAllowlist();

function combineLevel(score: number, current: VerdictLevel, thresholds = DEFAULT_THRESHOLDS): VerdictLevel {
  if (current === 'danger' || score >= thresholds.danger) return 'danger';
  if (score >= thresholds.suspicious) return 'suspicious';
  return 'safe';
}

export function scoreUrl(
  url: string,
  context?: EmailContext,
  options: ScoreOptions = {},
): LinkVerdict {
  const thresholds = options.thresholds ?? DEFAULT_THRESHOLDS;
  const finalUrl = unwrapUrl(url);
  const parsed = parseUrl(finalUrl);
  const sources: VerdictSource[] = [];
  const reasons: string[] = [];

  const shell = verdictShell(url, finalUrl, parsed);
  const allow = options.allowlist ?? DEFAULT_ALLOW;

  if (parsed && (allow.has(parsed.etld1) || allow.has(parsed.hostname))) {
    return {
      ...shell,
      level: 'safe',
      score: 0.02,
      reasons: [`${parsed.etld1} is on the trusted-domain allowlist`],
      source: ['allowlist'],
    };
  }

  const block = options.blocklist;
  const bloomHas = options.bloomHas;
  if (parsed) {
    const keys = [finalUrl.toLowerCase(), parsed.hostname, parsed.etld1];
    const blocked =
      keys.some((k) => block?.has(k)) || keys.some((k) => bloomHas?.(k));
    if (blocked) {
      return {
        ...shell,
        level: 'danger',
        score: 0.99,
        reasons: ['This link matches a known phishing / malware threat feed'],
        source: ['blocklist'],
      };
    }
  }

  const heur = runHeuristics(finalUrl, parsed);
  if (heur.reasons.length) {
    sources.push('heuristic');
    reasons.push(...heur.reasons);
  }
  let score = heur.score;
  let level = heur.level;

  if (options.modelPredict && parsed) {
    const features = extractFeatureVector(finalUrl);
    const modelScore = clamp01(options.modelPredict(features));
    sources.push('model');
    const blended = Math.max(score, modelScore * 0.9 + score * 0.1);
    if (modelScore >= thresholds.danger) {
      reasons.push(`On-device URL model scored this ${Math.round(modelScore * 100)}% likely phishing`);
    } else if (modelScore >= thresholds.suspicious) {
      reasons.push(`On-device URL model found suspicious URL structure (${Math.round(modelScore * 100)}%)`);
    }
    score = blended;
    level = combineLevel(score, level, thresholds);
  }

  if (parsed) {
    const ctx = applyEmailContext(parsed, context);
    if (ctx.delta > 0) {
      sources.push('context');
      reasons.push(...ctx.reasons);
      score = clamp01(score + ctx.delta);
      level = combineLevel(score, level, thresholds);
    }
  }

  if (options.safeBrowsingHit) {
    sources.push('safebrowsing');
    reasons.push('Google Safe Browsing lists this URL as a known threat');
    score = Math.max(score, 0.98);
    level = 'danger';
  }

  if (!reasons.length) {
    reasons.push('No high-risk URL patterns detected');
  }

  return {
    ...shell,
    level: combineLevel(score, level, thresholds),
    score: clamp01(score),
    reasons,
    source: sources.length ? sources : ['heuristic'],
  };
}

export function scoreEmailLinks(
  urls: string[],
  context?: EmailContext,
  options: ScoreOptions = {},
): LinkVerdict[] {
  const seen = new Set<string>();
  const out: LinkVerdict[] = [];
  for (const url of urls) {
    const key = unwrapUrl(url).toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(scoreUrl(url, context, options));
  }
  return out;
}

export function worstVerdict(verdicts: LinkVerdict[]): LinkVerdict | null {
  if (!verdicts.length) return null;
  const rank: Record<VerdictLevel, number> = { safe: 0, suspicious: 1, danger: 2 };
  return [...verdicts].sort((a, b) => rank[b.level] - rank[a.level] || b.score - a.score)[0];
}

function clamp01(n: number): number {
  if (Number.isNaN(n)) return 0;
  return Math.min(1, Math.max(0, n));
}
