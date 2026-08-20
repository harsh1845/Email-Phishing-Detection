export type VerdictLevel = 'safe' | 'suspicious' | 'danger';

export interface EmailContext {
  senderEmail?: string;
  senderDisplayName?: string;
  subject?: string;
}

export interface LinkVerdict {
  url: string;
  finalUrl: string;
  hostname: string;
  etld1: string;
  level: VerdictLevel;
  score: number;
  reasons: string[];
  source: VerdictSource[];
}

export type VerdictSource =
  | 'allowlist'
  | 'blocklist'
  | 'heuristic'
  | 'model'
  | 'context'
  | 'safebrowsing';

export interface ScoreOptions {
  allowlist?: Set<string>;
  blocklist?: Set<string>;
  bloomHas?: (value: string) => boolean;
  modelPredict?: (features: Float32Array) => number;
  safeBrowsingHit?: boolean;
  thresholds?: ScoreThresholds;
}

export interface ScoreThresholds {
  danger: number;
  suspicious: number;
}

export const DEFAULT_THRESHOLDS: ScoreThresholds = {
  danger: 0.85,
  suspicious: 0.45,
};

export interface ParsedUrl {
  href: string;
  protocol: string;
  hostname: string;
  host: string;
  port: string;
  pathname: string;
  search: string;
  hash: string;
  username: string;
  etld1: string;
  tld: string;
  registeredName: string;
  subdomain: string;
  labels: string[];
}

export type FeatureMap = Record<string, number>;
