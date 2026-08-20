import { coerceUrl } from './parse.js';

const MAX_UNWRAP = 4;

interface WrapperRule {
  hostSuffix: string;
  param: string;
}

const WRAPPERS: WrapperRule[] = [
  { hostSuffix: 'google.com', param: 'q' },
  { hostSuffix: 'google.com', param: 'url' },
  { hostSuffix: 'googleusercontent.com', param: 'q' },
  { hostSuffix: 'safelinks.protection.outlook.com', param: 'url' },
  { hostSuffix: 'protection.outlook.com', param: 'url' },
  { hostSuffix: 'l.facebook.com', param: 'u' },
  { hostSuffix: 'lm.facebook.com', param: 'u' },
  { hostSuffix: 'l.instagram.com', param: 'u' },
  { hostSuffix: 'linkedin.com', param: 'url' },
  { hostSuffix: 'youtube.com', param: 'q' },
  { hostSuffix: 'mandrillapp.com', param: 'p' },
];

function hostMatches(hostname: string, suffix: string): boolean {
  return hostname === suffix || hostname.endsWith(`.${suffix}`);
}

function unwrapOnce(raw: string): string | null {
  const url = coerceUrl(raw);
  if (!url) return null;

  const hostname = url.hostname.toLowerCase();
  for (const rule of WRAPPERS) {
    if (!hostMatches(hostname, rule.hostSuffix)) continue;
    const value = url.searchParams.get(rule.param);
    if (!value) continue;
    try {
      return decodeURIComponent(value);
    } catch {
      return value;
    }
  }

  if (hostname === 'www.google.com' && url.pathname === '/url') {
    const q = url.searchParams.get('q') || url.searchParams.get('url');
    if (q) {
      try {
        return decodeURIComponent(q);
      } catch {
        return q;
      }
    }
  }

  return null;
}

/** Unwrap Gmail / Outlook / social redirectors to the real destination. */
export function unwrapUrl(raw: string): string {
  let current = raw.trim();
  for (let i = 0; i < MAX_UNWRAP; i += 1) {
    const next = unwrapOnce(current);
    if (!next || next === current) break;
    current = next;
  }
  return current;
}
