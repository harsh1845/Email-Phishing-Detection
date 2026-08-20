import { isIpAddress, splitHost } from './psl.js';
import type { ParsedUrl } from './types.js';

const DANGEROUS_SCHEMES = new Set(['javascript:', 'data:', 'blob:', 'vbscript:']);

export function isDangerousScheme(raw: string): boolean {
  const lower = raw.trim().toLowerCase();
  for (const scheme of DANGEROUS_SCHEMES) {
    if (lower.startsWith(scheme)) return true;
  }
  return false;
}

export function coerceUrl(raw: string): URL | null {
  const trimmed = raw.trim();
  if (!trimmed || isDangerousScheme(trimmed)) return null;
  try {
    return new URL(trimmed);
  } catch {
    try {
      if (/^[\w.-]+\.[a-z]{2,}/i.test(trimmed) || trimmed.startsWith('www.')) {
        return new URL(`https://${trimmed}`);
      }
    } catch {
      return null;
    }
    return null;
  }
}

export function parseUrl(raw: string): ParsedUrl | null {
  const url = coerceUrl(raw);
  if (!url) return null;
  if (url.protocol !== 'http:' && url.protocol !== 'https:') return null;

  let hostname = url.hostname.toLowerCase();
  if (hostname.startsWith('[') && hostname.endsWith(']')) {
    hostname = hostname.slice(1, -1);
  }

  const parts = splitHost(hostname);
  return {
    href: url.href,
    protocol: url.protocol,
    hostname,
    host: url.host.toLowerCase(),
    port: url.port,
    pathname: url.pathname,
    search: url.search,
    hash: url.hash,
    username: url.username,
    etld1: parts.etld1,
    tld: parts.tld,
    registeredName: parts.registeredName,
    subdomain: parts.subdomain,
    labels: parts.labels,
  };
}

export { isIpAddress };
