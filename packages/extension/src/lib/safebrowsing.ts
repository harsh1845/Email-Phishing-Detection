/** Google Safe Browsing v5 hash-prefix lookup. Sends 4-byte SHA-256 prefixes, not raw URLs. */

function canonicalize(url: string): string {
  try {
    const u = new URL(url);
    u.hash = '';
    let host = u.hostname.toLowerCase();
    if (host.startsWith('www.')) host = host.slice(4);
    let path = u.pathname || '/';
    path = path.replace(/\/+/g, '/');
    return `${u.protocol}//${host}${path}${u.search}`;
  } catch {
    return url.toLowerCase();
  }
}

function toBase64Url(bytes: Uint8Array): string {
  let bin = '';
  for (const b of bytes) bin += String.fromCharCode(b);
  return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/g, '');
}

async function sha256Prefix(value: string): Promise<string> {
  const data = new TextEncoder().encode(value);
  const digest = await crypto.subtle.digest('SHA-256', data);
  return toBase64Url(new Uint8Array(digest).slice(0, 4));
}

export async function checkSafeBrowsing(url: string, apiKey: string): Promise<boolean> {
  if (!apiKey) return false;
  const canonical = canonicalize(url);
  const prefix = await sha256Prefix(canonical);
  const endpoint = `https://safebrowsing.googleapis.com/v5/hashes:search?key=${encodeURIComponent(apiKey)}&hashPrefixes=${prefix}`;
  try {
    const res = await fetch(endpoint);
    if (!res.ok) return false;
    const body = (await res.json()) as { fullHashes?: unknown[] };
    return Array.isArray(body.fullHashes) && body.fullHashes.length > 0;
  } catch {
    return false;
  }
}
