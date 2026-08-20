import { BRANDS, type Brand } from './data/brands.js';
import type { ParsedUrl } from './types.js';

const HOMOGLYPHS: Record<string, string> = {
  '0': 'o',
  '1': 'l',
  '3': 'e',
  '4': 'a',
  '5': 's',
  '7': 't',
  '8': 'b',
  '@': 'a',
  $: 's',
  а: 'a',
  е: 'e',
  о: 'o',
  р: 'p',
  с: 'c',
  у: 'y',
  х: 'x',
  і: 'i',
  ј: 'j',
  ѕ: 's',
  ԁ: 'd',
  ɡ: 'g',
  һ: 'h',
  ӏ: 'l',
  ո: 'n',
  ο: 'o',
  ρ: 'p',
  υ: 'u',
  ν: 'v',
  ω: 'w',
  ｚ: 'z',
  α: 'a',
  β: 'b',
  ε: 'e',
  κ: 'k',
  η: 'n',
  τ: 't',
};

export function normalizeHomoglyphs(input: string): string {
  let out = '';
  for (const ch of input.toLowerCase()) {
    out += HOMOGLYPHS[ch] ?? ch;
  }
  return out.replace(/rn/g, 'm');
}

export function damerauLevenshtein(a: string, b: string): number {
  if (a === b) return 0;
  const al = a.length;
  const bl = b.length;
  if (!al) return bl;
  if (!bl) return al;
  if (Math.abs(al - bl) > 2 && Math.min(al, bl) > 4) {
    return Math.abs(al - bl);
  }

  const inf = al + bl;
  const da: Record<string, number> = {};
  const h: number[][] = Array.from({ length: al + 2 }, () =>
    new Array<number>(bl + 2).fill(0),
  );
  h[0][0] = inf;
  for (let i = 0; i <= al; i += 1) {
    h[i + 1][0] = inf;
    h[i + 1][1] = i;
  }
  for (let j = 0; j <= bl; j += 1) {
    h[0][j + 1] = inf;
    h[1][j + 1] = j;
  }
  for (let i = 1; i <= al; i += 1) {
    let db = 0;
    for (let j = 1; j <= bl; j += 1) {
      const i1 = da[b[j - 1]] ?? 0;
      const j1 = db;
      let cost = 1;
      if (a[i - 1] === b[j - 1]) {
        cost = 0;
        db = j;
      }
      h[i + 1][j + 1] = Math.min(
        h[i][j] + cost,
        h[i + 1][j] + 1,
        h[i][j + 1] + 1,
        h[i1][j1] + (i - i1 - 1) + 1 + (j - j1 - 1),
      );
    }
    da[a[i - 1]] = i;
  }
  return h[al + 1][bl + 1];
}

export interface LookalikeHit {
  brand: Brand;
  kind: 'typosquat' | 'homograph' | 'combosquat' | 'subdomain-brand';
  distance: number;
  official: string;
}

export function shortestBrandDistance(registeredName: string): number {
  const name = normalizeHomoglyphs(registeredName.replace(/-/g, ''));
  let best = 10;
  for (const brand of BRANDS) {
    for (const token of brand.tokens) {
      if (token.length < 3) continue;
      const d = damerauLevenshtein(name, token);
      if (d < best) best = d;
    }
  }
  return best;
}

export function detectLookalike(parsed: ParsedUrl): LookalikeHit | null {
  const name = parsed.registeredName;
  const normalizedName = normalizeHomoglyphs(name.replace(/-/g, ''));
  const hostHaystack = `${parsed.subdomain}.${parsed.etld1}`.toLowerCase();
  const labels = parsed.labels;

  for (const brand of BRANDS) {
    const official = brand.domains.includes(parsed.etld1);
    if (official) continue;

    for (const token of brand.tokens) {
      if (token.length < 4) continue;
      const normToken = normalizeHomoglyphs(token);

      if (normalizedName === normToken && name !== token) {
        return { brand, kind: 'homograph', distance: 0, official: brand.domains[0] };
      }

      const dist = damerauLevenshtein(normalizedName, normToken);
      const threshold = token.length >= 8 ? 2 : 1;
      if (dist > 0 && dist <= threshold && Math.abs(normalizedName.length - normToken.length) <= threshold) {
        return { brand, kind: 'typosquat', distance: dist, official: brand.domains[0] };
      }

      if (
        (name.startsWith(`${token}-`) ||
          name.endsWith(`-${token}`) ||
          name.includes(`-${token}-`)) &&
        name !== token
      ) {
        return { brand, kind: 'combosquat', distance: 0, official: brand.domains[0] };
      }

      const inSubdomain = labels.slice(0, -2).some((label) => {
        const n = normalizeHomoglyphs(label.replace(/-/g, ''));
        return n === normToken || n.includes(normToken);
      });
      if (inSubdomain || parsed.subdomain.split('.').some((l) => normalizeHomoglyphs(l) === normToken)) {
        return { brand, kind: 'subdomain-brand', distance: 0, official: brand.domains[0] };
      }
    }

    void hostHaystack;
  }

  return null;
}

export function brandInSubdomain(parsed: ParsedUrl): boolean {
  if (!parsed.subdomain) return false;
  const labels = parsed.subdomain.split('.').filter((l) => l && l !== 'www');
  for (const brand of BRANDS) {
    if (brand.domains.includes(parsed.etld1)) continue;
    for (const token of brand.tokens) {
      if (token.length < 4) continue;
      const normToken = normalizeHomoglyphs(token);
      if (labels.some((l) => normalizeHomoglyphs(l.replace(/-/g, '')) === normToken || normalizeHomoglyphs(l).includes(normToken))) {
        return true;
      }
    }
  }
  return false;
}

export function brandInPath(pathname: string, etld1: string): boolean {
  const path = pathname.toLowerCase();
  for (const brand of BRANDS) {
    if (brand.domains.includes(etld1)) continue;
    for (const token of brand.tokens) {
      if (token.length < 5) continue;
      if (path.includes(`/${token}`) || path.includes(`-${token}`) || path.includes(`${token}-`)) {
        return true;
      }
    }
  }
  return false;
}

export function brandInHostnameNotEtld(parsed: ParsedUrl): boolean {
  return brandInSubdomain(parsed);
}
