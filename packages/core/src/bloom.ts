const FNV_OFFSET = 2166136261;
const FNV_PRIME = 16777619;

function fnv1a(str: string, seed: number): number {
  let h = (FNV_OFFSET ^ seed) >>> 0;
  for (let i = 0; i < str.length; i += 1) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, FNV_PRIME) >>> 0;
  }
  return h >>> 0;
}

export interface BloomFilterJSON {
  m: number;
  k: number;
  bits: string;
  n: number;
}

export class BloomFilter {
  readonly m: number;
  readonly k: number;
  readonly bits: Uint8Array;
  n = 0;

  constructor(m: number, k: number, bits?: Uint8Array) {
    this.m = m;
    this.k = k;
    this.bits = bits ?? new Uint8Array(Math.ceil(m / 8));
  }

  static fromCapacity(expected: number, fpRate = 0.01): BloomFilter {
    const m = Math.max(64, Math.ceil((-expected * Math.log(fpRate)) / Math.log(2) ** 2));
    const k = Math.max(1, Math.round((m / expected) * Math.log(2)));
    return new BloomFilter(m, k);
  }

  add(value: string): void {
    const v = value.toLowerCase();
    for (let i = 0; i < this.k; i += 1) {
      const idx = fnv1a(v, i + 1) % this.m;
      this.bits[idx >> 3] |= 1 << (idx & 7);
    }
    this.n += 1;
  }

  has(value: string): boolean {
    const v = value.toLowerCase();
    for (let i = 0; i < this.k; i += 1) {
      const idx = fnv1a(v, i + 1) % this.m;
      if ((this.bits[idx >> 3] & (1 << (idx & 7))) === 0) return false;
    }
    return true;
  }

  toJSON(): BloomFilterJSON {
    const bin: string[] = [];
    for (const b of this.bits) bin.push(String.fromCharCode(b));
    return {
      m: this.m,
      k: this.k,
      n: this.n,
      bits: btoa(bin.join('')),
    };
  }

  static fromJSON(json: BloomFilterJSON): BloomFilter {
    const raw = atob(json.bits);
    const bits = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i += 1) bits[i] = raw.charCodeAt(i);
    const bloom = new BloomFilter(json.m, json.k, bits);
    bloom.n = json.n;
    return bloom;
  }
}
