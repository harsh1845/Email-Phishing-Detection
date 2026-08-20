import { describe, expect, it } from 'vitest';
import { extractFeatureMap, FEATURE_NAMES } from '../src/features.js';
import { BloomFilter } from '../src/bloom.js';

describe('features', () => {
  it('emits every spec feature', () => {
    const map = extractFeatureMap('https://login.example.com/verify?id=1');
    for (const name of FEATURE_NAMES) {
      expect(map).toHaveProperty(name);
      expect(typeof map[name]).toBe('number');
    }
  });

  it('detects IP hosts and https', () => {
    const map = extractFeatureMap('http://192.168.1.8/login');
    expect(map.has_ip).toBe(1);
    expect(map.has_https).toBe(0);
    expect(map.has_http).toBe(1);
  });
});

describe('bloom', () => {
  it('round-trips membership', () => {
    const bloom = BloomFilter.fromCapacity(100, 0.01);
    bloom.add('evil.example');
    expect(bloom.has('evil.example')).toBe(true);
    expect(bloom.has('google.com')).toBe(false);
    const again = BloomFilter.fromJSON(bloom.toJSON());
    expect(again.has('evil.example')).toBe(true);
  });
});
