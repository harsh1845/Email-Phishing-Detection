import { describe, expect, it } from 'vitest';
import { unwrapUrl } from '../src/unwrap.js';

describe('unwrapUrl', () => {
  it('unwraps Gmail redirector', () => {
    const inner = 'https://paypa1-secure.net/login';
    const wrapped = `https://www.google.com/url?q=${encodeURIComponent(inner)}&sa=D&ust=1`;
    expect(unwrapUrl(wrapped)).toBe(inner);
  });

  it('unwraps Outlook SafeLinks', () => {
    const inner = 'https://evil.example/verify';
    const wrapped = `https://nam01.safelinks.protection.outlook.com/?url=${encodeURIComponent(inner)}&data=abc`;
    expect(unwrapUrl(wrapped)).toBe(inner);
  });

  it('returns original when not wrapped', () => {
    expect(unwrapUrl('https://example.com/a')).toBe('https://example.com/a');
  });
});
