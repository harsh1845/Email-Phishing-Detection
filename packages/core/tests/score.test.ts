import { describe, expect, it } from 'vitest';
import { detectLookalike } from '../src/lookalike.js';
import { parseUrl } from '../src/parse.js';
import { scoreUrl } from '../src/score.js';

describe('lookalike + score', () => {
  it('flags paypal typosquat as danger', () => {
    const v = scoreUrl('https://paypa1.com/signin');
    expect(v.level).toBe('danger');
    expect(v.reasons.join(' ')).toMatch(/PayPal/i);
  });

  it('flags brand-in-subdomain', () => {
    const parsed = parseUrl('https://paypal.com.evil-login.net/secure');
    expect(parsed).not.toBeNull();
    const hit = detectLookalike(parsed!);
    expect(hit?.kind).toBe('subdomain-brand');
    const v = scoreUrl('https://paypal.com.evil-login.net/secure');
    expect(v.level).toBe('danger');
  });

  it('allowlists real paypal', () => {
    const v = scoreUrl('https://www.paypal.com/signin');
    expect(v.level).toBe('safe');
    expect(v.source).toContain('allowlist');
  });

  it('flags javascript URIs', () => {
    const v = scoreUrl('javascript:alert(1)');
    expect(v.level).toBe('danger');
  });

  it('treats shorteners as suspicious, not danger', () => {
    const v = scoreUrl('https://bit.ly/abc123');
    expect(v.level).toBe('suspicious');
  });

  it('raises score when display name impersonates a brand', () => {
    const v = scoreUrl('https://account-review-login.net/verify', {
      senderEmail: 'alerts@randomhost.biz',
      senderDisplayName: 'PayPal Security',
      subject: 'Urgent: verify your account',
    });
    expect(v.score).toBeGreaterThan(0.2);
    expect(v.reasons.join(' ')).toMatch(/PayPal/i);
  });
});
