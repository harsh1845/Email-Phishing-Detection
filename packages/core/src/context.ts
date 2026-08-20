import { BRANDS } from './data/brands.js';
import { getEtldPlusOne } from './psl.js';
import { normalizeHomoglyphs } from './lookalike.js';
import type { EmailContext, ParsedUrl } from './types.js';

const URGENCY = [
  'urgent',
  'immediately',
  'verify your account',
  'account suspended',
  'unusual activity',
  'confirm your identity',
  'limited time',
  'act now',
  'password expires',
  'unauthorized login',
];

export interface ContextAdjustment {
  delta: number;
  reasons: string[];
}

export function applyEmailContext(
  parsed: ParsedUrl,
  context?: EmailContext,
): ContextAdjustment {
  const reasons: string[] = [];
  let delta = 0;
  if (!context) return { delta, reasons };

  const senderEmail = (context.senderEmail ?? '').trim().toLowerCase();
  const senderHost = senderEmail.includes('@') ? senderEmail.split('@')[1] : '';
  const senderEtld1 = senderHost ? getEtldPlusOne(senderHost) : '';
  const display = (context.senderDisplayName ?? '').trim().toLowerCase();
  const subject = (context.subject ?? '').toLowerCase();

  if (senderEtld1 && parsed.etld1 && senderEtld1 !== parsed.etld1) {
    const senderIsBrand = BRANDS.some((b) => b.domains.includes(senderEtld1));
    const linkIsBrandOfficial = BRANDS.some((b) => b.domains.includes(parsed.etld1));
    if (!linkIsBrandOfficial) {
      delta += 0.08;
      reasons.push(`Link domain ${parsed.etld1} does not match sender domain ${senderEtld1}`);
    }
    if (senderIsBrand && !linkIsBrandOfficial) {
      delta += 0.12;
    }
  }

  if (display) {
    for (const brand of BRANDS) {
      const displayHits = brand.tokens.some(
        (t) => t.length >= 4 && normalizeHomoglyphs(display).includes(t),
      );
      const fromOfficial = brand.domains.includes(senderEtld1);
      const linkOfficial = brand.domains.includes(parsed.etld1);
      if (displayHits && !fromOfficial) {
        delta += 0.2;
        reasons.push(`Display name impersonates ${brand.name} but sender is ${senderEtld1 || 'unknown'}`);
      }
      if (displayHits && !linkOfficial && parsed.etld1) {
        delta += 0.1;
      }
    }
  }

  if (URGENCY.some((p) => subject.includes(p))) {
    delta += 0.04;
    reasons.push('Subject uses urgency language often seen in phishing');
  }

  return { delta: Math.min(delta, 0.35), reasons };
}
