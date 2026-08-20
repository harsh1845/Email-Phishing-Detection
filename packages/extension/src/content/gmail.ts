import type { EmailContext, LinkVerdict } from '@phishguard/core';
import { collectMessageLinks, isMessageView, readOpenMessageContext } from './gmail-adapter';

const BADGE_ID = 'phishguard-thread-badge';

function applyLinkStyle(el: HTMLAnchorElement, verdict: LinkVerdict): void {
  el.classList.remove('pg-safe', 'pg-suspicious', 'pg-danger');
  el.classList.add(`pg-${verdict.level}`);
  el.dataset.pgLevel = verdict.level;
  el.dataset.pgScore = String(verdict.score);
  el.dataset.pgReasons = verdict.reasons.join(' | ');
  el.title = `PhishGuard: ${verdict.level} — ${verdict.reasons[0] ?? ''}`;
}

function upsertBadge(level: LinkVerdict['level'], count: number): void {
  const subject = document.querySelector('h2.hP, h2[data-thread-perm-id]');
  if (!subject?.parentElement) return;
  let badge = document.getElementById(BADGE_ID);
  if (!badge) {
    badge = document.createElement('span');
    badge.id = BADGE_ID;
    subject.parentElement.insertBefore(badge, subject.nextSibling);
  }
  badge.className = `pg-badge pg-badge-${level}`;
  if (level === 'danger') {
    badge.textContent = count === 1 ? 'Phishing link blocked' : `${count} phishing links`;
  } else if (level === 'suspicious') {
    badge.textContent = 'Suspicious links';
  } else {
    badge.textContent = 'Links look safe';
  }
}

async function scan(): Promise<void> {
  if (!isMessageView()) return;
  const anchors = collectMessageLinks();
  if (!anchors.length) return;
  const context: EmailContext = readOpenMessageContext();
  const urls = anchors.map((a) => a.href);
  const response = await chrome.runtime.sendMessage({
    type: 'SCORE_LINKS',
    urls,
    context,
  });
  const verdicts = (response?.verdicts ?? []) as LinkVerdict[];
  const byUrl = new Map(verdicts.map((v) => [v.url, v]));
  const byFinal = new Map(verdicts.map((v) => [v.finalUrl, v]));

  let danger = 0;
  let suspicious = 0;
  for (const a of anchors) {
    const v = byUrl.get(a.href) || byFinal.get(a.href);
    if (!v) continue;
    applyLinkStyle(a, v);
    if (v.level === 'danger') danger += 1;
    else if (v.level === 'suspicious') suspicious += 1;
  }

  const level = danger ? 'danger' : suspicious ? 'suspicious' : 'safe';
  upsertBadge(level, danger || suspicious);
}

function onClickCapture(event: MouseEvent): void {
  const target = event.target as HTMLElement | null;
  const a = target?.closest?.('a') as HTMLAnchorElement | null;
  if (!a?.classList.contains('pg-danger')) return;
  event.preventDefault();
  event.stopPropagation();
  chrome.runtime.sendMessage({
    type: 'OPEN_INTERSTITIAL',
    url: a.href,
    finalUrl: a.href,
    reasons: (a.dataset.pgReasons || '').split(' | ').filter(Boolean),
  });
}

let timer: number | undefined;
function scheduleScan(): void {
  if (timer) window.clearTimeout(timer);
  timer = window.setTimeout(() => {
    void scan();
  }, 250);
}

const observer = new MutationObserver(() => scheduleScan());
const main = document.querySelector('div[role="main"]') ?? document.body;
observer.observe(main, { childList: true, subtree: true });
document.addEventListener('click', onClickCapture, true);
scheduleScan();
