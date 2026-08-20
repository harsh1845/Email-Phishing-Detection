export const SELECTORS = {
  main: 'div[role="main"]',
  body: 'div.a3s, div[data-message-id] .a3s',
  sender: 'span.gD[email], span[email].gD, h3 span[email]',
  subject: 'h2.hP, h2[data-thread-perm-id]',
  links: 'div.a3s a[href], div[data-message-id] a[href]',
};

export interface GmailContext {
  senderEmail?: string;
  senderDisplayName?: string;
  subject?: string;
}

export function readOpenMessageContext(root: ParentNode = document): GmailContext {
  const senderEl = root.querySelector<HTMLElement>(SELECTORS.sender);
  const subjectEl = root.querySelector<HTMLElement>(SELECTORS.subject);
  return {
    senderEmail: senderEl?.getAttribute('email') || undefined,
    senderDisplayName: senderEl?.getAttribute('name') || senderEl?.textContent?.trim() || undefined,
    subject: subjectEl?.textContent?.trim() || undefined,
  };
}

export function collectMessageLinks(root: ParentNode = document): HTMLAnchorElement[] {
  const seen = new Set<string>();
  const out: HTMLAnchorElement[] = [];
  for (const el of root.querySelectorAll<HTMLAnchorElement>(SELECTORS.links)) {
    const href = el.getAttribute('href');
    if (!href || href.startsWith('#') || href.startsWith('mailto:')) continue;
    if (seen.has(href)) continue;
    seen.add(href);
    out.push(el);
  }
  return out;
}

export function isMessageView(root: ParentNode = document): boolean {
  return Boolean(root.querySelector(SELECTORS.body));
}
