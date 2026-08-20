/** Compact multi-part public suffixes used to compute eTLD+1. */
const MULTI_PART_TLDS = new Set([
  'ac.uk',
  'co.uk',
  'gov.uk',
  'ltd.uk',
  'me.uk',
  'net.uk',
  'org.uk',
  'plc.uk',
  'sch.uk',
  'com.au',
  'net.au',
  'org.au',
  'edu.au',
  'gov.au',
  'asn.au',
  'id.au',
  'co.nz',
  'net.nz',
  'org.nz',
  'govt.nz',
  'ac.nz',
  'co.jp',
  'or.jp',
  'ne.jp',
  'ac.jp',
  'go.jp',
  'co.kr',
  'or.kr',
  'go.kr',
  'ac.kr',
  'co.za',
  'org.za',
  'gov.za',
  'ac.za',
  'com.br',
  'org.br',
  'gov.br',
  'com.mx',
  'org.mx',
  'gob.mx',
  'co.in',
  'net.in',
  'org.in',
  'gov.in',
  'ac.in',
  'res.in',
  'firm.in',
  'gen.in',
  'ind.in',
  'com.cn',
  'net.cn',
  'org.cn',
  'gov.cn',
  'edu.cn',
  'com.tw',
  'org.tw',
  'gov.tw',
  'edu.tw',
  'com.hk',
  'org.hk',
  'gov.hk',
  'edu.hk',
  'com.sg',
  'org.sg',
  'gov.sg',
  'edu.sg',
  'com.my',
  'org.my',
  'gov.my',
  'edu.my',
  'co.id',
  'or.id',
  'go.id',
  'ac.id',
  'com.ar',
  'org.ar',
  'gov.ar',
  'com.tr',
  'org.tr',
  'gov.tr',
  'com.ua',
  'org.ua',
  'gov.ua',
  'co.il',
  'org.il',
  'gov.il',
  'ac.il',
  'com.pl',
  'org.pl',
  'gov.pl',
  'com.pt',
  'org.pt',
  'gov.pt',
  'co.th',
  'ac.th',
  'go.th',
  'or.th',
  'com.ph',
  'gov.ph',
  'edu.ph',
  'com.vn',
  'gov.vn',
  'edu.vn',
  'com.ng',
  'org.ng',
  'gov.ng',
  'edu.ng',
  'com.pk',
  'org.pk',
  'gov.pk',
  'edu.pk',
  'com.bd',
  'org.bd',
  'gov.bd',
  'com.sa',
  'org.sa',
  'gov.sa',
  'edu.sa',
  'com.ae',
  'org.ae',
  'gov.ae',
  'ac.ae',
  'co.ke',
  'or.ke',
  'go.ke',
  'ac.ke',
  'com.eg',
  'org.eg',
  'gov.eg',
  'edu.eg',
  'com.ru',
  'org.ru',
  'gov.ru',
  'net.ru',
  'com.cy',
  'gov.cy',
  'ac.cy',
  'azurewebsites.net',
  'cloudfront.net',
  'herokuapp.com',
  'netlify.app',
  'github.io',
  'gitlab.io',
  'pages.dev',
  'workers.dev',
  'web.app',
  'firebaseapp.com',
  'appspot.com',
  's3.amazonaws.com',
  'blob.core.windows.net',
]);

const IPV4 =
  /^(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)$/;
const IPV6_HOST = /^\[?[0-9a-f:]+\]?$/i;

export function isIpAddress(hostname: string): boolean {
  const host = hostname.replace(/^\[/, '').replace(/\]$/, '').toLowerCase();
  if (IPV4.test(host)) return true;
  if (host.includes(':') && IPV6_HOST.test(host)) return true;
  return false;
}

export function getEtldPlusOne(hostname: string): string {
  const host = hostname.replace(/\.$/, '').toLowerCase();
  if (!host) return '';
  if (isIpAddress(host)) return host;
  const parts = host.split('.').filter(Boolean);
  if (parts.length <= 1) return host;
  if (parts.length === 2) return host;

  const last2 = parts.slice(-2).join('.');
  const last3 = parts.slice(-3).join('.');
  if (MULTI_PART_TLDS.has(last3) && parts.length >= 4) {
    return parts.slice(-4).join('.');
  }
  if (MULTI_PART_TLDS.has(last2) && parts.length >= 3) {
    return last3;
  }
  return last2;
}

export function splitHost(hostname: string): {
  etld1: string;
  tld: string;
  registeredName: string;
  subdomain: string;
  labels: string[];
} {
  const host = hostname.replace(/\.$/, '').toLowerCase();
  const labels = host.split('.').filter(Boolean);
  const etld1 = getEtldPlusOne(host);
  if (isIpAddress(host)) {
    return { etld1: host, tld: '', registeredName: host, subdomain: '', labels };
  }
  const etld1Labels = etld1.split('.').filter(Boolean);
  const tld = etld1Labels.slice(1).join('.');
  const registeredName = etld1Labels[0] ?? host;
  const subLabels = labels.slice(0, Math.max(0, labels.length - etld1Labels.length));
  return {
    etld1,
    tld,
    registeredName,
    subdomain: subLabels.join('.'),
    labels,
  };
}
