import {
  BloomFilter,
  extractFeatureVector,
  scoreEmailLinks,
  type BloomFilterJSON,
  type EmailContext,
  type LinkVerdict,
  type ScoreOptions,
} from '@phishguard/core';
import { checkSafeBrowsing } from '../lib/safebrowsing';
import { loadSettings, type Settings } from '../lib/settings';

type ScoreRequest = {
  type: 'SCORE_LINKS';
  urls: string[];
  context?: EmailContext;
};

type InterstitialRequest = {
  type: 'OPEN_INTERSTITIAL';
  url: string;
  finalUrl?: string;
  reasons: string[];
};

let modelReady = false;
let bloom: BloomFilter | undefined;
let blocklist = new Set<string>();

async function ensureOffscreen(): Promise<boolean> {
  if (!chrome.offscreen?.createDocument) return false;
  try {
    const contexts = await chrome.runtime.getContexts({
      contextTypes: [chrome.runtime.ContextType.OFFSCREEN_DOCUMENT],
    });
    if (contexts.length) return true;
  } catch {
    /* Firefox or older Chrome */
  }
  try {
    await chrome.offscreen.createDocument({
      url: 'offscreen.html',
      reasons: [chrome.offscreen.Reason.BLOBS, chrome.offscreen.Reason.WORKERS],
      justification: 'Run the on-device phishing URL model',
    });
    return true;
  } catch {
    return false;
  }
}

async function modelPredict(features: Float32Array): Promise<number | null> {
  if (!modelReady) return null;
  try {
    const ok = await ensureOffscreen();
    if (!ok) return null;
    const response = await chrome.runtime.sendMessage({
      type: 'MODEL_PREDICT',
      features: Array.from(features),
    });
    return typeof response?.score === 'number' ? response.score : null;
  } catch {
    return null;
  }
}

async function loadBundledFeeds(): Promise<void> {
  try {
    const listUrl = chrome.runtime.getURL('assets/blocklist.json');
    const res = await fetch(listUrl);
    if (res.ok) {
      const hosts = (await res.json()) as string[];
      blocklist = new Set(hosts.map((h) => h.toLowerCase()));
    }
  } catch {
    blocklist = new Set();
  }
  try {
    const bloomUrl = chrome.runtime.getURL('assets/bloom.json');
    const res = await fetch(bloomUrl);
    if (res.ok) {
      bloom = BloomFilter.fromJSON((await res.json()) as BloomFilterJSON);
    }
  } catch {
    bloom = undefined;
  }
}

async function refreshLiveFeeds(): Promise<void> {
  const settings = await loadSettings();
  if (!settings.enableLiveFeeds) return;
  try {
    const res = await fetch('https://urlhaus.abuse.ch/downloads/hostfile/');
    if (!res.ok) return;
    const text = await res.text();
    const hosts: string[] = [];
    for (const line of text.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith('#')) continue;
      const host = trimmed.split(/\s+/).pop();
      if (host && host.includes('.')) hosts.push(host.toLowerCase());
      if (hosts.length >= 50000) break;
    }
    for (const h of hosts) blocklist.add(h);
    await chrome.storage.local.set({ liveBlocklist: hosts.slice(0, 20000), liveBlocklistAt: Date.now() });
  } catch {
    /* optional feed */
  }
}

async function scoreUrls(urls: string[], context: EmailContext | undefined, settings: Settings): Promise<LinkVerdict[]> {
  const userAllow = new Set((settings.userAllowlist ?? []).map((d) => d.toLowerCase()));
  const cached = await chrome.storage.local.get(['liveBlocklist']);
  const live = new Set<string>([...blocklist, ...((cached.liveBlocklist as string[]) || [])]);

  const options: ScoreOptions = {
    allowlist: undefined,
    blocklist: new Set([...live, ...[]]),
    bloomHas: bloom ? (v) => bloom!.has(v) : undefined,
  };

  const base = scoreEmailLinks(urls, context, options).map((v) => {
    if (userAllow.has(v.etld1) || userAllow.has(v.hostname)) {
      return { ...v, level: 'safe' as const, score: 0.01, reasons: ['Trusted by you'], source: ['allowlist' as const] };
    }
    return v;
  });

  const enhanced: LinkVerdict[] = [];
  for (const verdict of base) {
    if (verdict.level === 'safe' && verdict.source.includes('allowlist')) {
      enhanced.push(verdict);
      continue;
    }
    let next = verdict;
    const features = extractFeatureVector(verdict.finalUrl || verdict.url);
    const modelScore = await modelPredict(features);
    if (modelScore != null) {
      next = scoreEmailLinks([verdict.url], context, {
        ...options,
        modelPredict: () => modelScore,
      })[0];
    }
    if (settings.enableSafeBrowsing && settings.safeBrowsingApiKey) {
      const hit = await checkSafeBrowsing(next.finalUrl, settings.safeBrowsingApiKey);
      if (hit) {
        next = {
          ...next,
          level: 'danger',
          score: Math.max(next.score, 0.98),
          reasons: [...next.reasons, 'Google Safe Browsing lists this URL as a known threat'],
          source: [...next.source, 'safebrowsing'],
        };
      }
    }
    enhanced.push(next);
  }
  return enhanced;
}

chrome.runtime.onMessage.addListener((message: ScoreRequest | InterstitialRequest, _sender, sendResponse) => {
  if (message.type === 'SCORE_LINKS') {
    loadSettings()
      .then((settings) => scoreUrls(message.urls, message.context, settings))
      .then((verdicts) => sendResponse({ verdicts }))
      .catch((err) => sendResponse({ verdicts: [], error: String(err) }));
    return true;
  }
  if (message.type === 'OPEN_INTERSTITIAL') {
    const params = new URLSearchParams({
      url: message.url,
      reasons: message.reasons.join('\n'),
    });
    chrome.tabs.create({ url: chrome.runtime.getURL(`interstitial.html?${params.toString()}`) });
    sendResponse({ ok: true });
  }
  if ((message as { type?: string }).type === 'MODEL_READY') {
    modelReady = true;
    sendResponse({ ok: true });
  }
  return false;
});

chrome.runtime.onInstalled.addListener(() => {
  void loadBundledFeeds();
  chrome.alarms.create('phishguard-feeds', { periodInMinutes: 360 });
});

chrome.runtime.onStartup.addListener(() => {
  void loadBundledFeeds();
});

chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === 'phishguard-feeds') void refreshLiveFeeds();
});

void loadBundledFeeds();
void ensureOffscreen().then((ok) => {
  modelReady = ok;
});
