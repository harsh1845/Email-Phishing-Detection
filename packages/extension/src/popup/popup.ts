import { loadSettings, saveSettings } from '../lib/settings';

async function init(): Promise<void> {
  const settings = await loadSettings();
  const gsb = document.querySelector<HTMLInputElement>('#gsb')!;
  const key = document.querySelector<HTMLInputElement>('#gsb-key')!;
  const feeds = document.querySelector<HTMLInputElement>('#feeds')!;
  const allow = document.querySelector<HTMLTextAreaElement>('#allow')!;
  const status = document.querySelector<HTMLElement>('#status')!;

  gsb.checked = settings.enableSafeBrowsing;
  key.value = settings.safeBrowsingApiKey;
  feeds.checked = settings.enableLiveFeeds;
  allow.value = settings.userAllowlist.join('\n');

  document.querySelector('#save')!.addEventListener('click', async () => {
    await saveSettings({
      enableSafeBrowsing: gsb.checked,
      safeBrowsingApiKey: key.value.trim(),
      enableLiveFeeds: feeds.checked,
      userAllowlist: allow.value
        .split(/[\n,]/)
        .map((s) => s.trim().toLowerCase())
        .filter(Boolean),
    });
    if (feeds.checked) {
      try {
        await chrome.permissions.request({
          origins: ['https://urlhaus.abuse.ch/*', 'https://safebrowsing.googleapis.com/*'],
        });
      } catch {
        /* optional */
      }
    }
    status.textContent = 'Saved';
    setTimeout(() => {
      status.textContent = '';
    }, 1500);
  });
}

void init();
