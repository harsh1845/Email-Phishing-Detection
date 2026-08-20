export interface Settings {
  enableSafeBrowsing: boolean;
  safeBrowsingApiKey: string;
  enableLiveFeeds: boolean;
  userAllowlist: string[];
}

const DEFAULTS: Settings = {
  enableSafeBrowsing: false,
  safeBrowsingApiKey: '',
  enableLiveFeeds: false,
  userAllowlist: [],
};

export async function loadSettings(): Promise<Settings> {
  const stored = await chrome.storage.sync.get(DEFAULTS);
  return { ...DEFAULTS, ...stored };
}

export async function saveSettings(patch: Partial<Settings>): Promise<void> {
  const current = await loadSettings();
  await chrome.storage.sync.set({ ...current, ...patch });
}
