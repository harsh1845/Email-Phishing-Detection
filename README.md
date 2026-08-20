# PhishGuard

On-device phishing **link** shield for Gmail. Chrome Manifest V3 + Firefox. Email bodies never leave the browser.

The old CLI spam classifier (TF-IDF + logistic regression on Enron/Ling/SpamAssassin) is gone. This repo scores **URLs**, not email prose.

## How it works

When you open a Gmail message, PhishGuard unwraps redirectors (`google.com/url?q=`, Outlook SafeLinks), then scores each link:

1. Allowlist of well-known domains
2. Bundled threat-feed hostnames (optional live URLhaus refresh)
3. High-precision heuristics (lookalikes, IP hosts, punycode, brand-in-subdomain)
4. Tiny on-device LightGBM model (ONNX)
5. Sender / display-name mismatch (weak prior)
6. Optional Google Safe Browsing **hash-prefix** lookup

Dangerous links get a red outline; clicking them opens an interstitial with an explanation and a “proceed anyway” escape.

## Packages

| Path | Role |
|------|------|
| `packages/core` | Shared TypeScript scoring engine |
| `packages/model` | Download PhiUSIIL / feeds, train LightGBM, export ONNX |
| `packages/extension` | Chrome / Firefox extension |
| `packages/feeds` | Build compact blocklist + bloom filter (data, not code) |
| `docs/` | Privacy policy and store listing copy |

## Develop

```bash
npm install
npm test
npm run build
```

Load unpacked: Chrome → `chrome://extensions` → Developer mode → Load `packages/extension/dist`. Firefox → `about:debugging` → Load Temporary Add-on → `packages/extension/dist/manifest.json`.

Train the URL model (downloads PhiUSIIL from UCI; optional live feeds). On macOS, LightGBM needs OpenMP: `brew install libomp`.

```bash
packages/model/.venv/bin/python packages/model/train.py
python3 packages/feeds/build_feeds.py
npm run build
```

Package zips for the stores:

```bash
npm run package:chrome
npm run package:firefox
```

Artifacts land in `dist/phishguard-chrome.zip` and `dist/phishguard-firefox.zip`.

## Privacy

See [docs/privacy-policy.md](docs/privacy-policy.md). Default mode is fully on-device. Safe Browsing and URLhaus are opt-in.

## Store

Copy and permission justifications: [docs/store/listing.md](docs/store/listing.md).
