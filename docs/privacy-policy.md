PhishGuard Privacy Policy
=========================

Last updated: 20 August 2026

PhishGuard is a browser extension that inspects hyperlinks in Gmail messages you open, on your device, to warn about phishing.

What we collect
---------------
PhishGuard does **not** collect, transmit, or sell email content, sender addresses, or browsing history.

Processing happens locally:
- Link URLs from the open Gmail message
- Sender display name and email address (used only as on-device context)
- Optional list of domains you mark as trusted (stored in `chrome.storage` on your profile)

Optional network requests (off by default)
------------------------------------------
If you enable **live threat feeds**, the extension fetches hostname lists from URLhaus (`urlhaus.abuse.ch`). Those requests do not include your email.

If you enable **Google Safe Browsing** and provide an API key, the extension sends **4-byte SHA-256 hash prefixes** of canonicalized URLs to Google. It does not send the full URL, the email body, or your Google account. Safe Browsing is for non-commercial use; commercial deployments should use Google Web Risk. Google’s terms and attribution requirements apply to those lookups.

Storage
-------
Settings and cached feed hostnames stay in your browser profile. Uninstalling the extension removes them.

Sharing
-------
We do not sell data. We do not use analytics SDKs. We do not run ads.

Contact
-------
Open an issue at the project repository: https://github.com/harsh1845/Email-Phishing-Detection
