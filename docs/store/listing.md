# Chrome Web Store / Firefox AMO listing

## Single purpose
Warn the user about phishing links inside Gmail messages before they click.

## Short description
Checks links in opened Gmail messages for phishing. Scoring runs on-device.

## Detailed description
PhishGuard looks at the links in a Gmail message you have open and warns you before you click a dangerous one.

It unwraps Gmail redirectors, checks brand lookalikes and other high-precision heuristics, and can run a small on-device URL model. Email bodies never leave your browser.

Optional: Google Safe Browsing (hash prefixes only) and URLhaus hostname feeds. Both are off until you turn them on.

This extension only runs on mail.google.com.

## Permission justifications

storage
: Save your settings (trusted domains, Safe Browsing toggle) on this device.

alarms
: Refresh optional threat-feed data a few times per day.

offscreen (Chrome)
: Run the on-device ONNX model without blocking Gmail.

host mail.google.com
: Read hyperlinks and sender metadata in the open message so they can be scored. We do not send that content to a server.

optional urlhaus.abuse.ch
: Download public malware/phishing host lists when you enable live feeds.

optional safebrowsing.googleapis.com
: Hash-prefix threat lookup when you enable Safe Browsing.

## Review notes
- No remote executable code
- No analytics
- Heuristics work without the ONNX model
- Interstitial always offers “Go back” and “Proceed anyway”
