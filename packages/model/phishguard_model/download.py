"""Download PhiUSIIL, optional live threat feeds, and a compact URL-only PhreshPhish sample."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

PHIUSIIL_URL = "https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip"
URLHAUS_HOSTS = "https://urlhaus.abuse.ch/downloads/hostfile/"
OPENPHISH_FEED = "https://raw.githubusercontent.com/openphish/public_feed/refs/heads/main/feed.txt"
TRANCO_TOP = "https://tranco-list.eu/top-1m.csv.zip"


def download_file(url: str, dest: Path, timeout: int = 120) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1024:
        return dest
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "PhishGuard/1.0 (research)"})
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def load_phiusiil(data_dir: Path) -> pd.DataFrame:
    zip_path = download_file(PHIUSIIL_URL, data_dir / "phiusiil.zip", timeout=300)
    with zipfile.ZipFile(zip_path) as zf:
        csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        with zf.open(csv_name) as fh:
            df = pd.read_csv(fh)
    url_col = next(c for c in df.columns if c.lower() in {"url", "urls"})
    label_col = next(c for c in df.columns if c.lower() in {"label", "class", "target"})
    out = df[[url_col, label_col]].rename(columns={url_col: "url", label_col: "label_raw"})
    out["url"] = out["url"].astype(str)
    # PhiUSIIL: 1 = legitimate, 0 = phishing
    unique = set(out["label_raw"].dropna().unique()[:8])
    if unique <= {0, 1} or unique <= {0.0, 1.0}:
        phish_is_one = out["label_raw"].mean() < 0.55
        if phish_is_one:
            out["label"] = out["label_raw"].astype(int)
        else:
            out["label"] = (out["label_raw"].astype(int) == 0).astype(int)
    else:
        out["label"] = out["label_raw"].astype(int)
    return out[["url", "label"]]


def load_urlhaus_hosts(data_dir: Path, limit: int = 40000) -> pd.DataFrame:
    try:
        path = download_file(URLHAUS_HOSTS, data_dir / "urlhaus_hosts.txt", timeout=60)
    except Exception:
        return pd.DataFrame(columns=["url", "label"])
    hosts = []
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        host = parts[-1].strip()
        if host and "." in host:
            hosts.append(host)
        if len(hosts) >= limit:
            break
    if not hosts:
        return pd.DataFrame(columns=["url", "label"])
    return pd.DataFrame({"url": [f"http://{h}/" for h in hosts], "label": 1})


def load_openphish(data_dir: Path, limit: int = 10000) -> pd.DataFrame:
    try:
        path = download_file(OPENPHISH_FEED, data_dir / "openphish.txt", timeout=60)
    except Exception:
        return pd.DataFrame(columns=["url", "label"])
    urls = []
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if line.startswith("http"):
            urls.append(line)
        if len(urls) >= limit:
            break
    if not urls:
        return pd.DataFrame(columns=["url", "label"])
    return pd.DataFrame({"url": urls, "label": 1})


def load_tranco_benign(data_dir: Path, limit: int = 20000) -> pd.DataFrame:
    try:
        zip_path = download_file(TRANCO_TOP, data_dir / "tranco.zip", timeout=180)
        with zipfile.ZipFile(zip_path) as zf:
            name = zf.namelist()[0]
            with zf.open(name) as fh:
                raw = io.TextIOWrapper(fh, encoding="utf-8", errors="ignore")
                rows = []
                for i, line in enumerate(raw):
                    if i >= limit:
                        break
                    parts = line.strip().split(",")
                    domain = parts[-1].strip().lower()
                    if domain and "." in domain:
                        rows.append(domain)
        return pd.DataFrame({"url": [f"https://{d}/" for d in rows], "label": 0})
    except Exception:
        return pd.DataFrame(columns=["url", "label"])


def load_phreshphish_urls(limit: int = 80000) -> pd.DataFrame:
    """Optional Hugging Face load of URL+label only. Skips HTML payloads."""
    try:
        from datasets import load_dataset
    except ImportError:
        return pd.DataFrame(columns=["url", "label"])
    try:
        ds = load_dataset("phreshphish/phreshphish", split="train", streaming=True)
        rows = []
        for row in ds:
            url = row.get("url") or row.get("URL")
            label = row.get("label") or row.get("phish")
            if not url:
                continue
            if isinstance(label, str):
                y = 1 if label.lower() in {"phish", "phishing", "1", "true"} else 0
            else:
                y = int(label)
            rows.append((str(url), y))
            if len(rows) >= limit:
                break
        return pd.DataFrame(rows, columns=["url", "label"])
    except Exception:
        return pd.DataFrame(columns=["url", "label"])
