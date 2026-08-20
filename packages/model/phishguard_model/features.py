from __future__ import annotations

import math
import re
from urllib.parse import urlparse

from .constants import (
    BRANDS,
    CREDENTIAL_KEYWORDS,
    FEATURE_NAMES,
    MULTI_PART_TLDS,
    SHORTENERS,
)

IPV4 = re.compile(
    r"^(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)$"
)
HOMOGLYPHS = str.maketrans({
    "0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "@": "a",
})


def is_ip(hostname: str) -> bool:
    host = hostname.strip("[]").lower()
    return bool(IPV4.match(host) or (":" in host and all(c in "0123456789abcdef:" for c in host)))


def get_etld1(hostname: str) -> str:
    host = hostname.rstrip(".").lower()
    if not host or is_ip(host):
        return host
    parts = [p for p in host.split(".") if p]
    if len(parts) <= 2:
        return host
    last2 = ".".join(parts[-2:])
    last3 = ".".join(parts[-3:])
    if last3 in MULTI_PART_TLDS and len(parts) >= 4:
        return ".".join(parts[-4:])
    if last2 in MULTI_PART_TLDS and len(parts) >= 3:
        return last3
    return last2


def split_host(hostname: str):
    host = hostname.rstrip(".").lower()
    labels = [p for p in host.split(".") if p]
    etld1 = get_etld1(host)
    if is_ip(host):
        return etld1, "", host, "", labels
    etld1_labels = [p for p in etld1.split(".") if p]
    tld = ".".join(etld1_labels[1:])
    registered = etld1_labels[0] if etld1_labels else host
    sub_labels = labels[: max(0, len(labels) - len(etld1_labels))]
    return etld1, tld, registered, ".".join(sub_labels), labels


def coerce_parse(raw: str):
    text = (raw or "").strip()
    if not text:
        return None
    try:
        parsed = urlparse(text if "://" in text else "https://" + text)
    except Exception:
        return None
    if parsed.scheme not in ("http", "https"):
        return None
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        return None
    return parsed, hostname


def normalize_homoglyphs(value: str) -> str:
    return value.lower().translate(HOMOGLYPHS).replace("rn", "m")


def damerau(a: str, b: str) -> int:
    if a == b:
        return 0
    al, bl = len(a), len(b)
    if not al:
        return bl
    if not bl:
        return al
    d = [[0] * (bl + 1) for _ in range(al + 1)]
    for i in range(al + 1):
        d[i][0] = i
    for j in range(bl + 1):
        d[0][j] = j
    for i in range(1, al + 1):
        for j in range(1, bl + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + cost)
    return d[al][bl]


def shannon(value: str) -> float:
    if not value:
        return 0.0
    freq: dict[str, int] = {}
    for ch in value:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(value)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def longest_digit_run(value: str) -> int:
    best = cur = 0
    for ch in value:
        if ch.isdigit():
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def shortest_brand_distance(registered: str) -> int:
    name = normalize_homoglyphs(registered.replace("-", ""))
    best = 10
    for brand in BRANDS:
        for token in brand["tokens"]:
            if len(token) < 3:
                continue
            best = min(best, damerau(name, token))
    return best


def brand_in_subdomain(subdomain: str, etld1: str) -> bool:
    if not subdomain:
        return False
    labels = [l for l in subdomain.split(".") if l and l != "www"]
    for brand in BRANDS:
        if etld1 in brand["domains"]:
            continue
        for token in brand["tokens"]:
            if len(token) < 4:
                continue
            norm = normalize_homoglyphs(token)
            for lab in labels:
                n = normalize_homoglyphs(lab.replace("-", ""))
                if n == norm or norm in n:
                    return True
    return False


def brand_in_path(path: str, etld1: str) -> bool:
    lower = path.lower()
    for brand in BRANDS:
        if etld1 in brand["domains"]:
            continue
        for token in brand["tokens"]:
            if len(token) < 5:
                continue
            if f"/{token}" in lower or f"-{token}" in lower or f"{token}-" in lower:
                return True
    return False


def extract_feature_map(raw_url: str) -> dict[str, float]:
    parsed_pair = coerce_parse(raw_url)
    url = raw_url
    hostname = path = query = fragment = username = ""
    protocol = ""
    port = ""
    etld1 = tld = registered = subdomain = ""
    labels: list[str] = []
    if parsed_pair:
        parsed, hostname = parsed_pair
        url = parsed.geturl()
        path = parsed.path or ""
        query = ("?" + parsed.query) if parsed.query else ""
        fragment = ("#" + parsed.fragment) if parsed.fragment else ""
        username = parsed.username or ""
        protocol = parsed.scheme + ":"
        port = str(parsed.port) if parsed.port else ""
        etld1, tld, registered, subdomain, labels = split_host(hostname)

    letters = sum(ch.isalpha() for ch in url)
    digits = sum(ch.isdigit() for ch in url)
    vowels = sum(ch in "aeiou" for ch in hostname.lower())
    sensitive = sum(1 for kw in CREDENTIAL_KEYWORDS if kw in url.lower())

    return {
        "url_length": float(len(url)),
        "hostname_length": float(len(hostname)),
        "path_length": float(len(path)),
        "query_length": float(len(query)),
        "fragment_length": float(len(fragment)),
        "num_dots": float(url.count(".")),
        "num_hyphens": float(url.count("-")),
        "num_underscores": float(url.count("_")),
        "num_slashes": float(url.count("/")),
        "num_question_marks": float(url.count("?")),
        "num_equals": float(url.count("=")),
        "num_ampersands": float(url.count("&")),
        "num_percent": float(url.count("%")),
        "num_digits": float(digits),
        "digit_ratio": digits / len(url) if url else 0.0,
        "num_letters": float(letters),
        "letter_ratio": letters / len(url) if url else 0.0,
        "has_https": 1.0 if protocol == "https:" else 0.0,
        "has_http": 1.0 if protocol == "http:" else 0.0,
        "has_ip": 1.0 if hostname and is_ip(hostname) else 0.0,
        "has_at": 1.0 if "@" in url else 0.0,
        "has_double_slash_redirect": 1.0 if url.find("//", 8) != -1 else 0.0,
        "has_punycode": 1.0 if "xn--" in hostname else 0.0,
        "has_port": 1.0 if port else 0.0,
        "subdomain_count": float(len([l for l in subdomain.split(".") if l and l != "www"])) if subdomain else 0.0,
        "tld_length": float(len(tld)),
        "registered_domain_length": float(len(registered)),
        "path_depth": float(len([p for p in path.split("/") if p])),
        "num_query_params": float(len([p for p in query.lstrip("?").split("&") if p])) if query else 0.0,
        "sensitive_keyword_count": float(sensitive),
        "brand_in_subdomain": 1.0 if brand_in_subdomain(subdomain, etld1) else 0.0,
        "brand_in_path": 1.0 if brand_in_path(path, etld1) else 0.0,
        "brand_in_hostname_not_etld": 1.0 if brand_in_subdomain(subdomain, etld1) else 0.0,
        "shortest_brand_distance": float(min(10, shortest_brand_distance(registered))) if registered else 10.0,
        "is_shortener": 1.0 if etld1 in SHORTENERS else 0.0,
        "hostname_entropy": shannon(hostname),
        "url_entropy": shannon(url),
        "has_hex_escape": 1.0 if re.search(r"%[0-9a-fA-F]{2}", url) else 0.0,
        "vowel_ratio_hostname": vowels / len(hostname) if hostname else 0.0,
        "consecutive_digits": float(longest_digit_run(url)),
        "max_label_length": float(max((len(l) for l in labels), default=0)),
        "num_labels": float(len(labels)),
        "has_www": 1.0 if hostname == "www" or hostname.startswith("www.") else 0.0,
        "has_credential_in_userinfo": 1.0 if username else 0.0,
    }


def extract_vector(raw_url: str) -> list[float]:
    fmap = extract_feature_map(raw_url)
    return [float(fmap[name]) for name in FEATURE_NAMES]
