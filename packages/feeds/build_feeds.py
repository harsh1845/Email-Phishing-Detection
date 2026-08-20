#!/usr/bin/env python3
"""Build a compact hostname blocklist + bloom filter for the extension (data only)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent / "model"
sys.path.insert(0, str(MODEL))

from phishguard_model.download import load_openphish, load_urlhaus_hosts  # noqa: E402
from phishguard_model.features import coerce_parse, get_etld1  # noqa: E402

DATA = MODEL / "data"
OUT = ROOT.parent / "extension" / "src" / "assets"
SEED = [
    "paypa1.com",
    "appleid-secure-login.net",
    "login-microsoftonline.com",
]


def fnv1a(data: bytes, seed: int) -> int:
    h = (2166136261 ^ seed) & 0xFFFFFFFF
    for b in data:
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def host_of(url: str) -> str:
    parsed = coerce_parse(url)
    if not parsed:
        return url.lower().replace("http://", "").replace("https://", "").split("/")[0]
    return get_etld1(parsed[1])


def build_bloom(hosts: list[str], fp: float = 0.01) -> dict:
    n = max(1, len(hosts))
    m = max(64, math.ceil((-n * math.log(fp)) / (math.log(2) ** 2)))
    k = max(1, round((m / n) * math.log(2)))
    bits = bytearray((m + 7) // 8)
    for host in hosts:
        raw = host.lower().encode()
        for i in range(k):
            idx = fnv1a(raw, i + 1) % m
            bits[idx >> 3] |= 1 << (idx & 7)
    import base64

    return {"m": m, "k": k, "n": n, "bits": base64.b64encode(bytes(bits)).decode("ascii")}


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    hosts = set(SEED)
    try:
        uh = load_urlhaus_hosts(DATA, limit=25000)
        for url in uh["url"]:
            hosts.add(host_of(str(url)))
    except Exception as exc:
        print("urlhaus skipped:", exc)
    try:
        op = load_openphish(DATA, limit=8000)
        for url in op["url"]:
            hosts.add(host_of(str(url)))
    except Exception as crc:
        print("openphish skipped:", crc)

    hosts = {h for h in hosts if h and "." in h and " " not in h}
    compact = sorted(hosts)[:40000]
    (OUT / "blocklist.json").write_text(json.dumps(compact))
    (OUT / "bloom.json").write_text(json.dumps(build_bloom(compact)))
    print(f"Wrote {len(compact)} hosts → {OUT}")


if __name__ == "__main__":
    main()
