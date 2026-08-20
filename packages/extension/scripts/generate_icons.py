#!/usr/bin/env python3
"""Write simple shield PNG icons without third-party imaging libs."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "extension" / "icons"


def png(width: int, height: int, rgba: bytes) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    raw = b""
    stride = width * 4
    for y in range(height):
        raw += b"\x00" + rgba[y * stride : (y + 1) * stride]
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def draw(size: int) -> bytes:
    px = bytearray(size * size * 4)
    cx = cy = size / 2
    for y in range(size):
        for x in range(size):
            nx = (x + 0.5) / size
            ny = (y + 0.5) / size
            in_shield = (
                ny > 0.12
                and ny < 0.88
                and abs(nx - 0.5) < 0.38 * (1.05 - (ny - 0.12) ** 1.4)
            )
            i = (y * size + x) * 4
            if in_shield:
                px[i : i + 4] = bytes([21, 101, 192, 255])
                # check mark
                if 0.32 < nx < 0.48 and abs((ny - 0.52) - (nx - 0.32) * 0.9) < 0.06 / (size / 32):
                    px[i : i + 4] = bytes([255, 255, 255, 255])
                if 0.46 < nx < 0.72 and abs((ny - 0.58) + (nx - 0.48) * 1.1) < 0.07 / (size / 32):
                    px[i : i + 4] = bytes([255, 255, 255, 255])
            else:
                px[i : i + 4] = bytes([0, 0, 0, 0])
    return png(size, size, bytes(px))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for size in (16, 32, 48, 128):
        (OUT / f"icon{size}.png").write_bytes(draw(size))
    print("Wrote icons to", OUT)


if __name__ == "__main__":
    main()
