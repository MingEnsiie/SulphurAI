#!/usr/bin/env python3
import os, time, sys
from pathlib import Path

TEMP_DIR = Path("/home/mingzhang/Downloads/code/Assets/models/gemma-3-12b-it-bnb-4bit/._____temp")
FINAL_DIR = Path("/home/mingzhang/Downloads/code/Assets/models/gemma-3-12b-it-bnb-4bit")
TOTAL_BYTES = 7_800 * 1024 * 1024  # ~7.8 GB


def get_downloaded():
    total = 0
    if TEMP_DIR.exists():
        for f in TEMP_DIR.iterdir():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    for f in FINAL_DIR.glob("model-*.safetensors"):
        try:
            total += f.stat().st_size
        except OSError:
            pass
    return total


def bar(done, total, width=40):
    pct = done / total
    filled = int(width * pct)
    b = "█" * filled + "░" * (width - filled)
    return f"[{b}] {pct*100:.1f}%"


def fmt_size(b):
    return f"{b/1024/1024/1024:.2f} GB" if b >= 1e9 else f"{b/1024/1024:.1f} MB"


def fmt_time(s):
    s = int(s)
    if s < 60:
        return f"{s}s"
    elif s < 3600:
        return f"{s//60}m {s%60:02d}s"
    else:
        return f"{s//3600}h {(s%3600)//60:02d}m"


prev, prev_t = get_downloaded(), time.time()
time.sleep(2)

while True:
    now = get_downloaded()
    now_t = time.time()
    speed = (now - prev) / (now_t - prev_t) if now_t > prev_t else 0
    prev, prev_t = now, now_t

    remaining = max(0, TOTAL_BYTES - now)
    eta = remaining / speed if speed > 1024 else float("inf")

    line = (
        f"\r{bar(now, TOTAL_BYTES)}  "
        f"{fmt_size(now)}/{fmt_size(TOTAL_BYTES)}  "
        f"{fmt_size(speed)}/s  "
        f"ETA {fmt_time(eta) if eta != float('inf') else '---'}"
    )
    sys.stdout.write(line)
    sys.stdout.flush()

    if now >= TOTAL_BYTES * 0.999:
        print("\n下载完成！")
        break

    time.sleep(3)
