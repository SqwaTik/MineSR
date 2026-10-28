"""Measures flicker in still-camera frames from MineSRFlickerTest.

    python analyze_flicker.py <flicker dir> [--out flicker.png]

For each mode: mean and 99th-percentile frame-to-frame change, and how many pixels change by
more than 8/255. Writes a heat map per mode (bright = changes between frames) next to a frame.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out", default="flicker.png")
    args = ap.parse_args()
    rows = []
    print(f"{'mode':22} {'mean change':>12} {'99th pct':>9} {'pixels >8/255':>14}")
    for mode_dir in sorted(p for p in Path(args.dir).iterdir() if p.is_dir()):
        frames = [np.asarray(Image.open(f).convert("RGB"), dtype=np.float32) for f in sorted(mode_dir.glob("*.png"))]
        if len(frames) < 2:
            continue
        diffs = np.stack([np.abs(b - a).mean(-1) for a, b in zip(frames, frames[1:])])
        mean_map = diffs.mean(0)
        big = (diffs > 8).mean() * 100
        print(f"{mode_dir.name:22} {diffs.mean():12.3f} {np.percentile(diffs, 99):9.2f} {big:13.2f}%")
        heat = np.clip(mean_map * 16, 0, 255).astype(np.uint8)
        rows.append((mode_dir.name, frames[-1].astype(np.uint8), heat))
    w, h = 480, 270
    sheet = Image.new("RGB", (w * 2, (h + 20) * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (name, frame, heat) in enumerate(rows):
        y = i * (h + 20)
        sheet.paste(Image.fromarray(frame).resize((w, h)), (0, y + 20))
        sheet.paste(Image.fromarray(heat).convert("RGB").resize((w, h)), (w, y + 20))
        draw.text((4, y + 4), f"{name}: frame | where it changes between frames (x16)", fill="black")
    sheet.save(args.out)
    print(f"heat maps: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
