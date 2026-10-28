"""Summarizes all capture sessions under a folder and checks each one.

    python dataset_summary.py <captures dir> [--sheet sheet.png]

Per session: frames, consecutive pairs, size on disk, and the warp check from
check_capture.py on a sample of pairs. Writes a contact sheet with one frame per session.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import capture_data as cd


def check_pairs(pairs, preset="quality", samples=6):
    """Median warped/unwarped error ratio over evenly spaced pairs."""
    if not pairs:
        return float("nan"), float("nan")
    idx = np.linspace(0, len(pairs) - 1, min(samples, len(pairs))).astype(int)
    ratios, warped_errors = [], []
    for i in idx:
        prev, cur = pairs[i]
        target = cd.load_image(cur, "target")
        warped, valid = cd.warp_previous_target(cur, prev, preset)
        if not valid.any():
            continue
        e0 = np.abs(cd.load_image(prev, "target") - target).mean(axis=-1)[valid].mean()
        e1 = np.abs(warped - target).mean(axis=-1)[valid].mean()
        if e0 > 1e-4:  # skip pairs where nothing changed
            ratios.append(e1 / e0)
        warped_errors.append(e1 * 255)
    return (float(np.median(ratios)) if ratios else float("nan")), (float(np.median(warped_errors)) if warped_errors else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("captures")
    ap.add_argument("--sheet", default="sheet.png")
    args = ap.parse_args()

    sessions = sorted(p for p in Path(args.captures).iterdir() if p.is_dir() and p.name.startswith("session-"))
    total_frames = total_pairs = total_bytes = 0
    thumbs = []
    print(f"{'session':28} {'frames':>6} {'pairs':>6} {'MB':>6} {'warp ratio':>10} {'warped err':>10}")
    for s in sessions:
        runs = cd.sequences(s)
        frames = sum(len(r) for r in runs)
        pairs = [(r[i - 1], r[i]) for r in runs for i in range(1, len(r))]
        size = sum(f.stat().st_size for f in s.rglob("*") if f.is_file())
        ratio, err = check_pairs(pairs)
        total_frames += frames
        total_pairs += len(pairs)
        total_bytes += size
        flag = "" if ratio < 0.6 or np.isnan(ratio) else "  <- check this one"
        print(f"{s.name:28} {frames:6d} {len(pairs):6d} {size / 2**20:6.0f} {ratio:10.2f} {err:10.2f}{flag}")
        if runs:
            middle = runs[len(runs) // 2][len(runs[len(runs) // 2]) // 2]
            img = Image.open(f"{middle['path'].with_suffix('')}-target.png").convert("RGB")
            thumbs.append((s.name.replace("session-", ""), img.resize((320, 180))))
    print(f"\n{len(sessions)} sessions, {total_frames} frames, {total_pairs} consecutive pairs, {total_bytes / 2**30:.2f} GB")

    if thumbs:
        cols = 4
        rows = (len(thumbs) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * 320, rows * 200), "white")
        draw = ImageDraw.Draw(sheet)
        for i, (name, img) in enumerate(thumbs):
            x, y = (i % cols) * 320, (i // cols) * 200
            sheet.paste(img, (x, y + 20))
            draw.text((x + 4, y + 4), f"{i + 1}. {name}", fill="black")
        sheet.save(args.sheet)
        print(f"Contact sheet: {Path(args.sheet).resolve()}")


if __name__ == "__main__":
    main()
