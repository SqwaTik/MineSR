"""Checks a capture session is usable for training.

    python check_capture.py <session dir> [--out check.png]

For consecutive frames it rebuilds motion from depth and camera data exactly as the game's
upscaler does, warps the previous target image onto the current one, and compares. If depth,
matrices, jitter or row order were wrong, warping would not beat leaving the image alone.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import capture_data as cd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--out", default="check.png")
    ap.add_argument("--preset", default="quality", choices=["quality", "performance"])
    args = ap.parse_args()

    runs = cd.sequences(args.session)
    frames = sum(len(r) for r in runs)
    pairs = [(r[i - 1], r[i]) for r in runs for i in range(1, len(r))]
    print(f"{frames} frames in {len(runs)} sequence(s), {len(pairs)} consecutive pairs")
    if not pairs:
        sys.exit("Need at least two consecutive frames.")

    rows, ratios = [], []
    for prev, cur in pairs:
        target = cd.load_image(cur, "target")
        warped, valid = cd.warp_previous_target(cur, prev, args.preset)
        before = cd.load_image(prev, "target")
        err_none = np.abs(before - target).mean(axis=-1)
        err_warp = np.abs(warped - target).mean(axis=-1)
        moved = np.linalg.norm(cur["camera"] - prev["camera"])
        e0, e1 = err_none[valid].mean(), err_warp[valid].mean()
        ratios.append(e1 / max(e0, 1e-6))
        print(f"  frame {cur['frame']:6d}: camera moved {moved:5.3f} blocks, "
              f"error unwarped {e0 * 255:5.2f}, warped {e1 * 255:5.2f} (of 255), "
              f"{valid.mean() * 100:4.1f}% of pixels were on screen last frame")
        rows.append((before, warped, target, err_warp))

    median = float(np.median(ratios))
    print(f"\nWarped error is {median * 100:.0f}% of unwarped (median). ", end="")
    if median < 0.5:
        print("Motion data is consistent: OK for training.")
    else:
        print("Warping barely helps: depth, matrices or jitter are probably wrong (or the camera didn't move).")

    # Visual check from the pair with the most motion: previous | warped | current | error x4.
    i = int(np.argmax([np.linalg.norm(c["camera"] - p["camera"]) for p, c in pairs]))
    before, warped, target, err = rows[i]
    err_img = np.repeat(np.clip(err * 4, 0, 1)[..., None], 3, axis=-1)
    strip = np.concatenate([before, warped, target, err_img], axis=1)[::-1]  # back to top-down
    Image.fromarray((strip * 255).astype(np.uint8)).save(args.out)
    print(f"Wrote {Path(args.out).resolve()} (previous | previous warped | current | error x4)")


if __name__ == "__main__":
    main()
