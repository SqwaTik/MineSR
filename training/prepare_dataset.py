"""Converts capture sessions into arrays for fast training.

    python prepare_dataset.py <captures dir> <cache dir>

For every continuous sequence (see capture_data.sequences) and preset, writes .npy files
(bottom row first, like everything else here):
    <name>_target.npy             (T, H, W, 3) uint8      answer images
    <name>_<preset>_low.npy       (T, h, w, 3) uint8      low-res inputs
    <name>_<preset>_motion.npy    (T, h, w, 2) float16    previous uv - current uv (frame 0: zeros)
    <name>_<preset>_onscreen.npy  (T, h, w)    bool       previous position was on screen (frame 0: false)
    <name>_<preset>_jitter.npy    (T, 2)       float32
and index.json listing every sequence with its scene (session) name.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import capture_data as cd

PRESETS = ["quality", "performance"]
MIN_LENGTH = 4


def convert(job):
    session, run_index, run, out = job
    name = f"{session}_{run_index:02d}"
    t = len(run)
    np.save(out / f"{name}_target.npy", np.stack([(cd.load_image(m, "target") * 255 + 0.5).astype(np.uint8) for m in run]))
    for preset in PRESETS:
        p = run[0]["presets"][preset]
        low = np.zeros((t, p["height"], p["width"], 3), np.uint8)
        motion = np.zeros((t, p["height"], p["width"], 2), np.float16)
        onscreen = np.zeros((t, p["height"], p["width"]), bool)
        jitter = np.zeros((t, 2), np.float32)
        for i, m in enumerate(run):
            low[i] = (cd.load_image(m, preset) * 255 + 0.5).astype(np.uint8)
            jitter[i] = m["presets"][preset]["jitter"]
            if i > 0:
                mv, on = cd.motion(m, run[i - 1], preset)
                motion[i] = np.nan_to_num(mv, nan=0.0, posinf=0.0, neginf=0.0)
                onscreen[i] = on
        np.save(out / f"{name}_{preset}_low.npy", low)
        np.save(out / f"{name}_{preset}_motion.npy", motion)
        np.save(out / f"{name}_{preset}_onscreen.npy", onscreen)
        np.save(out / f"{name}_{preset}_jitter.npy", jitter)
    return {"name": name, "scene": session, "frames": t}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("captures")
    ap.add_argument("cache")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.cache)
    out.mkdir(parents=True, exist_ok=True)

    jobs = []
    for session in sorted(p for p in Path(args.captures).iterdir() if p.is_dir()):
        for k, run in enumerate(cd.sequences(session)):
            if len(run) >= MIN_LENGTH:
                jobs.append((session.name, k, run, out))
    print(f"{len(jobs)} sequences to convert")
    entries = []
    with ProcessPoolExecutor(args.workers) as pool:
        for n, entry in enumerate(pool.map(convert, jobs), 1):
            entries.append(entry)
            if n % 10 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)}")
    (out / "index.json").write_text(json.dumps({"presets": PRESETS, "sequences": entries}, indent=1))
    size = sum(f.stat().st_size for f in out.glob("*.npy"))
    print(f"{sum(e['frames'] for e in entries)} frames, {size / 2**30:.1f} GB in {out}")


if __name__ == "__main__":
    main()
