"""Reads MineSR capture sessions. Format is written by CaptureWriter.java.

All arrays returned here are BOTTOM row first, matching the GPU and the shader math
(texCoord.y = 0 at the bottom). Files on disk are top-down; we flip on load.
"""
import gzip
import json
from pathlib import Path

import numpy as np
from PIL import Image


def load_meta(json_path):
    meta = json.loads(Path(json_path).read_text())
    meta["viewProj"] = np.array(meta["viewProjColumnMajor"], dtype=np.float64).reshape(4, 4, order="F")
    meta["camera"] = np.array(meta["camera"], dtype=np.float64)
    meta["path"] = Path(json_path)
    return meta


def _base(meta):
    return meta["path"].with_suffix("")  # .../frame-000123


def load_image(meta, which):
    """'target', 'quality' or 'performance' -> float32 (H, W, 3) in [0, 1], bottom-up."""
    img = np.asarray(Image.open(f"{_base(meta)}-{which}.png").convert("RGB"), dtype=np.float32) / 255.0
    return img[::-1].copy()


def load_depth(meta, preset):
    p = meta["presets"][preset]
    with gzip.open(f"{_base(meta)}-{preset}-depth.f32.gz", "rb") as f:
        d = np.frombuffer(f.read(), dtype="<f4").reshape(p["height"], p["width"])
    return d[::-1].copy()


# Real movement is at most about a block per frame (elytra and fast flying included). A bigger
# jump is a teleport: the two frames show unrelated views, so the sequence is cut there.
TELEPORT_BLOCKS = 4.0
# Likewise a camera snap: no real mouse turns this far between two frames.
CUT_DEGREES = 60.0


def _forward(view_proj):
    inv = np.linalg.inv(view_proj)
    near, far = inv @ np.array([0, 0, 0.9, 1.0]), inv @ np.array([0, 0, 0.5, 1.0])
    d = far[:3] / far[3] - near[:3] / near[3]
    return d / np.linalg.norm(d)


def _is_cut(prev, cur):
    if np.linalg.norm(cur["camera"] - prev["camera"]) > TELEPORT_BLOCKS:
        return True
    cos = np.clip(_forward(prev["viewProj"]) @ _forward(cur["viewProj"]), -1.0, 1.0)
    return np.degrees(np.arccos(cos)) > CUT_DEGREES


def sequences(session_dir):
    """Lists of consecutive frames (frame numbers differ by 1, no teleport or camera snap between), oldest first."""
    metas = sorted((load_meta(p) for p in Path(session_dir).rglob("frame-*.json")), key=lambda m: m["frame"])
    runs, run = [], []
    for m in metas:
        if run and (m["frame"] != run[-1]["frame"] + 1 or m["burst"] != run[-1]["burst"]
                    or _is_cut(run[-1], m)):
            runs.append(run)
            run = []
        run.append(m)
    if run:
        runs.append(run)
    return runs


def motion(cur, prev, preset):
    """Per low-res pixel: (previous uv - current uv) and on-screen flag. Mirrors motion.fsh."""
    p = cur["presets"][preset]
    w, h = p["width"], p["height"]
    jx, jy = p["jitter"]
    depth = load_depth(cur, preset)

    # Closest (largest, reversed depth) value in each 3x3 neighbourhood.
    padded = np.pad(depth, 1, mode="edge")
    closest = np.max([padded[dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3)], axis=0)

    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    u = (xs + 0.5 - jx) / w
    v = (ys + 0.5 - jy) / h
    ndc_z = closest if cur["zZeroToOne"] else closest * 2.0 - 1.0
    clip = np.stack([u * 2 - 1, v * 2 - 1, ndc_z, np.ones_like(u)], axis=-1)  # (h, w, 4)

    inv = np.linalg.inv(cur["viewProj"])
    world = clip @ inv.T
    delta = cur["camera"] - prev["camera"]
    shifted = world.copy()
    shifted[..., :3] += delta * world[..., 3:4]  # homogeneous: sky (w = 0) only rotates
    prev_clip = shifted @ prev["viewProj"].T
    pw = prev_clip[..., 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        pu = prev_clip[..., 0] / pw * 0.5 + 0.5
        pv = prev_clip[..., 1] / pw * 0.5 + 0.5
    on_screen = (pw > 0) & (pu >= 0) & (pu <= 1) & (pv >= 0) & (pv <= 1)
    return np.stack([pu - u, pv - v], axis=-1), on_screen


def bilinear(img, u, v):
    """Sample (H, W, C) at normalized coords u, v (bottom-up), clamped to edge."""
    h, w = img.shape[:2]
    x = np.clip(u * w - 0.5, 0, w - 1)
    y = np.clip(v * h - 0.5, 0, h - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    top = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
    bottom = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
    return top * (1 - fy) + bottom * fy


def warp_previous_target(cur, prev, preset):
    """Previous target image moved to where each pixel is now, like the temporal pass does."""
    mv, on_screen = motion(cur, prev, preset)
    p = cur["presets"][preset]
    lw, lh = p["width"], p["height"]
    jx, jy = p["jitter"]
    tw, th = cur["target"]["width"], cur["target"]["height"]
    xs, ys = np.meshgrid(np.arange(tw), np.arange(th))
    tu, tv = (xs + 0.5) / tw, (ys + 0.5) / th
    nx = np.clip(np.floor(tu * lw + jx).astype(int), 0, lw - 1)
    ny = np.clip(np.floor(tv * lh + jy).astype(int), 0, lh - 1)
    pu = tu + mv[ny, nx, 0]
    pv = tv + mv[ny, nx, 1]
    valid = on_screen[ny, nx]
    return bilinear(load_image(prev, "target"), pu, pv), valid
