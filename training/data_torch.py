"""Loads the prepared arrays (prepare_dataset.py) as batches of sequence crops."""
import json
import random
from pathlib import Path

import numpy as np
import torch

from upscaler_torch import Frame


class Cache:
    def __init__(self, cache_dir):
        self.dir = Path(cache_dir)
        index = json.loads((self.dir / "index.json").read_text())
        self.sequences = index["sequences"]
        self._arrays = {}

    def array(self, name):
        if name not in self._arrays:
            self._arrays[name] = np.load(self.dir / f"{name}.npy", mmap_mode="r")
        return self._arrays[name]

    def split(self, val_scenes):
        train = [s for s in self.sequences if s["scene"] not in val_scenes]
        val = [s for s in self.sequences if s["scene"] in val_scenes]
        return train, val


def load_window(cache, seq, preset, t0, length, low_box, target_box, device):
    """Arrays for frames t0..t0+length-1 inside the given boxes -> list of Frame + targets."""
    name = seq["name"]
    lx, ly, lw, lh = low_box
    tx, ty, tw, th = target_box
    low = cache.array(f"{name}_{preset}_low")
    target = cache.array(f"{name}_target")
    sl = slice(t0, t0 + length)
    low_c = torch.from_numpy(np.ascontiguousarray(low[sl, ly:ly + lh, lx:lx + lw])).to(device)
    mot = torch.from_numpy(np.ascontiguousarray(cache.array(f"{name}_{preset}_motion")[sl, ly:ly + lh, lx:lx + lw])).to(device)
    on = torch.from_numpy(np.ascontiguousarray(cache.array(f"{name}_{preset}_onscreen")[sl, ly:ly + lh, lx:lx + lw])).to(device)
    jit = torch.from_numpy(np.ascontiguousarray(cache.array(f"{name}_{preset}_jitter")[sl])).to(device)
    tgt = torch.from_numpy(np.ascontiguousarray(target[sl, ty:ty + th, tx:tx + tw])).to(device)
    full_low = (low.shape[2], low.shape[1])
    full_target = (target.shape[2], target.shape[1])
    return low_c, mot, on, jit, tgt, full_low, full_target


def batch_frames(windows, low_origins, target_origins):
    """Stack per-sample windows into per-frame Frame objects (B along dim 0)."""
    low = torch.stack([w[0] for w in windows]).float().div(255).permute(1, 0, 4, 2, 3)   # T,B,3,h,w
    mot = torch.stack([w[1] for w in windows]).float().permute(1, 0, 4, 2, 3)            # T,B,2,h,w
    on = torch.stack([w[2] for w in windows]).float().unsqueeze(2).permute(1, 0, 2, 3, 4)  # T,B,1,h,w
    jit = torch.stack([w[3] for w in windows]).permute(1, 0, 2)                          # T,B,2
    tgt = torch.stack([w[4] for w in windows]).float().div(255).permute(1, 0, 4, 2, 3)   # T,B,3,H,W
    dev = low.device
    lo = torch.tensor(low_origins, device=dev)
    to = torch.tensor(target_origins, device=dev)
    full_low, full_target = windows[0][5], windows[0][6]
    frames = [Frame(low[t], mot[t], on[t], jit[t], lo, to, full_low, full_target) for t in range(low.shape[0])]
    return frames, tgt


def random_batch(cache, seqs, preset, batch, length, target_crop, device, rng=random):
    """Random sequence crops. The low-res crop is aligned so its target crop starts on a whole pixel."""
    windows, lo, to = [], [], []
    weights = [s["frames"] for s in seqs]
    while len(windows) < batch:
        seq = rng.choices(seqs, weights)[0]
        if seq["frames"] < length:
            continue
        low = cache.array(f"{seq['name']}_{preset}_low")
        tgt = cache.array(f"{seq['name']}_target")
        ratio = tgt.shape[2] / low.shape[2]   # 1.5 (quality) or 2 (performance)
        step = 2 if abs(ratio - 1.5) < 1e-6 else 1
        lc = int(round(target_crop / ratio))
        lx = rng.randrange(0, (low.shape[2] - lc) // step) * step
        ly = rng.randrange(0, (low.shape[1] - lc) // step) * step
        tx, ty = int(round(lx * ratio)), int(round(ly * ratio))
        t0 = rng.randrange(0, seq["frames"] - length + 1)
        windows.append(load_window(cache, seq, preset, t0, length, (lx, ly, lc, lc),
                                   (tx, ty, target_crop, target_crop), device))
        lo.append((lx, ly))
        to.append((tx, ty))
    return batch_frames(windows, lo, to)


def full_sequence(cache, seq, preset, length, device):
    """A whole-frame window from the start of a sequence (for evaluation)."""
    low = cache.array(f"{seq['name']}_{preset}_low")
    tgt = cache.array(f"{seq['name']}_target")
    length = min(length, seq["frames"])
    w = load_window(cache, seq, preset, 0, length, (0, 0, low.shape[2], low.shape[1]),
                    (0, 0, tgt.shape[2], tgt.shape[1]), device)
    return batch_frames([w], [(0, 0)], [(0, 0)])
