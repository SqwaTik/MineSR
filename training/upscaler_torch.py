"""The temporal upscale pass in PyTorch, mirroring temporal.fsh line for line.

Everything works in full-frame coordinates (uv in [0, 1], bottom row first) so that random
crops behave like the real frame: a crop is a window (tx, ty) into a target of size
(W, H) and a low-res window (lx, ly) into the low-res frame of size (w, h).

Two ways to make the per-pixel decisions:
  * hand_tuned: today's shader rules (the baseline to beat)
  * NeuralDecider: a tiny per-pixel network (what the shader will run next)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_FEATURES = 17
N_OUTPUTS = 4  # reconstruction sharpness, blend weight, rectify amount, memory


class Frame:
    """One frame's inputs for a batch of crops (all crops share sizes, not offsets)."""

    def __init__(self, low, motion, onscreen, jitter, low_origin, target_origin, low_full, target_full):
        self.low = low                    # (B, 3, hl, wl) float in [0, 1]
        self.motion = motion              # (B, 2, hl, wl) uv units
        self.onscreen = onscreen          # (B, 1, hl, wl) 0/1
        self.jitter = jitter              # (B, 2) low-res pixels
        self.low_origin = low_origin      # (B, 2) int: crop corner in the low-res frame
        self.target_origin = target_origin  # (B, 2) int: crop corner in the target frame
        self.low_full = low_full          # (w, h) of the whole low-res frame
        self.target_full = target_full    # (W, H) of the whole target frame


def _gather(img, ix, iy):
    """img (B, C, h, w); ix, iy (B, H, W) long, already clamped -> (B, C, H, W)."""
    b, c, h, w = img.shape
    flat = (iy * w + ix).view(b, 1, -1).expand(b, c, -1)
    return torch.gather(img.reshape(b, c, h * w), 2, flat).view(b, c, *ix.shape[1:])


def _sample(img, px, py):
    """Bilinear, clamp-to-edge, like texture() with a LINEAR sampler.
    px, py: (B, H, W) continuous texel coordinates in img (texel i spans [i, i+1))."""
    h, w = img.shape[2:]
    grid = torch.stack([px / w * 2 - 1, py / h * 2 - 1], dim=-1)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=False)


def step(frame, history, decide, out_size):
    """One frame of the temporal pass.

    history: (B, 4, H, W) previous output (rgb + memory) for this crop, or None on the first frame.
    decide: function(features dict) -> dict with sharpness, alpha, beta, memory (each (B,1,H,W)).
    Returns (output rgb (B,3,H,W), new history (B,4,H,W)).
    """
    th, tw = out_size
    dev = frame.low.device
    b = frame.low.shape[0]
    hl, wl = frame.low.shape[2:]
    w_full, h_full = frame.low_full
    tw_full, th_full = frame.target_full

    ys, xs = torch.meshgrid(torch.arange(th, device=dev), torch.arange(tw, device=dev), indexing="ij")
    tx = frame.target_origin[:, 0].view(b, 1, 1).float()
    ty = frame.target_origin[:, 1].view(b, 1, 1).float()
    u = (xs[None] + tx + 0.5) / tw_full  # texCoord
    v = (ys[None] + ty + 0.5) / th_full
    jx = frame.jitter[:, 0].view(b, 1, 1)
    jy = frame.jitter[:, 1].view(b, 1, 1)
    lx0 = frame.low_origin[:, 0].view(b, 1, 1).float()
    ly0 = frame.low_origin[:, 1].view(b, 1, 1).float()

    # x = texCoord * lowSize; nearest = floor(x + Jitter)  (full-frame low-res pixel units)
    x, y = u * w_full, v * h_full
    nx, ny = torch.floor(x + jx), torch.floor(y + jy)

    colors, dist2 = [], []
    for i in range(9):
        dx, dy = i % 3 - 1, i // 3 - 1
        txl, tyl = nx + dx, ny + dy
        ix = (txl - lx0).clamp(0, wl - 1).long()
        iy = (tyl - ly0).clamp(0, hl - 1).long()
        colors.append(_gather(frame.low, ix, iy))
        ddx, ddy = txl + 0.5 - jx - x, tyl + 0.5 - jy - y
        dist2.append((ddx * ddx + ddy * ddy).unsqueeze(1))
    colors = torch.stack(colors, 0)  # (9, B, 3, H, W)
    dist2 = torch.stack(dist2, 0)    # (9, B, 1, H, W)
    mean = colors.mean(0)
    sigma = (colors.pow(2).mean(0) - mean.pow(2)).clamp_min(0).sqrt()
    box_min, box_max = colors.min(0).values, colors.max(0).values

    cx = (nx - lx0).clamp(0, wl - 1).long()
    cy = (ny - ly0).clamp(0, hl - 1).long()
    mv = _gather(frame.motion, cx, cy)
    on = _gather(frame.onscreen, cx, cy)[:, 0] > 0.5
    pu, pv = u + mv[:, 0], v + mv[:, 1]
    on = on & (pu >= 0) & (pu <= 1) & (pv >= 0) & (pv <= 1)

    if history is not None:
        # Catmull-Rom from 5 bilinear taps (sampleHistory in temporal.fsh), in target pixels.
        pos_x, pos_y = pu * tw_full, pv * th_full
        cxp, cyp = torch.floor(pos_x - 0.5) + 0.5, torch.floor(pos_y - 0.5) + 0.5
        fx, fy = pos_x - cxp, pos_y - cyp

        def weights(f):
            w0 = f * (-0.5 + f * (1.0 - 0.5 * f))
            w1 = 1.0 + f * f * (-2.5 + 1.5 * f)
            w2 = f * (0.5 + f * (2.0 - 1.5 * f))
            w3 = f * f * (-0.5 + 0.5 * f)
            return w0, w1 + w2, w2 / (w1 + w2), w3

        w0x, w12x, o12x, w3x = weights(fx)
        w0y, w12y, o12y, w3y = weights(fy)
        # tap positions (target pixel units) -> crop texel coordinates
        t0x, t3x, t12x = cxp - 1.0 - tx, cxp + 2.0 - tx, cxp + o12x - tx
        t0y, t3y, t12y = cyp - 1.0 - ty, cyp + 2.0 - ty, cyp + o12y - ty
        middle = _sample(history, t12x, t12y)
        wc, wl_, wr = w12x * w12y, w0x * w12y, w3x * w12y
        wb, wt = w12x * w0y, w12x * w3y
        hist = (middle[:, :3] * wc.unsqueeze(1)
                + _sample(history[:, :3], t0x, t12y) * wl_.unsqueeze(1)
                + _sample(history[:, :3], t3x, t12y) * wr.unsqueeze(1)
                + _sample(history[:, :3], t12x, t0y) * wb.unsqueeze(1)
                + _sample(history[:, :3], t12x, t3y) * wt.unsqueeze(1))
        hist = (hist / (wc + wl_ + wr + wb + wt).unsqueeze(1)).clamp_min(0)
        memory = middle[:, 3:4]
        # A crop has no history outside itself; treat that like off-screen.
        inside = (t12x >= 0) & (t12x <= tw) & (t12y >= 0) & (t12y <= th)
        on = on & inside
    else:
        hist = torch.zeros_like(mean)
        memory = torch.zeros_like(mean[:, :1])
        on = torch.zeros_like(on)

    clip_min = torch.maximum(box_min, mean - 1.25 * sigma)
    clip_max = torch.minimum(box_max, mean + 1.25 * sigma)
    clipped = torch.minimum(torch.maximum(hist, clip_min), clip_max)
    have = on.unsqueeze(1).float()

    feats = {
        "_nearest": (cx, cy),
        "mean": mean, "sigma": sigma, "hist": hist, "clipped": clipped, "memory": memory * have,
        "have": have, "dist2_min": dist2.min(0).values, "motion_px": torch.sqrt(
            (mv[:, 0] * tw_full) ** 2 + (mv[:, 1] * th_full) ** 2).unsqueeze(1),
    }
    d = decide(feats)

    weights9 = torch.exp(-d["sharpness"].unsqueeze(0) * dist2)
    current = (colors * weights9).sum(0) / weights9.sum(0).clamp_min(1e-5)
    nearest_weight = weights9.max(0).values
    d = d["finish"](d, nearest_weight) if "finish" in d else d

    rectified = hist + (clipped - hist) * d["beta"]
    out = rectified + (current - rectified) * d["alpha"]
    out = have * out + (1 - have) * current
    new_memory = have * d["memory"] + (1 - have) * d.get("memory_fresh", d["memory"])
    return out, torch.cat([out, new_memory], 1)


# ---------------------------------------------------------------- hand-tuned (today's shader)

def hand_tuned(max_history=10.0):
    def decide(f):
        n_prev = f["memory"]
        sharp = 2.3 + (9.0 - 2.3) * (n_prev / 4.0).clamp(0, 1)
        change = (f["clipped"] - f["hist"]).norm(dim=1, keepdim=True) / (f["mean"].norm(dim=1, keepdim=True) + 0.05)
        n_hist = n_prev * (1.0 - change * 4.0).clamp(0.1, 1.0)

        def finish(d, nearest_weight):
            n = torch.minimum(n_hist + nearest_weight, torch.full_like(n_hist, max_history))
            alpha = nearest_weight / n.clamp_min(1e-4)
            return {"alpha": alpha, "beta": torch.ones_like(alpha), "memory": n, "memory_fresh": nearest_weight}

        return {"sharpness": sharp, "finish": finish}
    return decide


def spatial_only(sharpness=4.0):
    """No history: every frame rebuilt from its own low-res samples. Sharp per frame, but shimmers."""
    def decide(f):
        one = torch.ones_like(f["have"])
        return {"sharpness": one * sharpness, "alpha": one, "beta": one, "memory": one * 0}
    return decide


# ---------------------------------------------------------------- neural

class NeuralDecider(nn.Module):
    """Per-pixel MLP (1x1 convolutions): 17 features -> 16 -> 16 -> 4 decisions."""

    def __init__(self, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(N_FEATURES, hidden, 1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 1), nn.ReLU(),
            nn.Conv2d(hidden, N_OUTPUTS, 1))

    @staticmethod
    def features(f):
        """Order matters: the shader builds the same vector."""
        m = f["mean"]
        return torch.cat([
            m, f["sigma"], f["hist"] - m, f["clipped"] - m, f["memory"], f["have"],
            f["dist2_min"], torch.log1p(f["motion_px"]),
            (f["clipped"] - f["hist"]).abs().sum(1, keepdim=True),
        ], 1)

    def forward(self, f):
        o = self.net(self.features(f))
        return {
            "sharpness": 1.0 + 11.0 * torch.sigmoid(o[:, 0:1]),
            "alpha": torch.sigmoid(o[:, 1:2]),
            "beta": torch.sigmoid(o[:, 2:3]),
            "memory": torch.sigmoid(o[:, 3:4]),
        }


# ---------------------------------------------------------------- low-res decisions (v3)

def _neighborhood(img):
    """3x3 stats of img (B, 3, h, w) with edge clamping -> mean, sigma, min, max."""
    padded = F.pad(img, (1, 1, 1, 1), mode="replicate")
    h, w = img.shape[2:]
    taps = torch.stack([padded[:, :, dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3)], 0)
    mean = taps.mean(0)
    sigma = (taps.pow(2).mean(0) - mean.pow(2)).clamp_min(0).sqrt()
    return mean, sigma, taps.min(0).values, taps.max(0).values


def lowres_decisions(frame, history, model, out_size):
    """Runs the network once per LOW-RES pixel (decide.fsh). Returns (B, 4, h, w):
    sharpness, alpha, beta, memory."""
    th, tw = out_size
    b = frame.low.shape[0]
    hl, wl = frame.low.shape[2:]
    w_full, h_full = frame.low_full
    tw_full, th_full = frame.target_full
    dev = frame.low.device
    ys, xs = torch.meshgrid(torch.arange(hl, device=dev), torch.arange(wl, device=dev), indexing="ij")
    lx0 = frame.low_origin[:, 0].view(b, 1, 1).float()
    ly0 = frame.low_origin[:, 1].view(b, 1, 1).float()
    tx = frame.target_origin[:, 0].view(b, 1, 1).float()
    ty = frame.target_origin[:, 1].view(b, 1, 1).float()
    jx = frame.jitter[:, 0].view(b, 1, 1)
    jy = frame.jitter[:, 1].view(b, 1, 1)
    # Where this low-res sample actually is (its center minus the jitter), in full-frame uv.
    u = (xs[None] + lx0 + 0.5 - jx) / w_full
    v = (ys[None] + ly0 + 0.5 - jy) / h_full
    mean, sigma, box_min, box_max = _neighborhood(frame.low)
    mv = frame.motion
    on = frame.onscreen[:, 0] > 0.5
    pu, pv = u + mv[:, 0], v + mv[:, 1]
    on = on & (pu >= 0) & (pu <= 1) & (pv >= 0) & (pv <= 1)
    if history is not None:
        px, py = pu * tw_full - tx, pv * th_full - ty   # continuous texel coords in the crop
        s = _sample(history, px, py)
        hist, memory = s[:, :3], s[:, 3:4]
        on = on & (px >= 0) & (px <= tw) & (py >= 0) & (py <= th)
    else:
        hist = torch.zeros_like(mean)
        memory = torch.zeros_like(mean[:, :1])
        on = torch.zeros_like(on)
    clip_min = torch.maximum(box_min, mean - 1.25 * sigma)
    clip_max = torch.minimum(box_max, mean + 1.25 * sigma)
    clipped = torch.minimum(torch.maximum(hist, clip_min), clip_max)
    have = on.unsqueeze(1).float()
    feats = {
        "mean": mean, "sigma": sigma, "hist": hist, "clipped": clipped, "memory": memory * have, "have": have,
        "dist2_min": (jx * jx + jy * jy).unsqueeze(1).expand(b, 1, hl, wl),
        "motion_px": torch.sqrt((mv[:, 0] * tw_full) ** 2 + (mv[:, 1] * th_full) ** 2).unsqueeze(1),
    }
    d = model(feats)
    return torch.cat([d["sharpness"], d["alpha"], d["beta"], d["memory"]], 1)


def step_lowres(frame, history, model, out_size):
    """One frame: decisions at low resolution, then the full-resolution pass applies them
    (temporal_nn.fsh v3). Same history handling and reconstruction as step()."""
    decisions = lowres_decisions(frame, history, model, out_size)

    def decide(f):
        # Look up the decisions of the nearest low-res pixel (the same texel step() uses for motion).
        nearest = f["_nearest"]
        dd = _gather(decisions, nearest[0], nearest[1])

        def finish(d, nearest_weight):
            # Per output pixel: a sample far from this pixel counts for less.
            return {"alpha": dd[:, 1:2] * nearest_weight, "beta": dd[:, 2:3], "memory": dd[:, 3:4]}

        return {"sharpness": dd[:, 0:1], "finish": finish}

    return step(frame, history, decide, out_size)


def sharpen(img, strength=0.5):
    """Contrast-adaptive sharpening, exactly like sharpen.fsh (the last pass before the screen).
    img: (..., 3, H, W). Flicker is judged after this, because that is what the player sees."""
    shape = img.shape
    x = img.reshape(-1, 3, shape[-2], shape[-1])
    p = F.pad(x, (1, 1, 1, 1), mode="replicate")
    c = x
    n, s = p[:, :, 2:, 1:-1], p[:, :, :-2, 1:-1]
    e, w = p[:, :, 1:-1, 2:], p[:, :, 1:-1, :-2]
    mn = torch.minimum(c, torch.minimum(torch.minimum(n, s), torch.minimum(e, w)))
    mx = torch.maximum(c, torch.maximum(torch.maximum(n, s), torch.maximum(e, w)))
    amp = (torch.minimum(mn, 1.0 - mx) / mx.clamp_min(1e-4)).clamp(0, 1).sqrt()
    k = -amp / (8.0 + (5.0 - 8.0) * strength)
    out = (c + (n + s + e + w) * k) / (1.0 + 4.0 * k)
    return out.clamp(0, 1).reshape(shape)
