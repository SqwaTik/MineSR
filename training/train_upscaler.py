"""Trains the neural decider of the temporal upscaler.

    python train_upscaler.py --cache ../cache --out out

The network runs inside the same temporal pass as the game (upscaler_torch.py), unrolled over
sequences so it learns how its output feeds into later frames. Validation uses scenes it never
trained on and compares: bilinear upscaling, spatial-only reconstruction (no history), today's
hand-tuned rules, and the network, on accuracy (PSNR) and steadiness (flicker).
Writes out/upscaler_net.json (weights for the shader) and out/report.json.
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from data_torch import Cache, full_sequence, random_batch
from upscaler_torch import NeuralDecider, hand_tuned, sharpen, spatial_only, step, step_lowres

# One overworld, one Nether and one End scene stay out of training.
VAL_SCENES = {"session-20261026-070729", "session-20261026-071412", "session-20261026-071546"}
# Still / slow-camera scenes (overworld, Nether): where flicker is most visible. Reported separately.
CALM_VAL_SCENES = {"session-20261026-180803", "session-20261026-181331"}
BORDER = 16  # target pixels at crop edges lack history from outside the crop; not scored


def run_sequence(frames, decide, size, step_fn=step):
    outs, hist = [], None
    for f in frames:
        out, hist = step_fn(f, hist, decide, size)
        outs.append(out)
    return torch.stack(outs)  # T,B,3,H,W


def psnr(a, b):
    return (10 * torch.log10(1.0 / F.mse_loss(a.clamp(0, 1), b).clamp_min(1e-10))).item()


def flicker(outs, tgt):
    """Frame-to-frame change the reference doesn't have (shimmer, crawling edges), x255,
    measured after the game's sharpening pass: that is what reaches the screen."""
    o, t = sharpen(outs), sharpen(tgt)
    return ((o[1:] - o[:-1]) - (t[1:] - t[:-1])).abs().mean().item() * 255


@torch.no_grad()
def evaluate(cache, seqs, model, device, length=24):
    """Mean PSNR (higher = closer to the reference) and flicker (lower = steadier) per method."""
    results = {}
    for preset in ["quality", "performance"]:
        methods = {"spatial only": spatial_only(), "hand-tuned": hand_tuned(), "neural": model}
        psnr_sum = {"bilinear": 0.0, **{k: 0.0 for k in methods}}
        flicker_sum = {k: 0.0 for k in psnr_sum}
        frames_n = seq_n = 0
        for seq in seqs:
            frames, tgt = full_sequence(cache, seq, preset, length, device)
            size = tgt.shape[-2:]
            outs = {"bilinear": torch.stack([F.interpolate(f.low, size=size, mode="bilinear", align_corners=False)
                                             for f in frames]).clamp(0, 1)}
            for k, decide in methods.items():
                outs[k] = run_sequence(frames, decide, size, step_lowres if k == "neural" else step).clamp(0, 1)
            for k, o in outs.items():
                psnr_sum[k] += sum(psnr(o[t], tgt[t]) for t in range(len(frames)))
                flicker_sum[k] += flicker(o, tgt)
            frames_n += len(frames)
            seq_n += 1
        results[preset] = {k: {"psnr": psnr_sum[k] / frames_n, "flicker": flicker_sum[k] / seq_n} for k in psnr_sum}
    return results


def describe(res):
    lines = []
    for preset, r in res.items():
        lines.append(f"  {preset:11s} " + "  ".join(
            f"{k}: {v['psnr']:.2f} dB / flicker {v['flicker']:.2f}" for k, v in r.items()))
    return "\n".join(lines)


def export(model, path):
    layers = [m for m in model.net if isinstance(m, torch.nn.Conv2d)]
    # Version 2: decisions are made per low-res pixel (decide.fsh), applied at full resolution.
    doc = {"version": 2, "features": 17, "outputs": 4, "hidden": layers[0].out_channels,
           "layers": [{"weight": l.weight.detach().cpu()[:, :, 0, 0].tolist(), "bias": l.bias.detach().cpu().tolist()}
                      for l in layers]}
    Path(path).write_text(json.dumps(doc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="../cache")
    ap.add_argument("--out", default="out")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--length", type=int, default=16, help="frames per training sequence")
    ap.add_argument("--crop", type=int, default=144, help="target-resolution crop size")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--temporal", type=float, default=1.0, help="weight of the flicker term in the loss")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--hidden", type=int, default=16, help="units per hidden layer (multiple of 4)")
    ap.add_argument("--init", help="start from these weights (upscaler_net.pt) instead of random")
    args = ap.parse_args()

    torch.manual_seed(0)
    rng = random.Random(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Cache(args.cache)
    train, val_all = cache.split(VAL_SCENES | CALM_VAL_SCENES)
    val = [s for s in val_all if s["scene"] in VAL_SCENES]
    calm = [s for s in val_all if s["scene"] in CALM_VAL_SCENES]
    print(f"{len(train)} training sequences, {len(val)} moving + {len(calm)} calm validation sequences, on {device}")

    model = NeuralDecider(args.hidden).to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device))
        print(f"fine-tuning from {args.init}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)
    print(f"network: {sum(p.numel() for p in model.parameters())} parameters")
    print("before training, held-out moving scenes:\n" + describe(evaluate(cache, val, model, device)))
    print("before training, held-out calm scenes:\n" + describe(evaluate(cache, calm, model, device)))

    log = []
    start = time.time()
    for it in range(1, args.steps + 1):
        preset = "quality" if it % 2 else "performance"
        frames, tgt = random_batch(cache, train, preset, args.batch, args.length, args.crop, device, rng)
        outs = run_sequence(frames, model, tgt.shape[-2:], step_lowres)
        b = BORDER
        o, t = outs[..., b:-b, b:-b], tgt[..., b:-b, b:-b]
        # Per-frame accuracy plus steadiness: frame-to-frame change should match the reference's,
        # judged after sharpening, which amplifies small flicker into visible flicker.
        so, st = sharpen(o), sharpen(t)
        loss = (o - t).abs().mean() + args.temporal * ((so[1:] - so[:-1]) - (st[1:] - st[:-1])).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % 100 == 0:
            print(f"step {it:5d}  loss {loss.item():.4f}  ({(time.time() - start) / it:.2f} s/step)")
        if it % args.eval_every == 0 or it == args.steps:
            res = evaluate(cache, val, model, device)
            res_calm = evaluate(cache, calm, model, device)
            log.append({"step": it, "moving": res, "calm": res_calm})
            print(f"  held-out moving scenes at step {it}:\n" + describe(res))
            print(f"  held-out calm scenes at step {it}:\n" + describe(res_calm))
            export(model, out / "upscaler_net.json")
            torch.save(model.state_dict(), out / "upscaler_net.pt")
    (out / "report.json").write_text(json.dumps({"validation_scenes": sorted(VAL_SCENES),
                                                 "calm_validation_scenes": sorted(CALM_VAL_SCENES), "log": log}, indent=1))
    print(f"done in {(time.time() - start) / 60:.1f} min -> {out / 'upscaler_net.json'}")


if __name__ == "__main__":
    main()
