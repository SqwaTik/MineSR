# MineSR

Temporal upscaler for Minecraft **26.2** (Fabric), built toward an FSR 4-style neural upscaler.
Client-only: works in singleplayer and on any server, Hypixel included.

The world is drawn at a lower resolution with a different sub-pixel camera offset each frame.
A temporal pass rebuilds full resolution from the current frame plus the previous result, moved
to where each pixel is now. The hand and GUI are drawn afterwards at full resolution, so text
and the crosshair stay sharp.

**Upscaling, not frame generation:** the game draws fewer pixels and MineSR rebuilds the full
image. Frame generation (inserting extra frames, like FSR 3 / DLSS Frame Generation) is not
implemented yet.

## Install

1. Minecraft Java **26.2** with [Fabric Loader](https://fabricmc.net/use/) and Fabric API.
2. Build with JDK 25: `gradlew build`, then copy `build/libs/minesr-<version>.jar` to `.minecraft/mods/`.
3. In game: `/minesr` shows the status. Works in singleplayer and on servers (client-only).

Best in GPU-heavy situations (improved transparency, high resolution, weaker GPUs). With
default graphics Minecraft is usually limited by the CPU, and upscaling costs FPS there.

## Status: stage 1 of 3

1. **Rendering scaffold + hand-tuned temporal upscaler**: done, optimized (v2 pass).
2. **Training data capture**: done (`/minesr capture`), verified with `training/check_capture.py`.
3. **Neural network** replacing the hand-tuned rules: trained and running in game (`temporal_nn.fsh`).

Verified with an automated in-game test (OpenGL, NVIDIA): no geometry offsets, sky and fog
correct, no ghosting while turning, leaf texture detail close to native in quality and
performance modes. Not yet verified: Vulkan backend, Fabulous graphics, shader packs.

## Measured results

RTX 3050 laptop, OpenGL, 1600x900, render distance 12. Median of 10 interleaved cycles
(every mode measured once per cycle, so GPU clock swings hit all modes alike):

| mode | default graphics | improved transparency |
|---|---|---|
| off (vanilla) | 454 FPS | 271 FPS |
| quality, old (v1) pass | 389 | 251 |
| quality | 427 (-6%) | 316 (+17%) |
| balanced | 450 (-1%) | 359 (+32%) |
| performance | 475 (+5%) | 489 (+80%) |

With default graphics the GPU spends only ~0.6 ms per frame on the world; the game is
limited by the CPU, so rendering fewer pixels can't help much. Upscaling pays off when the
GPU is the limit: improved transparency, higher resolutions, weaker GPUs.

The v2 pass (motion computed once per low-res pixel, 5-tap history) costs 0.48 ms against
1.00 ms for v1 in the heavy scene. `/minesr` shows live GPU time on your own setup.

## Recording training data

`/minesr capture start`, play normally, `/minesr capture stop`. Files go to
`.minecraft/minesr-captures/session-<date>/`.

While capturing, the world renders at 1920x1080 (expect lower FPS) and you see a downscaled
copy. Every 10 seconds a burst of 32 consecutive frames is saved, about 1.8 MB per frame.
A session stops saving at 4 GB, which is about 11 minutes of play at default settings
(`captureBurstFrames`, `captureBurstIntervalSeconds`, `captureMaxMegabytes` in `config/minesr.json`).

Per frame:
- `target.png` (960x540): the answer, 4 samples per pixel
- `quality.png` + `quality-depth.f32.gz` (640x360) and `performance.*` (480x270): the low-res inputs
  the upscaler sees in those modes, each cut out at a known sub-pixel jitter
- `.json`: camera position, view-projection matrix, jitter, sizes

What to record: variety beats hours. Different biomes, day and night, caves, water, the Nether
and the End; walking, sprinting, flying, fast turns; mobs, players, particles, rain. The hand
and GUI are never captured.

### Automatic recording (bot)

`MINESR_TEST=dataset gradlew runClientGameTest` launches a dev client that records a dataset by
itself: 14 scenes (8 far-apart overworld spots, 3 Nether, 3 around the End island) with varied
time, weather and mobs, walking, sprint-jumping, flying, looking around, fast turns and strafing.
It turns away from walls, auto-jumps steps, and climbs before flying. Output goes to
`minesr/captures/` (outside the test folder, which Loom wipes before every test run).
`MINESR_DIMENSION=the_end` (or a comma list) re-records only some dimensions.

Check a session before training on it:

```
cd training
python check_capture.py <session dir>
```

`python dataset_summary.py ../captures` checks every session and writes a contact sheet.
`check_capture.py` rebuilds motion from depth and camera data the way the game does, warps each frame onto the
next, and reports whether that lines up. On the test capture, warping cut the error to 27% of
unwarped (about 1.3 out of 255), leaving only object edges and newly revealed areas.

## The neural upscaler

Following FSR 4's idea, a small trained network makes the per-pixel decisions the hand-tuned
pass makes with fixed rules: how sharp to rebuild the current frame, how much history to keep,
how hard to clamp it, and a learned memory value carried to the next frame. It is a per-pixel
MLP (17 inputs -> H -> H -> 4) evaluated in the fragment shader, so it runs on OpenGL and
Vulkan alike. The mod ships a trained network; `config/minesr/upscaler_net.json` overrides it.

Training (`training/`):

```
pip install -r requirements.txt            # CUDA build of PyTorch recommended
python prepare_dataset.py ../captures ../cache
python train_upscaler.py --cache ../cache --out out [--hidden 16]
python export_parity.py out/upscaler_net.pt out/parity.json
```

`upscaler_torch.py` is the temporal pass in PyTorch, line for line like the shader, so the
network trains in exactly the conditions it runs in, unrolled over 8-frame sequences. The loss
is per-frame accuracy plus a flicker term. Three scenes (overworld, Nether, End) never seen in
training are used for validation.

Held-out results. The shipped network is H = 8 (252 parameters): same quality as H = 16 at
2.8x fewer multiply-adds per pixel.

| | quality: accuracy / flicker | performance: accuracy / flicker |
|---|---|---|
| bilinear | 37.03 dB / 2.52 | 35.36 dB / 3.15 |
| spatial only (no history) | 38.66 dB / 2.02 | 36.61 dB / 2.64 |
| hand-tuned | 39.10 dB / 1.86 | 37.69 dB / 2.19 |
| neural, H = 16 | 40.25 dB / 1.52 | 38.57 dB / 1.86 |
| neural, H = 8 (shipped) | **40.23 dB / 1.54** | **38.54 dB / 1.90** |

Current design (network format v2): the network runs once per low-res pixel (`decide.fsh`) and
the full-resolution pass applies its decisions (`temporal_nn.fsh`). The game jitters on the same
sub-pixel grid the training data uses, and training judges flicker after the sharpening pass.

Flicker with a still camera, in game (share of pixels changing by more than 8/255 per frame):

| mode | hand-tuned | neural |
|---|---|---|
| quality | 0.20% | 0.27% |
| performance | 0.24% | 0.39% |

(Vanilla: 0.01%. The first neural version flickered on 3.4-4.4% of pixels: it had been trained
almost only on moving footage, without sharpening in the loss.)

FPS at 1600x900, render distance 12, GPU-heavy scene (improved transparency), same run:

| mode | FPS without the mod | FPS with the mod | change |
|---|---|---|---|
| quality, hand-tuned | 77 | 80 | +4% |
| quality, neural | 77 | 71 | -8% |
| balanced, neural | 77 | 81 | +5% |
| performance, hand-tuned | 77 | 97 | +25% |
| performance, neural | 77 | 88 | +14% |

With default graphics the game is CPU-limited and the mod costs FPS in every mode.

`/minesr neural on|off` switches live.

## Commands

| Command | |
|---|---|
| `/minesr` | Status, resolutions, live GPU time of world and upscale |
| `/minesr quality` / `balanced` / `performance` | Render at 67% / 59% / 50% |
| `/minesr native` | Full resolution, temporal anti-aliasing only |
| `/minesr off` / `on` | |
| `/minesr sharpness <0-1>` | Sharpening after upscale (default 0.5) |
| `/minesr capture start` / `stop` | Record training data (see above) |
| `/minesr capture` | Capture progress: frames, MB, skipped frames |
| `/minesr retry` | Re-enable after a shader error (e.g. after fixing a resource pack) |

## How it works

- `GameRendererMixin`: jitters the world projection, swaps in the low-res target for the world
  pass only, runs the upscale right after it.
- `SkyRendererMixin`: the sky renderer stores its target once; this makes it follow the swap.
- `ShaderManagerMixin` + `GlobalSettingsUniformMixin`: texture LOD bias. At 67% the GPU would
  pick blurrier mipmaps and the detail would never reach the upscaler. Terrain derivatives are
  scaled by the render scale through one extra float in the `Globals` uniform block. Edits are
  exact-text matches; if a resource pack changed those shaders, the bias is skipped safely.
- `temporal.fsh`: re-projects history using depth and camera motion, clips it to colors
  plausible in the current neighborhood, blends with a filter that narrows as history builds up.
  Moving mobs and players have no motion vectors yet; the clipping limits their ghosting.
- `sharpen.fsh`: contrast-adaptive sharpening into the game's main image.
- `GpuTimer`: timestamp queries around the world and the upscale, read back a few frames later.

## Testing

- `gradlew build`: unit tests (jitter sequence, presets, shader patch).
- `gradlew runClientGameTest`: launches a dev client, creates a world, screenshots every mode
  into `build/run/clientGameTest/screenshots`, and benchmarks every mode at 1600x900 in two graphics settings (results in
  `build/run/clientGameTest/minesr-benchmark.json`).

## Next steps

- Train the network on that data; run it as fragment-shader passes on OpenGL and Vulkan alike.
- Motion vectors for entities, to stop ghosting on moving players and mobs.
