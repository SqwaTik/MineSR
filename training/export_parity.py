"""Writes a test fixture: random network inputs and what PyTorch computes for them.

    python export_parity.py out/upscaler_net.pt out/parity.json

NeuralWeightsTest (Java) packs upscaler_net.json the way the shader reads it, evaluates the
same inputs, and must match: that pins the weight layout and indexing used by temporal_nn.fsh.
"""
import json
import sys

import torch

from upscaler_torch import N_FEATURES, NeuralDecider


def main():
    state = torch.load(sys.argv[1], map_location="cpu")
    model = NeuralDecider(hidden=state["net.0.weight"].shape[0])
    model.load_state_dict(state)
    torch.manual_seed(1)
    x = torch.rand(16, N_FEATURES) * 2 - 0.5
    with torch.no_grad():
        raw = model.net(x.view(16, N_FEATURES, 1, 1)).view(16, -1)
    json.dump({"inputs": x.tolist(), "outputs": raw.tolist()}, open(sys.argv[2], "w"))
    print(f"wrote {len(x)} cases to {sys.argv[2]}")


if __name__ == "__main__":
    main()
