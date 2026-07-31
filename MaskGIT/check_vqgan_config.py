"""
Run this from inside your MaskGIT repo folder:

    cd ./MaskGIT
    python3 check_vqgan_config.py

It answers three things we need before fixing transformer.py:

1. What shape does your trained VQGAN's encoder actually produce for a
   256x256 image? (this tells us the real num_image_tokens and the
   downsample_factor sketch_encoder.py must match)
2. What's the codebook embedding dimension? (needed for indices_to_image's
   reshape, currently hardcoded wrong as 32)
3. What checkpoint files do you actually have saved, and in what format?
   (state_dict vs. full model, so we load it correctly)
"""

import os
import torch

import sys
sys.path.insert(0, os.getcwd())

print("=" * 60)
print("1. Checkpoint files available:")
print("=" * 60)
for root, dirs, files in os.walk("./checkpoints"):
    for f in files:
        full = os.path.join(root, f)
        size_mb = os.path.getsize(full) / (1024 * 1024)
        print(f"  {full}  ({size_mb:.1f} MB)")

print()
print("=" * 60)
print("2. Inspecting a checkpoint's raw structure (no model class needed):")
print("=" * 60)
# Just peek at the raw state dict keys/shapes without needing to correctly
# construct the VQGAN class first -- this avoids guessing its __init__ args.
ckpt_dir = "./checkpoints"
candidates = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")] if os.path.isdir(ckpt_dir) else []
if not candidates:
    print("  No .pt files found in ./checkpoints -- update the path if yours saved elsewhere.")
else:
    latest = sorted(candidates)[-1]
    path = os.path.join(ckpt_dir, latest)
    print(f"  Loading {path} ...")
    state = torch.load(path, map_location="cpu")

    if isinstance(state, dict) and not any(hasattr(v, "shape") for v in state.values()):
        print("  Top-level keys (looks like it might be a nested checkpoint):")
        for k in state.keys():
            print(f"    {k}")
    else:
        print(f"  Looks like a flat state_dict with {len(state)} tensors. Sample keys:")
        for i, (k, v) in enumerate(state.items()):
            if i > 15:
                print("    ...")
                break
            shape = tuple(v.shape) if hasattr(v, "shape") else type(v)
            print(f"    {k}: {shape}")

        # try to find the codebook embedding specifically
        for k, v in state.items():
            if "codebook" in k.lower() and "embed" in k.lower() and hasattr(v, "shape"):
                print()
                print(f"  >>> Found codebook embedding: {k} with shape {tuple(v.shape)}")
                print(f"      (num_codebook_vectors={v.shape[0]}, embedding_dim={v.shape[1]})")

print()
print("=" * 60)
print("3. Actual encoder output shape for a 256x256 image:")
print("=" * 60)
try:
    from vqgan import VQGAN
    import argparse
    # minimal args namespace matching training_vqgan.py's known flags
    args = argparse.Namespace(
        latent_dim=256, image_size=256, num_codebook_vectors=8192,
        beta=0.25, image_channels=3, device="cpu"
    )
    model = VQGAN(args)
    if candidates:
        state = torch.load(os.path.join(ckpt_dir, sorted(candidates)[-1]), map_location="cpu")
        try:
            model.load_state_dict(state)
            print("  Loaded checkpoint successfully into VQGAN class.")
        except Exception as e:
            print(f"  Could not load checkpoint directly into VQGAN class: {e}")
            print("  (structure info from step 2 above is still useful)")

    model.eval()
    dummy = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        quant_z, _, (_, _, indices) = model.encode(dummy)
    print(f"  quant_z shape: {tuple(quant_z.shape)}")
    print(f"  indices shape: {tuple(indices.shape)}")
    grid = quant_z.shape[-1]
    print(f"  ==> latent grid appears to be {grid}x{grid} = {grid*grid} tokens")
    print(f"  ==> embedding dim appears to be {quant_z.shape[1]}")
except Exception as e:
    print(f"  Could not run this check automatically: {e}")
    print("  The info from steps 1-2 above should still be enough to proceed.")