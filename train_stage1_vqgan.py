"""
Stage 1 launcher for the VQGAN tokenizer.

Rather than reimplementing dome272/MaskGIT-pytorch's training_vqgan.py loop
(which would risk drifting from their tested GAN/perceptual loss setup),
this script:

    1. Extracts the "target" (real image) half from every combined A|B file
       in your paired dataset into a flat folder of plain images.
    2. Launches their existing training_vqgan.py CLI unmodified, pointed at
       that folder — exactly what a "thin wrapper" should do.

Sketches are NOT used here at all. Stage 1 only ever sees target images.
"""

import argparse
import os
import subprocess
import sys

from PIL import Image


def extract_targets(source_dir, dest_dir, direction="AtoB", exts=(".png", ".jpg", ".jpeg")):
    os.makedirs(dest_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(source_dir) if f.lower().endswith(exts))
    if not files:
        raise RuntimeError(f"No images found in {source_dir}")

    for fname in files:
        path = os.path.join(source_dir, fname)
        combined = Image.open(path).convert("RGB")
        w, h = combined.size
        half_w = w // 2
        if half_w * 2 != w:
            raise ValueError(
                f"{path}: width {w} is not evenly divisible into two halves."
            )
        left = combined.crop((0, 0, half_w, h))
        right = combined.crop((half_w, 0, w, h))
        target_img = right if direction == "AtoB" else left

        out_path = os.path.join(dest_dir, fname)
        target_img.save(out_path)

    print(f"Extracted {len(files)} target images from {source_dir} -> {dest_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=str, default="./data/train",
                         help="Folder of combined A|B images (your paired dataset).")
    parser.add_argument("--targets-dir", type=str, default="./data/vqgan_targets",
                         help="Where to write the extracted target-only images.")
    parser.add_argument("--direction", type=str, default="AtoB", choices=["AtoB", "BtoA"],
                         help="AtoB: sketch=left, target=right. BtoA: sketch=right, target=left.")
    parser.add_argument("--maskgit-repo", type=str, default="./MaskGIT-pytorch",
                         help="Path to the cloned MaskGIT-pytorch repo.")
    parser.add_argument("--skip-extract", action="store_true",
                         help="Skip re-extracting targets if you've already run this once.")

    # pass-through hyperparameters forwarded to their training_vqgan.py.
    # Check `python training_vqgan.py --help` in your repo copy to confirm exact
    # flag names/defaults before a long run — argparse flag names occasionally
    # drift between repo versions.
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=2.25e-05)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--num-codebook-vectors", type=int, default=8192)

    args = parser.parse_args()

    if not args.skip_extract:
        extract_targets(args.source_dir, args.targets_dir, direction=args.direction)
    else:
        print(f"Skipping extraction, using existing folder: {args.targets_dir}")

    training_script = os.path.join(args.maskgit_repo, "training_vqgan.py")
    if not os.path.isfile(training_script):
        raise FileNotFoundError(
            f"Could not find training_vqgan.py at {training_script}. "
            "Check --maskgit-repo points at your cloned repo."
        )

    cmd = [
        sys.executable, training_script,
        "--dataset-path", os.path.abspath(args.targets_dir),
        "--image-size", str(args.image_size),
        "--batch-size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--learning-rate", str(args.learning_rate),
        "--latent-dim", str(args.latent_dim),
        "--num-codebook-vectors", str(args.num_codebook_vectors),
    ]

    print("Launching Stage 1 VQGAN training:")
    print(" ", " ".join(cmd))
    subprocess.run(cmd, cwd=args.maskgit_repo, check=True)


if __name__ == "__main__":
    main()