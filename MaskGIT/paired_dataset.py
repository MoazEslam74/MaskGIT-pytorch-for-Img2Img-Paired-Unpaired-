"""
Paired sketch/real-image dataset for MaskGIT-pytorch.

Expects data in the classic pix2pix format: each file in the dataset folder
is a single image of width = 2 * height, with the sketch (A) and the real
image (B) placed side by side.

    +----------------+----------------+
    |   sketch (A)   |  real image (B)|
    +----------------+----------------+

If your sketch is on the right instead of the left, set direction="BtoA"
(that just swaps which half is loaded as condition vs. target).
"""

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PairedABDataset(Dataset):
    def __init__(self, root_dir, image_size=256, direction="AtoB", exts=(".png", ".jpg", ".jpeg")):
        """
        root_dir:    folder containing the combined A|B images (e.g. data/train)
        image_size:  final square size for BOTH the sketch and the target image
        direction:   "AtoB" -> left half = sketch (condition), right half = target
                     "BtoA" -> right half = sketch (condition), left half = target
        """
        self.root_dir = root_dir
        self.direction = direction
        self.files = sorted(
            f for f in os.listdir(root_dir) if f.lower().endswith(exts)
        )
        if len(self.files) == 0:
            raise RuntimeError(f"No images found in {root_dir}")

        # VQGAN in dome272/MaskGIT-pytorch expects inputs roughly in [-1, 1].
        # Keep this transform identical for sketch and target so both encoders
        # (VQGAN + sketch encoder) see consistently scaled inputs.
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),                       # [0, 1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # -> [-1, 1]
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.root_dir, self.files[idx])
        combined = Image.open(path).convert("RGB")

        w, h = combined.size
        half_w = w // 2
        if half_w * 2 != w:
            raise ValueError(
                f"{path}: width {w} is not evenly divisible into two halves. "
                "Check that every file is exactly 2x wider than it is tall."
            )

        left = combined.crop((0, 0, half_w, h))
        right = combined.crop((half_w, 0, w, h))

        if self.direction == "AtoB":
            sketch_img, target_img = left, right
        elif self.direction == "BtoA":
            sketch_img, target_img = right, left
        else:
            raise ValueError("direction must be 'AtoB' or 'BtoA'")

        sketch = self.transform(sketch_img)
        target = self.transform(target_img)

        return {
            "sketch": sketch,   # condition, feed to sketch_encoder
            "target": target,   # feed to VQGAN encoder for ground-truth tokens
            "filename": self.files[idx],
        }


if __name__ == "__main__":
    # Quick sanity check: run this file directly to confirm halves split correctly.
    import sys
    import torchvision.utils as vutils

    root = sys.argv[1] if len(sys.argv) > 1 else "./data/train"
    ds = PairedABDataset(root, image_size=256, direction="AtoB")
    print(f"Found {len(ds)} paired images in {root}")

    sample = ds[0]
    print("sketch shape:", sample["sketch"].shape, "target shape:", sample["target"].shape)

    # Save a denormalized preview so you can eyeball that A/B split is correct
    def denorm(t):
        return (t * 0.5 + 0.5).clamp(0, 1)

    vutils.save_image(denorm(sample["sketch"]), "preview_sketch.png")
    vutils.save_image(denorm(sample["target"]), "preview_target.png")
    print("Saved preview_sketch.png and preview_target.png — check these match visually.")