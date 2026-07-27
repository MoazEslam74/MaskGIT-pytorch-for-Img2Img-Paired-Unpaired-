"""
Sketch encoder for conditional MaskGIT.

Turns a sketch image into a spatial grid of embeddings that has the SAME
height/width as the VQGAN's latent token grid, so each sketch embedding can
be prepended 1:1 alongside the image tokens as a fixed, never-masked prefix
for the bidirectional transformer.

Why match the latent grid size (not just produce one global vector)?
A single pooled vector throws away spatial layout — the transformer would
know "there's a sketch of a shoe" but not "there's a strap here, a sole
there." Keeping it spatial lets the transformer attend to the right region
of the sketch for each image token it's predicting.

Downsampling factor must match your VQGAN's downsampling factor. dome272's
default VQGAN downsamples by 16x (see vq_f16.py) — if you're using a
different config, change `downsample_factor` to match, or this will
silently misalign the sketch grid with the image token grid.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Strided conv + norm + activation, halves spatial size once."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


class SketchEncoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        base_channels=64,
        embed_dim=768,          # must match the transformer's hidden dim
        downsample_factor=16,   # must match the VQGAN's downsampling factor
    ):
        super().__init__()

        assert downsample_factor in (4, 8, 16, 32), "use a power-of-2 downsample factor"
        num_downsamples = downsample_factor.bit_length() - 1  # log2(downsample_factor)

        channels = [in_channels] + [
            min(base_channels * (2 ** i), 512) for i in range(num_downsamples)
        ]

        layers = []
        for i in range(num_downsamples):
            layers.append(ConvBlock(channels[i], channels[i + 1]))
        self.downsample = nn.Sequential(*layers)

        # project final feature channels -> transformer embedding dim
        self.proj = nn.Conv2d(channels[-1], embed_dim, kernel_size=1)

    def forward(self, sketch):
        """
        sketch: (B, 3, H, W) tensor, normalized the same way as paired_dataset.py
                (i.e. roughly in [-1, 1])

        returns: (B, N, embed_dim) sequence of conditioning tokens, where
                 N = (H / downsample_factor) * (W / downsample_factor) —
                 this must equal the VQGAN's latent token count so it lines
                 up with the image token sequence in the transformer.
        """
        feat = self.downsample(sketch)      # (B, C, H', W')
        feat = self.proj(feat)              # (B, embed_dim, H', W')
        B, E, Hp, Wp = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)  # (B, H'*W', embed_dim)
        return tokens


if __name__ == "__main__":
    # Sanity check: confirm the token count matches your VQGAN's latent grid.
    # Example: image_size=256, downsample_factor=16 -> latent grid = 16x16 = 256 tokens.
    image_size = 256
    downsample_factor = 16
    embed_dim = 768

    encoder = SketchEncoder(embed_dim=embed_dim, downsample_factor=downsample_factor)
    dummy_sketch = torch.randn(2, 3, image_size, image_size)
    tokens = encoder(dummy_sketch)

    expected_grid = image_size // downsample_factor
    expected_tokens = expected_grid * expected_grid

    print("Output shape:", tokens.shape)
    print(f"Expected token count: {expected_tokens} ({expected_grid}x{expected_grid} grid)")
    assert tokens.shape[1] == expected_tokens, (
        "Token count mismatch — check downsample_factor against your VQGAN's "
        "actual latent grid size before wiring this into the transformer."
    )
    print("OK — sketch token count matches expected VQGAN latent grid size.")