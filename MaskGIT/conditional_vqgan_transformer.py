"""
Corrected, sketch-conditioned replacement for transformer.py's VQGANTransformer.
 
Fixes three confirmed bugs found by inspecting the actual trained checkpoint
and vqgan.py source (see check_vqgan_config.py output):
 
  1. load_vqgan() loaded a DIFFERENT, incompatible model (vq_f16.VQModel with
     a pretrained checkpoint you don't have) instead of your actual trained
     vqgan.py VQGAN + vqgan_epoch_N.pt checkpoint.
  2. indices_to_image() reshaped to embedding dim 32 (hardcoded), but your
     codebook's real embedding dim is 256 (confirmed:
     codebook.embedding.weight shape = (8192, 256)).
  3. encode_to_z() unpacked vqgan.encode(x) as a nested tuple
     (quant_z, _, (_, _, indices)), but the real signature is a flat 3-tuple:
     codebook_mapping, codebook_indices, q_loss = self.encode(x)
 
Confirmed real config (from your checkpoint + a live encode() call):
    num_codebook_vectors = 8192
    embedding_dim         = 256
    latent grid            = 16x16 -> num_image_tokens = 256
 
On top of the fixes, this also adds sketch conditioning: the transformer is
ConditionalBidirectionalTransformer (from conditional_transformer.py) instead
of the original BidirectionalTransformer, and every forward/sampling call
threads a sketch-derived conditioning sequence through.
"""
 
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
 
from vqgan import VQGAN
from conditional_transformer import ConditionalBidirectionalTransformer
from sketch_encoder import SketchEncoder
 
_CONFIDENCE_OF_KNOWN_TOKENS = torch.Tensor([torch.inf])
 
 
class ConditionalVQGANTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = args.device
 
        self.num_image_tokens = args.num_image_tokens   # 256 (16x16 grid) -- confirmed
        self.latent_grid_size = int(math.sqrt(self.num_image_tokens))  # 16
        self.latent_dim = args.latent_dim                # 256 -- confirmed embedding dim
        self.sos_token = args.num_codebook_vectors + 1
        self.mask_token_id = args.num_codebook_vectors
        self.choice_temperature = 4.5
        self.gamma = self.gamma_func("cosine")
 
        self.transformer = ConditionalBidirectionalTransformer(
            num_codebook_vectors=args.num_codebook_vectors,
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            num_image_tokens=self.num_image_tokens,
        )
        self.sketch_encoder = SketchEncoder(
            embed_dim=args.dim,
            downsample_factor=16,   # confirmed: matches VQGAN's own downsampling
        )
 
        self.vqgan = self.load_vqgan(args)
 
        print(f"Transformer parameters: {sum(p.numel() for p in self.transformer.parameters())}")
        print(f"Sketch encoder parameters: {sum(p.numel() for p in self.sketch_encoder.parameters())}")
 
