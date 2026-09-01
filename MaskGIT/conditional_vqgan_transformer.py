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

    def load_checkpoint(self, epoch, checkpoint_dir="checkpoints"):
        path = os.path.join(checkpoint_dir, f"transformer_epoch_{epoch}.pt")
        self.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Loaded transformer checkpoint: {path}")

    @staticmethod
    def load_vqgan(args):
        """Loads YOUR actual trained VQGAN (vqgan.py + training_vqgan.py output),
        not the incompatible external vq_f16 model the original file used."""
        model = VQGAN(args)
        state = torch.load(args.checkpoint_path, map_location=args.device)
        model.load_state_dict(state)
        model.eval()
        # freeze -- Stage 2 must never update the tokenizer
        for p in model.parameters():
            p.requires_grad = False
        return model

    @torch.no_grad()
    def encode_to_z(self, x):
        """Fixed to match vqgan.py's REAL encode() signature (flat 3-tuple)."""
        quant_z, indices, q_loss = self.vqgan.encode(x)
        indices = indices.view(quant_z.shape[0], -1)   # (B, 256)
        return quant_z, indices

    def forward(self, x, sketch):
        """
        x:      (B, 3, H, W) target/real images
        sketch: (B, 3, H, W) paired sketches (condition)
        """
        _, z_indices = self.encode_to_z(x)
        cond_tokens = self.sketch_encoder(sketch)

        sos_tokens = torch.ones(x.shape[0], 1, dtype=torch.long, device=z_indices.device) * self.sos_token

        r = math.floor(self.gamma(np.random.uniform()) * z_indices.shape[1])
        sample = torch.rand(z_indices.shape, device=z_indices.device).topk(r, dim=1).indices
        mask = torch.zeros(z_indices.shape, dtype=torch.bool, device=z_indices.device)
        mask.scatter_(dim=1, index=sample, value=True)

        masked_indices = self.mask_token_id * torch.ones_like(z_indices, device=z_indices.device)
        a_indices = mask * z_indices + (~mask) * masked_indices
        a_indices = torch.cat((sos_tokens, a_indices), dim=1)

        target = torch.cat((sos_tokens, z_indices), dim=1)

        logits = self.transformer(a_indices, cond_tokens)

        return logits, target

    def gamma_func(self, mode="cosine"):
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: np.cos(r * np.pi / 2)
        elif mode == "square":
            return lambda r: 1 - r ** 2
        elif mode == "cubic":
            return lambda r: 1 - r ** 3
        else:
            raise NotImplementedError

    def create_input_tokens_normal(self, num):
        blank_tokens = torch.ones((num, self.num_image_tokens), device=self.device)
        masked_tokens = self.mask_token_id * blank_tokens
        return masked_tokens.to(torch.int64)

    def tokens_to_logits(self, seq, cond_tokens):
        return self.transformer(seq, cond_tokens)

    def mask_by_random_topk(self, mask_len, probs, temperature=1.0):
        confidence = torch.log(probs) + temperature * torch.distributions.gumbel.Gumbel(0, 1).sample(probs.shape).to(probs.device)
        sorted_confidence, _ = torch.sort(confidence, dim=-1)
        cut_off = torch.take_along_dim(sorted_confidence, mask_len.to(torch.long), dim=-1)
        masking = (confidence < cut_off)
        return masking

    @torch.no_grad()
    def sample_good(self, sketch, inputs=None, num=1, T=11, mode="cosine"):
        """
        sketch: (B, 3, H, W) -- REQUIRED now, since generation is conditional.
        """
        N = self.num_image_tokens
        if inputs is None:
            inputs = self.create_input_tokens_normal(num)
        else:
            inputs = torch.hstack((
                inputs,
                torch.zeros((inputs.shape[0], N - inputs.shape[1]), device=self.device, dtype=torch.int).fill_(self.mask_token_id)
            ))

        sos_tokens = torch.ones(inputs.shape[0], 1, dtype=torch.long, device=inputs.device) * self.sos_token
        inputs = torch.cat((sos_tokens, inputs), dim=1)

        cond_tokens = self.sketch_encoder(sketch)   # encode once, reuse every step

        unknown_number_in_the_beginning = torch.sum(inputs == self.mask_token_id, dim=-1)
        gamma = self.gamma_func(mode)
        cur_ids = inputs

        for t in range(T):
            logits = self.tokens_to_logits(cur_ids, cond_tokens)
            sampled_ids = torch.distributions.categorical.Categorical(logits=logits).sample()

            unknown_map = (cur_ids == self.mask_token_id)
            sampled_ids = torch.where(unknown_map, sampled_ids, cur_ids)

            ratio = 1. * (t + 1) / T
            mask_ratio = gamma(ratio)

            probs = F.softmax(logits, dim=-1)
            selected_probs = torch.squeeze(torch.take_along_dim(probs, torch.unsqueeze(sampled_ids, -1), -1), -1)
            selected_probs = torch.where(unknown_map, selected_probs, _CONFIDENCE_OF_KNOWN_TOKENS.to(probs.device))

            mask_len = torch.unsqueeze(torch.floor(unknown_number_in_the_beginning * mask_ratio), 1)
            mask_len = torch.maximum(
                torch.zeros_like(mask_len),
                torch.minimum(torch.sum(unknown_map, dim=-1, keepdim=True) - 1, mask_len)
            )

            masking = self.mask_by_random_topk(mask_len, selected_probs, temperature=self.choice_temperature * (1. - ratio))
            cur_ids = torch.where(masking, self.mask_token_id, sampled_ids)

        return cur_ids[:, 1:]   # drop sos token

    def indices_to_image(self, indices, p1=None, p2=None):
        """Fixed: real embedding dim is 256 (was hardcoded 32), real grid is 16x16."""
        p1 = p1 or self.latent_grid_size
        p2 = p2 or self.latent_grid_size
        ix_to_vectors = self.vqgan.codebook.embedding(indices).reshape(
            indices.shape[0], p1, p2, self.latent_dim
        )
        ix_to_vectors = ix_to_vectors.permute(0, 3, 1, 2)
        image = self.vqgan.decode(ix_to_vectors)
        return image

    @torch.no_grad()
    def generate(self, sketch, T=11, mode="cosine"):
        """Convenience method for inference.py: sketch -> generated image."""
        index_sample = self.sample_good(sketch, num=sketch.shape[0], T=T, mode=mode)
        image = self.indices_to_image(index_sample)
        return image

    @torch.no_grad()
    def log_images(self, x, sketch, mode="cosine"):
        """
        Used by the training loop to save preview images each epoch.
        x:      (B, 3, H, W) real/target image
        sketch: (B, 3, H, W) paired sketch (condition)
        Returns a dict of named images plus a single concatenated tensor
        (sketch | real | reconstruction | generated) for easy side-by-side saving.
        """
        log = dict()

        _, z_indices = self.encode_to_z(x)

        # reconstruction: encode the real image, decode it straight back
        # (sanity check that the frozen VQGAN + codebook lookup path works)
        x_rec = self.indices_to_image(z_indices)

        # full generation: start from all-masked, condition on the sketch only
        index_sample = self.sample_good(sketch, num=x.shape[0], T=11, mode=mode)
        x_new = self.indices_to_image(index_sample)

        log["sketch"] = sketch
        log["input"] = x
        log["rec"] = x_rec
        log["new_sample"] = x_new
        return log, torch.cat((sketch, x, x_rec, x_new), dim=0)