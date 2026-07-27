"""
Conditional bidirectional transformer for sketch -> image MaskGIT.

Based on dome272/MaskGIT-pytorch's transformer.py / bidirectional_transformer.py.
Key facts pulled directly from their source that this depends on:

    mask_token_id = args.num_codebook_vectors           # e.g. 8192
    sos_token     = args.num_codebook_vectors + 1        # e.g. 8193
    vocab size for the embedding layer = num_codebook_vectors + 2   (codes + mask + sos)

    In VQGANTransformer.forward():
        sos_tokens = torch.ones(B, 1, dtype=torch.long) * self.sos_token
        a_indices  = mask * z_indices + (~mask) * masked_indices
        a_indices  = torch.cat((sos_tokens, a_indices), dim=1)   # (B, 1 + N)
        logits     = self.transformer(a_indices)

This module REPLACES `self.transformer` in that class. It has the same
call signature — `forward(token_ids)` -> logits of shape (B, 1+N, vocab_size)
— but internally prepends the sketch condition tokens, runs everything
through the transformer body, then strips the condition positions back off
before returning. That means you do NOT need to touch the masking logic,
sos-token logic, or sampling loop in transformer.py at all — only the two
lines that construct `self.transformer` and call it (see integration notes
at the bottom of this file).
"""

import torch
import torch.nn as nn


class ConditionalBidirectionalTransformer(nn.Module):
    def __init__(
        self,
        num_codebook_vectors=8192,   # must match args.num_codebook_vectors
        dim=768,                     # must match args.dim
        hidden_dim=3072,             # must match args.hidden_dim (MLP width)
        n_layers=24,                 # must match args.n_layers
        n_heads=12,
        num_image_tokens=256,        # latent grid size, e.g. 16x16 for f16 @ 256px
        dropout=0.1,
    ):
        super().__init__()

        self.mask_token_id = num_codebook_vectors
        self.sos_token = num_codebook_vectors + 1
        vocab_size = num_codebook_vectors + 2   # codes + [MASK] + [SOS]

        self.dim = dim
        self.seq_len = 1 + num_image_tokens     # sos + image tokens (no cond tokens counted here)

        # --- token embedding for the discrete image/sos/mask sequence ---
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.token_pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, dim))

        # --- positional embedding for the (variable-length) sketch condition prefix ---
        # sized generously; sliced down to actual cond length at forward time
        max_cond_len = 1024
        self.cond_pos_embed = nn.Parameter(torch.zeros(1, max_cond_len, dim))

        nn.init.trunc_normal_(self.token_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cond_pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # pre-LN, more stable to train than post-LN
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.norm_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, token_ids, cond_tokens):
        """
        token_ids:   (B, 1+N) long tensor — [sos, masked/unmasked image tokens...]
                     exactly what VQGANTransformer already builds.
        cond_tokens: (B, C, dim) float tensor from SketchEncoder — the sketch
                     conditioning sequence. Never masked.

        returns: logits of shape (B, 1+N, vocab_size) — SAME shape as the
                 original BidirectionalTransformer produced, so nothing else
                 in transformer.py needs to change.
        """
        B, L = token_ids.shape
        assert L == self.seq_len, (
            f"token_ids length {L} != expected {self.seq_len}. "
            "num_image_tokens must match your VQGAN's actual latent token count."
        )

        tok_emb = self.token_embed(token_ids) + self.token_pos_embed  # (B, 1+N, dim)

        C = cond_tokens.shape[1]
        cond_emb = cond_tokens + self.cond_pos_embed[:, :C, :]        # (B, C, dim)

        x = torch.cat([cond_emb, tok_emb], dim=1)   # (B, C+1+N, dim)
        x = self.encoder(x)                          # bidirectional, no attention mask = full attention
        x = self.norm_out(x)

        # strip the condition positions back off — only return logits for [sos, image tokens...]
        x = x[:, C:, :]
        logits = self.head(x)
        return logits


"""
INTEGRATION NOTES for transformer.py (VQGANTransformer class)
---------------------------------------------------------------
1. In __init__, replace:
       self.transformer = BidirectionalTransformer(args)
   with:
       from conditional_transformer import ConditionalBidirectionalTransformer
       self.transformer = ConditionalBidirectionalTransformer(
           num_codebook_vectors=args.num_codebook_vectors,
           dim=args.dim,
           hidden_dim=args.hidden_dim,
           n_layers=args.n_layers,
           num_image_tokens=args.num_image_tokens,
       )
       from sketch_encoder import SketchEncoder
       self.sketch_encoder = SketchEncoder(embed_dim=args.dim, downsample_factor=16)

2. In forward(x), change signature to forward(x, sketch) and change:
       logits = self.transformer(a_indices)
   to:
       cond_tokens = self.sketch_encoder(sketch)
       logits = self.transformer(a_indices, cond_tokens)

3. In tokens_to_logits(seq), change signature to tokens_to_logits(seq, cond_tokens) and change:
       logits = self.transformer(seq)
   to:
       logits = self.transformer(seq, cond_tokens)
   Then update sample_good() to accept + thread `sketch` through to tokens_to_logits
   (encode the sketch once at the top of sample_good, reuse cond_tokens across all T steps).

This is exactly what train_stage2_transformer.py and inference.py (next files) will call.
"""