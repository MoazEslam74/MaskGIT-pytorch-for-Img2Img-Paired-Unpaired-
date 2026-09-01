"""
Stage 2 training script for sketch-conditioned MaskGIT.

Fixes vs. the uploaded training_transformer.py:
  1. `sketches` was referenced but never defined -- now comes from
     paired_dataset.PairedABDataset via a proper DataLoader.
  2. load_data(args) (their single-image loader) can't produce paired
     (sketch, target) batches -- replaced with PairedABDataset.
  3. The optimizer only included self.model.transformer.parameters() --
     the sketch_encoder is a fresh module and MUST be in the optimizer or
     it stays randomly initialized forever, silently breaking conditioning
     while the rest of training looks like it's working.

Run from inside ./MaskGIT (same folder as conditional_vqgan_transformer.py,
sketch_encoder.py, conditional_transformer.py, vqgan.py), with paired_dataset.py
copied into that same folder.
"""

import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import utils as vutils
from torch.utils.tensorboard import SummaryWriter

from conditional_vqgan_transformer import ConditionalVQGANTransformer
from paired_dataset import PairedABDataset
from lr_schedule import WarmupLinearLRSchedule


class TrainTransformer:
    def __init__(self, args):
        self.model = ConditionalVQGANTransformer(args).to(device=args.device)
        self.optim = self.configure_optimizers(args)
        self.lr_schedule = WarmupLinearLRSchedule(
            optimizer=self.optim,
            init_lr=1e-6,
            peak_lr=args.learning_rate,
            end_lr=0.,
            warmup_epochs=10,
            epochs=args.epochs,
            current_step=args.start_from_epoch,
        )

        if args.start_from_epoch > 1:
            self.model.load_checkpoint(args.start_from_epoch)
            print(f"Loaded Transformer from epoch {args.start_from_epoch}.")

        self.logger = SummaryWriter(f"./runs/{args.run_name}" if args.run_name else None)

        os.makedirs("results", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

        self.train(args)

    def configure_optimizers(self, args):
        # BUG FIX: original only included self.model.transformer.parameters().
        # The sketch_encoder is freshly initialized and must be trained too,
        # or sketch conditioning silently never learns anything.
        params = list(self.model.transformer.parameters()) + list(self.model.sketch_encoder.parameters())
        optimizer = torch.optim.Adam(params, lr=args.learning_rate, betas=(0.9, 0.96), weight_decay=4.5e-2)
        return optimizer

    def train(self, args):
        dataset = PairedABDataset(args.dataset_path, image_size=args.image_size, direction=args.direction)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        len_train_dataset = len(loader)
        step = args.start_from_epoch * len_train_dataset

        for epoch in range(args.start_from_epoch + 1, args.epochs + 1):
            print(f"Epoch {epoch}:")
            last_target, last_sketch = None, None
            with tqdm(range(len(loader))) as pbar:
                self.lr_schedule.step()
                for i, batch in zip(pbar, loader):
                    target = batch["target"].to(device=args.device)
                    sketch = batch["sketch"].to(device=args.device)
                    last_target, last_sketch = target, sketch

                    logits, target_tokens = self.model(target, sketch)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_tokens.reshape(-1))
                    loss.backward()

                    if step % args.accum_grad == 0:
                        self.optim.step()
                        self.optim.zero_grad()
                    step += 1

                    pbar.set_postfix(Transformer_Loss=np.round(loss.cpu().detach().numpy().item(), 4))
                    pbar.update(0)
                    self.logger.add_scalar("Cross Entropy Loss", loss.cpu().detach().item(), (epoch * len_train_dataset) + i)

            # save a preview: sketch | real target | reconstruction | generated sample
            try:
                log, preview = self.model.log_images(last_target[0:1], last_sketch[0:1])
                vutils.save_image(preview.add(1).mul(0.5), os.path.join("results", f"{epoch}.jpg"), nrow=4)
            except Exception as e:
                # surfaced instead of silently swallowed -- worth seeing during early runs
                print(f"  [warning] could not save preview image this epoch: {e}")

            if epoch % args.ckpt_interval == 0:
                torch.save(self.model.state_dict(), os.path.join("checkpoints", f"transformer_epoch_{epoch}.pt"))
            torch.save(self.model.state_dict(), os.path.join("checkpoints", "transformer_current.pt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sketch-conditioned MaskGIT transformer")
    parser.add_argument('--run-name', type=str, default=None)
    parser.add_argument('--dataset-path', type=str, required=True,
                         help='Folder of combined A|B (sketch|target) images -- e.g. ./data/train')
    parser.add_argument('--direction', type=str, default='AtoB', choices=['AtoB', 'BtoA'])
    parser.add_argument('--checkpoint-path', type=str, required=True,
                         help='Path to your trained Stage 1 VQGAN checkpoint, e.g. ./checkpoints/vqgan_epoch_18.pt')
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--latent-dim', type=int, default=256, help='Confirmed real embedding dim -- do not change unless you retrain Stage 1 differently.')
    parser.add_argument('--num-codebook-vectors', type=int, default=8192, help='MUST match Stage 1 training exactly.')
    parser.add_argument('--beta', type=float, default=0.25)
    parser.add_argument('--image-channels', type=int, default=3)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--accum-grad', type=int, default=25)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--start-from-epoch', type=int, default=0)
    parser.add_argument('--ckpt-interval', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=1e-4)

    parser.add_argument('--n-layers', type=int, default=24)
    parser.add_argument('--dim', type=int, default=768)
    parser.add_argument('--hidden-dim', type=int, default=3072)
    parser.add_argument('--num-image-tokens', type=int, default=256, help='Confirmed 16x16 grid -- do not change unless Stage 1 config changes.')

    args = parser.parse_args()

    train_transformer = TrainTransformer(args)