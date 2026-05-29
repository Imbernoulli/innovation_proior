# Masked Autoencoders (MAE)

## Problem

Self-supervised pre-training transformed NLP — hide part of the input, predict it — but in vision the practical pre-training paradigm stayed supervised and label-hungry. Porting BERT-style masked modeling to images directly underperforms. The reason is a property of the signal: images are spatially redundant, so a low masking ratio makes hole-filling a trivial interpolation that forces no real understanding. MAE is a self-supervised pre-training method for vision that fixes this and scales to the largest ViTs using only unlabeled ImageNet-1K.

## Key idea

Mask a very high fraction of image patches at random and reconstruct the missing *pixels*, with an **asymmetric** design:

- **High masking ratio (75%).** Removing most patches destroys local redundancy, so a missing patch cannot be recovered by copying neighbors; the model must infer broader object and scene structure. The default ratio is far higher than BERT's 15% and higher than prior vision work (20-50%).
- **Encoder on visible patches only, no mask tokens.** The ViT encoder processes just the ~25% visible patches. The attention-matrix term scales as sequence length squared, so reducing the sequence to 0.25L makes that term about `(0.25)^2 = 1/16` of full-sequence attention; total wall-clock savings are smaller because MLPs, the decoder, and data movement are not quadratic. Dropping mask tokens also removes the train/deploy mismatch of feeding the encoder placeholder tokens it never sees on clean images at deployment.
- **Lightweight, separate decoder.** After encoding, the full token set is rebuilt: encoded visible tokens plus a shared, learned mask token at every masked position, with positional embeddings so each mask token knows its location. A small Transformer decoder (default 8 blocks, 512-dim) reconstructs the image and is discarded after pre-training, keeping reconstruction-specific computation out of the encoder as much as possible.
- **Pixel reconstruction target, MSE on masked patches only.** No tokenizer, no extra data, no adversarial loss. The implementation can optionally normalize each patch target as `(x - mean) / sqrt(var + eps)` over that patch's pixel values with `norm_pix_loss`.

After pre-training the decoder is thrown away; the encoder is applied to full, uncorrupted images for recognition (fine-tuning, linear probing, or transfer).

## Algorithm

1. Patchify the image; embed every patch with a linear projection and add fixed 2D sine-cosine positional embeddings.
2. Randomly shuffle the patch tokens, keep the first `(1 - mask_ratio)` fraction (= uniform sampling without replacement), drop the rest, and set `ids_restore = argsort(ids_shuffle)` as the inverse permutation.
3. Run the ViT encoder on the kept tokens (plus a class token).
4. Project to decoder width; append shared mask tokens; unshuffle so every token returns to its original position aligned with its target; add decoder positional embeddings.
5. Run the lightweight decoder; a final linear layer outputs `patch_size² · channels` pixel values per patch (`16² * 3 = 768` for 16x16 RGB patches).
6. Loss = MSE between predicted and (optionally per-patch-normalized) target pixels, averaged over masked patches only.

The canonical pre-training setup uses 400 epochs by default, random resized crop plus horizontal flip, AdamW with betas `(0.9, 0.95)`, weight decay `0.05`, base learning rate `blr=1e-3` with `lr = blr * effective_batch_size / 256`, cosine scheduling with 40 warmup epochs, default `mask_ratio=0.75`, and effective batch size computed as `batch_size * accum_iter * world_size`.

## Code

```python
from functools import partial
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block
from util.pos_embed import get_2d_sincos_pos_embed


class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with a VisionTransformer backbone."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        # encoder
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,
                  qk_scale=None, norm_layer=norm_layer)
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)
        # decoder (lightweight, discarded after pre-training)
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True,
                  qk_scale=None, norm_layer=norm_layer)
            for _ in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        pos = get_2d_sincos_pos_embed(self.pos_embed.shape[-1],
                                      int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))
        dpos = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                       int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(dpos).float().unsqueeze(0))
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], 3, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        return x.reshape(imgs.shape[0], h * w, p**2 * 3)

    def unpatchify(self, x):
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        x = x.reshape(x.shape[0], h, w, p, p, 3)
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(x.shape[0], 3, h * p, h * p)

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)        # small=keep, large=remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # inverse permutation
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones([N, L], device=x.device)       # 1=remove, 0=keep
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        x = torch.cat((cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x), mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)                       # no cls
        x_ = torch.gather(x_, 1, ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)                                 # restore cls
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x[:, 1:, :]                                                      # drop cls

    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        loss = ((pred - target) ** 2).mean(dim=-1)          # per-patch MSE
        return (loss * mask).sum() / mask.sum()             # masked patches only

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    return MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


def mae_vit_large_patch16_dec512d8b(**kwargs):
    return MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    return MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b
```

```python
import math
import sys
import torch
import util.lr_sched as lr_sched


def train_one_epoch(model, data_loader, optimizer, device, epoch, loss_scaler,
                    log_writer=None, args=None):
    model.train(True)
    accum_iter = args.accum_iter
    optimizer.zero_grad()

    for data_iter_step, (samples, _) in enumerate(data_loader):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss / accum_iter
        loss_scaler(
            loss, optimizer, parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
```
