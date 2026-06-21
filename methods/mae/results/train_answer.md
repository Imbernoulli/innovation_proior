The thing that keeps bothering me is the gap between how language and vision models are fed. In NLP the recipe is settled: take a giant pile of raw text, hide part of it, train a model to predict the hidden part, and out comes a representation that transfers everywhere and scales to absurd model sizes — GPT does it autoregressively, BERT by masking out roughly 15% of the tokens and predicting them from both sides. No labels, free data, and the bigger the model the better it gets. In vision none of this is the practical reality. The strongest backbones are still pre-trained *supervised*, on labeled data, and the appetite for labels is enormous — a Vision Transformer only really shines after something like three hundred million labeled images, most of which nobody outside a few labs can access. Models can already overfit a million images without breaking a sweat, so the bottleneck is not capacity, it is supervision. And the maddening part is that the cure is sitting right there: hiding part of the input and reconstructing it is a special case of the denoising autoencoder (Vincent et al. 2008), older than BERT, and vision even had masked-image work before BERT — Pathak's Context Encoder (2016) inpaints a large missing region of an image with a convnet. Yet years after BERT detonated in NLP, hide-and-reconstruct pre-training in vision still trails. The Vision Transformer line already tried the straight BERT move — mask some patches, predict them — and got about two points over training from scratch, still around four points behind supervised pre-training. iGPT predicts pixels but at tiny resolution and ruinous compute. BEiT predicts discrete visual tokens from a separately trained dVAE tokenizer, which needs an extra pre-training stage on ~250M images and a large convolutional net adding real per-step overhead. Everyone is circling the same idea and it just does not pay off the way it does in language.

So the real question is not "how do I hide and reconstruct image content," it is *what is different about vision that makes the obvious port fail*. Three differences separate the signals. Architecture used to be the scapegoat — convnets run on regular grids and it is awkward to inject a mask placeholder or add positional embeddings the way Transformers do — but ViT closed that gap: cut the image into patches, project each to a token, add a positional embedding, run a standard Transformer, and a mask token is just another token. So architecture is no longer the obstacle. The second difference is decisive. Language is a human invention where every word is a deliberate, high-entropy, semantic unit, so blanking 15% genuinely requires understanding; the task is hard *because the signal is information-dense*. An image is recorded light, a natural signal drowning in spatial redundancy — neighboring patches are nearly the same thing. Blank out one patch and leave its neighbors and you can recover it by smearing them in, extending edges, copying texture, with no understanding of "this is a dog's ear" at all. Porting BERT's 15% straight over therefore builds a task whose solution is *interpolation*: the model aces it with low-level statistics and is never forced to learn about objects or scenes. The third difference is subtler: in BERT the prediction target is a word, a rich semantic token, so a trivial decoder suffices; in vision a pixel-reconstructing decoder outputs a *low* semantic-level signal, far below a recognition label, and whatever machinery turns an abstract latent into concrete pixels is reconstruction-specialized — the opposite of what I want a transferable representation to be specialized for.

The method that resolves all three is the Masked Autoencoder, MAE. The reframing is that the job is to *destroy the redundancy* so that filling holes requires understanding, and the lever is the masking ratio. Keep only a sparse scatter of patches and any missing patch is far from every visible one — there is no neighbor to copy, and the only way to guess what was there is a holistic sense of the whole object. So the ratio should not be 15%; it should be genuinely high. With a $224\times224$ image and $16\times16$ patches there are $196$ tokens: keeping 25% leaves 49 visible patches, a sparse but still broad sketch; 15% leaves about 29, which starts to feel starved; half leaves 98, dense enough that many holes still have close neighbors. So I mask 75% at random — aggressive but reconstructable, far above BERT's 15% and the 20–50% of earlier vision attempts. The masking is uniform random per sample rather than block-wise (which at a high ratio can erase whole objects, leaving too little to condition on) or grid-wise (which keeps a regular downsampled image and hands the interpolation shortcut right back); random scatters the holes everywhere, leaves no low-resolution image, and still keeps anchors across the scene.

The high ratio then pays a dividend I was not even chasing. Self-attention is quadratic in sequence length, so if the encoder sees only the visible 25% of patches, the attention term goes from $L\times L$ to $(L/4)\times(L/4)$ — about $(0.25)^2 = 1/16$ of full-sequence attention (total wall-clock savings are smaller, since MLPs, projections, and the decoder are not quadratic). The thing that makes the task hard is the same thing that makes the encoder cheap. But BERT's way is to put a mask token in the *encoder* at every masked slot, so the encoder processes a full-length sequence that is three-quarters artificial placeholders — and at deployment the encoder sees clean, complete images with *no* mask tokens, so it would have spent most of its pre-training capacity learning to process a token it never sees again. That is a train/deploy mismatch baked into the very weights I keep. The fix is to simply not give the encoder mask tokens: drop the masked patches entirely, so the encoder only ever sees real patches, exactly as at deployment, and as a bonus the sequence is short. The architecture is therefore *asymmetric*, and the asymmetry is forced by two pressures pointing the same way — keep the encoder on real patches only, and keep it short.

Someone still has to reconstruct the patches the encoder never saw, and that is the decoder, the place where the third difference is resolved. After the encoder I have latent vectors for the visible patches; I build the *full* token set by inserting, at every masked slot, a single shared learned mask token whose only job is to say "something belongs here, predict it." A bare mask token has no idea *where* it is, so I add positional embeddings to the whole set, and a small Transformer decoder attends over it, letting information flow from the encoded visible tokens into the mask positions. Crucially the decoder is where pixel-painting specialization gets *quarantined*: if I forced that job into the encoder with a trivial decoder, the encoder's output would be dragged toward pixel-level detail, ruining the representation I plan to keep. The decoder is used only during pre-training and thrown away, so its design is free — make it lightweight: narrow ($512$ dimensions, since pixel reconstruction does not need the recognition backbone's width) and a handful of blocks (eight by default; the minimum is one, because mask positions need at least one round of attention to read from the visible tokens, but enough depth keeps reconstruction work out of the latent). The expensive encoder runs short; the full-length work happens only in the cheap decoder.

The target is the raw pixels of each masked patch — the final decoder layer is a linear projection to $\text{patch\_size}^2\cdot C$ values ($16^2\cdot 3 = 768$ for RGB) — scored by mean squared error, computed on the masked patches *only*. Masked-only matters: the visible patches were handed to the model, so asking it to reproduce them rewards an identity path and dilutes the gradient on the hard part; the objective I want is the conditional problem of recovering the missing patches from the visible evidence. With predicted patches $\hat p_i$ and targets $p_i$ and a binary mask $m_i$ ($1$ if patch $i$ was removed), the loss is
$$\mathcal{L} = \frac{\sum_i m_i \,\lVert \hat p_i - p_i\rVert^2 / D}{\sum_i m_i},$$
the per-patch MSE averaged over removed patches. I deliberately do *not* go BEiT's route of predicting discrete visual tokens: that buys a more "semantic" target only by paying for an extra tokenizer stage, extra data, and a large per-step convnet — and if the 75% masking already forces the model to understand the scene, the difficulty is coming from the masking, not the target, so pixels suffice and keep everything self-contained. One optional refinement is a `norm_pix_loss` switch that normalizes each patch by its own mean and variance, $(x-\mu)/\sqrt{\sigma^2+\epsilon}$ over that patch's pixels, shifting weight toward local contrast and texture; a low-frequency target such as PCA coefficients would do the opposite, making the output easier and throwing away the fine structure that dense pixel supervision provides.

The last piece is making the sparse bookkeeping run on ordinary dense ops, with no custom sparse kernel. Tokenize *every* patch and add its positional embedding; draw one random number per patch and `argsort` it for a uniform random permutation — taking the first `len_keep` of that order is exactly sampling that fraction without replacement, and those are the kept tokens, gathered into the short sequence the encoder sees. To put everything back I need the inverse permutation, and `argsort(ids_shuffle)` is precisely it: if `ids_shuffle[k]=j` then original position $j$ moved to shuffled slot $k$, so `ids_restore[j]=k`, and gathering by `ids_restore` returns each token to its home slot, aligned with its target. After encoding I append the right number of shared mask tokens, gather by the restore indices to unshuffle, add the decoder positional embeddings, decode, and project to pixels. The binary loss mask is built the same way — zeros on the first `len_keep`, ones after, then unshuffled by the same indices. A class token is prepended to the encoder (kept compatible with standard ViT recognition code), and both encoder and decoder use fixed 2D sine-cosine positional embeddings, no learned relative-position machinery, no layer scaling. The recommended pre-training is 800 epochs (scaling further to 1600; the repo's argparse default is 400), random resized crop plus horizontal flip, AdamW with betas $(0.9, 0.95)$, weight decay $0.05$, base learning rate $\text{blr}=1.5\times10^{-4}$ with $\text{lr}=\text{blr}\cdot\text{batch}/256$ at batch size 4096, cosine schedule with 40 warmup epochs, and `mask_ratio=0.75`. Hide most of it, encode the little that is left, reconstruct the rest cheaply — and the representation that falls out is what I keep.

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
