Vision Transformers split an image into a sequence of patches and process it with a standard Transformer encoder. They achieve strong accuracy, but only when trained on very large labeled datasets; on ImageNet-1K alone, trained from random initialization, they are unstable and underperform convolutional networks. The architecture has no built-in locality or translation equivariance, so it must learn all visual priors from data. The question is whether a vision Transformer can be pretrained on unlabeled images with a self-supervised objective that produces a transferable encoder for classification and dense downstream tasks.

The natural starting point is BERT-style masked-token denoising. In language, that works because tokens are discrete symbols from a fixed vocabulary, so predicting a masked token is a clean classification problem. Image patches are continuous, so there is no vocabulary to classify into. The literal substitute, regressing the raw pixels of masked patches, is dominated by short-range correlations and high-frequency texture. A model can minimize that loss by local smoothing and copying without learning semantics, and empirical attempts at pixel-level masked-patch prediction underperform. What is needed is a prediction target that is discrete and semantically abstract, so the masked-prediction game forces the encoder to reason about objects and layout rather than interpolate pixels.

The method is BEiT, which stands for BERT Pre-Training of Image Transformers. BEiT gives each image two aligned views. The first view is the usual set of 16 by 16 patches, linearly embedded and fed to the Transformer; this keeps raw pixels as the encoder input. The second view is produced by a frozen, pretrained discrete variational autoencoder tokenizer, the same kind used in DALL-E, which maps an image to a grid of integer codebook indices. With a 224 by 224 patch view and a 112 by 112 tokenizer view, both grids are 14 by 14, so every patch has a corresponding visual token. The codebook has 8192 entries, and those entries serve as a visual vocabulary learned to reconstruct images through a bottleneck, which means they discard high-frequency detail and retain abstraction.

During pretraining, roughly 40 percent of the patches are masked. Instead of masking isolated patches, BEiT masks contiguous rectangular blocks, so the task cannot be solved by copying from immediate neighbors. Masked patch positions are replaced with a learnable mask embedding. The corrupted sequence is fed through the Transformer, and at each masked position the output vector is passed through a linear classifier over the 8192 codebook entries. The loss is cross-entropy summed only over the masked positions. Because the target is discrete, the objective is exactly the masked-language-modeling loss, now applied to visual tokens.

This choice also has a variational interpretation. Let x be the image, x tilde the masked image, and z the discrete visual tokens. The log-likelihood of recovering x from x tilde can be lower-bounded by an ELBO in which the tokenizer plays the role of the posterior q phi of z given x, a decoder p psi reconstructs x from z, and the BEiT encoder p theta predicts z from x tilde. The tokenizer is trained first by reconstruction with a uniform prior. Then it is frozen, the argmax token grid is taken as a hard target, and only the p theta term is optimized. That term factorizes over masked positions and becomes the cross-entropy used during pretraining.

The encoder is a standard vision Transformer, so it can later be fine-tuned with the same recipe as supervised ViT. For classification the pretrained encoder is frozen or fine-tuned end-to-end with a linear head on top of the pooled patch features; for semantic segmentation a lightweight upsampling decoder is attached to the patch grid. The only pretraining-specific component is the linear prediction head. To keep deep training stable, weights are initialized with a small truncated normal and the output projections of the l-th block are rescaled by one over the square root of two l, damping residual-stream growth at initialization.

```python
import math, random
from functools import partial
import numpy as np
import torch, torch.nn as nn
from timm.models.layers import trunc_normal_ as _trunc_normal

def trunc_normal_(tensor, std=0.02):
    return _trunc_normal(tensor, std=std, a=-std, b=std)


class PatchGridMaskGenerator:
    """Blockwise masking over the h×w patch grid until ~num_masking_patches are masked."""
    def __init__(self, input_size, num_masking_patches,
                 min_num_patches=16, max_num_patches=None,
                 min_aspect=0.3, max_aspect=None):
        self.height, self.width = (input_size, input_size) if isinstance(input_size, int) else input_size
        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches
        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)   # s
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))        # r
            h = int(round(math.sqrt(target_area * aspect_ratio)))                  # a = sqrt(s*r)
            w = int(round(math.sqrt(target_area / aspect_ratio)))                  # b = sqrt(s/r)
            if w < self.width and h < self.height:
                top  = random.randint(0, self.height - h)
                left = random.randint(0, self.width  - w)
                num_masked = mask[top:top + h, left:left + w].sum()
                if 0 < h * w - num_masked <= max_mask_patches:        # count newly-masked (overlap ok)
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1; delta += 1
                if delta > 0:
                    break
        return delta

    def __call__(self):
        mask = np.zeros((self.height, self.width), dtype=int)
        mask_count = 0
        while mask_count < self.num_masking_patches:
            max_mask_patches = min(self.num_masking_patches - mask_count, self.max_num_patches)
            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            mask_count += delta
        return mask                                                   # (h, w) boolean grid


class TwoViewPretrainingTransform:
    def __init__(self, args):
        mean = IMAGENET_INCEPTION_MEAN if not args.imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not args.imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
        self.common_transform = transforms.Compose([
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(p=0.5),
            RandomResizedCropAndInterpolationWithTwoPic(
                size=args.input_size, second_size=args.second_input_size,
                interpolation=args.train_interpolation,
                second_interpolation=args.second_interpolation),
        ])
        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        ])
        if args.discrete_vae_type == "dall-e":
            self.visual_token_transform = transforms.Compose([transforms.ToTensor(), map_pixels])
        elif args.discrete_vae_type == "customized":
            self.visual_token_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
            ])
        else:
            raise NotImplementedError()
        self.masked_position_generator = PatchGridMaskGenerator(
            args.window_size, num_masking_patches=args.num_mask_patches,
            max_num_patches=args.max_mask_patches_per_block,
            min_num_patches=args.min_mask_patches_per_block)

    def __call__(self, image):
        for_patches, for_tokens = self.common_transform(image)
        return (self.patch_transform(for_patches),
                self.visual_token_transform(for_tokens),
                self.masked_position_generator())


class PatchSequencePretrainer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, vocab_size=8192,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, attn_head_dim=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
                 init_std=0.02, **kw):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_features = self.embed_dim = embed_dim
        self.init_std = init_std
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token  = nn.Parameter(torch.zeros(1, 1, embed_dim))      # [S]
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))      # learnable e_[M]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim)) if use_abs_pos_emb else None
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.rel_pos_bias = (
            RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
            if use_shared_rel_pos_bias else None
        )
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                  norm_layer=norm_layer, init_values=init_values,
                  window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
                  attn_head_dim=attn_head_dim)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)                  # softmax over codebook

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=self.init_std)
        trunc_normal_(self.cls_token,  std=self.init_std)
        trunc_normal_(self.mask_token, std=self.init_std)
        trunc_normal_(self.lm_head.weight, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        # rescale the l-th block's output projections by 1/sqrt(2*l) for stability
        for layer_id, blk in enumerate(self.blocks):
            blk.attn.proj.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))
            blk.mlp.fc2.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def forward_features(self, x, bool_masked_pos):
        x = self.patch_embed(x, bool_masked_pos=bool_masked_pos)          # (B, N, D)
        B, N, _ = x.shape
        mask_token = self.mask_token.expand(B, N, -1)
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w                                 # substitute e_[M] at masked patches
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)
        return self.norm(x)

    def forward(self, x, bool_masked_pos, return_all_tokens=False):
        x = self.forward_features(x, bool_masked_pos)[:, 1:]
        if return_all_tokens:
            return self.lm_head(x)
        return self.lm_head(x[bool_masked_pos])                          # logits at masked positions only


def train_one_step(model, tokenizer, batch, optimizer):
    samples, token_view, bool_masked_pos = batch
    bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)          # (B, N)

    with torch.no_grad():
        input_ids = tokenizer.get_codebook_indices(token_view).flatten(1)  # argmax codes ẑ
        labels = input_ids[bool_masked_pos]                                # targets at masked positions

    logits = model(samples, bool_masked_pos=bool_masked_pos)               # (#masked, |V|)
    loss = nn.CrossEntropyLoss()(logits, labels)                           # = -Σ_{i∈M} log p_MIM(ẑ_i | x^M)

    optimizer.zero_grad(); loss.backward(); optimizer.step()
    return loss.item()
```
