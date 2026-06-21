The goal is to build a diffusion denoiser on a standard Vision Transformer backbone so that image generation inherits the predictable scaling laws that transformers exhibit in language and vision recognition. A vanilla ViT block — pre-norm multi-head self-attention followed by an MLP, each wrapped in a residual — has no natural place for the global, per-example side information a diffusion model needs: the noise level timestep t and, for class-conditional generation, the class label c. The conditioning must reach every block, add almost no compute, leave the standard block untouched, and keep a deep stack stable at initialization.

The obvious ports are poor fits. Appending t and c as extra tokens forces self-attention to re-route a constant global signal at every layer, wasting attention capacity on a structureless two-element table. Cross-attention is genuinely useful when the conditioning is itself a sequence, but paying a whole extra attention sublayer per block for a single global class label is overkill: the keys and values have no internal token structure to attend over. What matches a global, per-channel, location-agnostic signal is a global, per-channel, location-agnostic operation — the affine that LayerNorm already applies after normalization.

The method is adaLN-Zero, adaptive layer norm with zero-initialized residual gates, the conditioning mechanism used in the Diffusion Transformer (DiT). It replaces each LayerNorm's learned per-channel affine with a scale and shift regressed from the conditioning vector, and adds a per-sublayer output gate also regressed from the conditioning and initialized to zero. The conditioning vector is simply the sum of the timestep embedding and the class embedding, both at the transformer's hidden width: cond = t_emb + c_emb. A small SiLU-then-Linear regressor takes cond and emits six length-d vectors per block: shift, scale, and gate for the attention sublayer, and shift, scale, and gate for the MLP sublayer. The modulate helper is u * (1 + scale) + shift, applied to the affine-free LayerNorm output; the 1 + scale form makes zero-valued modulation leave the normalized activation unchanged. The gate multiplies the sublayer output immediately before the residual add, so when the modulation Linear is zero-initialized the entire residual branch is switched off and every block starts as the exact identity map. The gateless final decode head uses only shift and scale, and its output linear is also zero-initialized so the denoiser starts by predicting zero noise.

Everything outside the conditioning path is kept standard: pre-norm blocks, multi-head self-attention with qkv bias, a 4x-wide GELU MLP, and fixed sine-cosine positional embeddings on the patch tokens. This minimal perturbation is the point: the design inherits transformer scaling by touching only the normalization affines and residual gates.

```python
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        import math
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
        ).to(t.device)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob=0.1):
        super().__init__()
        use_cfg = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels):
        drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        return torch.where(drop_ids, self.num_classes, labels)

    def forward(self, labels, train):
        if train and self.dropout_prob > 0:
            labels = self.token_drop(labels)
        return self.embedding_table(labels)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **kw):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **kw)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden,
                       act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class DiT(nn.Module):
    def __init__(self, input_size=32, patch_size=2, in_channels=4, hidden_size=1152,
                 depth=28, num_heads=16, mlp_ratio=4.0, class_dropout_prob=0.1,
                 num_classes=1000, learn_sigma=True):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_basic_init)

        pos = get_2d_sincos_pos_embed(self.pos_embed.shape[-1],
                                      int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(x.shape[0], c, h * p, h * p)

    def forward(self, x, t, y):
        x = self.x_embedder(x) + self.pos_embed
        t = self.t_embedder(t)
        y = self.y_embedder(y, self.training)
        c = t + y
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        return self.unpatchify(x)
```
