# adaLN-Zero, distilled

adaLN-Zero (adaptive layer norm, zero-initialized) is the conditioning mechanism of the Diffusion Transformer
(DiT). It injects a global conditioning signal — the diffusion timestep `t` and the class label `c` — into a
standard pre-norm transformer block by replacing each LayerNorm's learned affine with a *scale* and *shift*
regressed from the conditioning, and by adding a *gate* on each residual branch that is zero-initialized so
every residual block starts as the exact identity map while the final head starts at a zero prediction. It is
the cheapest of the conditioning options (one small MLP per block, negligible Gflops), applies the *same*
function to every token (the right inductive bias for a global signal), and keeps the rest of the ViT block
standard so transformer scaling transfers.

## Problem it solves

A transformer diffusion denoiser must condition on the timestep `t` and class label `c` at every block, but a
vanilla ViT block (LN → MHSA → residual; LN → MLP → residual) has no natural socket for a global, per-example
conditioning vector. The mechanism must (1) reach every block, (2) barely perturb the standard block so its
scaling properties survive, (3) add little compute, and (4) keep a deep stack stable to optimize.

## Key idea

Three converging moves:

1. **Condition by modulating normalization (adaptive LayerNorm).** A LayerNorm's learned per-channel affine
   `gamma · x_hat + beta` is a global, per-channel, every-token knob. Make it a *function of the conditioning*
   instead of a learned constant. The helper is
   `modulate(x, shift, scale) = x * (1 + scale) + shift`, applied to `LN(x)` in each sublayer. With `shift = 0`,
   the `1 +` makes `scale = 0` leave the normalized activation unchanged, so the regressor only learns a
   deviation from "no modulation." This is
   FiLM / AdaIN / ADM-AdaGN ported from convolutional blocks to a transformer's LayerNorm.

2. **Sum the conditionings.** `cond = t_emb + c_emb` (a sinusoidal-then-MLP timestep embedding plus a learned
   class embedding, both at hidden width `d`). One MLP then produces all modulation parameters.

3. **Zero-initialized residual gates (the "Zero").** Borrowing the zero-init-residual trick (zeroing the last
   residual scale so a block starts as the identity), add a per-sublayer **gate** applied right before each
   residual add, also regressed from `cond`. Zero-initialize the whole modulation projection so at start
   `scale = shift = gate = 0`: each residual block is exactly the identity, the `N`-block transformer body is
   the identity map, and conditioning grows from a no-op — which stabilizes deep-stack training.

Per block the modulation MLP emits **six** vectors — `(shift, scale, gate)` for the attention sublayer and for
the MLP sublayer — via one `Linear(d, 6d)`. The gateless final decode head needs only `(shift, scale)` (a
`Linear(d, 2d)`), and its output linear is zero-initialized too, so the model starts by predicting zero.

## Why this over the alternatives

- **In-context conditioning** (append `t`, `c` as extra tokens): zero block change, but forces self-attention
  to re-route a constant global signal at every layer — a poor structural match for per-example conditioning.
- **Cross-attention** (extra cross-attn sublayer keyed on `t`, `c`): most expressive, but pays a whole extra
  attention sublayer for a content lookup over a structureless two-element table — overkill for a global label.
- **adaLN-Zero**: least Gflops, matches the global/per-channel shape of the signal (same function on all
  tokens), and the zero-init gates give a clean identity start. Plain adaLN (the four-vector, no-gate version)
  conditions fine but gives up the identity initialization.

## Block (the core operation)

```
modulate(x, shift, scale) = x * (1 + scale) + shift        # broadcast (N,d) over tokens

cond = t_emb + c_emb                                        # (N, d)
shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = SiLU_then_Linear(cond, 6d).chunk(6)

x = x + gate_a ⊙ MHSA( modulate(LN1(x), shift_a, scale_a) )   # affine-free LN1
x = x + gate_m ⊙ MLP ( modulate(LN2(x), shift_m, scale_m) )   # affine-free LN2
```

Initialization: zero the modulation `Linear` (weight + bias) ⇒ scale = shift = gate = 0 ⇒ each residual block
is identity at start. Everything else (MHSA with qkv bias, `4×` GELU-tanh MLP, sin-cos positional embeddings,
pre-norm) is a standard ViT block, untouched.

## Working code

```python
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


def modulate(x, shift, scale):
    # x: (N, T, D); shift, scale: (N, D) -> broadcast over the token axis
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Scalar diffusion timestep -> vector: sinusoidal frequency features + 2-layer MLP."""
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
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
    """Class label -> vector (learned table; one extra row = null label for classifier-free guidance)."""
    def __init__(self, num_classes, hidden_size, dropout_prob=0.1):
        super().__init__()
        use_cfg = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        return torch.where(drop_ids, self.num_classes, labels)

    def forward(self, labels, train, force_drop_ids=None):
        if (train and self.dropout_prob > 0) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


class DiTBlock(nn.Module):
    """Pre-norm transformer block with adaLN-Zero conditioning."""
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
    """Conditioned decode head: shift/scale only (no residual to gate)."""
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class DiT(nn.Module):
    """Diffusion transformer with adaLN-Zero conditioning."""
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

        # fixed sin-cos positional embedding
        pos = get_2d_sincos_pos_embed(self.pos_embed.shape[-1],
                                      int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))

        # patch-embed like a Linear; small-std init for label/timestep embeddings
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # adaLN-Zero: zero the modulation projections -> every residual block starts as identity
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        # final head starts by predicting zero
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
        x = self.x_embedder(x) + self.pos_embed   # (N, T, D)
        t = self.t_embedder(t)                    # (N, D)
        y = self.y_embedder(y, self.training)     # (N, D)
        c = t + y                                 # (N, D): sum the conditionings
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)                # (N, T, p*p*out_channels)
        return self.unpatchify(x)                 # (N, out_channels, H, W)
```

(`get_2d_sincos_pos_embed` is the standard fixed sine-cosine 2D positional-embedding helper.)

## Notes on the design choices

- **`(1 + scale)`, not `scale`** — so `scale = 0, shift = 0` leaves the normalized activation unchanged and the
  regressor learns a delta.
- **`cond = t + y`** — fixed dim, one MLP emits all params; symmetric treatment of timestep and class.
- **SiLU before the modulation Linear** — lets the modulation depend nonlinearly on `cond`, capturing
  timestep×class interactions (FiLM permits arbitrary regressors).
- **Gate position (right before the residual add)** — this is what makes a zero gate switch the *whole branch*
  off and give the exact-identity initialization; a scale of zero alone does not (it leaves `LN(x) ≠ 0` flowing
  into the sublayer).
- **6 vectors per block / 2 at the head** — two residual sublayers each need shift/scale/gate; the decode head
  has no residual to gate, so shift/scale suffice.
- **Zero-init the modulation projections and the final linear** — the residual block stack is the identity at
  start and the denoiser predicts zero noise, so a deep stack trains stably and grows its conditioning from a
  no-op.
