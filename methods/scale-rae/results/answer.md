# Scale-RAE

## What it is

A large-scale text-to-image generator that runs flow-matching diffusion **directly in the frozen
semantic representation space** of a pretrained image encoder, rather than in a compressed VAE latent.
A pretrained encoder (SigLIP-2 So400M, patch 14, 224x224) is frozen and emits N=256 tokens of
dimension d=1152; a lightweight ViT decoder, the Representation Autoencoder (RAE) decoder, is trained
to invert those tokens to pixels. A pretrained LLM (Qwen2.5), conditioned on the text prompt plus a
block of 256 learnable query tokens placed at image-generation slots (the MetaQuery interface),
produces per-token conditioning that drives a DiT to generate the 256x1152 representation, which the
RAE decoder renders to pixels. The same frozen encoder feeds the understanding path, so perception and
generation share one space.

The recipe is **simplified by scale**: of the design choices that made
representation-space diffusion work on ImageNet, only the two grounded in high-dimensional-noise
mathematics survive at scale; the rest are small-model crutches and are dropped.

## Key ideas

1. **Diffuse in a frozen representation latent.** One high-dimensional semantic space serves both
   understanding and generation — no second encoder, no compression ceiling. Encoder frozen, only a
   decoder trained.

2. **Width >= token dimension (kept).** A denoiser of width `d < n` (token dim) has an irreducible
   loss floor `L >= sum_{i=d+1}^{n} lambda_i` (tail eigenvalues of `cov(eps - x)`), because diffusion
   noise inflates the data manifold to full rank. So the DiT hidden size is always > 1152, even at
   0.5B (hidden 1280).

3. **Dimension-dependent noise schedule shift (kept).** Per-element SNR scales like
   `sigma(t,n) = (t/(1-t)) sqrt(1/n)`, so larger effective dimension under-noises the target at a given
   `t`. Shift the schedule by `t_m = alpha t_n / (1 + (alpha-1) t_n)`, `alpha = sqrt(m/n)`, using the
   effective data dimension `m = N x d` and base `n = 4096`. Removing it sharply degrades generation.

4. **Drop the wide denoising head.** It existed only to buy width when backbones (~1024) were narrower
   than the latent (1152). Large T2I DiTs (hidden >= 2048) already exceed the latent dimension, so the
   bottleneck is gone; its benefit vanishes past ~2.4B.

5. **Drop noise-augmented decoding.** It is early-training regularization for the train/inference token
   gap; once the generator converges (quickly, at scale) the gap and its benefit saturate.

6. **MetaQuery interface, finetuned LLM.** 256 learnable queries (one per latent token) occupy the
   image-generation slots; the LLM's query-token outputs, optionally projected to the DiT condition
   width, are the DiT condition. Conditioning is applied **per token** (query `i` modulates latent
   token `i`) via adaLN, not as a single pooled vector. The LLM is finetuned instead of frozen.

7. **Decoder quality is a data-composition problem.** Web data for breadth, synthetic for clean style,
   text-rendering data for glyphs; matched data, not raw scale, fixes the hard domains (especially
   text). Decoder loss is the standard L1 + LPIPS + Gram + adversarial, encoder frozen.

8. **Latent-space test-time scaling (free payoff).** Because generations land in the LLM's own space,
   the LLM scores its own candidates directly (prompt-confidence or a yes/no logit) and keeps best-of-N,
   with no decode/re-encode.

## Final objective

Rectified flow with linear path and velocity prediction:

- Path: `x_t = (1 - t) x + t eps`, `eps ~ N(0, I)` (data at `t=0`, noise at `t=1`).
- Target: `v = eps - x` (constant along the path); loss `E ||v_theta(x_t, t) - (eps - x)||^2`.
- Timestep: `t = sigmoid(u)`, `u ~ N(mu, sigma)`, then shifted by `t <- alpha t/(1+(alpha-1)t)`,
  `alpha = sqrt(m/n)`.
- Sampling: Euler, 50 steps, from noise (`t=1`) to data (`t=0`):
  `x_{t-} = x_t + (sigma_{t-} - sigma_t) v_theta(x_t, sigma_t)` with `sigma_t = t`.

Separate optimizer groups: pretrained LLM/projector parameters use lr ~5e-5 with Adam beta2 0.999;
the from-scratch diffusion head uses lr ~5e-4 with beta2 0.95.

## Code

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def tokens_to_grid(x):
    # (B, L, C) -> (B, C, H, H), matching the diffusion head's channel-first path.
    b, l, c = x.shape
    h = int(math.sqrt(l))
    assert h * h == l
    return x.view(b, h, h, c).permute(0, 3, 1, 2).contiguous()


def grid_to_tokens(x):
    # (B, C, H, H) -> (B, L, C)
    return x.permute(0, 2, 3, 1).contiguous().view(x.shape[0], -1, x.shape[1])


class FlowMatching:
    """Rectified flow: x_t=(1-t)x + t*eps, target velocity eps-x."""
    def __init__(self, num_tokens, token_dim, base_dim=4096, steps=1000):
        m = num_tokens * token_dim
        self.side = int(math.sqrt(num_tokens))
        assert self.side * self.side == num_tokens
        self.size_ratio = math.sqrt(base_dim / m)        # 1 / sqrt(m / base_dim)
        self.steps = steps

    def shift(self, t):
        alpha = 1.0 / self.size_ratio                    # sqrt(m / base_dim)
        return alpha * t / (1 + (alpha - 1) * t)

    def sample_timestep(self, batch, device, dtype, mean=0.0, std=1.0):
        u = torch.normal(mean, std, size=(batch,), device=device)
        return self.shift(torch.sigmoid(u)).to(dtype)       # logit-normal, then dimension shift

    def training_loss(self, denoiser, x_tokens, cond):
        x0 = tokens_to_grid(x_tokens)
        x1 = torch.randn_like(x0)
        t = self.sample_timestep(x0.shape[0], x0.device, x0.dtype)
        a, s = (1 - t).view(-1, 1, 1, 1), t.view(-1, 1, 1, 1)
        xt = a * x0 + s * x1
        pred = denoiser(xt, s.flatten(), cond)
        target = x1 - x0
        return ((pred - target) ** 2).flatten(1).mean(1)

    @torch.no_grad()
    def euler_sample(self, denoiser, cond, batch, token_dim, device, num_steps=50):
        x = torch.randn(batch, token_dim, self.side, self.side, device=device)
        ts = self.shift(torch.linspace(1 / num_steps, 1, num_steps, device=device))
        for t_cur, t_next in zip(reversed(ts[1:]), reversed(ts[:-1])):
            cur = t_cur.repeat(batch).to(x.dtype)
            nxt = t_next.repeat(batch).to(x.dtype)
            v = denoiser(x, cur, cond)
            x = x + (nxt - cur).view(-1, 1, 1, 1) * v     # negative step: noise -> data
        return grid_to_tokens(x)


def modulate(x, shift, scale):
    if len(shift.shape) < len(x.shape):
        shift = shift.unsqueeze(1)
    if len(scale.shape) < len(x.shape):
        scale = scale.unsqueeze(1)
    return x * (1 + scale) + shift


def gate(x, value):
    if len(value.shape) < len(x.shape):
        value = value.unsqueeze(1)
    return x * value


class DenoiserBlock(nn.Module):
    """Modern DiT block: RMSNorm, RoPE+QK-norm attention, SwiGLU FFN, adaLN(shift,scale,gate)."""
    def __init__(self, width, heads, cond_dim, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.attn  = NormAttention(width, heads, qkv_bias=True, qk_norm=True, use_rmsnorm=True)
        self.norm2 = RMSNorm(width)
        self.mlp   = SwiGLUFFN(width, int(2 / 3 * mlp_ratio * width))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * width))

    def forward(self, x, c, rope):
        sh1, sc1, g1, sh2, sc2, g2 = self.adaLN(c).chunk(6, dim=-1)
        x = x + gate(self.attn(modulate(self.norm1(x), sh1, sc1), rope=rope), g1)
        x = x + gate(self.mlp(modulate(self.norm2(x), sh2, sc2)), g2)
        return x


class Denoiser(nn.Module):
    """Plain DiT over the 16x16 representation grid. No wide head. Width kept > token dim."""
    def __init__(self, num_tokens, token_dim, width, depth, heads, cond_dim, dropout_prob=0.1):
        super().__init__()
        side = int(math.sqrt(num_tokens))
        self.cond_dim, self.token_dim, self.side = cond_dim, token_dim, side
        self.x_embed = PatchEmbed(side, 1, token_dim, width, bias=True)
        self.t_embed = GaussianFourierEmbedding(cond_dim)
        self.y_embed = ConditionEmbedder(cond_dim, dropout_prob)
        self.pos = nn.Parameter(sincos_2d(width, num_tokens), requires_grad=False)
        self.blocks = nn.ModuleList([DenoiserBlock(width, heads, cond_dim) for _ in range(depth)])
        self.final = FinalLayer(width, patch_size=1, out_channels=token_dim,
                                z_channels=cond_dim, use_rmsnorm=True)
        self.rope = VisionRotaryEmbeddingFast(width // heads // 2, pt_seq_len=side)

    def unpatchify(self, x):
        b, n, c = x.shape
        h = int(math.sqrt(n))
        return x.view(b, h, h, c).permute(0, 3, 1, 2).contiguous()

    def forward(self, x, t, y):                  # x:(B,D,H,H) t:(B,) y:(B,N,cond_dim)
        s = self.x_embed(x) + self.pos
        t = self.t_embed(t)
        y = self.y_embed(y, self.training)
        if len(t.shape) < len(y.shape):
            t = t.unsqueeze(1).expand(-1, y.shape[1], -1)
        c = F.silu(t + y)                                       # per-token condition
        for blk in self.blocks:
            s = blk(s, c, self.rope)
        return self.unpatchify(self.final(s, c))                 # velocity grid


class GenerationModel(nn.Module):
    """LLM + frozen encoder + diffusion head, joined by learnable image-slot queries."""
    def __init__(self, llm, encoder, decoder, denoiser, flow, num_queries=256, cond_dim=2048):
        super().__init__()
        self.llm, self.encoder, self.decoder = llm, encoder, decoder
        self.denoiser, self.flow = denoiser, flow
        self.latent_queries = nn.Parameter(torch.randn(num_queries, llm.hidden_size) / llm.hidden_size ** 0.5)
        self.mm_projector = MLPProjector(encoder.dim, llm.hidden_size)
        self.diff_head_projector = nn.Linear(llm.hidden_size, cond_dim)
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def forward(self, prompt_ids, target_image, image_token_indices, answer_image_mask):
        with torch.no_grad():
            target = self.encoder(target_image)                 # (B*M, 256, 1152)

        b = prompt_ids.shape[0]
        m = answer_image_mask.shape[1]
        q = self.latent_queries.unsqueeze(0).expand(b, -1, -1).repeat(1, m, 1)
        input_embeds = insert_image_slot_queries(self.llm, prompt_ids, q, image_token_indices)
        hidden = self.llm(inputs_embeds=input_embeds, output_hidden_states=True).hidden_states[-1]
        patch_h = gather_image_slots(hidden, image_token_indices)        # (B, M*256, hidden)
        cond = self.diff_head_projector(patch_h.view(b * m, self.latent_queries.shape[0], -1))

        target = target.view(b * m, self.latent_queries.shape[0], -1)
        loss = self.flow.training_loss(self.denoiser, target, cond).view(b, m)
        return (loss * answer_image_mask.float()).sum() / (answer_image_mask.sum() + 1e-8)


def build_optimizer(model, lr=5.65e-5, diff_lr=5.65e-4, weight_decay=0.0):
    diff_names = [n for n, _ in model.named_parameters()
                  if "denoiser" in n or "diff_head_projector" in n]
    base_decay, base_nodecay, diff_decay, diff_nodecay = [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        no_decay = p.ndim < 2 or name.endswith(".bias")
        is_diff = name in diff_names
        group = diff_nodecay if is_diff and no_decay else diff_decay if is_diff else base_nodecay if no_decay else base_decay
        group.append(p)
    return torch.optim.AdamW(
        [
            {"params": base_decay, "weight_decay": weight_decay},
            {"params": base_nodecay, "weight_decay": 0.0},
            {"params": diff_decay, "lr": diff_lr, "weight_decay": weight_decay, "betas": (0.9, 0.95)},
            {"params": diff_nodecay, "lr": diff_lr, "weight_decay": 0.0, "betas": (0.9, 0.95)},
        ],
        lr=lr,
        betas=(0.9, 0.999),
    )
```

Diffusion head configs keep hidden width above the 1152 token dimension at every scale: 0.5B (1280),
2.4B (2048), 3.3B (2304), 5.5B (3072), 9.8B (4096), scaling width before depth.
