Latent diffusion has settled on a comfortable but awkward division of labor: train a variational autoencoder to compress an image into a small, low-dimensional latent, then run the diffusion model in that latent instead of in pixel space. The latent is cheap to generate in, but it is shaped entirely by a pixel-reconstruction objective, so it is not semantic. The moment I want a single model that both *understands* an image and *generates* one, this bites: understanding wants the high-dimensional, semantically structured features of a representation encoder, while generation wants the compressed VAE latent, and the two spaces are different. That forces a two-tower design, and on top of it aggressive compression imposes a hard reconstruction ceiling — information discarded at encode time never comes back. The representation-space alternative is to diffuse directly in the frozen token space of a pretrained vision encoder, so that one space serves both seeing and generating, with no second encoder and no compression ceiling. In a controlled setting — class-conditional ImageNet, encoder frozen, only a decoder trained — this already converges faster and reconstructs better than the VAE pipeline. But ImageNet is a best case: fixed resolution, curated object-centric content, a single class label as the entire condition. The real question is whether the win is an artifact of that easy regime or whether it carries to large-scale, freeform text-to-image generation, with open prompts, billion-parameter models, rendered text and typography, and an LLM as the conditioning source.

I propose Scale-RAE: a large-scale text-to-image generator that runs flow-matching diffusion directly in the frozen semantic representation space of a pretrained image encoder, with the recipe deliberately *simplified by scale*. A SigLIP-2 So400M encoder at patch size 14 and 224x224 input is frozen and emits $N = 256$ tokens of dimension $d = 1152$; a lightweight ViT decoder, the Representation Autoencoder (RAE) decoder, is trained to invert those tokens back to pixels while the encoder stays frozen; a pretrained Qwen2.5 LLM is conditioned on the text prompt plus a block of 256 learnable query tokens placed at the image-generation slots, and the query-token hidden states drive a DiT that generates the $256 \times 1152$ representation, which the RAE decoder renders to pixels. The same frozen encoder feeds the understanding path, so perception and generation share one space.

The defining objective is rectified flow with a straight path and velocity prediction. I put data at $t=0$ and noise at $t=1$, draw $\varepsilon \sim \mathcal{N}(0, I)$, and define
$$x_t = (1-t)\,x + t\,\varepsilon,$$
whose time-derivative is the constant velocity $\varepsilon - x$, so the network $v_\theta$ is trained with plain MSE to that target,
$$\mathcal{L} = \mathbb{E}\,\lVert v_\theta(x_t, t) - (\varepsilon - x) \rVert^2,$$
and sampling integrates the ODE backward from noise with Euler steps $x_{t^-} = x_t + (\sigma_{t^-} - \sigma_t)\,v_\theta(x_t, \sigma_t)$ where $\sigma_t = t$, so the step size is negative and we march from noise to data over a fixed 50-step budget.

Two of the design choices that made representation-space diffusion work survive scaling because they are properties of high-dimensional noising, not of any model size, and I keep both. The first is a hard width constraint. Diffusion adds Gaussian noise across the entire latent at every noise level, which spreads the data's support over the whole ambient space — whatever thin manifold the real tokens lie on, the noised distribution is full-rank — so the denoiser needs capacity that scales with the *full* token dimension, not the intrinsic dimension. Make this sharp: if the denoiser's effective width is $d$ and the token dimension is $n$, then at high noise the target is essentially the noise direction $\varepsilon - x$; a width-$d$ model can represent only a $d$-dimensional output subspace, so the best it can do is capture the top $d$ eigen-directions of $\mathrm{cov}(\varepsilon - x)$ and pay for the rest, giving a squared-error floor
$$\mathcal{L} \ge \sum_{i=d+1}^{n} \lambda_i$$
in the tail eigenvalues. If $d < n$ that floor is strictly positive and no training removes it. The practical reading is non-negotiable: the DiT hidden width must stay above the 1152 token dimension at every scale, so even my smallest 0.5B model uses hidden width 1280 rather than the more common 1024.

The second survivor is a dimension-dependent noise-schedule shift. Recovering a roughly constant signal from $n$ noisy coordinates has uncertainty that pools down like $1/\sqrt{n}$; in the rectified-flow parameterization the noise-to-signal ratio at time $t$ is $t/(1-t)$, so the per-element uncertainty after pooling $n$ coordinates goes like $\sigma(t,n) = \frac{t}{1-t}\sqrt{1/n}$. For a *fixed* nominal $t$, more coordinates means the target is effectively *less* corrupted — an easier problem than the timestep label claims — so to keep difficulty calibrated as the dimension grows I must push the schedule toward more noise. The resolution-shift trick from rectified-flow text-to-image does this for spatial resolution; the key generalization is that "more coordinates" should mean more *effective data dimension*, and a representation token has both many tokens $N$ and many channels $d$, so the controlling quantity is $m = N \times d$, the total number of scalar coordinates being noised. Against a base dimension $n = 4096$ I set $\alpha = \sqrt{m/n}$ and reparameterize timesteps by
$$t_m = \frac{\alpha\,t_n}{1 + (\alpha - 1)\,t_n}.$$
This is not optional polish: dropping it makes generation quality collapse, with the compositional prompt-following score falling to roughly half. Concretely I draw $t$ from a logit-normal — $u \sim \mathcal{N}(\mu, \sigma)$, $t = \mathrm{sigmoid}(u)$, concentrating timesteps in the middle of the path where the signal is richest — and then apply the shift.

What makes Scale-RAE *Scale*-RAE is that the remaining two ImageNet ingredients turn out to be small-model crutches, and I drop them. The wide denoising head existed only because ImageNet-scale denoisers were often around width 1024, narrower than the 1152 latent, and widening the whole backbone costs quadratically; the cheap patch was a shallow, very wide head bolted onto a normal-width backbone to buy the required width locally. But its justification is purely "the backbone was too narrow," and modern T2I transformers at billions of parameters already carry hidden widths of 2048 and up. So I predict — and confirm — that the head helps only when the backbone barely clears 1152 (a double-digit boost at 0.5B) and its advantage all but vanishes once the hidden size is 2048+ (by ~2.4B). I therefore use a plain DiT with no special head, respecting only the width floor. Likewise, noise-augmented decoding — training the decoder on perturbed tokens $z + n$, $n \sim \mathcal{N}(0, \sigma^2 I)$ with $\sigma \sim |\mathcal{N}(0, \tau^2)|$ — was there to close the gap between the decoder's clean training tokens and the slightly off-manifold tokens a diffusion sampler emits. That gap is largest early, when the generator is weak, and shrinks as it converges near the manifold; the gains are visible before ~15k steps and negligible after, and $\tau$ cannot be cranked up because too much perturbation destabilizes decoder training (it caps around 0.2). At scale the model blows past the early phase quickly, so I drop the augmentation too. Scale simplifies the recipe exactly where its justification was weakest.

The decoder I keep as the standard frozen-encoder reconstruction stack — L1 for pixel fidelity, LPIPS for perceptual quality, a Gram/texture term, and an adversarial term for sharpness — but the real lesson is that its open-world quality is a *data-composition* problem, not a pure scaling one. Scaling decoder training data from ImageNet to web and synthetic imagery barely moves ImageNet reconstruction (already in-distribution), gives moderate gains on diverse web photos, and does almost nothing for rendered text, because generic web and synthetic images carry very little crisp glyph structure. Only *matched* data fixes the hard domain: adding a rendered-text corpus drops text reconstruction error by roughly a third (rFID from ~2.6 to ~1.6). So I compose the decoder's training set deliberately — web for breadth, synthetic for clean style, text-rendering for glyphs. One web-scale wrinkle: the strong DINO-S/8 patch discriminator that was stable on curated ImageNet does not converge on the messier web distribution, so I step down to a weaker, more stable DINO-S/16 and turn the adversarial term on only after the reconstruction terms have settled.

The genuinely new part is the text condition, and I reuse the LLM rather than inventing a bespoke text encoder. Following the MetaQuery interface, I place a block of 256 learnable query tokens at the image-generation slots, let the LLM process the prompt together with those queries, and read out the query positions' hidden states as the conditioning; a small MLP connector maps frozen visual tokens into the LLM's space for the understanding path, and a linear projector maps the query hidden states into the DiT's conditioning width. The symmetry is the structural payoff: both paths hang off the same frozen encoder — the LLM reads its tokens to see, and the denoiser is trained to *produce* its tokens — so it is one shared space. I use exactly 256 queries because the denoiser generates 256 tokens, and that one-to-one alignment changes the wiring: instead of collapsing the condition to a single pooled vector that adaLN broadcasts identically (as in class-conditional DiT), I keep the condition as a length-256 sequence and combine it per token, so query $i$ modulates latent site $i$. The conditioning entering each adaLN block is $c = \mathrm{SiLU}(t_{\text{embed}} + y_{\text{query}})$, with the timestep part broadcast and the query part per-token, giving localized control at no extra cost because the shapes already line up. Contrary to the received wisdom that scaling the LLM does not help, I finetune the LLM rather than freeze it, so it can reshape its latent space toward generation, and pair it with a large denoiser that can exploit richer text representations.

The denoiser itself reshapes the 256 tokens into their natural 16x16 grid — $(B, 256, 1152) \to (B, 1152, 16, 16)$ — patch-embeds with patch size 1 so every representation token stays its own spatial site, adds fixed 2D sin-cos positional embeddings, runs a stack of modern LightningDiT-style blocks (RMSNorm, QK-normalized attention with rotary position encoding, SwiGLU feed-forward, Gaussian Fourier timestep embeddings), and unpatchifies a final head back to a velocity grid of the same shape, with the adaLN modulation and final layers zero-initialized so the head starts gently. I scale width before depth, since the width floor is the binding constraint: 0.5B (1280), 2.4B (2048), 3.3B (2304), 5.5B (3072), 9.8B (4096), always above 1152. One practical detail that bit me: the LLM is pretrained while the denoiser is from scratch, so a single learning rate either blows the LLM apart or starves the denoiser; I decouple the optimizer groups, giving the pretrained LLM and connector a small learning rate (~5.65e-5) with Adam $\beta_2 = 0.999$ and the from-scratch diffusion head a large one (~5.65e-4) with $\beta_2 = 0.95$.

There is a dividend I get for free, and it is the reason the shared space was worth all this. Because the denoiser emits tokens in the *same* space the LLM understands, the LLM can read its own generations directly, without decoding to pixels and re-encoding. So test-time scaling happens entirely in latent space: generate several candidates, feed each candidate's latent back into the LLM, score it — by the LLM's aggregate confidence on the prompt tokens conditioned on the candidate, or by asking a yes/no "does this image match the prompt?" and reading the probability of "yes" — and keep best-of-$N$, no render and no re-encode. A VAE pipeline cannot do this, because its generation latent is alien to the understanding encoder; it would have to decode and re-encode just to look at what it made. That asymmetry is the concrete payoff of generating where you perceive.

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
