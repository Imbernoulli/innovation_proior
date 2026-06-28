OK, let me think this through from scratch. I have a result I trust in a narrow setting and I want to know if it survives in a wide one. The narrow setting is the Representation Autoencoders line: class-conditional ImageNet, where I diffuse not in a VAE latent but directly in the frozen token space of a pretrained representation encoder — freeze the encoder, train a small decoder to invert its tokens to pixels, and run a diffusion transformer on those tokens. In that controlled world it converges faster and reconstructs better than the VAE pipeline. The wide setting I actually care about: freeform text-to-image, open prompts, billions of parameters, an LLM in the loop. ImageNet is a best case — fixed resolution, curated content, one class label as the entire condition. So the honest question is whether anything about that win was an artifact of the easy regime.

Let me first be clear about why I even want to leave the VAE behind, because if the only reason were "it's a bit faster on ImageNet" I wouldn't bother re-engineering a whole pipeline. The deeper pain is this: a VAE latent is low-dimensional and shaped purely for pixel reconstruction. It is not semantic. So if I want one model that both *understands* an image and *generates* one, I'm stuck running two different visual spaces — a high-dimensional semantic encoder for perception and a separate compressed VAE latent for generation. That's the two-tower awkwardness everyone lives with. And compression has a hard cost: information thrown away at encode time can never come back, so reconstruction has a ceiling. If instead I could generate directly in the *same* semantic token space the model already uses to see, both problems dissolve at once: one space, no second encoder, and no compression ceiling because the tokens are high-dimensional. So the representation-space route isn't a minor efficiency tweak — it's the candidate that could collapse understanding and generation onto a single space. Whether it's worth the trouble depends entirely on whether the ImageNet win survives scale, so let me not assume it does.

Now, why did people think you *can't* generate in such a space? The folklore was that these features are too "abstract." But there's a real reason underneath, and I should state it precisely because it tells me what will break. A diffusion model adds Gaussian noise across the whole latent during training, at every noise level. That noise spreads the data's support over the entire ambient space — whatever thin manifold the real tokens lie on, the noised distribution is full-rank. So the denoiser has to have capacity that scales with the *full* token dimensionality, not the intrinsic dimension of the data. For a VAE latent with a few dozen channels that's free. For a 1152-dimensional semantic token it is not. Let me make the consequence sharp, because it might constrain the architecture before I even start. Suppose my denoiser's effective output rank is `d` and the token dimension is `n`. At the highest noise level the target is essentially the noise direction `eps - x`; look at the covariance `C = cov(eps - x)` with eigenvalues `lambda_1 >= ... >= lambda_n`. A model whose output lives in a `d`-dimensional subspace can only place its prediction in `d` of those directions; the best it can do is align that subspace with the top `d` eigen-directions and eat the energy in the rest.

Let me actually check that this gives the floor I think it does, rather than just assert it, because the whole architecture rule hangs off it. Take a tiny case, `n = 3`, with `cov(eps - x)` having eigenvalues `(0.7, 0.2, 0.1)`. The expected squared norm of the target is `E||eps - x||^2 = trace(C) = 0.7 + 0.2 + 0.1 = 1.0`. A width-1 (rank-1) predictor can fully capture exactly one eigen-direction; aligning it with the top one captures `0.7`, leaving residual `1.0 - 0.7 = 0.3`. And `0.3` is precisely `lambda_2 + lambda_3` — the sum of the tail eigenvalues below the cutoff `d = 1`. So the best-case residual is `sum_{i=d+1}^{n} lambda_i`, exactly. For `d = 2` it would be `lambda_3 = 0.1`, and only at `d = 3 = n` does it hit zero. So the squared-error floor is `L >= sum_{i=d+1}^{n} lambda_i`: strictly positive whenever `d < n`, and no amount of training removes it because it's the energy the model structurally cannot represent. The practical reading is blunt — the denoiser's effective width must be at least the token dimension, ~1152 here, or it sits above a nonzero loss forever. That's rule one, and it's a consequence of the eigenvalue arithmetic, not a hyperparameter I can tune away.

That immediately created a tension in the ImageNet recipe, and I want to remember how it was patched, because the patch is exactly the kind of thing that might be a small-scale crutch. ImageNet-scale denoisers were often around width 1024 — *narrower* than 1152. Widening the whole backbone to clear 1152 is expensive: transformer compute grows quadratically in width. The fix was a shallow but very wide head bolted onto a normal-width backbone — the backbone produces a conditioning signal `z_t = M(x_t | t, y)` and a wide head produces the velocity `v_t = H(x_t | z_t, t)`, with the head's width (~2688) comfortably above 1152. So you buy the required width locally and cheaply instead of paying for it across every backbone layer. Clever for ImageNet. But notice *why* it exists: only because the backbone happened to be narrower than the latent. Hold that thought.

Next ImageNet ingredient: the noise schedule. There's a subtle dimension effect I have to get right or training silently fails. Think about recovering a roughly constant signal buried in noise from `n` independent measurements. Average them and the uncertainty drops like `1/sqrt(n)`. In the rectified-flow parameterization where `x_t = (1-t) x + t eps`, the ratio of noise scale to signal scale at time `t` is `t/(1-t)`, and the per-element uncertainty after pooling `n` elements goes like `sigma(t, n) = (t/(1-t)) * sqrt(1/n)`. Let me make the dependence concrete by plugging numbers at a fixed nominal `t = 0.5`: at `n = 256`, `sigma = 1 * sqrt(1/256) = 0.0625`; at `n = 1024`, `0.03125`; at `n = 4096`, `0.015625`. So going from 256 to 4096 elements at the *same* timestep cuts the effective corruption by `sqrt(256/4096) = 0.25` — a factor of four. For a fixed nominal `t`, more elements means the signal is effectively *less* corrupted; the model sees an easier problem than the timestep claims. To keep the difficulty calibrated when the effective dimension grows, I must push the schedule toward more noise.

The resolution-shift trick from the rectified-flow text-to-image line is built for exactly this, for spatial resolution: reparameterize timesteps by `t_m = alpha t_n / (1 + (alpha-1) t_n)` with `alpha = sqrt(m/n)`. The generalization I want is that "more elements" shouldn't only mean more pixels — it should mean more *effective data dimension*. A representation token has both many tokens `N` and many channels `d`, so the quantity that controls corruption is `m = N x d`, the total number of scalar coordinates being noised, not just the spatial count. So I'll use `m = N x d` against a base dimension (taken as `n = 4096`), `alpha = sqrt(m/n)`, and apply the same shift. Before I trust it, let me check that this formula actually does the thing the SNR argument demands — that the shifted timestep is the one which restores the base-dimension difficulty — because if it doesn't, I'm patching the symptom with an unrelated curve.

Concretely: at the base dimension `n = 4096`, nominal `t = 0.5` gives `sigma = 0.015625`. My representation latent has `m = 256 x 1152 = 294912`. What timestep `t*` at dimension `m` would reproduce that *same* `sigma`? Solve `(t*/(1-t*)) sqrt(1/m) = 0.015625`, i.e. `t*/(1-t*) = 0.015625 * sqrt(294912) = 8.4853`, so `t* = 8.4853 / 9.4853 = 0.8946`. Now run the shift formula on the same nominal `t = 0.5`: `alpha = sqrt(294912/4096) = sqrt(72) = 8.4853`, and `t_m = 8.4853 * 0.5 / (1 + 7.4853 * 0.5) = 4.2426 / 4.7426 = 0.8946`. They agree to every digit. That's not a coincidence I assumed — the shift's `alpha = sqrt(m/n)` is exactly the factor that turns "the SNR I'd have at `t` in dimension `n`" into "the same SNR at the shifted `t` in dimension `m`." So the dimension shift *is* the SNR-matching reparameterization, not a heuristic that happens to help. The formula also passes the boundary sanity check: `t = 0` maps to `0` and `t = 1` maps to `1`, so it only redistributes interior timesteps and never breaks the endpoints. And it's monotone upward — checking a few points, `0.25 -> 0.739`, `0.5 -> 0.895`, `0.75 -> 0.962` — every interior `t` moves toward more noise, which is the direction the SNR argument said I needed. Its justification is purely the mathematics of high-dimensional noising, which doesn't go away as models get bigger, so I'd expect this one to remain essential at scale; I'll come back and pressure-test that expectation rather than bank it.

Third ImageNet ingredient: noise-augmented decoding. The decoder is trained on *true* encoder tokens `z`. But at generation time it's fed tokens produced by the diffusion sampler, which are never exactly on the true-token manifold — they're a little off. Train/inference mismatch. The fix is to train the decoder on perturbed tokens `z + n`, with `n ~ N(0, sigma^2 I)` and `sigma` itself drawn from a half-normal `|N(0, tau^2)|`, so the decoder learns to tolerate small deviations. It's a robustness/regularization trick. Reasonable. But it's the kind of thing that should matter most when the diffusion samples are *bad* — i.e., early in training or for a weak generator — and less once the generator is strong and lands close to the manifold. Another candidate crutch.

So before touching text-to-image I have four inherited pieces: (1) width >= token dimension, justified by the loss-floor arithmetic I just worked; (2) the dimension-dependent noise shift, justified by the SNR-matching identity I just checked; (3) the wide denoising head, justified only by "backbones were too narrow"; (4) noise-augmented decoding, justified by a train/inference gap that shrinks as the generator improves. Two of these rest on dimension mathematics that hold for any model size; two rest on regime-specific weaknesses. My working hypothesis is that scale leaves (1) and (2) standing and erodes (3) and (4) — but that's a hypothesis from the structure of the justifications, and each one needs to be reasoned about in the new regime rather than waved through.

Start with the decoder, because nothing else matters if I can't render the tokens to pixels in the open world. On ImageNet the decoder only ever had to invert curated object-centric images at one resolution. Freeform T2I has to handle web photos, synthetic art, and — the nasty one — rendered text and typography, where a one-pixel error turns an 'e' into a 'c'. My first instinct is "just train the decoder on much more data and it'll generalize." Let me pressure-test that instinct rather than trust it, by reasoning per domain about what more data should and shouldn't fix. On ImageNet itself the decoder was already in-distribution, so more data barely moves reconstruction there. On diverse natural images like a broad web set, moderate improvement — the decoder sees more of that world. And on rendered text? Here the instinct should break: generic web and synthetic images contain very little crisp glyph structure, so adding them gives the decoder almost no signal about how to place sharp edges in a glyph — it should still smear text. The thing that *should* fix text is text-specific data, and the recorded measurement matches that prediction: adding a rendered-text corpus drops the held-out text reconstruction rFID by roughly a third (from ~2.64 with ImageNet-only data to ~1.62 once text data is included), while ImageNet-only scaling left it essentially flat. So the lesson isn't "more data"; it's "matched data." That reframes decoder training as a *data-composition* problem — web for breadth, synthetic for clean style, text-rendering for glyphs — rather than a pure scaling problem. And it lets me keep the decoder loss as the standard reconstruction stack (L1 for pixel fidelity, LPIPS for perceptual, a Gram/texture term, and an adversarial term for sharpness), encoder frozen throughout. One wrinkle I hit at web scale: the adversarial recipe that was stable on ImageNet isn't. The strong patch discriminator that worked on curated images (a DINO-S/8) won't converge on the messier web distribution; I have to step down to a weaker, more stable discriminator (DINO-S/16), and start the adversarial term only after the reconstruction terms have settled. That's a training-stability detail, not a change to the idea.

Now the four inherited pieces, in the new regime. The noise shift first, since I expected it to be fundamental. The way to test that is to ask what the math predicts if I *don't* apply it. Without the shift, at the larger effective dimension `m = 294912` the model experiences, at every nominal `t`, an effective `sigma` that's `sqrt(m/4096) = 8.49x` smaller than the base — by the very numbers I computed above, `t = 0.5` behaves like a much cleaner image than its label says. So it never trains at the noise regime it'll actually be sampled in, and the deficit isn't marginal, it's nearly an order of magnitude in effective SNR. I'd therefore predict a large, not small, degradation without the shift — and the recorded ablation is consistent with that: prompt-following roughly halves when the shift is removed (order ~24 vs ~50 on the compositional metric). The size of that gap is in line with an SNR mismatch that big, so (2) holds at scale exactly as the dimension math says it should. Keep it, base `n = 4096`, `alpha = sqrt(m/n)`.

The wide head, (3). Its whole reason for being was: backbone width < latent dimension, and widening the backbone is too expensive. So the clean test is whether that premise is even true for a *large* T2I denoiser. Modern T2I transformers at billions of parameters have hidden widths of 2048 and up — already far above the 1152 token dimension. The bottleneck the wide head was invented to fix doesn't exist there; the backbone is naturally wide enough. By rule (1) the head should only matter while the backbone width barely clears 1152, and stop mattering once it's comfortably wider. That's a prediction with a shape: a large effect at the smallest model, fading to nothing as width grows. The ablation across denoiser sizes matches that shape — at 0.5B, where width only just exceeds the token dimension, the head gives a double-digit gain (~+11 on the compositional metric); at 2.4B and beyond, where hidden size is 2048+, the advantage all but vanishes. So (3) reads as a capacity crutch tied to the small regime, not a fundamental requirement, and I drop it for scalable training — a plain denoiser, no special head, as long as I respect rule (1) and keep hidden width above 1152 even at the small end (so my 0.5B uses hidden 1280 > 1152, not 1024).

Noise-augmented decoding, (4). The gap it closes is between the decoder's clean training tokens and the diffusion sampler's slightly-off tokens. Early in training the sampler is bad and its tokens are far off-manifold, so the augmentation helps; as training proceeds and the generator lands close to the true manifold, there's little gap left to close. So the prediction is gains concentrated early and saturating later — and the measurements show benefit before ~15k steps and negligible after. At T2I scale the model blows past that early phase quickly, so the lifetime benefit is small, and I can't crank `tau` to force more, because too much perturbation makes the decoder itself hard to train (cap ~0.2). Net: (4) is optional at scale, drop it too. So scale *simplifies*: keep the dimension-justified pieces (width >= dim, noise shift), discard the regime-specific crutches (wide head, noise augmentation). The recipe gets cleaner exactly where its justification was weakest — which is the asymmetry I hypothesized from the structure of the four justifications, now borne out piece by piece.

Now the genuinely new part: text-to-image needs a *text* condition, and an LLM is the natural source. How do I turn a text LLM into a conditioner for a representation-space denoiser? I don't want to invent a bespoke cross-attention text encoder; I want to reuse the LLM's own representations. The MetaQuery interface gives a clean route: a block of learnable query tokens placed where the image tokens will be generated. Let the LLM process the prompt plus those image-slot queries, then read out the query positions' hidden states. Those become my conditioning. The queries learn to "interrogate" the prompt and summarize it into exactly the form the denoiser needs. How many queries? The denoiser is going to produce `N = 256` representation tokens, so it's natural to use `Q = 256` query tokens — one conditioning slot per generated token. That alignment turns out to matter more than it first looks, and I'll come back to it.

A small MLP connector maps the LLM's hidden dimension to the denoiser's conditioning dimension, and a separate small projector maps frozen visual tokens *into* the LLM's space for the understanding path. Note the symmetry that falls out: the understanding path and the generation path both hang off the *same frozen encoder*. The LLM reads that encoder's tokens to see, and the denoiser is trained to *produce* that same encoder's tokens. One shared space — which is the whole reason I committed to representation-space in the first place, and it costs nothing extra here.

There's a received bit of wisdom I want to challenge. An earlier version of this query-interface idea reported that scaling the LLM didn't help much. But that was with a frozen LLM and modest denoisers. If I *finetune* the LLM so it can reshape its latent space toward generation, and pair it with a *large* denoiser that can actually exploit richer text representations, the earlier null result may not transfer. I'll keep the LLM trainable rather than frozen, and treat "LLM scaling is futile" as an open question rather than a settled fact.

Now back to the alignment between the 256 queries and the 256 latent tokens, because it changes how I wire the conditioning into the denoiser. In a plain class-conditional DiT the condition is a single pooled vector that adaLN broadcasts identically to every token. But here my condition is a *sequence* of 256 query outputs, naturally in one-to-one correspondence with the 256 latent tokens the denoiser is generating. So instead of collapsing them to one vector, I can keep the conditioning *per token*: the timestep embedding is one global vector, but the query-derived condition is a length-256 sequence, and I add them position-wise so token `i` of the latent is modulated by query `i`'s representation. That gives spatially/semantically localized control instead of a single global knob, and it costs nothing because the shapes already line up. Concretely the conditioning fed to each adaLN block is `c = t_embed + query_embed`, broadcast in the timestep part and per-token in the query part, then a SiLU before the modulation projection.

Let me assemble the denoiser itself, now that the constraints are fixed. The 256 latent tokens have a natural 16 by 16 patch grid, so I reshape `(B, 256, 1152)` into `(B, 1152, 16, 16)` before the diffusion head, exactly the layout a vision DiT expects. Patch size 1 means every representation token remains its own spatial site. I patch-embed that grid, add fixed 2D sin-cos positional embeddings, run a stack of transformer blocks, and unpatchify the final head back to a velocity grid with the same shape as the input. Each block is a modern DiT block: normalize, attention, normalize, MLP, with adaLN modulation driving (shift, scale, gate) around both sublayers and the modulation/final layers zero-initialized so the diffusion head starts gently. I'll use the LightningDiT-style block — RMSNorm instead of LayerNorm, a SwiGLU FFN instead of plain GELU-MLP, QK-normalized attention, rotary position encoding on the token grid, and Gaussian Fourier timestep embeddings — because those are the choices that make DiTs converge fast, and they're orthogonal to everything I've reasoned about. The conditioning dimension into adaLN is the LLM/query channel; the hidden width is chosen per model size but always kept above 1152 (rule 1), and I scale by adding width before depth, since the width floor is the binding constraint and wide-over-deep is what the vision-scaling evidence favors.

The objective is rectified-flow. Define the straight path with data at `t=0` and noise at `t=1`: `x_t = (1-t) x + t eps`, `eps ~ N(0, I)`. The velocity along this path is `dx_t/dt = eps - x`, a constant in `t`, so the target is simply `eps - x` and the loss is plain MSE between the network's prediction and that target. The only nonstandard thing is *which* `t` I draw and how I shift it. I sample `t` from a logit-normal — draw `u ~ N(mu, sigma)`, set `t = sigmoid(u)` so timesteps concentrate in the middle of the path where the learning signal is richest — and then apply the dimension shift `t <- alpha t / (1 + (alpha-1) t)` with `alpha = sqrt(m/n)`. Sampling is the ODE run backward: start from noise at `t=1`, take Euler steps down to `t=0`. Let me trace the sampler direction once to make sure the signs march the right way, because an off-by-sign here silently inverts noise and data. With `sigma_t = t`, the step is `x_{t-} = x_t + (sigma_{t-} - sigma_t) v_theta`. Building the shifted grid for a tiny 5-step run gives ascending shifted timesteps `[0.68, 0.85, 0.93, 0.97, 1.0]`; the loop walks them in reverse as `(cur, next)` pairs, so the first step is `cur=1.0, next=0.97` with `next - cur = -0.029`, then `-0.044`, `-0.077`, `-0.170` — every step negative, `t` strictly decreasing from ~1 toward 0. So the integration does run noise -> data over the fixed budget (50 steps in practice), as intended.

One more thing about wiring the latent into the diffusion loss inside the unified model. During training I encode the target image with the frozen encoder to get the clean tokens `x`, insert the 256 learnable queries into the answer image-token slots, run the LLM, gather the hidden states at those slots, optionally project them to the diffusion condition width, and run flow matching on `x` conditioned on that. If I use the block variant, I add an attention bias that unmasks future attention only among tokens inside the same image block while the rest of the sequence stays causal; in the plain query path I can keep the normal causal pass and still get all 256 query hidden states at once. The LLM is trained with its usual language loss on the text path and the denoiser with the flow-matching loss on the image path, jointly.

And the optimizer groups — this bit me in practice so it's worth stating. The LLM is *pretrained*; the denoiser is *from scratch*. If I put them under one learning rate, either the LLM gets blown apart by a denoiser-sized learning rate or the denoiser starves under an LLM-sized one. So I decouple the parameter groups: a small learning rate for the LLM and connector (~5.66e-5) and a large one for the diffusion head (~5.66e-4), and I also give the diffusion head a lower Adam second-moment beta, 0.95, while the pretrained side keeps 0.999. One optimizer can carry those groups, but the update scales are different where they need to be different.

There's a payoff at inference I didn't have to design but get for free, and it falls directly out of the shared space. Because the denoiser produces tokens in the *same* space the LLM understands, the LLM can read its own generations *directly*, without decoding to pixels and re-encoding. So I can do test-time scaling entirely in latent space: generate several candidates, feed each candidate's latent back into the LLM, and score it — either by the LLM's aggregate confidence on the prompt tokens when conditioned on the candidate, or by asking the LLM a yes/no "does this image match the prompt?" and reading the probability of "yes" — then keep the best of `N`. No render, no re-encode, all in feature space. A VAE pipeline can't do this: its generation latent is alien to the understanding encoder, so it would have to decode to pixels and re-encode just to let the model look at what it made. That asymmetry is the concrete dividend of generating where you perceive.

Let me write the core of it as code.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def tokens_to_grid(x):
    # Feature lists are converted to a channel-first square grid before denoising.
    b, l, c = x.shape
    h = int(math.sqrt(l))
    assert h * h == l
    return x.view(b, h, h, c).permute(0, 3, 1, 2).contiguous()


def grid_to_tokens(x):
    return x.permute(0, 2, 3, 1).contiguous().view(x.shape[0], -1, x.shape[1])


# Straight path: x_t=(1-t)x + t eps. Data is t=0, noise is t=1.
class FlowMatching:
    def __init__(self, num_tokens, token_dim, base_dim=4096, steps=1000):
        m = num_tokens * token_dim
        self.side = int(math.sqrt(num_tokens))
        assert self.side * self.side == num_tokens
        self.size_ratio = math.sqrt(base_dim / m)        # 1/sqrt(m/base_dim)
        self.steps = steps

    def shift(self, t):
        alpha = 1.0 / self.size_ratio                    # sqrt(m/base_dim)
        return alpha * t / (1 + (alpha - 1) * t)

    def sample_timestep(self, batch, device, dtype, mean=0.0, std=1.0):
        u = torch.normal(mean, std, size=(batch,), device=device)
        return self.shift(torch.sigmoid(u)).to(dtype)

    def training_loss(self, denoiser, x_tokens, cond):
        x0 = tokens_to_grid(x_tokens)
        x1 = torch.randn_like(x0)
        t = self.sample_timestep(x0.shape[0], x0.device, x0.dtype)
        a, s = (1 - t).view(-1, 1, 1, 1), t.view(-1, 1, 1, 1)
        xt = a * x0 + s * x1
        pred = denoiser(xt, s.flatten(), cond)
        target = x1 - x0                                  # velocity eps - x
        return ((pred - target) ** 2).flatten(1).mean(1)

    @torch.no_grad()
    def euler_sample(self, denoiser, cond, batch, token_dim, device, num_steps=50):
        x = torch.randn(batch, token_dim, self.side, self.side, device=device)
        ts = self.shift(torch.linspace(1 / num_steps, 1, num_steps, device=device))
        for t_cur, t_next in zip(reversed(ts[1:]), reversed(ts[:-1])):
            cur = t_cur.repeat(batch).to(x.dtype)
            nxt = t_next.repeat(batch).to(x.dtype)
            v = denoiser(x, cur, cond)
            x = x + (nxt - cur).view(-1, 1, 1, 1) * v      # negative: noise -> data
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

    def forward(self, x, t, y):                  # x:(B,D,H,H), y:(B,N,cond_dim)
        s = self.x_embed(x) + self.pos
        t = self.t_embed(t)
        y = self.y_embed(y, self.training)
        if len(t.shape) < len(y.shape):
            t = t.unsqueeze(1).expand(-1, y.shape[1], -1)
        c = F.silu(t + y)                         # query i conditions latent site i
        for blk in self.blocks:
            s = blk(s, c, self.rope)
        return self.unpatchify(self.final(s, c))


class GenerationModel(nn.Module):
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
            target = self.encoder(target_image)

        b = prompt_ids.shape[0]
        m = answer_image_mask.shape[1]
        q = self.latent_queries.unsqueeze(0).expand(b, -1, -1).repeat(1, m, 1)
        input_embeds = insert_image_slot_queries(self.llm, prompt_ids, q, image_token_indices)
        hidden = self.llm(inputs_embeds=input_embeds, output_hidden_states=True).hidden_states[-1]
        patch_h = gather_image_slots(hidden, image_token_indices)
        cond = self.diff_head_projector(patch_h.view(b * m, self.latent_queries.shape[0], -1))

        target = target.view(b * m, self.latent_queries.shape[0], -1)
        loss = self.flow.training_loss(self.denoiser, target, cond).view(b, m)
        return (loss * answer_image_mask.float()).sum() / (answer_image_mask.sum() + 1e-8)


def build_optimizer(model, lr=5.66e-5, diff_lr=5.66e-4, weight_decay=0.0):
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

So the chain, end to end: I wanted one space for seeing and generating, which pushed me to diffuse in a frozen high-dimensional semantic latent instead of a compressed VAE latent. The noising-fills-the-manifold argument, checked against a tiny eigenvalue example, forced the denoiser's width above the token dimension; the per-element SNR argument, checked by matching the base-dimension difficulty exactly, forced a dimension-dependent timestep shift keyed to the effective dimension `m = N x d` — both pieces are properties of high-dimensional noising, not of any particular model size, and the scale ablations move in the directions that math predicts. The wide head and noise-augmented decoding, by contrast, only ever fixed small-model weaknesses (a too-narrow backbone, a bad early-training generator), so as the denoiser grows past 1152 in width and converges quickly their measured benefit fades, and I drop them — scale simplifies the recipe. For the text condition I reuse the LLM through 256 learnable queries aligned one-to-one with the 256 latent tokens, which lets me condition the denoiser per token rather than with a single pooled vector; I finetune the LLM and decouple the optimizer groups for the pretrained side and the from-scratch diffusion head. The decoder I keep as the standard frozen-encoder reconstruction stack, but its open-world quality is a *data-composition* problem — web for breadth, synthetic for clean style, text-rendering for glyphs — not a pure scaling one. And because the generator emits tokens in the very space the LLM understands, the model can score and select its own generations entirely in latent space, no decode-re-encode — the dividend of generating where you perceive.
