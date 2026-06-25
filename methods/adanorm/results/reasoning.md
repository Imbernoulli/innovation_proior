Let me start from what is actually bugging me. Every diffusion image model I look at is built on a convolutional U-Net, and that U-Net has barely changed since Ho et al. first borrowed it. Meanwhile transformers have eaten language, eaten recognition, eaten reinforcement learning — and the reason people keep reaching for them isn't just accuracy, it's that they *scale predictably*: make the thing deeper or wider, or feed it more tokens, and the loss comes down along a smooth curve you can plan compute around. Image generation has none of that discipline; it has a grab-bag of U-Net tweaks. So the thing I want is plain: take a standard Vision Transformer — patchify the image (or, cheaper, the VAE latent) into tokens, add positional embeddings, run a stack of ordinary pre-norm blocks — and use *that* as the diffusion denoiser, so the transformer's scaling behaviour comes along for free. The whole bet is "stay as close to the standard block as possible," because every bespoke modification I make is a chance to break the very scaling I'm trying to inherit.

And immediately I hit the thing that makes a denoiser not just an image network. The model isn't a function of the noisy image alone. At every reverse step it has to be told the diffusion timestep `t` — how much noise is in `x_t`, i.e. where on the schedule we are — and for class-conditional generation it also has to be told the label `c`. The training loss is the friendly one, `||eps_theta(x_t, t) - eps||^2`, an MSE between predicted and true noise, but `eps_theta` is conditioned: the same noisy latent at the same noise level must denoise differently toward a "golden retriever" than toward a "volcano." So I have two side inputs, `t` and `c`, that have to reach *every* block, and a vanilla transformer block — `x = x + MHSA(LN(x))`, `x = x + MLP(LN(x))` — has no obvious socket for them. That's the real problem. Everything else is standard ViT. The question is just: how do I plumb `(t, c)` into the block without wrecking it?

First, the easy parts so I can set them aside. `t` is a scalar, so I do what diffusion models always do: a sinusoidal frequency embedding to spread the scalar across many channels, then a little two-layer MLP to mix it into a vector at the transformer's hidden width `d`. `c` is a categorical label, so a learned embedding table, one row per class, also at width `d`. (I'll keep one spare row for a "null" label so I can do classifier-free guidance later — drop the label with some probability during training, learn the null embedding, extrapolate at sampling time. That's orthogonal to the block design.) Now I have two vectors per example, `t_emb` and `c_emb`, both `(N, d)`. The block-conditioning question is what to do with them.

Option one is the most literal reading of "transformers consume sequences." Just turn `t_emb` and `c_emb` into two extra *tokens*, append them to the image-token sequence, run completely unmodified ViT blocks, and slice the extra tokens off at the end. This is exactly the `cls`-token trick from ViT, and it's seductive because the block doesn't change at all — zero new parameters, basically zero new flops, and whatever scaling the standard block has is trivially preserved because there *is* no modification. So why does my gut say no? Because the conditioning is *global*. It's the same `(t, c)` for every spatial location. If I make it a token, the only way it influences an image token is for self-attention to attend from each image token over to the conditioning tokens, at every layer, and re-derive "ah yes, we're at noise level `t`, class volcano" again and again. I'm asking the attention mechanism — an expensive, content-routing, all-pairs operation — to carry a constant. It can do it, but it spends representational capacity doing something that has no token structure to exploit. It feels like the wrong structural match. Let me hold it as the floor and look for something that matches the *shape* of the signal: one global modulation applied identically to every token.

Option two goes the other direction — make the conditioning maximally expressive. Keep the image tokens as queries, and after the self-attention sublayer add a *cross-attention* sublayer whose keys and values are the conditioning embeddings, the way the original encoder-decoder transformer attends from decoder to encoder, the way latent diffusion attends image features over text tokens. Now each image token can attend to `t` and `c` directly. This is genuinely the right hammer when the conditioning is itself a sequence with structure — a sentence, a set of region features. But look at what it costs: a whole extra multi-head attention block in *every* transformer block. And what am I buying with it? The "sequence" I'm attending over is two vectors with no internal structure, the same two for every token and every spatial position. I'm paying for a content-addressable lookup over a two-element, structureless table. The expressiveness is real but almost entirely wasted on a global class label — it's the most flops for the least-suited task, so it goes on the shelf next to the floor.

Both extremes sit wrong for the same underlying reason, and that reason is worth naming because it tells me what to look for: the conditioning signal is *global and structureless*, identical across all tokens. A mechanism that fits a global, per-channel, location-agnostic signal would take the one conditioning vector and use it to *modulate every channel of every token in the same way* — no token routing at all. Is there already a per-channel, every-token operation in the block I could hijack instead of bolting on a new sublayer? The normalization layer. Every block normalizes, and right after normalizing it applies a learned per-channel affine: LayerNorm computes `x_hat = (x - mu)/sigma` per token over the feature dimension, then returns `gamma * x_hat + beta` with learned vectors `gamma, beta` of length `d`. Those `gamma` and `beta` are exactly per-channel scale and shift knobs applied identically to every token. Right now they're *learned constants* — the same for every input, so they can't condition on anything. But what if I stop learning them as constants and instead *produce them from the conditioning vector*?

That's not a wild idea; it's a primitive several lines of work already converged on, and now I see why they all landed there. Perez et al.'s FiLM made it a general layer: feature-wise linear modulation, `FiLM(F | gamma, beta) = gamma * F + beta`, where `gamma = f(x)` and `beta = h(x)` are *regressed from a conditioning input* `x` by a small network, and in practice one network emits the whole `(gamma, beta)` vector. Their whole point was that a conditional, per-channel affine — applied the same way regardless of spatial position — is enough to steer a network's behaviour on the conditioning. The generative-model crowd had the same primitive as adaptive instance norm, `AdaIN(x, y) = y_s (x - mu)/sigma + y_b`, where a style vector `y` supplies the post-normalization scale and bias; StyleGAN drives its whole generator that way. And most on-the-nose for me, Dhariwal & Nichol, improving diffusion U-Nets, ablated exactly how to feed `(t, c)` into a residual block and settled on adaptive group norm, `AdaGN(h, y) = y_s · GroupNorm(h) + y_b`, with `y = [y_s, y_b]` a linear projection of the combined timestep-and-class embedding — and they reported it beat the older "add the embedding, then normalize" recipe. So in the diffusion setting specifically, the answer to "how do I inject `(t, c)`" was already "modulate the normalization's affine from `(t, c)`." The only catch is that all of this is written for *convolutional* blocks — group norm, conv branches. My job is to port the same move into a transformer block, where the norm is LayerNorm.

Let me write that port. The transformer's LayerNorm has a `gamma * x_hat + beta`; I'll strip out the learned affine (set LayerNorm to do the normalization only, no built-in scale/shift) and supply the affine from the conditioning instead. Take the conditioning vector — call it `cond` — push it through a small MLP, and have that MLP emit two affine controls, each of length `d`. The literal FiLM write-down would be `gamma * LN(x) + beta`, with both `gamma` and `beta` regressed from `cond`. But I should not freeze that parameterization yet, because the value that means "leave the normalized activation alone" matters.

Now, two refinements I want to nail down before going further. First, how do I form `cond` from `t_emb` and `c_emb`? I could concatenate them, but then the MLP input is `2d` wide and I've doubled a matrix for no real reason — and worse, I'd have to decide a fixed split. The cleaner move, and the one AdaGN uses, is just to *sum* them: `cond = t_emb + c_emb`, both already at width `d`. Summation keeps the dimension fixed, lets a single MLP produce every modulation parameter, and treats `t` and `c` symmetrically as two contributions to one conditioning state. Good — `cond = t_emb + c_emb`.

Second refinement, and this one I want to get exactly right because it bites at initialization. If I use the raw regressed value as the gain and the regressor happens to start near zero — which is the natural place for a freshly-initialized linear layer to sit — then the gain is near zero and I've *annihilated* the normalized activations on the very first step. Let me actually compute the two parameterizations on a concrete normalized vector to be sure I have the algebra right and not just a vibe. Take `u = LN(x)` for `x = [1, 2, 3, 4]`; normalizing gives `u = [-1.342, -0.447, 0.447, 1.342]`. With the raw form `scale·u + shift` at `scale = 0, shift = 0` I get `[0, 0, 0, 0]` — the activation is gone. With a *deviation-from-one* form,

  `modulate(u, shift, scale) = u * (1 + scale) + shift`, with `u = LN(x)`,

the same `scale = 0, shift = 0` gives `u * (1 + 0) + 0 = [-1.342, -0.447, 0.447, 1.342]` — exactly `LN(x)`, unchanged. So the `1+` is the difference between "start by destroying the normalized signal" and "start by leaving it alone." The regressor's job becomes "learn how far to push scale and shift *away from the identity*," which is both a better-conditioned learning problem and a saner initialization. I'll use `(1 + scale)`, not `scale`. Tiny algebraic choice, and I just watched it flip the init from zero to identity.

The transformer block has *two* sublayers, multi-head self-attention and the MLP, each with its own pre-norm. So I need a `(shift, scale)` pair for each: that's four vectors of length `d`, and a natural single MLP that takes `cond` and emits `4d` outputs, chunked into `shift_attn, scale_attn, shift_mlp, scale_mlp`. The block becomes

  `x = x + MHSA( modulate(LN1(x), shift_attn, scale_attn) )`,
  `x = x + MLP ( modulate(LN2(x), shift_mlp,  scale_mlp ) )`.

This is a conditional, per-channel affine driven through the normalization — adaptive layer norm. It's cheap — one small MLP per block, no extra attention, negligible flops — and it has the inductive bias I was after: a single function of `cond` applied per-channel, identically across all tokens. Compared to the in-context floor and the cross-attention ceiling, its cost and structure match a global signal. I'll treat this as the conditioning core and pressure-test it before committing.

The pressure point is initialization, because I'm about to stack a *lot* of these blocks and deep residual stacks are where training quietly goes wrong. With the `(1 + scale)` parameterization and the regressor near zero, `modulate(LN(x), shift, scale) ≈ LN(x)`, fine — but then I still compute `x = x + MHSA(LN1(x))` and `x = x + MLP(LN2(x))` with the attention and MLP weights at their *random* initialization. So at step zero each block adds a random, nonzero perturbation to its input, and the perturbations accumulate down the stack. How badly? Let me put a number on it instead of hand-waving "the signal gets pushed around." Model each block's branch output at init as a random per-channel contribution with per-channel standard deviation `s` (LN normalizes its input to roughly unit scale, and a random linear map keeps it order-one), added to a running signal that starts at unit per-channel power. If the contributions are roughly uncorrelated with the running signal, the per-channel power should grow like `1 + N·s²`. I simulate `d = 1152`, `N = 28` blocks, averaging over a couple thousand draws: for `s = 0.5` the mean per-channel power after 28 blocks comes out `8.0` (additive prediction `1 + 28·0.25 = 8.0`), and for `s = 1.0` it comes out `29.0` (prediction `1 + 28 = 29.0`). So the signal at the top of the stack has 8–30× the power it started with, purely from random init, before a single gradient step. That is exactly the regime where deep transformers are known to be finicky — the loss spikes, the warmups, the careful schedules people bolt on. I'd much rather have the *opposite* starting point: a residual stack that at initialization is the *identity map*, so the signal passes through at unit power and the model grows its conditioning and nonlinearity gently from there. Adaptive layer norm conditions, but it does nothing to tame that init-time variance blow-up — the `1 + N·s²` growth happens whether or not the affine is conditioned. So the conditioning core is not yet enough. I need a second ingredient that switches each residual branch *off* at the start.

How do I make a residual block start as the identity? I've seen this exact trick, and now I understand why it works. Goyal et al., training ImageNet in an hour at huge batch sizes, initialized the *last* batch-norm scale `gamma` of each residual block to **zero**. With that last `gamma = 0`, the residual branch outputs exactly zero, so `x = x + branch(x) = x + 0 = x` — the block is the identity, and the forward and backward signal initially flow only through the skip connection. They found it eased optimization at the start, especially for large-batch training, and diffusion U-Nets do the sibling thing, zero-initializing the final convolution of each residual block before the add. The principle is general: put a learnable multiplier at the *very end* of each residual branch, right before it's added back, and initialize that multiplier to zero, so the whole branch is off at the start and the network is a stack of identities. It then learns to "turn on" each branch as needed.

So I add exactly that to my block. After the sublayer computes its output, multiply by a per-channel **gate** — call it `gate`, a third length-`d` vector — *immediately before* the residual addition:

  `x = x + gate_attn ⊙ MHSA( (1 + scale_attn) ⊙ LN1(x) + shift_attn )`,
  `x = x + gate_mlp  ⊙ MLP ( (1 + scale_mlp ) ⊙ LN2(x) + shift_mlp )`.

And `gate` is just another thing I can regress from `cond`, like `shift` and `scale`, which keeps the conditioning path uniform. So the modulation MLP would emit *six* vectors per block: `(shift, scale, gate)` for the attention sublayer and `(shift, scale, gate)` for the MLP sublayer — a single `Linear(d, 6d)`, chunked into `shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp`. The init move is then to **zero-initialize that entire modulation Linear**, weight and bias, so at step zero every one of the six outputs is exactly zero.

Before I trust the claim that this makes each block the identity, I want to trace one block by hand at init, because the obvious-looking shortcut here is wrong and I nearly took it: surely `scale = 0` already does the job? Take a single token `x = [1, 2, 3, 4]`. Normalize: `LN(x) = [-1.342, -0.447, 0.447, 1.342]`, whose norm is `2.0` — emphatically *not* zero. Now run the modulated norm with `scale = 0, shift = 0`: `modulate(LN(x), 0, 0) = LN(x)`, still that nonzero vector. Push it through a fixed sublayer map (any order-one linear `W` at random init) and I get a nonzero branch output, e.g. `[-0.201, -0.134, 0.0, 0.470]`. Without a gate the block computes `x + branch = [0.799, 1.866, 3.0, 4.470]` — *changed* from `[1, 2, 3, 4]`. So `scale = 0` alone does **not** give the identity: it neutralizes the modulation but `LN(x)` is still a live nonzero signal that the random sublayer happily transforms and adds back. That was the trap — and the variance check from a moment ago already told me a stack of these non-identity blocks blows the signal up `8`–`30×`. Now redo the same trace with the gate at zero: the block computes `x + gate · branch = x + 0 · branch = [1, 2, 3, 4]` — *exactly* the input, identity to the digit. So the gate is the piece that buys the identity, and it is genuinely a different knob from scale: scale and shift modulate the sublayer's *input* (and leave `LN(x)` nonzero), while the gate multiplies the sublayer's *output* by zero, killing the whole branch. That also pins down *where* the gate must sit — right at the residual add, outside the sublayer. If I'd tucked it inside, say between attention and its output projection, a zero there wouldn't cleanly zero the branch contribution the way a zero at the add does. So six vectors, not four, and the gate's position is load-bearing.

With the gate zeroed in every block, each block is the identity, so the whole `N`-block stack is the identity map at init: the signal enters at unit power and exits at unit power, exactly the clean start the variance check said I wanted, instead of the `8`–`30×` blow-up. The transformer body starts as a no-op and grows conditioning out of it — adaptive layer norm with zero-initialized gates. (Plain adaptive layer norm with just the four vectors — `shift, scale` per sublayer, no gate — is a real fallback, and it conditions fine, but it gives up exactly this identity start. Whether that matters in final FID I can't settle on paper; the variance argument makes me expect the gated version trains more stably on a deep stack, and I'd want to confirm it by training both. I'll carry the gated version as the design and the four-vector version as the thing it improves on.)

Now the final decode head. After the last block I have to turn the token sequence into a noise prediction (and, if I'm learning the reverse-process covariance, also a covariance prediction) shaped like the input — for each token a `p × p × (channels or 2·channels)` patch, then rearrange to spatial. I want to condition this head too, by the same adaptive-norm logic: a final LayerNorm whose affine is regressed from `cond`. But here there's no residual branch to gate — it's a straight decode, not `x = x + branch`. So I only need `shift` and `scale`, not a gate: a `Linear(d, 2d)` emitting `(shift, scale)`, and `decode = Linear( (1 + scale) ⊙ LN(x) + shift )`. And by the same start-as-a-no-op spirit, I zero-initialize this head's output Linear as well, so at initialization the model predicts exactly zero — a clean, neutral starting noise prediction rather than random patches. Both adaptive-norm Linears (the per-block `6d` one and the final `2d` one) and the final decode Linear are zero-initialized; everything else uses standard transformer init.

Let me also pin down the small choices inside the modulation MLP, because I don't want any of them to be arbitrary. Why a nonlinearity before the modulation Linear, and why SiLU? FiLM allows the regressor `f, h` to be *arbitrary* functions, not just linear — and there's a reason to want nonlinearity: `cond = t_emb + c_emb` is a sum of two embeddings, and the right `(shift, scale, gate)` for, say, "high noise level *and* class volcano" need not be the linear sum of the per-factor modulations. A nonlinearity lets the regressor express interactions between `t` and `c`. SiLU is the smooth gated activation already used in the timestep MLP and throughout diffusion conditioning paths, so it's the natural, consistent choice — `cond → SiLU → Linear(d, 6d)`. And everything outside the conditioning path I deliberately leave *standard*: pre-norm blocks, multi-head self-attention with qkv bias, an MLP `4×` wide with a (tanh-approx) GELU, sinusoidal positional embeddings on the patch tokens. The entire point of the architecture was to inherit transformer scaling, so I touch nothing I don't have to; the conditioning lives *only* in the modulated norms and gates.

Let me write out the block as code I'd actually run, because I want to see that it's nothing but a standard ViT block plus the modulation. The modulation helper is `modulate(x, shift, scale) = x * (1 + scale) + shift`, broadcasting the `(N, d)` modulation across the token axis. The block:

```python
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


def modulate(x, shift, scale):
    # x: (N, T, d); shift, scale: (N, d) -> broadcast over the token axis T
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """Standard pre-norm transformer block, but the two LayerNorm affines and a
    per-sublayer output gate are regressed from the conditioning vector c = t_emb + c_emb."""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        # LayerNorm with NO learned affine -- the affine comes from the conditioning instead
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)   # standard MHSA
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)                                # 4x wide
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden,
                       act_layer=approx_gelu, drop=0)
        # one MLP regresses all 6 modulation vectors: (shift, scale, gate) x (attn, mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),     # zero-initialized (see init)
        )

    def forward(self, x, c):
        # c: (N, d) conditioning. Split into the six per-channel modulation vectors.
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        # attention sublayer: modulate the (affine-free) norm, gate the output before the residual add
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        # MLP sublayer: same pattern
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
```

The final layer, conditioned by shift/scale only (no gate — no residual to switch off), and zero-initialized so it starts by predicting zero:

```python
class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),    # only (shift, scale)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)
```

And the surrounding network, which is just patchify + positional embedding + the block stack + decode, with the one conditioning line `c = t + y` (the sum I argued for), a block stack that starts as the identity, and a final head that starts by predicting zero:

```python
class DiT(nn.Module):
    def __init__(self, input_size, patch_size, in_channels, hidden_size, depth,
                 num_heads, num_classes, mlp_ratio=4.0, class_dropout_prob=0.1,
                 learn_sigma=True):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels   # noise (+ covariance)
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)     # sinusoidal features + MLP (SiLU)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)  # +null row for CFG
        num_patches = self.x_embedder.num_patches
        # fixed (non-learned) sin-cos positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) the fixed sine-cosine positional embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1],
                                            int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table and timestep embedding MLP:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero the adaLN projections so every residual block starts as the identity:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)   # -> scale=shift=gate=0
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)     #    => each residual block = identity

        # Final head starts by predicting zero:
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
        x = self.x_embedder(x) + self.pos_embed          # (N, T, d) patch tokens + positions
        t = self.t_embedder(t)                           # (N, d)
        y = self.y_embedder(y, self.training)            # (N, d)
        c = t + y                                        # (N, d): sum the two conditionings
        for block in self.blocks:
            x = block(x, c)                              # adaLN-Zero conditioning in every block
        x = self.final_layer(x, c)                       # conditioned decode -> patch pixels
        return self.unpatchify(x)                        # (N, out_channels, H, W)
```

Let me trace the causal chain back so I'm sure each piece earned its place. I wanted a transformer diffusion backbone to inherit transformer scaling, which meant keeping the standard block and only solving one thing: how to inject the global, structureless conditioning `(t, c)` into every block. Treating `(t, c)` as extra tokens makes attention re-derive a constant at every layer — weak; cross-attention buys real expressiveness but pays a whole extra attention sublayer for a content lookup over a two-element structureless table — wasteful. Both fail because they ignore that the signal is *global and per-channel*. The operation that already is global and per-channel is the normalization's learned affine, so I make that affine a *function of the conditioning* instead of a constant — the FiLM / AdaIN / AdaGN move, ported from convolutional blocks to LayerNorm: `modulate(u, shift, scale) = u * (1 + scale) + shift`, applied with `u = LN(x)`, with `(shift, scale)` regressed per sublayer from `cond = t_emb + c_emb`, the `1+` so that zero modulation leaves the normalized activation unchanged. That conditions cheaply with the right inductive bias, but at initialization a deep stack of such blocks still perturbs the signal with random residual branches — the regime where deep transformers are hard to train. So I borrow the zero-initialized-residual trick (Goyal's last-`gamma`-zero, the U-Net's zero-last-conv): add a per-sublayer output gate regressed from the same `cond`, position it right before the residual add, and zero-initialize the whole modulation projection so `scale = shift = gate = 0` and every residual block starts as the exact identity, growing its conditioning from a no-op. Six modulation vectors per block (shift/scale/gate × two sublayers, a single `Linear(d, 6d)`), two for the gateless final head (`Linear(d, 2d)`, also zero-init), and a zero-initialized output linear so the denoiser's initial prediction is zero rather than a random patch field. The result is a bog-standard ViT block plus one small zero-initialized modulation MLP — negligible extra flops, the same function applied to every token, identity-initialized residual blocks — which is exactly the conditioning mechanism I was looking for.
