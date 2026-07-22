Let me start from the gap that actually bothers me. I have a diffusion model that is everything I want a generator to be — likelihood-based, trained by one stationary mean-squared-error objective with no discriminator to balance, covering the data distribution, scaling cleanly with compute. On CIFAR-10 it is the best thing going. And yet on ImageNet at 128×128 and 256×256, on LSUN, its FID sits behind BigGAN-deep. The samples are good but not GAN-good on the hard datasets. So the question is: where is the quality going, and is it something I can recover without giving up what makes diffusion diffusion?

I want to be careful not to touch the wrong thing. The training target is ε-prediction with the simple MSE, plus the learned-variance interpolation and the hybrid loss; the sampler is DDPM, or DDIM when I want few steps. Those are settled, and I have good reasons to keep them — they are exactly the parts that give me likelihood and stable training. If I start swapping the loss or the sampler I am no longer answering "is the architecture leaving quality on the table," I am changing the experiment. So I hold everything fixed except the one object that computes ε_θ(x_t, t): the denoiser network. That is the only thing I will let myself edit.

Why suspect the architecture at all? Because the asymmetry is suspicious. GAN generators and discriminators have been refined for years — channel schedules, normalization, residual block shapes, attention placement all swept and re-swept. The diffusion denoiser, by contrast, is essentially the UNet that made diffusion work on CIFAR-10: a contracting stack of residual blocks with downsampling, a symmetric expanding stack with upsampling, skip connections at equal resolution, a single global self-attention layer at the 16×16 feature map with one head, and the timestep embedding projected and *added* into each residual block. That design was never stress-tested on large, diverse, higher-resolution data. And I already have a hint it is not saturated: continuous-time score models showed that further UNet changes lower FID on CIFAR-10 and CelebA-64. The open question is whether such changes carry to ImageNet scale, where the images are bigger and far more varied. My working hypothesis is that they do, and that the gap to GANs is partly an architecture gap. So I am going to treat the denoiser as a design space and sweep it, one change at a time, on ImageNet 128×128, holding the model size roughly constant and reading FID at two points in training so I do not chase a transient. I want to be honest with myself up front: I cannot settle these on paper, because FID is a property of millions of generated samples against the full training set. What I *can* settle on paper is whether each change is even coherent — whether it does what I think it does to the shapes, the cost, and the initialization — before I spend the compute to measure it. So that is the discipline: reason each change from a concrete weakness, check the parts that are checkable here, and mark clearly which claims still owe a measurement.

Now, what are the candidate changes, and — more importantly — *why* would each one help? I refuse to throw knobs at the wall. Let me reason each one from a concrete weakness of the current denoiser.

Start with attention, because the current placement looks like the most obvious under-use. Self-attention is the operation that lets any spatial position influence any other in a single layer, with content-dependent weights — exactly what convolution cannot do in one step, because a conv kernel only mixes a local neighborhood and long-range coupling has to be smeared across many layers. The current denoiser has this operation, but only at one feature resolution, 16×16. Think about what that means. On a 128×128 image, after the contracting path, "16×16" is a coarse feature map several downsamplings in; the denoiser can coordinate global structure there and nowhere else. But large, diverse images have long-range structure at more than one scale — the gross layout of an object at a coarse scale, the agreement of parts at a somewhat finer scale, the consistency of texture regions at a finer scale still. If attention lives at exactly one feature resolution, every other scale's global coordination has to be carried indirectly through the convolutional stack, which is the slow, brittle path. So the move to try is: put self-attention at several feature resolutions, not just one — at the 32×32, 16×16, and 8×8 feature maps. Let global mixing happen at coarse and middling scales alike.

But I cannot put attention everywhere for free, and I should be precise about why, because the reason draws the line for me. Attention forms an N×N weight matrix over the N = H·W spatial positions, so its cost grows like N². Let me actually put numbers on the tiers I am choosing between. The finest map is 128×128, so N = 16384 and N² ≈ 2.68×10⁸. The 32×32 feature map has N = 1024 and N² ≈ 1.05×10⁶; the 8×8 has N = 64 and N² = 4096. The ratios are the thing: N²(128)/N²(32) = 256 and N²(128)/N²(8) = 65536. So an attention layer at the input resolution costs 256× one at 32×32 and 65536× one at 8×8 — the quadratic genuinely *explodes* at the fine end and is essentially free at the coarse end. That settles the question of where attention can live: the three coarsest tiers (32, 16, 8) where N is already small, and never the 128 or 64 maps where the quadratic would dominate the whole network's compute. And that is also fine on the modeling side — local texture at the fine map is convolution's job anyway. So the rule that falls out is: attention at the coarse tiers where global structure lives and the cost is cheap; convolution alone at the fine tier where the cost would explode. Whether this lowers FID I still have to measure; what I have established here is that it is *affordable*, which is exactly the precondition that the single-resolution baseline was respecting too conservatively.

Next, the attention *heads*. The current layer uses a single head. A single head forces every output position to use one relevance pattern — one "which other positions matter to me" map, shared across whatever relations the image actually contains. But the relations are plural: the symmetric half of an object, the far edge of the same shape, a repeated texture region. One softmax map has to be a compromise across all of them. The fix is multi-head attention: split the channel embedding into several heads, run an independent scaled-dot-product attention in each lower-dimensional subspace, concatenate. Different heads specialize to different relations at once, and it is nearly free because each head is a fraction of the width, so the total cost is comparable to one full-width attention. So more heads should help. The question is *how* to set the head count, and there are two ways to parameterize it: fix the *number of heads* (so the per-head width grows with the channel count), or fix the *channels per head* (so the number of heads grows with the channel count). The second is the Transformer convention, and it has a cleaner scaling story — a head always operates in a fixed-dimensional subspace regardless of how wide the feature map is, so the comparison logits live in a consistent dimension across resolutions. Let me reason about which is better by what the head is actually doing: the per-head query/key dot product is a *comparison* in the head's subspace; if I let that subspace get very wide (few heads on a wide map) the comparison is high-dimensional and I have fewer distinct relevance patterns; if I make it narrow (many heads) I get many patterns but each comparison is in a low-dimensional space that may be too cramped to discriminate. So there is a sweet spot in channels-per-head, and I lean toward fixing it rather than fixing the head count. If I set channels-per-head to 64, I can read off what that does across the three attention tiers of my UNet: at the 32×32 map the width is 256, so num_heads = 256//64 = 4; at the 16×16 map width 384 gives 6 heads; at the 8×8 map width 512 gives 8 heads. That is the behavior I wanted — the head count automatically rises on the wider, coarser maps where I most want many distinct relevance patterns, while the per-head comparison dimension stays pinned at 64, the same subspace size that works in language models. I will *default* to 64 channels per head; whether 32 or 128 would have done better on FID is genuinely an empirical question I would sweep, but 64 is the principled starting point and it keeps every attention layer's per-head geometry identical. (One practical caveat I note for the sweep: very narrow heads are statistically fine but the kernel launches and reshapes get inefficient, so wall-clock, not just FID, will help pick the value.)

Now the resolution changes themselves — how the denoiser goes from one feature scale to the next. In the current design, downsampling and upsampling are separate operations sitting *outside* the residual blocks: a plain pooling or strided convolution on the way down, an interpolation followed by a convolution on the way up, with the residual blocks operating only within a resolution. That is a missed opportunity for the residual structure. A residual block carries an identity path that keeps gradients alive and lets the block learn a correction rather than a full transform; if the resolution change happens outside that structure, the resampling step has no residual path and is a plain, gradient-unfriendly transform stuck between two residual stages. The alternative is to fold the resolution change *into* a residual block: the channel-changing convolution inside the block also resamples, and the block's skip path resamples in parallel, so the up/downsampling itself gets a residual identity path. This is the BigGAN-style residual block for up/downsampling. The reason it should help is precisely the reason residual connections help anywhere — the resampling is now a learned residual correction with a clean gradient path, instead of a plain bottleneck the optimizer has to push everything through. It is a candidate to keep because it improves the *spine* of the network — every resolution transition — rather than one localized component; whether it actually moves FID is, again, for the ablation.

Then there is residual-branch rescaling. As residual branches accumulate down a deep stack, the activation variance grows — each residual sum adds the branch's variance on top of the identity's. The standard remedy, used in style-based GANs and in the continuous-time score models, is to divide each residual sum by √2, on the reasoning that if the two summands are roughly unit-variance and uncorrelated, the sum has variance 2, and dividing by √2 brings it back to 1. Drawing two independent standard-normal vectors a, b confirms it: empirically var(a+b) ≈ 2.02 and var((a+b)/√2) ≈ 1.01, so the rescale does restore unit variance when the branches are uncorrelated and unit-variance, exactly as advertised. But that is a conditional win: it only helps if growing variance is actually the bottleneck. And here it probably is not, because group normalization sits in every block and *re-standardizes* the activations regardless of how the residual sums accumulate — so the variance the √2 was meant to control is already being reset downstream. In that setting the rescale is at best redundant and at worst fights the normalization. I cannot prove which on this page (it is an FID question), but my expectation is that in this already-normalized denoiser the √2 rescale will not help and may hurt; I will keep it in the sweep as a thing to *test* rather than adopt, and only carry it forward if FID actually improves.

Last among the structural sweeps: depth versus width, at fixed model size. I can spend a fixed parameter budget on a deeper, narrower network or a shallower, wider one. Deeper networks are more expressive per parameter in principle, so added depth might well lower final FID at fixed size. But depth has a cost that does not show up in the parameter count: a deeper network has a longer sequential dependency, trains slower per step and takes more steps to reach a given quality, and the wall-clock-to-target-FID is what I actually care about under a compute budget. So this is a trade I expect to come out against depth on the metric I care about — but it is a wall-clock measurement, not something I can settle here. My plan is to spend capacity on width unless the depth sweep shows it reaching the same FID *faster*, which I doubt.

So I have five structural changes, each motivated by a specific weakness: multi-resolution attention (one-scale coordination → multi-scale, and I have shown it is affordable at the coarse tiers), more heads / fixed channels-per-head (one relevance pattern → many, with the head count rising correctly across tiers), BigGAN up/downsampling residual blocks (resampling outside the residual structure → folded in with an identity path), 1/√2 residual rescale (variance compounding — real arithmetic, but likely redundant under group norm), and depth-vs-width (a compute-efficiency trade I expect to favor width). I read them one at a time off a common baseline, on FID at two checkpoints. My priors going in: keep multi-resolution attention, more heads, and BigGAN up/downsampling; expect to drop the depth increase on wall-clock grounds and the √2 rescale because group norm already does its job. The sweep is what turns those priors into decisions — I am not entitled to the conclusions until the FID numbers land — but the priors are what I would bet on.

Now there is one more change, and it is not about where attention goes or how resolution changes — it is about how the timestep gets into the network, and I think it may be quietly important. The denoiser is one network that must handle *every* noise level t; the timestep embedding is the only thing telling it which level it is on, and it has to modulate the whole network accordingly. The current design *adds* a projection of the embedding into each residual block. Addition is the weakest possible conditioning: it shifts the activations by a t-dependent bias and nothing more. It cannot, for instance, *scale* a channel up or down as a function of t — and intuitively the denoising behavior should change in *gain*, not just offset, between a near-clean image (small corrections) and near-pure noise (large structure to invent). What I want is for the embedding to control both a scale and a shift of the activations, per channel, per timestep. That is the FiLM idea, and its image-network incarnation is adaptive instance / group normalization: let the conditioning vector produce a per-channel scale and shift applied to *normalized* activations. So I define an adaptive group normalization: after the group normalization inside the residual block, instead of the usual learned affine, apply a t-dependent affine,

  AdaGN(h, y) = y_s · GroupNorm(h) + y_b,

where h is the block's activations after the first convolution and [y_s, y_b] is a linear projection of the timestep embedding (and, in a class-conditional model, the class embedding folded in). Why this should beat additive injection: it gives the conditioning multiplicative control over the normalized signal — y_s can amplify or suppress each channel as a function of the noise level, and y_b still supplies the shift — so the single shared network can re-gain itself across the noise schedule rather than only re-bias itself. Group normalization is the right normalization to modulate because it is batch-independent and so stable at the small batches a large image model trains with. This is the change I am most hopeful about, because it touches every block and addresses a deficiency — fixed per-channel gain across all noise levels — that none of the structural changes touch; but "most hopeful" is a prior, and it goes in the same sweep as the rest.

Let me think about where AdaGN sits in the block so the modulation lands on the right signal. The residual block is: normalize, activate, convolve (in-layers); inject the embedding; normalize, activate, dropout, convolve (out-layers); add the skip. The embedding modulation should act on the *second* normalization — the one in the out-layers — so it scales and shifts the activations right before the final convolution that writes the block's output. Concretely, the embedding projection produces *two* tensors when AdaGN is on (a scale and a shift, hence twice the output width in the projection), and the out-layer normalization's output is multiplied by (1 + scale) and offset by shift before the final conv. The "1 +" is deliberate, and I want to make sure I have its consequence right rather than just asserting it is "identity at init." Two things have to line up for the block to be identity at init. First, the final conv in the out-layers is zero-initialized (zero_module), so whatever the AdaGN branch produces, the conv writes zeros at init — let me confirm: feeding a random h and a random embedding through exactly this path (group-norm, ×(1+scale)+shift, zero conv) gives a residual branch whose maximum absolute entry is 0.0 at init. So at init the block output is exactly skip_connection(x), the conditioning contributes nothing, and the modulation only grows in as the conv learns nonzero weights. Second — and this is what the "1 +" specifically buys — once the conv *is* nonzero, the scale enters as (1 + scale) rather than bare scale, so when the embedding projection is still near zero the gain applied to the normalized signal is ≈ 1 rather than ≈ 0: GroupNorm(h) has std ≈ 1.000 and (1+scale)·GroupNorm(h) preserves it, instead of a bare-scale form that would multiply the signal by ≈ 0 and suppress it early in training. So the modulation is a *perturbation of identity*, the same identity-at-init discipline that makes adding any new mechanism to a working network safe. When AdaGN is off, the projection produces a single tensor and it is simply added after the first convolution, recovering the old additive injection — that is the exact control I compare against in the sweep, differing from the AdaGN block in only the conditioning mechanism.

I want to check the whole network is internally coherent before I commit to it — that the pieces I have chosen actually compose into a UNet with the right input/output contract and the attention firing where I said it would. I instantiate the design at 128×128 with channel_mult (1,1,2,3,4), two res-blocks per resolution, attention at (32,16,8), 64 channels per head, and push a batch of two random images and timesteps through it. Tracing the feature shapes down the encoder: it holds 128×128 for the first tier, then 64×64, then 32×32 (where the width steps to 256), 16×16 (width 384), 8×8 (width 512) at the bottleneck, and the decoder mirrors it back up to 128×128 — and the final output is (2, 3, 128, 128), the same shape as the input. So the I/O contract — noised image and timestep in, same-shaped noise prediction out — is satisfied. Checking *where* attention attaches: with image_size 128 and attention_resolutions (32, 16, 8), the code converts these to downsample factors {128/32, 128/16, 128/8} = {4, 8, 16}, and attaches an AttentionBlock exactly at the tiers whose cumulative downsample is in that set — i.e. the 32×32, 16×16, and 8×8 maps, plus the 8×8 bottleneck — and nowhere on the 128×128 or 64×64 maps. That is precisely the "coarse tiers only" rule I argued for from the N² costs, now confirmed by where the blocks land in the actual module list. The model has ≈ 1.05×10⁸ parameters at this configuration. None of this is FID — but it tells me the design is not internally broken before I spend the compute, which is the thing a paper-only check *can* establish.

So the full picture of the improved denoiser is: a UNet whose width is the capacity knob (not added depth), with two residual blocks per resolution; multi-head self-attention with a fixed 64 channels per head at the 32, 16, and 8 feature resolutions (and at the bottleneck), convolution-only at the finest map; BigGAN-style residual blocks that fold up/downsampling into the residual structure; adaptive group normalization injecting the timestep (and class) embedding as a per-channel scale and shift after the block's normalization; and — pending the variance/normalization interaction I expect — no 1/√2 residual rescale. Everything outside the network — the ε-prediction MSE with learned variance, the DDPM/DDIM sampler, AdamW with β₁ = 0.9, β₂ = 0.999, EMA 0.9999, mixed precision — stays exactly as it was, because the whole point is to isolate the architecture's contribution. The structural priors (keep attention/heads/BigGAN, drop depth/√2) are what the FID sweep at two checkpoints is there to confirm or overturn; the coherence checks above are what let me trust that what the sweep measures is the design I intended.

There is a second, separate direction available — using gradients from a classifier at sampling time to trade diversity for fidelity — but that changes the sampler, not the denoiser, and it is orthogonal to the architecture question I am answering here. I note it and set it aside.

Let me write the network in the form the field actually uses, grounded in the standard denoiser implementation. The residual block has the in-layers / embedding projection / out-layers / skip structure, with the AdaGN scale-shift branch and the BigGAN up/down resampling folded in. The attention block computes multi-head scaled-dot-product attention over the flattened spatial positions, with the head count derived from a fixed channels-per-head, and a zero-initialized output projection so it is identity at init. The UNet wires these together: a contracting path that, at each resolution, runs the residual blocks (attaching an attention block where the resolution is in the attention set) and then a BigGAN downsampling residual block; a bottleneck of resblock–attention–resblock; and a symmetric expanding path that concatenates the equal-resolution encoder skips and runs residual blocks (with attention where applicable) and a BigGAN upsampling residual block. The timestep embedding is a sinusoidal embedding pushed through a small MLP and fed to every residual block's AdaGN.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)


def timestep_embedding(t, dim, max_period=10000):
    # sinusoidal embedding of the scalar noise level: one shared net reads off t at every scale
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def normalization(channels, groups=32):
    return nn.GroupNorm(min(groups, channels), channels)


class TimestepBlock(nn.Module):
    def forward(self, x, emb):
        raise NotImplementedError


class TimestepEmbedSequential(nn.Sequential):
    def forward(self, x, emb):
        for layer in self:
            x = layer(x, emb) if isinstance(layer, TimestepBlock) else layer(x)
        return x


class Upsample(nn.Module):
    def __init__(self, channels, use_conv, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = conv3(channels, self.out_channels)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x) if self.use_conv else x


class Downsample(nn.Module):
    def __init__(self, channels, use_conv, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.op = conv3(channels, self.out_channels, stride=2) if use_conv else nn.AvgPool2d(2, 2)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """Time-conditioned residual block with AdaGN, optionally folding up/down resampling in."""

    def __init__(self, channels, emb_channels, out_channels=None, dropout=0.0,
                 use_scale_shift_norm=True, use_conv=False, up=False, down=False):
        super().__init__()
        self.out_channels = out_channels or channels
        self.use_scale_shift_norm = use_scale_shift_norm
        self.updown = up or down

        self.in_layers = nn.Sequential(
            normalization(channels), nn.SiLU(), conv3(channels, self.out_channels)
        )

        if up:                                   # BigGAN-style: resample h and x inside the block
            self.h_upd = Upsample(channels, use_conv=False)
            self.x_upd = Upsample(channels, use_conv=False)
        elif down:
            self.h_upd = Downsample(channels, use_conv=False)
            self.x_upd = Downsample(channels, use_conv=False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        # AdaGN: project the embedding to a per-channel scale AND shift (2x width)
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            zero_module(conv3(self.out_channels, self.out_channels)),
        )
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv3(channels, self.out_channels)
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x, emb):
        if self.updown:                          # resample before the channel-changing conv
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:                                    # control: additive timestep injection (DDPM-style)
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class QKVAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = torch.einsum("bts,bcs->bct", weight, v)
        return out.reshape(bs, -1, length)


class AttentionBlock(nn.Module):
    """Multi-head self-attention over spatial positions; head count from fixed channels/head."""

    def __init__(self, channels, num_head_channels=64):
        super().__init__()
        assert channels % num_head_channels == 0
        self.num_heads = channels // num_head_channels
        self.norm = normalization(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)
        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x):
        B, C, H, W = x.shape
        x_in = x.reshape(B, C, H * W)
        h = self.attention(self.qkv(self.norm(x_in)))
        h = self.proj_out(h)
        return (x_in + h).reshape(B, C, H, W)


class Denoiser(nn.Module):
    """Improved diffusion UNet: width as capacity, multi-resolution multi-head attention,
    BigGAN up/downsampling residual blocks, AdaGN conditioning. No 1/sqrt(2) rescale."""

    def __init__(self, in_channels=3, out_channels=3, base_channels=128,
                 channel_mult=(1, 1, 2, 3, 4), num_res_blocks=2,
                 attention_resolutions=(32, 16, 8), num_head_channels=64,
                 dropout=0.0, image_size=128, use_scale_shift_norm=True,
                 resblock_updown=True):
        super().__init__()
        emb_ch = base_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(base_channels, emb_ch), nn.SiLU(), nn.Linear(emb_ch, emb_ch))
        self.base_channels = base_channels
        attention_ds = {image_size // r for r in attention_resolutions}

        ch = int(channel_mult[0] * base_channels)
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv3(in_channels, ch))])
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                blocks = [ResBlock(ch, emb_ch, out_ch, dropout, use_scale_shift_norm)]
                ch = out_ch
                if ds in attention_ds:
                    blocks.append(AttentionBlock(ch, num_head_channels))
                self.input_blocks.append(TimestepEmbedSequential(*blocks))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:           # BigGAN downsampling resblock
                down = ResBlock(ch, emb_ch, ch, dropout, use_scale_shift_norm, down=True)
                self.input_blocks.append(TimestepEmbedSequential(down if resblock_updown else Downsample(ch, True)))
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, emb_ch, ch, dropout, use_scale_shift_norm),
            AttentionBlock(ch, num_head_channels),
            ResBlock(ch, emb_ch, ch, dropout, use_scale_shift_norm))

        self.output_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            out_ch = base_channels * mult
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                blocks = [ResBlock(ch + ich, emb_ch, out_ch, dropout, use_scale_shift_norm)]
                ch = out_ch
                if ds in attention_ds:
                    blocks.append(AttentionBlock(ch, num_head_channels))
                if level != 0 and i == num_res_blocks:   # BigGAN upsampling resblock
                    up = ResBlock(ch, emb_ch, ch, dropout, use_scale_shift_norm, up=True)
                    blocks.append(up if resblock_updown else Upsample(ch, True))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*blocks))

        self.out = nn.Sequential(
            normalization(ch), nn.SiLU(),
            zero_module(conv3(ch, out_channels)))

    def forward(self, x, timestep):
        emb = self.time_embed(timestep_embedding(timestep, self.base_channels))
        hs = []
        h = x
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        return self.out(h)
```

The result is a UNet that is wider rather than deeper, multi-head attentive at 32/16/8 with 64 channels per head, residual through every resolution change via BigGAN-style up/downsampling, and conditioned by AdaGN's per-channel gain instead of a bare additive bias — every piece answering a named weakness of the prior denoiser, with the ε-prediction loss, the sampler, and everything outside the network left untouched so the comparison isolates architecture alone. I've checked that the assembled network is shape-correct and that attention lands exactly on the coarse tiers the N² argument said it should; whether each change actually earns its keep on FID is exactly what the sweep, not this page, has to decide.
