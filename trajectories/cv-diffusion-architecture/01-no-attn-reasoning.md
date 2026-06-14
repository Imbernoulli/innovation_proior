The architecture is the whole task, and before I can ask whether self-attention earns its place I
need a floor — the most stripped-down denoiser the harness will accept, so that every later rung
measures the *value of attention* against a clean convolution-only baseline. So the pain to start
from is just building a denoiser at all, inside this scaffold, that satisfies the contract: take a
noised `[B,3,32,32]` image and a timestep, return a same-shaped ε prediction with a `.sample`
attribute. Everything around it — ε-prediction, the linear 1000-step schedule, 50-step DDIM, AdamW,
the EMA, clean-fid — is frozen. The only thing I get to write is `build_model`.

What is the natural backbone for a denoiser? The input and output are both same-resolution images,
and the job is image-to-image: given a noisy 32×32 image, produce a 32×32 noise field. That is
exactly the dense-prediction shape the UNet was built for — a contracting path of
convolution-and-downsample stages to build a receptive field, a symmetric expanding path of
upsample stages to recover resolution, and skip connections that concatenate equal-resolution
encoder maps onto the decoder so a convolution can fuse fine localization with coarse context. The
skip connections matter especially for a denoiser, because removing noise is fundamentally about
*restoring fine detail*, and a pure bottleneck would crush exactly the high-frequency content the
output has to reconstruct; the skips let that detail bypass the bottleneck. So a time-conditioned
UNet over residual blocks, with group normalization and a sinusoidal timestep embedding injected
into every block, is the obvious substrate — and it is precisely what `diffusers`' `UNet2DModel`
gives me, with the contract already satisfied (`.sample` is the predicted ε). I do not need to
hand-roll the module; I configure it.

Now the design lever this task is actually about: where, across the UNet's four resolution levels
(32→16→8→4 feature maps), do I place self-attention? Self-attention is the operation that couples
any two spatial positions in a single layer with a content-dependent weight — the thing convolution
cannot do, because a 3×3 kernel only ever mixes a local neighborhood, and a dependency between two
distant regions has to be smeared across a long chain of conv-and-downsample layers before their
receptive fields overlap. That smearing is expensive to represent and brittle to optimize. So
attention is the natural cure for long-range coherence. But it is not free: its cost is O(N²) in the
number of spatial positions N = H·W. At the full 32×32 feature map N = 1024 and N² ≈ 10⁶ per
attention layer; at the coarsest 4×4 map N = 16 and the quadratic cost is nothing. That single fact
is going to govern the whole ladder.

Given that, the *floor* I want is the lean hypothesis: that per-resolution self-attention is not
necessary at all on CIFAR-10. The UNet already gives every output pixel a large receptive field
through its pooling, so in principle convolution alone could propagate global structure; the
question is whether it suffices. The cheapest, fastest architecture — and the cleanest test of "is
attention worth its O(N²) cost" — is to strip per-resolution attention out entirely: every down
stage and every up stage purely convolutional, `DownBlock2D`/`UpBlock2D` at all four resolution
levels, no `AttnDownBlock2D`/`AttnUpBlock2D` anywhere. This is the smallest and fastest denoiser the
contract permits, and it is the right starting rung because it isolates the variable the whole task
turns on: it says "convolution only," and every later rung adds attention back somewhere and pays to
find out whether that attention buys lower FID.

There is one subtlety I should be honest about, because it is in the scaffold and not in my control.
When I select pure convolutional down/up blocks, `UNet2DModel` still builds its default mid-block —
`UNetMidBlock2D` — which includes one self-attention layer at the bottleneck (the 4×4 resolution).
So "no-attn" does not mean *zero* attention; it means no attention at any *per-resolution* stage,
with a single attention layer surviving only at the smallest, cheapest feature map where N² is
trivial. That is actually the right shape for the lean hypothesis anyway: it tests whether
convolution suffices at every resolution while keeping one global mixing layer where it costs almost
nothing. I am not paying the quadratic price anywhere it would hurt, and I am asking whether that is
enough.

The rest of the configuration I match to the standard DDPM CIFAR settings so that attention
placement is the *only* thing that varies across the ladder — otherwise a later rung's win could be
a normalization or embedding artifact rather than the attention I am studying. Four resolution
levels with the channel schedule supplied by `BLOCK_OUT_CHANNELS` (so the same definition scales
across Small/Medium/Large), `layers_per_block` from the env (default 2), group normalization with 32
groups, `norm_eps=1e-6`, SiLU activation. The timestep embedding settings are the ones that match
`google/ddpm-cifar10-32`: `time_embedding_type="positional"`, `flip_sin_to_cos=False`,
`freq_shift=1`, and `downsample_padding=0` for the convolutional downsampling. These are not free
choices I am making — they are the fixed harness vocabulary that all three baselines share, so that
the comparison is apples to apples. The only field that moves between this rung and the next is the
`down_block_types`/`up_block_types` tuple.

Let me reason about what the residual blocks are doing so the convolution-only claim is concrete. A
`DownBlock2D` is a stack of `layers_per_block` Wide-ResNet blocks (group norm, SiLU, conv; the
timestep embedding projected and added in; group norm, SiLU, dropout, conv; residual add) followed
by a downsampling convolution. `UpBlock2D` is the mirror: concatenate the same-resolution encoder
skip, run the residual blocks, then upsample. With pure conv blocks at every level, the only
mechanism for one region of the image to influence a distant region is the growth of the receptive
field through successive downsamplings — by the 4×4 bottleneck every position has, in principle, seen
the whole image, and the single mid-block attention can coordinate at that scale. On the way back up,
the decoder reconstructs detail from the concatenated high-resolution encoder skips. So the network
*can* express some global structure; the open question this rung poses is whether convolution's
indirect, multi-layer coupling is as good as direct per-resolution attention for the kind of global
coherence CIFAR images need.

Two details inside those blocks are worth making explicit, because they are the reason a
convolution-only denoiser is even competitive and not just a strawman. First, the residual
structure: each Wide-ResNet block computes a *correction* added onto an identity shortcut, so a deep
stack of them stays trainable — the identity path keeps gradients alive through the whole contracting
and expanding arms, which is exactly what lets the network be deep enough to grow a large receptive
field in the first place. Without it, a denoiser this deep would be hard to optimize and the
"indirect coupling through many layers" story would collapse before it even got a chance to carry
long-range signal. Second, the timestep conditioning: the same network must denoise at *every* noise
level t, from near-clean (tiny corrections) to near-pure-noise (invent structure), and the only thing
telling it which level it is on is the sinusoidal timestep embedding projected and added into every
residual block. Adding it into *every* block — rather than once at the input — means every layer can
self-modulate to the current noise scale, which matters because the denoising task changes character
across the schedule. Group normalization rather than batch norm is the right normalization here for a
specific reason: the effective batch is small (and at the large channel scale, batch 64), and the
per-sample statistics vary with the noise level t, so batch statistics would be unreliable; group
norm computes its statistics over channel groups within a single example, independent of batch size
and of t, so it normalizes consistently across the whole schedule and across all three channel
tiers. These are not choices I am free to vary on this rung — they are the shared harness vocabulary
— but they are why the floor is a *real* floor and not a crippled baseline: the convolutional
denoiser is given every advantage except per-resolution attention, so the gap a later rung opens is
attributable to attention and nothing else.

I should also be clear about how the channel schedule interacts with the convolution-only choice,
because it bears on where I expect this rung to be weakest. The channels double (or hold) as
resolution halves down the contracting path — `(64,128,128,128)` at the small tier up to
`(256,512,512,512)` at large — so capacity grows as the grid shrinks, which is the standard way to
keep representational capacity roughly constant while the spatial extent collapses and abstraction
rises. A convolution-only denoiser spends all of that capacity on local filters plus the indirect
receptive-field growth; an attention-bearing one could spend some of it on direct long-range
coupling. So at the larger tiers, where there is the most capacity to *misallocate* toward indirect
coupling that attention would do better, I expect the convolution-only model to leave the most on
the table relative to an attention-bearing competitor — even though, in absolute FID, the larger
tiers will look best simply because there is more capacity overall.

Now reason about what this floor should do, because that is the entire point of running it. CIFAR-10
images are small and texturally rich; a lot of what FID rewards is local — sharp edges, plausible
color, convincing texture — and that is exactly what a convolutional denoiser is good at. The skip
connections preserve the fine detail, the bottleneck attention handles whatever global coordination
is cheap, and the network should produce recognizable, locally-coherent samples. But the things a
purely-convolutional denoiser is structurally worst at are precisely the long-range relationships:
the agreement of two distant parts of an object, the global layout of a scene, symmetries across the
image. Those have to be carried indirectly through the conv stack, and on a budget that indirect
path is where I expect quality to leak. So I expect this rung to be *competent but not the best* —
the floor against which adding attention is measured. And I expect the gap to be most visible at the
larger channel scales, where the model has the capacity to exploit attention if it had it, so a
purely-convolutional model leaves the most on the table there relative to an attention-bearing one.

Concretely, my edit is the minimal one: fill `build_model` to return a `UNet2DModel` with
`down_block_types = ("DownBlock2D",)×4` and `up_block_types = ("UpBlock2D",)×4` — pure convolution
at every resolution, the default bottleneck attention the only attention in the network — with the
shared DDPM configuration for everything else (the distilled module is in the answer). If this floor
already produces good FID, it would say per-resolution attention is unnecessary on CIFAR-10; if it
trails, the gap names the next move exactly — add self-attention back at some resolution and find out
where, and how much, it pays. Either way, this is the right rung to stand on first, because it is the
one architecture on the ladder that buys nothing from attention beyond the free bottleneck layer,
and so it is the floor every later rung must beat to justify the O(N²) cost it pays.
