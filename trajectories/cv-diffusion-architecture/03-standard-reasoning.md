Full-attention paid off, and the way it paid off names the next move precisely. It came in at 17.44
small, 8.82 medium, 7.31 large — beating no-attn's 21.39/11.25/9.55 at every scale. So the diagnosis
from rung one was right: the convolution-only floor was a long-range-coherence failure, and adding a
direct, content-addressed, every-to-every coupling at every resolution recovered a large chunk of
the lost quality. The biggest relative win landed exactly where I expected, at the small scale —
21.39 to 17.44, the model that could not afford to grow its way to coherence got the most out of a
parameter-efficient coordinating mechanism. Good. But now look harder at the *cost-benefit* of where
that attention sits, because that is the question this rung has to answer. Full-attn put
attention-bearing blocks at all four resolutions, including the 32×32 and 16×16 feature maps where
the O(N²) price is highest — at 32×32, N = 1024 and the attention matrix is ~10⁶ entries per layer.
It is the slowest, most memory-hungry architecture on the ladder. And I have a specific suspicion
about it: that most of the FID it bought came from attention at the *coarser* feature maps, and the
expensive fine-resolution attention at 32×32 was largely idle — paying a heavy quadratic price for
little coherence, because the global structure it could coordinate at the finest map is thin.

Why thin? Think about what attention at 32×32 is actually doing on a CIFAR image. The 32×32 feature
map is the input resolution, barely processed — the features there are local: edges, color, fine
texture. Attention at that map lets every one of 1024 positions attend to every other, but the
relationships that *matter* for image coherence — the gross layout, the agreement of an object's
parts, large-scale symmetry — are not yet legible in those shallow local features; they only emerge
after the contracting path has built abstraction, at the coarser maps. So at 32×32 the attention is
mostly coordinating texture against texture, which convolution already handles locally and cheaply,
while paying the maximal quadratic cost to do it. Meanwhile the long-range coordination that actually
helps is happening at 16×16 and below, where the features are abstract enough to carry global
meaning and N is small enough that attention is affordable. If that picture is right, then full-attn
is over-provisioned: I could drop the attention at the fine maps, keep it where it earns its cost,
and match — or even beat — full-attn's FID at a fraction of the compute. That is the hypothesis this
rung tests.

So the move is to find the *sweet spot* placement, and to reason about which single resolution is the
sweet spot rather than guess. Attention is O(N²) in spatial positions, so it is expensive at the
fine maps and nearly free at the coarse ones; but at the very coarsest map, 4×4 with N = 16, there is
almost no global structure left to coordinate — 16 positions is barely a layout. So the placement
trades two things against each other along the resolution axis: cost falls as I go coarser, but the
amount of meaningful global structure to coordinate first rises (as features become abstract enough
to carry layout) and then falls (as the grid gets too coarse to hold layout at all). That trade-off
has an interior optimum. The 16×16 feature map is where it lands: coarse enough that the quadratic
cost is modest (N = 256, a 65k-entry attention matrix, a fraction of 32×32's 10⁶), and the features
are abstract enough that there is real global layout and symmetry to coordinate across the image,
yet fine enough that the grid still holds genuine spatial structure (256 positions, not 16). Below
16×16 the structure thins toward the trivial bottleneck; above it the cost explodes for features too
local to need global mixing. So 16×16 is the resolution where attention buys the most coherence per
unit of quadratic cost. Put a single attention-bearing block there, and leave the local texture at
the finer resolutions — 32×32 — to convolution, which is what convolution is good at and cheap at.

This is exactly the original DDPM placement, and now I can see *why* it is the placement rather than
just inheriting it: it is the conclusion of the cost-vs-structure trade-off along the resolution
axis, the targeted version of full-attn's unrationed everywhere. The reverse process is, at heart, a
denoiser whose input and output are same-resolution images, so a U-Net over Wide-ResNet blocks is
the backbone — group normalization (batch- and noise-level independent, so it normalizes
consistently across timesteps and small batches), a sinusoidal timestep embedding pushed through a
small MLP and added into every residual block (so every layer can self-modulate to the current noise
scale), skip connections so the high-frequency detail a denoiser must restore bypasses the
bottleneck — and self-attention at exactly one feature resolution, 16×16, between the convolutional
residual blocks. Concretely on a 32×32 image that means four feature resolutions, 32→16→8→4, with
attention only at the second one. Group norm rather than weight or batch norm because it is
independent of batch size and of the per-sample noise scale, so it normalizes consistently across
all t and across the small batches used at higher resolution. The timestep embedding injected
*everywhere* rather than only at the normalization layers or only at the output, so every block can
condition on the noise level rather than relying on a single conditioning point.

I want to be careful about what is genuinely under my control here versus what the harness fixes,
because the whole derivation behind this denoiser — the ε-prediction objective, the closed-form
forward marginal that lets training jump to any t in one shot, the conditioning of the forward
posterior on x₀ that turns the variational bound into closed-form Gaussian KLs, the ε-parameterization
that makes the network predict the noise, the simplified unit-weight MSE, the linear β schedule with
T = 1000 — all of that is the fixed substrate. The scaffold gives me the ε-MSE loss, the 1000-step
linear schedule, 50-step DDIM, AdamW, and the EMA, none of which I touch. What I am editing is
solely the architecture, and within the architecture the only thing that distinguishes this rung
from full-attn is *where* the attention sits: this is `("DownBlock2D", "AttnDownBlock2D",
"DownBlock2D", "DownBlock2D")` on the way down and the mirror `("UpBlock2D", "UpBlock2D",
"AttnUpBlock2D", "UpBlock2D")` on the way up — attention-bearing blocks only at the 16×16 level,
pure convolution at 32×32, 8×8, and 4×4 (with the bottleneck's default mid-block attention still
present, as in every rung). Everything else — the channel schedule from `BLOCK_OUT_CHANNELS`,
`layers_per_block`, group norm with 32 groups, `norm_eps=1e-6`, SiLU, the positional time embedding
with `flip_sin_to_cos=False`, `freq_shift=1`, `downsample_padding=0` — is the shared DDPM
configuration, identical across the ladder, so the only variable is attention placement (the full
scaffold module is in the answer). This is the configuration `google/ddpm-cifar10-32` uses, and it
is the standard reference precisely because it is the targeted placement the cost-structure
trade-off points to.

So the delta from full-attn is a *removal*: drop the attention at 32×32, 8×8, and 4×4
per-resolution stages, keep it only at 16×16. Reading full-attn's numbers, here is what I expect and
where I am unsure. If my suspicion is right — that the fine-resolution attention was idle and the
coarse attention did the work — then standard should *match or beat* full-attn at the medium and
large scales while costing far less compute, because at those scales the model had the capacity to
exploit attention wherever it helped, and concentrating attention at the one resolution where it
helps removes wasted quadratic computation without removing the coherence. I would specifically
expect standard to edge ahead at medium and large, where the disciplined single-resolution placement
is enough and the everywhere-placement was over-provisioned. The small scale is where I am least
sure: there the model is thin, and it is possible that the extra attention at the fine maps in
full-attn was actually doing something — a small model has less convolutional capacity to handle
even the local texture, so the 32×32 attention might be substituting for missing conv capacity rather
than coordinating genuine long-range structure. If that is the case, full-attn could hold a small-scale
edge over standard even as standard wins the larger scales. So the falsifiable prediction is sharp:
standard ≥ full-attn at medium and large (the targeted placement matches the everywhere placement at
a fraction of the cost, confirming the fine-resolution attention was over-provisioned), with the
small scale the one place full-attn's extra attention might still pay. If standard wins the larger,
more capacity-rich scales, that is the evidence that 16×16-only is the sweet spot — the most
coherence per unit of quadratic cost — and that full-attn was paying for attention it did not need.

The causal chain in one breath: full-attn's 17.44/8.82/7.31 beat no-attn at every scale, confirming
the coherence cure, but it placed attention at all four resolutions including the 32×32 map where
O(N²) is maximal and the features are too local to carry the long-range structure attention is for →
the cost-vs-structure trade-off along the resolution axis has an interior optimum, because cost falls
toward coarse maps while meaningful global structure first rises (features grow abstract) then falls
(grid grows too coarse), and that optimum is 16×16: modest quadratic cost, features abstract enough
to hold layout, grid fine enough to hold spatial structure → so concentrate attention at 16×16 only,
leaving 32×32 texture to cheap convolution, which is exactly the standard DDPM placement, now derived
rather than inherited → expecting standard to match or beat full-attn at medium and large (the
targeted placement is the sweet spot; full-attn was over-provisioned at the fine maps), with the
small scale the one place full-attn's extra attention might still substitute for missing conv
capacity.
