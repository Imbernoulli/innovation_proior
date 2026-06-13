GPTQ's numbers came back and they half-confirm, half-redirect me. At INT3-g128 it cut RTN's blow-up
from 6.7341 (degradation 1.8270) down to 6.1011 (degradation 1.1940) — a real recovery, exactly the
direction I bet on: error compensation gave the deferred residual somewhere to go, so the thin-grid
case stopped compounding as badly. At INT4-g128 it moved RTN's 5.1343 to 5.0711 (degradation 0.1640),
a genuine but smaller gain, and at INT4-g64 it landed 5.0435 (degradation 0.1363), the smallest gap of
the three — also as predicted, because fine grouping already does much of the work compensation would
otherwise do, so the two mechanisms overlap. But two things in the GPTQ result bother me. First, INT3
is *still* the open wound: 1.1940 degradation is better than RTN but it is an order of magnitude worse
than INT4's 0.16, so the 8-level grid is plainly not solved — compensation softened the blow without
removing the cause. Second, GPTQ paid roughly 220s per setting against RTN's ~22s, a full order of
magnitude, for that recovery; it builds and inverts a dense $\mathbf H$ and sweeps columns per layer.
So I have a method that is accurate-ish but heavy, and the heaviness is structural — the inverse and
the sequential sweep are intrinsic to error feedback. The diagnosis I flagged when I built GPTQ now
bites: if INT3 does not fully recover under compensation, the bottleneck at 8 levels may not be the
rounding *objective* but the grid coarseness itself, and the move is to attack the rounding *before* it
happens — to protect the weights that matter rather than clean up after rounding them — without an
inverse Hessian at all.

So let me ask the question GPTQ never asks: are all the weights in a layer equally important to the
output? GPTQ implicitly says no — it weights each column's error by $\mathbf H^{-1}$ — but it spends a
dense inverse to express that. There is a cheaper signal. The diagnostic is to quantize a layer hard
(INT3) but keep a small fraction of weight *channels* in FP16, and ask how much accuracy returns per
channel kept. Keeping a tiny fraction — order 1% — recovers most of the lost accuracy, which means a
small salient set is doing the heavy lifting. The real question is which 1%. The obvious guess is the
large-magnitude weights; selecting by weight norm barely beats keeping a random 1%, so weight magnitude
is *not* what makes a weight important. The other signal is the activation: if I instead keep the weight
channels that multiply the largest-magnitude input features — salient by *activation* magnitude — the
same 1% budget gives a large recovery. The reading is simple and it is the same structure GPTQ's
$\mathbf X^\top\mathbf X$ encoded: an input feature with large magnitude contributes a lot to the output,
so the weights that process it matter a lot. But where GPTQ needed the full second-moment matrix and its
inverse, saliency only needs the per-channel activation magnitude — a vector, not a matrix, and no
inverse.

A working recipe falls out: keep the activation-salient channels in FP16, quantize the rest. But that
is a mixed-precision tensor — some channels FP16, most INT3 — and that is a hardware nightmare: irregular
layout, scattered access, custom kernels. The whole point of low-bit weights was a uniform grid an
on-device kernel can dequantize and multiply. I need to protect the salient weights *without* storing
them at a different precision. So look hard at where one channel's quantization error comes from and
whether I can shrink it at fixed bit-width. Take a group of weights with output $y=\mathbf w x$,
quantized symmetrically as $Q(\mathbf w)=\Delta\cdot\mathrm{round}(\mathbf w/\Delta)$ with
$\Delta=\max(|\mathbf w|)/q_{\max}$. The absolute rounding residual per element is roughly uniform — about
a quarter of a step on average — and it is a grid-position error, not a function of the weight's size.
Now scale one high-activation input channel's weights by $s>1$ before quantizing and divide the matching
activation by $s$ afterward. Before rounding this is an exact equivalence:
$\mathbf W\mathbf x=(\mathbf W\,\mathrm{diag}(\mathbf s))(\mathrm{diag}(\mathbf s)^{-1}\mathbf x)$. After
rounding, that element's contribution becomes $Q(ws)\cdot(x/s)$, whose absolute error is
$\Delta'\cdot\mathrm{RoundErr}(ws/\Delta')\cdot|x|/s$. Compared to the original $\Delta\cdot\mathrm{RoundErr}\cdot|x|$,
the expected residual is unchanged (~0.25 either way), and if scaling a small fraction of channels by a
modest $s$ leaves the group's max — and hence $\Delta'$ — about where it was, the error ratio is
$\approx 1/s$. The salient channel gets effectively finer resolution, purely by an equivalence transform,
with no FP16 side table and no bit-width change. That is the escape from the mixed-precision type, and it
is cheaper than GPTQ's inverse: a per-channel scale, not a matrix solve.

Now the trade-off that decides everything, and the reason I cannot just crank $s$. The error I just wrote,
$(\Delta'/\Delta)(1/s)$, keeps shrinking for the salient channel — but I forgot the other side. Every
*non-salient* weight in the same group has error proportional to $\Delta'$. Push $s$ high enough that the
scaled salient weight becomes the group maximum and $\Delta'$ grows, and then $\Delta'/\Delta>1$ amplifies
the error of everyone else in the group. Protecting the salient weights by overscaling quietly damages the
bulk — the very weights RTN was already starving. So there is a genuine balance, and the right scale must
account for *both* the protected channel and the collateral on the rest. The honest objective is to pick
the whole per-channel scale vector $\mathbf s$ to minimize the layer's quantized output error, not one
channel's error in isolation. But $Q$ has a non-differentiable round in it, and I refuse to bring back
gradient-based optimization — that is exactly the backprop GPTQ's error-feedback regression replaced with
a closed form, and the constraint here forbids touching weights with gradients anyway.

I do not need a free search over $\mathbb R^{C_{in}}$. I already know the one thing that sets saliency —
the per-channel activation magnitude — so I need a one-parameter family that dials "how hard do I scale
the high-activation channels," and I can grid-search a single scalar cheaply. Let $\mathbf x_{\max}$ be
the per-input-channel *average* magnitude of $|X|$ from a calibration pass (average, not max — I want the
typical importance of a channel, and averaging is robust to a few calibration outliers). Define
$\mathbf s=\mathbf x_{\max}^{\alpha}$ and search $\alpha\in[0,1)$. At $\alpha=0$, $\mathbf s=\mathbf 1$ —
plain RTN, no scaling. At $\alpha\to1$ the scaling tracks activation magnitude most aggressively — maximum
protection, maximum risk of inflating $\Delta$. $\alpha$ slides between "protect nobody" and "protect the
salient channels hard," which is precisely the balance I diagnosed. Because it is one scalar I sweep it
over a fine grid (`N_ALPHA=20` ratios), and for each candidate I quantize the scaled weights, undo the
scale, and measure the actual output error — the true objective, evaluated directly, no gradient. To keep
the scales numerically tame I normalize $\mathbf s$ by the geometric mean of its max and min before
applying, so it neither blows the weights up nor collapses them.

Here I must be careful, because this task's edit surface shapes the method in ways that differ from the
most general form, and I want the reasoning to land *the code that runs*. The harness hands me **one linear
layer at a time** — no cross-layer context, no adjacent operator to fold the inverse scale into. The
general form of activation-aware scaling absorbs $\mathrm{diag}(\mathbf s)^{-1}$ into the *previous*
operator (a LayerNorm or the preceding linear), keeping the network function exactly unchanged. I cannot do
that here: I only own this one `quantize()`. So the equivalence is realized *within the layer* — I scale
$\mathbf W$ by $\mathbf s$ along input channels, quantize, dequantize, and then divide the dequantized
weight back by $\mathbf s$ (`W_dq / s`). The scaling is undone in the stored weight itself rather than
pushed onto a neighbor, so the layer's output is mathematically the dequantized-and-unscaled $\mathbf W$ —
self-contained, same shape, same dtype. The loss I score each $\alpha$ against is therefore computed at
*linear-layer* granularity, not full-block: I keep a reservoir of real input tokens $X$ from `add_batch`
(stride-sampled to a few hundred rows to bound memory, kept on CPU across layers and moved to device at
quantize-time) and score $\alpha$ by $\lVert X(\mathbf W-\mathbf W_{\text{final}})^\top\rVert^2$ averaged
over tokens — the true per-layer output MSE on real activations, the honest objective evaluated directly.
And the grid is **symmetric** here ($q_{\min}=-2^{b-1}$, $q_{\max}=2^{b-1}-1$, zero point 0), the same
convention RTN and GPTQ used on this surface — not an affine zero-point quantizer — so the whole comparison
stays on one grid and the $1/s$ argument is the symmetric-step version. I should not import the
asymmetric-zero-point or cross-layer-migration story into how I describe what runs; what runs is
"per-linear symmetric scale-search, undone in-weight, scored on sampled real inputs."

There is a second knob this surface exposes and the bulk-damage trade-off motivates: after fixing the
per-channel scale, the group step is still set by the post-scale group *max*, and that max can be a lone
outlier dragging the whole group's $\Delta$ up. So I add a per-group clip search on the scaled weights —
shrink each group's max by $1-i/N$ for $i$ up to `CLIP_MAX_SHRINK·N_CLIP_GRID` (half the grid, the default
shrink ceiling), and for each candidate clip the weights to $\pm\text{max}$, quantize, and measure the
*per-(output-channel, group)* output error against the unclipped scaled weights using the same sampled
$X$ — an einsum $\sum_c W_{r,g,c}X_{t,g,c}$ for the original and the quantized-clipped version, choosing
per group the clip that minimizes it. Clipping trades a little extra rounding error on the outlier for a
tighter $\Delta$ on the bulk — directly the bulk-vs-outlier balance, now resolved per group rather than per
channel. It is batched over output channels (`OC_BATCH`) for memory. Then the final quantization uses the
clipped per-group scales, and the channel scale is undone. If there were no calibration samples at all the
clip search falls back to the unclipped max — but the harness always streams 128 sequences, so it runs.

So the delta from GPTQ is a different philosophy on the same grid: GPTQ rounds then *compensates* the
residual through an inverse Hessian; AWQ *protects* the salient weights before rounding with a per-channel
scale, picks that scale and a per-group clip by directly minimizing the real output MSE on sampled
activations, and never forms a Hessian or an inverse. Both consume the calibration stream RTN ignored —
GPTQ as the full second moment $\mathbf X^\top\mathbf X$, AWQ as the per-channel average $|X|$ plus a small
reservoir of raw tokens — but AWQ's is a vector and a search, not a matrix and a solve. On this surface it
is self-contained per-linear and symmetric, with the equivalence undone in the stored weight rather than
migrated to a neighbor.

Reading GPTQ's shape, here is what I expect, falsifiably, against its numbers. The whole bet is that
protecting salient channels before rounding recovers resolution at 8 levels better than compensating after,
so INT3-g128 is the setting I am watching: GPTQ left it at 6.1011 (degradation 1.1940), still the wound,
and if salience-aware scaling plus clipping does what I think, INT3 should drop clearly below GPTQ's — into
the high-5s, cutting that 1.19 degradation toward the high-0.8s, because the salient channels that carried
the output now get finer effective grids while clipping keeps the bulk's $\Delta$ in check. At INT4-g128
the grid is already fat (GPTQ at 5.0711, degradation 0.1640), so I expect a real but smaller improvement,
into the low-5.0s under GPTQ. At INT4-g64 GPTQ was strongest (5.0435, degradation 0.1363); fine grouping
has already protected the bulk, so AWQ and GPTQ should be closest here, and it is plausible AWQ does *not*
beat GPTQ at this one finest-grid setting — that would be consistent with the mechanisms overlapping where
the grid is least coarse. The cost I expect to *win* on relative to GPTQ: no inverse, no column sweep, just
two cheap grid searches over sampled activations, so quant-time should fall well under GPTQ's ~220s, back
toward the low end. The decisive test is INT3: if AWQ beats GPTQ there, protecting-before-rounding is the
right lever at extreme low bit-width; if it does not, the bottleneck at 8 levels is the uniform grid itself
and the next move would have to leave uniform quantization behind entirely. (The full scaffold module is in
the answer.)
