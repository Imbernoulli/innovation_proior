GPTQ's numbers came back and they half-confirm, half-redirect me. At INT3-g128 it cut RTN's blow-up
from 6.7341 (degradation 1.8270) down to 6.1011 (degradation 1.1940) — a real recovery, exactly the
direction I bet on: error compensation gave the deferred residual somewhere to go, so the thin-grid case
stopped compounding as badly. Let me read that recovery quantitatively before I decide what is left to
do, because the size of what compensation *did not* fix is the whole brief for this rung. In absolute
terms GPTQ erased $1.8270-1.1940=0.6330$ of INT3 degradation, a $34.6\%$ cut; at INT4-g128 it took
$0.2271$ down to $0.1640$, a $27.8\%$ cut; at INT4-g64, $0.1819$ to $0.1363$, $25.1\%$. So compensation's
*fractional* bite is actually largest exactly where the grid is thinnest — INT3 — which is encouraging,
it means the mechanism does aim where the damage is. But the absolute residual tells the opposite story:
the INT3-to-INT4 degradation ratio was $1.8270/0.2271=8.0$ under RTN and is $1.1940/0.1640=7.3$ under
GPTQ — barely narrowed. INT3 is still an order of magnitude worse than INT4, and compensation moved that
ratio by less than a tenth. So the reading is that error feedback attacks a real axis but not *the* axis
that separates 8 levels from 16: it softened the blow without removing the cause. And it was not cheap.
GPTQ paid $219.9$, $220.6$, $219.1$ seconds across the three settings — flat to within a second, a full
order of magnitude over RTN's $\sim22$s. That the cost is *identical* across bit-widths and group sizes is
itself a diagnosis: the price is the dense $\mathbf H$, its Cholesky inverse, and the sequential column
sweep, none of which care how many levels the grid has. So I am holding a method that is accurate-ish but
structurally heavy, and the heaviness buys a recovery that leaves INT3 unsolved. The move that suggests
itself is to stop cleaning up after rounding and attack the rounding *before* it happens — protect the
weights that matter rather than compensate the ones that drifted — and to do it without an inverse Hessian
at all.

So let me ask the question GPTQ never asks: are all the weights in a layer equally important to the
output? GPTQ implicitly says no — it weights each column's error by $\mathbf H^{-1}$ — but it spends a
dense inverse to express that no. There is a cheaper signal, and GPTQ half-revealed where it lives. The
Hessian it built is $\mathbf H=\mathbf X^\top\mathbf X$, and its *diagonal* $\mathbf H_{ii}=\sum_t x_{t,i}^2$
is just the total energy calibration text puts through input channel $i$ — a per-channel scalar, no
inversion required. The diagnostic that isolates whether a small set carries the layer is to quantize hard
(INT3) but keep a tiny fraction of weight *channels* in FP16 and watch how much accuracy returns per
channel kept. Keeping order $1\%$ recovers most of the lost accuracy — so a small salient set is doing the
heavy lifting, and the question collapses to *which* $1\%$. The obvious guess is the large-magnitude
weights; but selecting by weight norm barely beats keeping a random $1\%$, so weight magnitude is *not*
what makes a weight important. Select instead the channels that multiply the largest-magnitude input
features — salient by *activation* magnitude, i.e. large $\mathbf H_{ii}$ — and the same $1\%$ budget gives
a large recovery. The reading is the same structure GPTQ's $\mathbf X^\top\mathbf X$ encoded, now read off
its diagonal instead of its inverse: an input feature with large magnitude contributes a lot to the
output, so the weights that process it matter a lot. But where GPTQ needed the full second moment and its
inverse, saliency needs only the per-channel activation magnitude — a vector, not a matrix, and no solve.

A working recipe falls out of that: keep the activation-salient channels in FP16, quantize the rest. But
that is a mixed-precision tensor — a few channels FP16, most INT3 — and that is a hardware nightmare:
irregular layout, scattered memory access, a custom kernel to stitch the two precisions per matmul. The
whole reason to go low-bit was a uniform grid a stock kernel can dequantize and multiply in one sweep; a
side table of FP16 columns throws that away. So I need to protect the salient weights *without* storing
them at a different precision. Look hard at where a channel's quantization error comes from and whether I
can shrink it at fixed bit-width. Take a group with output $y=\mathbf w x$, quantized symmetrically as
$Q(\mathbf w)=\Delta\cdot\mathrm{round}(\mathbf w/\Delta)$ with $\Delta=\max(|\mathbf w|)/q_{\max}$. Scale
one high-activation channel's weights by $s>1$ before quantizing and divide the matching activation by $s$
after. Before rounding this is an exact identity,
$\mathbf W\mathbf x=(\mathbf W\,\mathrm{diag}(\mathbf s))(\mathrm{diag}(\mathbf s)^{-1}\mathbf x)$. After
rounding, that element's *effective* weight is $\Delta\cdot\mathrm{round}(ws/\Delta)/s$, and its residual is
$\tfrac1s\lvert ws-\Delta\,\mathrm{round}(ws/\Delta)\rvert\le\Delta/(2s)$ — the worst-case weight error on
the salient channel is $\Delta/(2s)$ instead of $\Delta/2$, a factor $1/s$, *provided* scaling it does not
move the group max and so leaves $\Delta$ where it was.

I have written enough algebra that I want to watch it happen on numbers before I build on it. Take a
four-weight group $\mathbf w=[0.90,\,0.10,\,-0.08,\,0.05]$ with channel 1 (weight $0.10$) the salient one
— a small weight but a big activation. At INT4, $q_{\max}=7$ and $\Delta=0.90/7=0.128571$; channel 1
rounds as $\mathrm{round}(0.10/\Delta)=\mathrm{round}(0.778)=1$, dequantizing to $0.128571$, a residual of
$0.028571$. Now scale channel 1 by $s=3$: the scaled weight is $0.30$, still below the group max $0.90$, so
$\Delta'=\Delta$ is untouched; $\mathrm{round}(0.30/\Delta)=\mathrm{round}(2.333)=2$, dequant $0.257143$,
undo the scale to $0.257143/3=0.085714$, residual $\lvert0.10-0.085714\rvert=0.014286$ — exactly halved.
Not the factor of three the $1/s$ bound advertises: the discrete rounding shifted the fractional landing
from $0.222$ of a step to $0.333$, eating part of the gain. Push $s=5$ and the same weight lands at
$\mathrm{round}(3.889)=4$, residual $0.002857$ — now a factor of *ten*, a lucky landing the other way. So
the per-weight shrink is real but noisy, swinging around $1/s$ with the discreteness of where each weight
falls; it is the average over a channel's hundreds of output rows that converges to $1/s$, not any single
weight. That is already a design instruction: I must not trust the $1/s$ formula to *pick* the scale — I
have to measure the real output error over many rows and tokens. And the collateral is exactly bounded in
this toy: channels 0, 2, 3 keep their identical codes and residuals for every $s$ up to the tipping point
$s\cdot0.10=0.90$, i.e. $s=9$, beyond which the salient channel becomes the group max, $\Delta'=0.10s/7$
starts to grow, and every other weight's error grows with it. So the four weights hand me the whole trade
in one picture: inside $s\in[1,9]$ the salient channel gains toward $1/s$ at zero cost to the bulk; past
$9$ I start paying the bulk to protect the one channel.

That tipping point is the whole game, and it is why I cannot just crank $s$. The $1/s$ shrink applies only
to the protected channel; every non-salient weight in its group carries error proportional to $\Delta'$,
so once a scaled channel becomes the group max and inflates $\Delta'$, the ratio $\Delta'/\Delta>1$
amplifies everyone else — protecting the salient weights quietly damages the bulk that RTN was already
starving. The honest objective is therefore to choose the whole per-channel scale vector $\mathbf s$ to
minimize the layer's quantized *output* error, both sides accounted. But $Q$ carries a non-differentiable
round, and I refuse to reintroduce gradient optimization — that is exactly the backprop GPTQ's closed form
replaced, and the constraint here forbids touching the weights with gradients anyway.

Before I settle on how to choose $\mathbf s$ I should look at what else is on the table at this junction,
because "protect before rounding" has more than one realization and I want the cheapest one that actually
tests the hypothesis. One option is a non-uniform grid — a per-group codebook fit to the weight
distribution, spending levels where the mass is instead of uniformly. But that breaks the uniform-grid
contract the driver and any dequantize kernel assume, needs an index table plus stored centroids per
group, and fitting centroids means either an iterative k-means over the layer's millions of groups or a
gradient — expensive and off-surface. A second option is learned rounding: relax the round to a soft,
per-weight up/down decision and optimize it. That reintroduces a per-layer optimization loop over the 224
linears and a non-differentiable operator I would have to smooth — precisely the machinery the closed-form
line was built to avoid. A third, tempting because it looks strictly stronger, is to *stack* scaling on top
of GPTQ: scale the salient channels, then run the Hessian compensation on the remainder. But that inherits
GPTQ's $\sim220$s — the cost I am here to shed — and the diagnosis says compensation attacks the wrong axis
at 8 levels, so once the salient channels are protected the residual GPTQ would compensate is already
small; I would pay the full inverse for a shrinking return. Cleaner to test protect-before-round on its
own first, and it is by far the cheapest. So I keep the uniform symmetric grid and choose $\mathbf s$ by
search.

I do not need a free search over $\mathbb R^{C_{in}}$: the one thing that sets saliency is the per-channel
activation magnitude, so a one-parameter family suffices. Let $\mathbf x_{\max}$ be the per-input-channel
*average* $|X|$ over calibration — average, not max, because a max over $128\times2048\approx2.6\times10^5$
tokens is dominated by a single rare spike, whereas saliency is about a channel's *typical* excitation, and
the average is what the diagonal-energy reading pointed at. Define $\mathbf s=\mathbf x_{\max}^{\alpha}$,
$\alpha\in[0,1)$. At $\alpha=0$, $\mathbf s=\mathbf 1$ — plain RTN, no scaling. As $\alpha\to1$ the scaling
tracks activation magnitude hardest — maximum protection, maximum risk of inflating $\Delta$. The exponent
tempers how much of the raw activation disparity turns into weight scaling: two channels whose average
$|X|$ differ 100-fold get scale ratio $100^{\alpha}$, which at $\alpha=0.5$ is $10$ and at $\alpha=0.25$
only $3.16$ — so $\alpha$ is a dial from "ignore the disparity" to "honor it fully," which is exactly the
bulk-versus-salient balance the toy drew. Because it is one scalar I sweep it on a fine grid, `N_ALPHA=20`
ratios $\alpha=i/20$ for $i=0..19$ (so $\alpha\in\{0,0.05,\dots,0.95\}$), and for each candidate I scale
$\mathbf W$, quantize, undo the scale, and score the *actual* output error directly — the true objective,
no gradient, which is exactly what the noisy per-weight $1/s$ trace told me I must do rather than trust a
formula. To keep the scales tame I normalize $\mathbf s$ by the geometric mean of its extremes,
$\mathbf s\leftarrow\mathbf s/\sqrt{s_{\max}s_{\min}}$: that maps the range to
$[\sqrt{s_{\min}/s_{\max}},\,\sqrt{s_{\max}/s_{\min}}]$, geometrically centered on $1$, so the hardest
scale-up equals the hardest scale-down in log units — e.g. a raw range $[0.5,8]$ becomes $[0.25,4]$ — and
$\mathbf s$ neither blows the weights up nor collapses them into the floor.

Before I lean on that normalization I should check what it actually changes, because it would be easy to
imagine it reshaping the objective when it does not. Rescale the whole vector $\mathbf s\to c\mathbf s$ and
follow it through: the scaled weight becomes $c\,\mathbf W\mathrm{diag}(\mathbf s)$, `find_scale_zero` reads
its per-group max and so returns $\Delta\to c\Delta$, and the code
$\mathrm{round}(cWs/(c\Delta))=\mathrm{round}(Ws/\Delta)$ is untouched; dequantizing gives $c$ times the
old $W_{\text{dq}}$, and the in-weight undo divides by $c\mathbf s$, cancelling the $c$ — so
$\mathbf W_{\text{final}}$, and therefore the output MSE I rank $\alpha$ by, is *invariant* to a global
rescale of $\mathbf s$. The geometric-mean normalization moves no rounding decision on its own. What it
earns is numerical: the search runs $\mathbf s=\mathbf x_{\max}^{\alpha}$ through a `clamp(min=1e-4)`, the
normalized vector through a `clamp(min=1e-5)`, and the per-group scale through `clamp(min=1e-12)`, and those
floors are *not* scale-invariant — an un-normalized $\mathbf s$ that drifts very large or very small with
$\alpha$ would let one of them bite asymmetrically and quietly corrupt a candidate. Centering the range on
$1$ keeps every $\alpha$ clear of the floors, so the twenty candidates are compared on equal numerical
footing. So the normalization is hygiene, not objective — worth having established, because it means the
*shape* of the search is set entirely by $\alpha$ and the real per-linear MSE, exactly as I intended.

This task's edit surface shapes the rest, and I want the reasoning to land *the code that runs*. The
harness hands me one linear layer at a time — no cross-layer context, no adjacent operator to fold
$\mathrm{diag}(\mathbf s)^{-1}$ into. The general form of activation-aware scaling absorbs that inverse
scale into the *previous* operator — a LayerNorm or the preceding linear — keeping the network function
exactly unchanged. I cannot: I own only this one `quantize()`. So I realize the equivalence *within the
layer* — scale $\mathbf W$ along input channels, quantize, dequantize, then divide the dequantized weight
back by $\mathbf s$ (`W_dq / s`). The scaling is undone in the stored weight itself rather than migrated to
a neighbor, so the layer's effective output is the dequantized-and-unscaled $\mathbf W$ — self-contained,
same shape, same dtype. That means the loss I score each $\alpha$ against is at *linear-layer* granularity,
not full-block: in `add_batch` I accumulate the per-channel sum of $|X|$ (averaged to $\mathbf x_{\max}$ at
quantize-time) and reservoir-sample real input tokens, and I score $\alpha$ by
$\lVert X(\mathbf W-\mathbf W_{\text{final}})^\top\rVert^2$ over those samples — the true per-layer output
MSE on real activations. (If a layer ever saw no calibration — it never does here, but the code guards it —
the score falls back to the per-weight squared error weighted by $\mathbf x_{\max}^2$, which is exactly the
diagonal, uncorrelated-input approximation of that same output MSE:
$\lVert X\,\Delta\mathbf W^\top\rVert^2$ averaged over tokens equals
$\sum_{r,j}\Delta W_{rj}^2\,\mathbb E[x_j^2]$ once the cross-channel terms $\mathbb E[x_jx_k]$ are dropped,
and $\mathbb E[x_j^2]$ is of order $x_{\max,j}^2$. So the two loss paths are one objective at two
fidelities, and the real-sample path is the one that runs.) The reservoir needs a budget: I keep `N_SAMPLE_TOKEN=256` tokens, stride-sampled
from up to $4\times$ that many candidates so they span the calibration set rather than clustering in the
first sequences, held on CPU across layers and moved to device only at quantize-time. For Mistral's widest
input, $4096$ channels, that reservoir is $256\times4096\times4$ bytes $\approx4$ MB — trivial, and the
reason it lives on CPU is that it must survive across all 224 layers without competing for GPU memory with
the block that is currently resident. And the grid is *symmetric* here ($q_{\min}=-2^{b-1}$,
$q_{\max}=2^{b-1}-1$, zero point $0$), the same convention RTN and GPTQ used on this surface, so the $1/s$
argument is its symmetric-step version and the whole comparison stays on one grid; this is per-linear
symmetric scale-search undone in-weight, not the asymmetric-zero-point or cross-layer-migration form.

There is a second knob this surface exposes, and the bulk-damage trade-off motivates it directly: after the
per-channel scale is fixed, the group step is still set by the post-scale group *max*, and that max can be a
lone outlier dragging the whole group's $\Delta$ up. So I add a per-group clip search on the scaled
weights — shrink each group's max by $1-i/N$ for $i$ up to `CLIP_MAX_SHRINK`$\cdot$`N_CLIP_GRID`
($0.5\times20=10$ steps, so the max can be pulled in by at most half), clip the weights to $\pm$max,
quantize, and measure the resulting per-group output error on the same sampled $X$. The shape of that
measurement is worth pinning down, because it is where the clip earns the right to be *per group*. Reshape
$X$ to $(T,G,g)$ with $G=\text{in}/g$ groups of $g$ columns, and the scaled weight to $(\text{out},G,g)$;
the group-$g$ contribution of output row $r$ at token $t$ is
$\mathrm{org\_out}[r,t,g]=\sum_c W_{r,g,c}X_{t,g,c}$, an `einsum('rgc,tgc->rtg')`. A clip on group $g$'s max
changes *only* that group's quantized weights, so the induced output error localizes:
$\mathrm{err}[r,g]=\mathrm{mean}_t(\mathrm{cur\_out}-\mathrm{org\_out})^2$ with $\mathrm{cur\_out}$ the same
einsum on the clipped-and-quantized weights, and I pick per $(r,g)$ the clip minimizing it — an independent
decision per output-channel-and-group, exactly the shape of `best_max`, $(\text{out},G,1)$. Clipping trades
a little extra rounding error on the outlier for a tighter $\Delta$ on the bulk — the same
outlier-versus-bulk balance the toy drew, now resolved at group granularity rather than per channel. It is
batched over output channels (`OC_BATCH`) because $\mathrm{org\_out}$ is $(\text{out},T,G)$ and
materializing all $\text{out}=14336$ rows of an MLP layer at once against 256 tokens is too much memory.

Two ordering questions I should settle rather than assume. Why scale first, then clip — not the reverse?
Because the clip refines the group $\Delta$ *given* the weight distribution, and the scale reshapes that
distribution; clip first and the later scaling re-expands the range and undoes the clip's tightening. And
why two sequential searches instead of one joint sweep over $(\alpha,\text{clip})$? A joint grid is
$20\times10=200$ full quantize-and-score passes against $\approx180$ for the two coordinate searches, an
order more work, and the two knobs are nearly separable — the scale decides *which channels* get
resolution, the clip decides *how tight* each group's step is — so coordinate descent (scale, then clip on
the scaled weights) captures almost all of the joint optimum at a fraction of the cost. If there were no
calibration samples the clip search falls back to the unclipped max, but the harness always streams 128
sequences, so it runs.

So the delta from GPTQ is a different philosophy on the same grid: GPTQ rounds then *compensates* the
residual through an inverse Hessian; this method *protects* the salient weights before rounding with a
per-channel scale, picks that scale and a per-group clip by directly minimizing the real per-linear output
MSE on sampled activations, and never forms a Hessian or an inverse — a vector and two cheap searches, not a
matrix and a solve. Both consume the calibration stream RTN ignored, but where GPTQ read it as the full
second moment $\mathbf X^\top\mathbf X$, this reads it as the per-channel average $|X|$ plus a small
reservoir of raw tokens.

Reading GPTQ's shape, here is what I expect, falsifiably, against its numbers. The whole bet is that
protecting salient channels before rounding recovers resolution at 8 levels better than compensating after,
so INT3-g128 is the setting I am watching: GPTQ left it at $6.1011$ (degradation $1.1940$), still the
wound, and if salience-aware scaling plus clipping does what the toy predicts, INT3 should drop clearly
below GPTQ's — into the high-5s, cutting that $1.19$ degradation toward the high-$0.8$s, because the salient
channels that carried the output now get finer effective grids while the clip keeps the bulk's $\Delta$ in
check. At INT4-g128 the grid is already fat (GPTQ $5.0711$, degradation $0.1640$), so I expect a real but
smaller improvement, into the low-$5.0$s under GPTQ. At INT4-g64 GPTQ was strongest ($5.0435$, degradation
$0.1363$); fine grouping has already protected the bulk the way clipping would, so this is the one setting
where the two mechanisms most overlap, and it is entirely plausible I do *not* beat GPTQ here — that would
be consistent with the diagnosis, not against it. On cost I expect to win, but I want to be precise about
*why*, because "no inverse" is not the whole story. Counting flops, the scale search is $20$ output-MSE
passes of $\approx T\cdot\text{out}\cdot\text{in}=256\cdot4096\cdot4096\approx8.6\times10^{10}$ for a square
projection, and the clip search adds a comparable pile of einsums, so the raw arithmetic is in the same
ballpark as GPTQ's $4096^3\approx6.9\times10^{10}$ inverse plus its sweep — this is *not* obviously fewer
FLOPs. The win is structural in a different way: every one of my passes is a dense GEMM or einsum that
saturates the tensor cores, and my Python loops run $20+10\times16\approx180$ times; GPTQ's column sweep is
a bandwidth-bound rank-one update inside a Python loop that turns roughly $4096/128=32$ blocks into
thousands of small sequential steps, gated by a serial Cholesky inverse. Same order of arithmetic, far
fewer sequential host round-trips and no serial inverse, so I expect quant-time to fall well under GPTQ's
$\sim220$s. The decisive test is INT3: if this beats GPTQ there, protecting-before-rounding is the right
lever at extreme low bit-width; if it does not, the bottleneck at 8 levels is the uniform grid itself and
the next move would have to leave uniform quantization behind entirely. (The full scaffold module is in the
answer.)
