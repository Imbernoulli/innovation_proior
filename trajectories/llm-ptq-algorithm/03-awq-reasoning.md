GPTQ's numbers came back and they half-confirm, half-redirect me. At INT3-g128 it cut RTN's blow-up
from 6.7341 (degradation 1.8270) down to 6.1011 (degradation 1.1940) — a real recovery, exactly the
direction I bet on: error compensation gave the deferred residual somewhere to go, so the thin-grid case
stopped compounding as badly. In absolute terms GPTQ erased $1.8270-1.1940=0.6330$ of INT3 degradation, a
$34.6\%$ cut; at INT4-g128 it took $0.2271$ to $0.1640$, $27.8\%$; at INT4-g64, $0.1819$ to $0.1363$,
$25.1\%$. So compensation's *fractional* bite is largest exactly where the grid is thinnest — encouraging,
the mechanism aims where the damage is. But the absolute residual tells the opposite story: the
INT3-to-INT4 degradation ratio was $1.8270/0.2271=8.0$ under RTN and is $1.1940/0.1640=7.3$ under GPTQ —
barely narrowed. INT3 is still an order of magnitude worse than INT4, and compensation moved that ratio by
less than a tenth. So error feedback attacks a real axis but not *the* axis that separates 8 levels from
16: it softened the blow without removing the cause. And it was not cheap — GPTQ paid 219.9, 220.6, 219.1
seconds across the three settings, flat to within a second, a full order of magnitude over RTN's $\sim22$s.
That the cost is *identical* across bit-widths is itself a diagnosis: the price is the dense $\mathbf H$,
its Cholesky inverse, and the sequential column sweep, none of which care how many levels the grid has. So
I hold a method that is accurate-ish but structurally heavy, and the heaviness buys a recovery that leaves
INT3 unsolved. The move that suggests itself is to stop cleaning up after rounding and attack the rounding
*before* it happens — protect the weights that matter rather than compensate the ones that drifted — and to
do it without an inverse Hessian at all.

So the question GPTQ never asks: are all the weights in a layer equally important to the output? GPTQ
implicitly says no — it weights each column's error by $\mathbf H^{-1}$ — but it spends a dense inverse to
express that no. There is a cheaper signal, and GPTQ half-revealed where it lives. The Hessian it built is
$\mathbf H=\mathbf X^\top\mathbf X$, and its *diagonal* $\mathbf H_{ii}=\sum_t x_{t,i}^2$ is the total
energy calibration text puts through input channel $i$ — a per-channel scalar, no inversion required.
Suppose I probed this: quantize hard at INT3 but keep a tiny fraction of weight channels in FP16 and see
how much accuracy returns. If a small set — order $1\%$ — carried most of the layer, the loss should come
back fast, and the question would collapse to *which* $1\%$. The obvious guess is the large-magnitude
weights, but that guess is likely wrong: weight magnitude says nothing about how hard the *input* it
multiplies is excited, and an error on a weight multiplying a near-dead input costs the output almost
nothing. The channels that matter are the ones multiplying the largest-magnitude input features — salient
by *activation* magnitude, i.e. large $\mathbf H_{ii}$. That is the same structure GPTQ's
$\mathbf X^\top\mathbf X$ encoded, now read off its diagonal instead of its inverse: a vector, not a matrix,
and no solve.

A working recipe would be to keep the activation-salient channels in FP16 and quantize the rest. But that
is a mixed-precision tensor — a few channels FP16, most INT3 — and a hardware nightmare: irregular layout,
scattered access, a custom kernel to stitch two precisions per matmul. The whole reason to go low-bit was a
uniform grid a stock kernel can dequantize and multiply in one sweep. So I need to protect the salient
weights *without* storing them at a different precision. Look at where a channel's quantization error comes
from at fixed bit-width. Take a group with output $y=\mathbf w x$, quantized symmetrically as
$Q(\mathbf w)=\Delta\cdot\mathrm{round}(\mathbf w/\Delta)$ with $\Delta=\max(|\mathbf w|)/q_{\max}$. Scale
one high-activation channel's weights by $s>1$ before quantizing and divide the matching activation by $s$
after. Before rounding this is an exact identity,
$\mathbf W\mathbf x=(\mathbf W\,\mathrm{diag}(\mathbf s))(\mathrm{diag}(\mathbf s)^{-1}\mathbf x)$. After
rounding, that element's *effective* weight is $\Delta\cdot\mathrm{round}(ws/\Delta)/s$, and its residual is
$\tfrac1s\lvert ws-\Delta\,\mathrm{round}(ws/\Delta)\rvert\le\Delta/(2s)$ — the worst-case weight error on
the salient channel is $\Delta/(2s)$ instead of $\Delta/2$, a factor $1/s$, *provided* scaling it does not
move the group max and so leaves $\Delta$ where it was.

The $1/s$ bound is a worst case, and the discreteness of rounding makes the actual per-weight shrink noisy
around it: scaling a salient weight by $s=3$ might land it so its residual halves, by $s=5$ so it drops
tenfold, depending on where the scaled value falls between grid points — it is the average over a channel's
hundreds of output rows that converges to $1/s$, not any single weight. That is already a design
instruction: I must not trust the $1/s$ formula to *pick* the scale, I have to measure the real output
error. And the collateral has a sharp boundary. As long as a scaled channel stays below its group's max,
the group step $\Delta$ is untouched and every other weight keeps its exact code; but the moment $s$ pushes
it past the max and inflates $\Delta$, that $\Delta'/\Delta>1$ amplifies the error of every non-salient
weight in the group — protecting the salient weights quietly damages the bulk RTN was already starving.
Inside the safe range the salient channel gains toward $1/s$ at zero cost; past it I pay the bulk to
protect the one channel. The honest objective is therefore to choose the whole per-channel scale vector
$\mathbf s$ to minimize the layer's quantized *output* error, both sides accounted. But $Q$ carries a
non-differentiable round, and I refuse to reintroduce the gradient optimization the closed-form line was
built to avoid — the constraint here forbids touching the weights with gradients anyway.

Protect-before-rounding has more than one realization, and I want the cheapest that tests the hypothesis. A
non-uniform grid — a per-group codebook fit to the weight distribution — breaks the uniform-grid contract
the driver and any dequantize kernel assume and needs k-means or a gradient to fit, off-surface. Learned
rounding reintroduces a per-layer optimization loop over the 224 linears and a non-differentiable operator
to smooth. Stacking scaling on top of GPTQ inherits its $\sim220$s and pays the full inverse for a
shrinking return once the salient channels are already protected. Cleanest is to test protect-before-round
on its own: keep the uniform symmetric grid and choose $\mathbf s$ by search.

I do not need a free search over $\mathbb R^{C_{in}}$: the one thing that sets saliency is the per-channel
activation magnitude, so a one-parameter family suffices. Let $\mathbf x_{\max}$ be the per-input-channel
*average* $|X|$ over calibration — average, not max, because a max over $128\times2048\approx2.6\times10^5$
tokens is dominated by a single rare spike, whereas saliency is about a channel's *typical* excitation, and
the average is what the diagonal-energy reading pointed at. Define $\mathbf s=\mathbf x_{\max}^{\alpha}$,
$\alpha\in[0,1)$. At $\alpha=0$, $\mathbf s=\mathbf 1$ — plain RTN, no scaling. As $\alpha\to1$ the scaling
tracks activation magnitude hardest — maximum protection, maximum risk of inflating $\Delta$. The exponent
tempers how much of the raw activation disparity turns into weight scaling: two channels whose average
$|X|$ differ 100-fold get scale ratio $100^{\alpha}$, which at $\alpha=0.5$ is $10$ and at $\alpha=0.25$
only $3.16$ — a dial from "ignore the disparity" to "honor it fully," exactly the bulk-versus-salient
balance the scaling argument laid out. Because it is one scalar I sweep it on a fine grid, `N_ALPHA=20`
ratios $\alpha=i/20$, and for each candidate I scale $\mathbf W$, quantize, undo the scale, and score the
*actual* output error directly — the true objective, no gradient, which is what the noisy per-weight $1/s$
trace told me I must do rather than trust a formula. To keep the scales tame I normalize $\mathbf s$ by the
geometric mean of its extremes, $\mathbf s\leftarrow\mathbf s/\sqrt{s_{\max}s_{\min}}$, centering the range
on $1$ in log units so the hardest scale-up equals the hardest scale-down and $\mathbf s$ neither blows the
weights up nor collapses them into the floor.

That normalization is pure hygiene, not objective. A global rescale $\mathbf s\to c\mathbf s$ leaves the
output invariant — `find_scale_zero` returns $\Delta\to c\Delta$, so $\mathrm{round}(cWs/(c\Delta))$ is
unchanged, and the in-weight undo divides the $c$ back out. What centering earns is that the clamp floors
the candidates pass through (`1e-4` on $\mathbf s$, `1e-5` on the normalized vector, `1e-12` on the
per-group scale) are *not* scale-invariant, so an un-normalized $\mathbf s$ drifting large or small with
$\alpha$ would let one bite asymmetrically and corrupt a candidate; centering keeps every $\alpha$ clear of
the floors so the twenty candidates are compared on equal footing.

The harness hands me one linear layer at a time — no cross-layer context, no adjacent operator to fold
$\mathrm{diag}(\mathbf s)^{-1}$ into. The general form of activation-aware scaling absorbs that inverse
scale into the *previous* operator — a LayerNorm or the preceding linear — keeping the network function
exactly unchanged. I cannot; I own only this one `quantize()`. So I realize the equivalence *within the
layer*: scale $\mathbf W$ along input channels, quantize, dequantize, then divide the dequantized weight
back by $\mathbf s$ (`W_dq / s`). The scaling is undone in the stored weight itself rather than migrated to
a neighbor, so the layer's effective output is the dequantized-and-unscaled $\mathbf W$ — self-contained,
same shape, same dtype. That means the loss I score each $\alpha$ against is at *linear-layer* granularity:
in `add_batch` I accumulate the per-channel sum of $|X|$ (averaged to $\mathbf x_{\max}$ at quantize-time)
and reservoir-sample real input tokens, and I score $\alpha$ by
$\lVert X(\mathbf W-\mathbf W_{\text{final}})^\top\rVert^2$ over those samples — the true per-layer output
MSE on real activations. (A no-calibration guard falls back to the per-weight squared error weighted by
$\mathbf x_{\max}^2$, the diagonal uncorrelated-input approximation of that same MSE, but the harness
always streams 128 sequences so the real-sample path runs.) The reservoir keeps `N_SAMPLE_TOKEN=256`
tokens, stride-sampled from up to $4\times$ that many candidates so they span the calibration set rather
than clustering in the first sequences, held on CPU so the $\approx4$ MB survives across all 224 layers
without competing for GPU memory with the resident block. The grid stays symmetric here ($q_{\min}=-2^{b-1}$,
$q_{\max}=2^{b-1}-1$, zero point $0$), the same convention as RTN and GPTQ, so the whole comparison stays on
one grid.

There is a second knob the bulk-damage trade-off motivates directly: after the per-channel scale is fixed,
the group step is still set by the post-scale group *max*, and that max can be a lone outlier dragging the
whole group's $\Delta$ up. So I add a per-group clip search on the scaled weights — shrink each group's max
by $1-i/N$ for $i$ up to `CLIP_MAX_SHRINK`$\cdot$`N_CLIP_GRID` ($0.5\times20=10$ steps, so the max can be
pulled in by at most half), clip the weights to $\pm$max, quantize, and measure the resulting per-group
output error on the same sampled $X$. The measurement is where the clip earns the right to be *per group*.
Reshape $X$ to $(T,G,g)$ with $G=\text{in}/g$ groups of $g$ columns and the scaled weight to
$(\text{out},G,g)$; group $g$'s contribution to output row $r$ at token $t$ is
$\mathrm{org\_out}[r,t,g]=\sum_c W_{r,g,c}X_{t,g,c}$, an `einsum('rgc,tgc->rtg')`. A clip on group $g$'s max
changes *only* that group's quantized weights, so the induced error localizes to
$\mathrm{err}[r,g]=\mathrm{mean}_t(\mathrm{cur\_out}-\mathrm{org\_out})^2$ and I pick per $(r,g)$ the clip
minimizing it — an independent decision per output-channel-and-group, the shape of `best_max`,
$(\text{out},G,1)$. Clipping trades a little extra rounding error on the outlier for a tighter $\Delta$ on
the bulk — the same outlier-versus-bulk balance, now at group granularity. It is batched over output
channels (`OC_BATCH`) because materializing all $\text{out}=14336$ rows of an MLP layer at once against 256
tokens is too much memory.

Two ordering questions rather than assumptions. Why scale first, then clip? Because the clip refines the
group $\Delta$ *given* the weight distribution, and the scale reshapes that distribution; clip first and the
later scaling re-expands the range and undoes the clip's tightening. And why two sequential searches instead
of one joint sweep over $(\alpha,\text{clip})$? A joint grid is $20\times10=200$ full quantize-and-score
passes against $\approx180$ for the two coordinate searches, and the knobs are nearly separable — the scale
decides *which channels* get resolution, the clip decides *how tight* each group's step is — so coordinate
descent captures almost all of the joint optimum at a fraction of the cost. The full module is in the
answer.

Against GPTQ's numbers: the whole bet is that protecting salient channels before rounding recovers
resolution at 8 levels better than compensating after, so INT3-g128 is the setting I am watching. GPTQ left
it at $6.1011$ (degradation $1.1940$), still the wound, and if salience-aware scaling plus clipping does
what the mechanism predicts, INT3 should drop clearly below GPTQ's $6.1011$, into the high-$5$s, because the
salient channels now get finer effective grids while the clip keeps the bulk's $\Delta$ in check. At
INT4-g128 the grid is already fat (GPTQ $5.0711$), so I expect a real but smaller improvement into the
low-$5.0$s. At INT4-g64 GPTQ was strongest ($5.0435$); fine grouping has already protected the bulk the way
clipping would, so this is the setting where the two mechanisms most overlap, and it is entirely plausible I
do *not* beat GPTQ here — which would be consistent with the diagnosis, not against it. On cost I expect to
win, but not simply for "no inverse": counting flops, the scale search is $20$ output-MSE passes of
$\approx256\cdot4096\cdot4096\approx8.6\times10^{10}$ for a square projection and the clip search adds a
comparable pile, in the same ballpark as GPTQ's $4096^3\approx6.9\times10^{10}$ inverse plus its sweep — not
obviously fewer FLOPs. The win is structural: every one of my passes is a dense GEMM or einsum saturating
the tensor cores and my Python loops run $\approx180$ times, where GPTQ's column sweep is a bandwidth-bound
rank-one update turning $\approx32$ blocks into thousands of small sequential steps gated by a serial
Cholesky. Same order of arithmetic, far fewer sequential host round-trips and no serial inverse, so I expect
quant-time well under GPTQ's $\sim220$s. The decisive test is INT3: if this beats GPTQ there,
protecting-before-rounding is the right lever at extreme low bit-width; if not, the bottleneck at 8 levels
is the uniform grid itself, and the next move would have to leave uniform quantization behind entirely.
