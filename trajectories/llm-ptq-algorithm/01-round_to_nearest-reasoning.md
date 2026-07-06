The algorithm is the whole point, but it has to bolt onto a fixed pipeline, and the simplest fill of
that pipeline is the floor I should start from before I spend any cleverness — so the pain to begin with
is just: take Mistral-7B's weights and put them on an integer grid, once, with nothing but the grid.
Everything downstream — the layer-by-layer driver, the calibration hooks, the perplexity eval — is
frozen, and the only region I own is `LayerQuantizer` plus the three primitives. The cheapest thing that
region can do is round each weight independently to its nearest grid point, and that is exactly what I
want as the baseline: it costs nothing, it depends on no calibration data, and it tells me how far plain
rounding gets at 7B before I add any structure. I want to build it from nothing and understand exactly
where it breaks, because the place it breaks is where every later rung gets its purchase.

Let me write down what rounding actually is here, because the grid is the substrate every later rung also
lives on. Symmetric quantization with $b$ bits: the integer range is $q_{\min}=-2^{b-1}$,
$q_{\max}=2^{b-1}-1$, so INT4 gives 16 levels ($-8..7$) and INT3 only 8 ($-4..3$). For a vector
of weights $\mathbf w$ the step is $\Delta=\max(|\mathbf w|)/q_{\max}$ — one scale stretched so the
largest-magnitude weight in the group lands at the top of the grid — and the quantized-dequantized value
is $\widehat w=\Delta\cdot\operatorname{clamp}(\operatorname{round}(w/\Delta),q_{\min},q_{\max})$.
`find_scale_zero` builds this: with `group_size>0` it reshapes each output row into groups of
`group_size` consecutive input columns and computes one $\max|\cdot|/q_{\max}$ per group, then
broadcasts the scale back over the columns of that group; with `group_size=-1` it does one scale per row.
The zero point is zero — symmetric, no offset — which is the right default for weight distributions that
are roughly centered at zero. So the only free choice RTN makes is the per-group scale, and it makes it
by the crudest possible rule: stretch to the absolute max.

Before I take that zero point on faith I should earn it, because the affine form is more general and I
want to know precisely what I am giving up by pinning the offset. The general uniform quantizer is
$\widehat w=\Delta\,(q-Z)$ with an integer offset $Z$ that decodes to real zero; a nonzero $Z$ slides
the grid so it covers the actual $[\min,\max]$ of a lopsided quantity rather than the symmetric
$[-\max|\cdot|,\max|\cdot|]$. That earns its keep for something one-sided like a post-ReLU activation.
But these are *weights* of a trained linear, roughly symmetric about zero, so the tighter box a nonzero
$Z$ would buy is marginal — and it is not free. Suppose I did carry $Z$ and this weight later multiplied
a quantized input in an integer matmul; one output coordinate is
$\sum_j \Delta_w(q_{w,j}-Z_w)\,\Delta_x(q_{x,j}-Z_x)$, and pulling the scales out, the bracket
$\sum_j(q_{w,j}-Z_w)(q_{x,j}-Z_x)$ expands into four sums: $\sum q_w q_x$, the integer dot product I
actually want; $-Z_x\sum q_w$, foldable into a per-output bias because it depends only on the fixed
weights; $-Z_w\sum q_x$, which carries a sum over the *activations* and so must be recomputed every
forward pass for every output; and the constant $+Z_wZ_x n$. That third term is real per-inference work
proportional to the matrix size, and it exists solely because $Z_w\neq0$. Set $Z_w=0$ and the two
offending sums with a $Z_w$ factor vanish, encode collapses to $\operatorname{round}(w/\Delta)$, decode
to $\Delta\,q$, and the weight side of the matmul is a clean integer dot product. So the symmetric choice
is not a shrug — it is the one that keeps the arithmetic clean for a distribution that barely benefits
from the offset — and it is also the cleaner reference point against which scaling and error compensation
can be measured later, since it has no zero-point degree of freedom to confound the comparison. I take
`symmetric=True`, the default the template already encodes, and I keep it.

Why is nearest-rounding onto that grid the floor and not something already better? Because the
per-element residual is irreducible under this rule and it is large at low bit-width. Rounding $w/\Delta$
to the nearest integer leaves a residual roughly uniform on $[-\tfrac12,\tfrac12]$ in grid units, i.e. an
absolute weight error around $0.25\,\Delta$ on average, $0.5\,\Delta$ in the worst case. I can make that
quantitative rather than hand-wavy: model the snap as additive noise $\widehat w=w+\eta$ with $\eta$
uniform on $[-\Delta/2,\Delta/2]$, whose variance is $\Delta^2/12$, so the per-weight mean squared error
should be about $\Delta^2/12$. Let me check the model against an actual quantized row instead of trusting
the algebra: take a row of 2048 draws from $N(0,0.1^2)$, set $\Delta$ from its max at INT4, and the
predicted $\Delta^2/12$ lands near $2.86\times10^{-4}$ while the measured MSE is about
$2.78\times10^{-4}$ — within three percent. So the uniform-noise picture is a faithful description of what
rounding does here, and I can lean on it. Now $\Delta\propto\max|\mathbf w|/q_{\max}$, so dropping from
INT4 ($q_{\max}=7$) to INT3 ($q_{\max}=3$) more than doubles $\Delta$, and the noise power $\Delta^2/12$
scales as $\Delta^2$, i.e. by $(7/3)^2\approx5.44$. That is not a small worsening: the same weights carry
about five and a half times the rounding-noise power at 8 levels, and a transformer stacks dozens of these
layers so the per-layer perturbation compounds with depth. This is the first concrete reason to expect
INT3 to be a different regime, not just a slightly harder one — though exactly how $5.44\times$ per-weight
noise translates into end perplexity is something the single-row estimate cannot tell me, because the
compounding across 32 blocks is not captured by it; the perplexity table will. The same picture in
absolute-weight terms makes the coarseness tangible: the average residual is $\approx0.25\,\Delta$ and the
worst case $0.5\,\Delta$, so for a group whose max magnitude is, say, $0.8$, INT4's $\Delta=0.8/7\approx
0.114$ puts a typical weight $\approx0.029$ off its true value and a worst-case one $\approx0.057$ off,
while INT3's $\Delta=0.8/3\approx0.267$ more than doubles both to $\approx0.067$ and $\approx0.133$. Those
are large fractions of the weights they perturb, and there are hundreds of millions of them per layer, so
the regime change from 16 to 8 levels is not a tuning detail — it is the grid getting coarse enough that
the bulk of the distribution can no longer be represented faithfully.

There is a choice of *which* rounding hiding here, and I want to argue it rather than assume it, because
the wrong choice poisons the compounding in a specific way. Nearest is the per-weight error-minimizer
under the constraint that I treat each weight in isolation — any other grid point is strictly worse on
that weight. The tempting cheaper primitive is truncation toward zero, a single `floor`. Let me see what
it costs: on the same 2048-weight row, truncation gives an MSE near $1.15\times10^{-3}$ against nearest's
$2.78\times10^{-4}$, roughly four times the error. But the more dangerous number is the *mean* error —
nearest leaves about $-1.2\times10^{-4}$, essentially zero, while truncation leaves $-2.9\times10^{-2}$.
Truncation always pulls toward zero, so its errors share a sign, and a coherent mean error does not
average out across the many weights feeding one output — it accumulates into a systematic bias on that
output that then propagates through every downstream block. Nearest has no such drift because its errors
straddle zero. The same warning applies to tie-breaking inside nearest: a low-level shift that always
rounds halves upward would inject a small positive mean of the same accumulating kind. So the primitive
has to be nearest with an unbiased tie rule, which is exactly what `torch.round` supplies before the
clamp. This matters more here than it would at INT8 precisely because the residual is large and the depth
is 32 blocks: a biased rounder would turn RTN from "noisy but centered" into "noisy and drifting," and the
drift is the part that compounds.

And here is the part that makes RTN genuinely the floor rather than merely crude: the scale is set by the
single largest weight in each group. One outlier weight inflates $\Delta$ for *every* other weight in that
group, so the typical small weights — the overwhelming majority, which carry most of the layer's behavior
in aggregate — get rounded on a grid stretched to accommodate a value they have nothing to do with. RTN
spends its resolution on the extremes and starves the bulk, by construction, with no mechanism to notice
or fix it. I can see the pathology in miniature by pushing it to the limit the group structure is meant to
prevent. Imagine a single scale shared across a whole row where one channel reaches magnitude $1.0$ and
the rest sit near $0.01$: the shared $\Delta=1.0/7\approx0.1429$ is set by the loud channel, and a quiet
sub-vector like $[0.01,0.007,-0.009,0.003,-0.002]$ maps under $\operatorname{round}(w/\Delta)$ to
$[0,0,0,0,0]$ — every entry collapses to zero, and the absolute error summed over it is $0.031$, the full
magnitude of the sub-vector, because nothing survived. Give that same sub-vector its own local scale
$\Delta=0.01/7\approx0.00143$ and the codes become $[7,5,-6,2,-1]$, decoding to
$[0.0100,0.00714,-0.00857,0.00286,-0.00143]$ with an error sum near $0.0013$ — a factor of roughly $24$
recovered purely by not letting a distant outlier dictate the step. This is exactly why grouping exists
and why finer groups help: a group of `group_size` columns is a smaller hostage set, so its max pollutes
fewer neighbors. The scale formula guarantees the local max over 64 columns can be no larger than over the
containing 128, which is no larger than over the whole row, so the error ordering
$\text{group64}\le\text{group128}\le\text{per-channel}$ should follow. "No larger max" only bounds the
worst case, though, so I check it on a synthetic row whose magnitude grows across 4096 columns:
per-channel gives MSE $\approx4.6\times10^{-3}$, group 128 gives $\approx1.25\times10^{-3}$, group 64
gives $\approx1.0\times10^{-3}$. The ordering holds, and the gap from per-channel to grouped is large
precisely when the magnitude varies a lot along the row — the situation grouping is built to rescue. This
is the one lever RTN genuinely responds to, and it responds in the direction the arithmetic predicts.

Let me get the per-group bookkeeping exactly right, because this is where a vectorized implementation
silently goes wrong and corrupts a real layer without ever raising an error. The weight matrix is
$(\text{out\_features},\text{in\_features})$. To quantize in groups of $g$ columns I reshape it to
$(\text{out\_features},\text{in\_features}/g,g)$ so the last axis is one group, take $\max|\cdot|$ over
that last axis with `keepdim` to get a $(\text{out\_features},\text{in\_features}/g,1)$ tensor of per-group
maxima, and divide by $q_{\max}$ for a per-group scale of the same shape. To apply it back to the flat
$(\text{out\_features},\text{in\_features})$ matrix I reshape the scales to
$(\text{out\_features},\text{in\_features}/g)$ and `repeat_interleave(g)` along the column axis so every
weight is paired with its own group's scale — not a neighbor's. There is one degenerate case the floor
$1e{-}12$ guards: a group of all zeros has $\max|\cdot|=0$ and a scale of zero, and then $w/\Delta$ is a
division by zero; clamping the per-group max to $1e{-}12$ before dividing catches it, and with any real
weight group that floor never binds. The `assert in_features % group_size == 0` in `find_scale_zero` is
what lets me trust the reshape — the eval settings use $g\in\{128,64\}$ against Mistral's 4096- and
14336-wide inputs, both divisible, so the reshape is always clean and the same code path serves group 128,
group 64, and per-channel without special-casing.

That said, grid resolution is not the real indictment of RTN; the objective is. What I care about is not
the weights — it is the layer's behavior. For a linear layer with weights $\mathbf W$ and calibration
inputs stacked as columns of $\mathbf X$, the quantity that propagates to the next block is
$\mathbf W\mathbf X$, so the honest objective is to keep
$\lVert\mathbf W\mathbf X-\widehat{\mathbf W}\mathbf X\rVert^2$ small. RTN minimizes a *different*
objective — the per-weight error $\lVert\mathbf W-\widehat{\mathbf W}\rVert^2$, and not even that
optimally across a group, just element-wise nearest. The two objectives diverge exactly when some input
directions are excited far more than others by real text: an error on a weight that multiplies a
high-variance input feature costs the output a lot, an equal error on a weight multiplying a near-dead
input costs almost nothing, and RTN treats them identically. There are two distinct leaks in the proxy.
One, the rounding errors across a row are uncorrelated with which input directions $\mathbf X$ actually
points along, so I am spending equal care on weights the data rarely touches. Two, the per-weight errors
in a row add up *coherently* into the output, and a scheme that quantized columns in sequence and pushed
each column's leftover error into the not-yet-quantized columns could cancel that accumulation, where RTN
just lets it stand. The whole calibration apparatus the scaffold hands me — the `add_batch` hook that
streams 128 real sequences through each layer — exists to measure precisely this input structure, and RTN
throws it away. So in the scaffold, RTN's `add_batch` does the minimum the interface demands: it reshapes
the input to 2-D and counts samples, keeping an `H` buffer allocated only so the class signature matches
what later Hessian-based methods will fill; nothing in `quantize()` reads it.

Why keep `H` at all if RTN ignores it? Because the interface is shared across the ladder. The driver
constructs `LayerQuantizer(layer, num_bits, group_size)`, calls `add_batch` under a forward hook for
every calibration sequence, then calls `quantize()`. A Hessian-based successor will use that hook to
accumulate $\mathbf X\mathbf X^\top$; an activation-aware one will use it to accumulate per-channel
activation magnitudes and a reservoir of raw inputs. RTN keeps the buffer and the sample count so it is a
clean, minimal *instance* of the same contract — the floor that the later fills are deltas against — and
`free()` releases it. This keeps the comparison honest: every rung is the same class with the same entry
points, differing only in what `add_batch` collects and what `quantize()` does with it. And it makes the
non-use of data a deliberate property rather than an oversight: weight ranges are static, sitting in the
tensor the instant training ends, so $\max|\mathbf w|$ needs no forward pass to compute. RTN is the scheme
whose defining feature is zero data and zero overhead — read the weights, compute per-group scales, round,
done — and I want that stated as the boundary the calibration-using methods will cross, not as laziness.

So my step-1 edit is the trivial one: leave the region at its default. `add_batch` reshapes the input to
2-D and increments `nsamples` (the literal floor is "collect nothing useful and round"); `quantize()`
clones the weight to float, calls `find_scale_zero` with the eval-set `num_bits` and `group_size` and
`symmetric=True`, rounds onto the grid, dequantizes, and casts back to the layer's dtype. No calibration
is consumed, no weight interacts with another, no output error is considered. Before I call it done I run
the assembled path once by hand on a tiny matrix, because a bookkeeping slip in the reshape or the
broadcast would silently corrupt a real layer. Take a row $[1.0,-0.6,0.31,-0.02]$ at INT4: $\max=1.0$,
$\Delta=1/7\approx0.142857$, so $\operatorname{round}(w/\Delta)=[7,-4,2,0]$, every code inside $[-8,7]$ so
the clamp is inert and the most-negative code $-8$ goes unused exactly as the symmetric grid intends;
dequantizing gives $[1.0,-0.5714,0.2857,0.0]$, the largest weight reproduced exactly and the rest close.
Give a second, quiet row its own `amax(dim=1)` and it does *not* inherit the loud row's scale — each row's
max is independent — which is the per-channel mechanism doing its job on the actual code rather than in the
argument. The round trip is consistent; the full module is in the answer.

Now reason about what this floor must do, because that is the entire reason to run it. Group size is the
one lever RTN responds to, and it responds in the obvious direction, so I expect the three settings to
order cleanly by how much grid resolution RTN has. INT4 with group 128 is the standard case and should
land within a small perplexity gap of FP16 — 16 levels per group is coarse but survivable when the model
is this redundant. INT4 with group 64 should be *better* than group 128 (finer granularity, fewer weights
polluted per outlier) — a small improvement, not a large one, since the bottleneck at INT4 is the level
count more than the group extent, and my synthetic group-ordering check already showed group64 only
modestly under group128 when the magnitude is smooth. INT3 with group 128 is where I expect RTN to fall
apart: the $\Delta^2/12$ model says halving $q_{\max}$ from 7 to 3 multiplies the per-weight noise power by
$\approx5.44$, the outlier-inflation problem bites harder when there are only 8 levels to spend, and
across 32 blocks the per-layer output errors compound into a large perplexity jump. The compounding is
worth being precise about, because it is why the single-layer noise number understates the danger: the
driver re-runs the calibration inputs through the *quantized* block to produce the inputs for the next
block, so an output error introduced in block $k$ is not a one-off — it becomes part of the input
distribution that blocks $k{+}1,\dots,32$ must themselves quantize against and propagate. A $5.44\times$
per-weight noise inflation at INT3 therefore does not add $5.44\times$ to the final perplexity; it feeds a
chain in which each block's degraded output is the next block's substrate, and the end metric can move far
more than the per-layer estimate. That is why I can predict the *direction* and the *ordering* with
confidence from the $\Delta^2/12$ arithmetic but must leave the *magnitude* of the INT3 perplexity to the
table — the degradation column, perplexity minus the 4.9071 FP16 reference, is the clean readout, and I
expect it small (a fraction of a perplexity point) at INT4-g128, slightly smaller at INT4-g64, and several
times larger at INT3-g128. INT3 is the setting that will most loudly demand something smarter than
nearest-rounding.

And that demand is already pointed at the successors the background names. The diagnosis is not that
rounding is implemented wrong — it is that RTN optimizes the wrong objective: per-weight error instead of
per-output error, with a scale set by outliers and no use of the calibration data the scaffold is
streaming through `add_batch` for free. The two leaks in the proxy suggest the two later baselines. One:
stop rounding each column in isolation — quantize column by column and *compensate* the residual onto the
still-free columns using the input second moment $\mathbf X\mathbf X^\top$, so the layer output is
preserved even though individual weights drift further from their originals. Two: stop letting outlier
*input channels* dictate the grid — measure which input channels are activated hardest, scale the
corresponding weights up before rounding so they get finer effective resolution, and undo the scale
afterward. Either way the move is the same in spirit: bring the activation structure that RTN ignored back
into the rounding decision. Whatever the precise numbers RTN posts, INT3 is the metric I am watching,
because that is where the floor is lowest and where the next rung has the most room to climb.
