The algorithm is the whole point, but it has to bolt onto a fixed pipeline, and the simplest fill of
that pipeline is the floor I should start from before I spend any cleverness — so the pain to begin with
is just: take Mistral-7B's weights and put them on an integer grid, once, with nothing but the grid.
Everything downstream — the layer-by-layer driver, the calibration hooks, the perplexity eval — is
frozen, and the only region I own is `LayerQuantizer` plus the three primitives. The cheapest thing that
region can do is round each weight independently to its nearest grid point, and that is exactly what I
want as the baseline: it costs nothing, it depends on no calibration data, and it tells me how far plain
rounding gets at 7B before I add any structure — and where it breaks is where everything I build later
gets its purchase.

Let me write down what rounding actually is here, because the grid is the substrate everything later also
lives on. Symmetric quantization with $b$ bits: the integer range is $q_{\min}=-2^{b-1}$,
$q_{\max}=2^{b-1}-1$, so INT4 gives 16 levels ($-8..7$) and INT3 only 8 ($-4..3$). For a vector
of weights $\mathbf w$ the step is $\Delta=\max(|\mathbf w|)/q_{\max}$ — one scale stretched so the
largest-magnitude weight in the group lands at the top of the grid — and the quantized-dequantized value
is $\widehat w=\Delta\cdot\operatorname{clamp}(\operatorname{round}(w/\Delta),q_{\min},q_{\max})$.
`find_scale_zero` builds this: with `group_size>0` it reshapes each output row into groups of
`group_size` consecutive input columns and computes one $\max|\cdot|/q_{\max}$ per group, then
broadcasts the scale back over the columns of that group; with `group_size=-1` it does one scale per row.
The zero point is zero — symmetric, no offset — the right default for weight distributions roughly
centered at zero. So the only free choice RTN makes is the per-group scale, and it makes it by the
crudest possible rule: stretch to the absolute max.

The affine form is more general — $\widehat w=\Delta\,(q-Z)$ with an integer offset $Z$ that slides the
grid to cover the actual $[\min,\max]$ rather than the symmetric $[-\max|\cdot|,\max|\cdot|]$ — and it
earns its keep for something one-sided like a post-ReLU activation. But these are *weights* of a trained
linear, roughly symmetric about zero, so the tighter box a nonzero $Z$ would buy is marginal and it costs
a stored zero point per group. And the symmetric grid is the cleaner reference against which scaling and
error compensation can be measured later, with no zero-point degree of freedom to confound the comparison.
I take `symmetric=True`, the default the template already encodes.

Why is nearest-rounding onto that grid the floor and not something already better? Because the
per-element residual is irreducible under this rule and it is large at low bit-width. Rounding $w/\Delta$
to the nearest integer leaves a residual roughly uniform on $[-\tfrac12,\tfrac12]$ in grid units, i.e. an
absolute weight error around $0.25\,\Delta$ on average, $0.5\,\Delta$ in the worst case. Model the snap
as additive noise $\widehat w=w+\eta$ with $\eta$ uniform on $[-\Delta/2,\Delta/2]$, whose variance is
$\Delta^2/12$, so the per-weight mean squared error is about $\Delta^2/12$. Now
$\Delta\propto\max|\mathbf w|/q_{\max}$, so dropping from INT4 ($q_{\max}=7$) to INT3 ($q_{\max}=3$) more
than doubles $\Delta$, and the noise power scales as $\Delta^2$, i.e. by $(7/3)^2\approx5.44$. The same
weights carry about five and a half times the rounding-noise power at 8 levels, and a transformer stacks
dozens of these layers so the per-layer perturbation compounds with depth. That is the first concrete
reason to expect INT3 to be a different regime, not just a slightly harder one — though exactly how the
$5.44\times$ per-weight noise translates into end perplexity the single-row estimate cannot tell me,
because the compounding across 32 blocks is not captured by it. The regime change from 16 to 8 levels is
not a tuning detail — it is the grid getting coarse enough that the bulk of the distribution can no
longer be represented faithfully.

Nearest is the per-weight error-minimizer when each weight is treated in isolation — any other grid point
is strictly worse on that weight — but the deeper reason to insist on it, over a cheaper truncation toward
zero, is the *sign* of the error. Truncation always pulls toward zero, so its residuals share a sign and
their mean does not average out across the many weights feeding one output; it accumulates into a
systematic output bias that then propagates through every downstream block. Nearest's errors straddle
zero, so there is no such drift. The same warning applies to tie-breaking: a rule that always rounds
halves upward would inject a small coherent positive mean of the same accumulating kind, so the primitive
has to be nearest with an unbiased tie rule — exactly what `torch.round` supplies before the clamp. This
matters more at 3–4 bits over 32 blocks than at INT8: the residual is large, and a biased rounder turns
RTN from noisy-but-centered into noisy-and-drifting, and the drift is the part that compounds.

And here is what makes RTN genuinely the floor rather than merely crude: the scale is set by the single
largest weight in each group. One outlier weight inflates $\Delta$ for *every* other weight in that group,
so the typical small weights — the overwhelming majority, which carry most of the layer's behavior in
aggregate — get rounded on a grid stretched to accommodate a value they have nothing to do with. RTN
spends its resolution on the extremes and starves the bulk, by construction, with no mechanism to notice
or fix it. Push it to the limit the grouping is meant to prevent: a single scale shared across a whole row
where one channel reaches magnitude $1.0$ and the rest sit near $0.01$. The shared $\Delta\approx0.143$ is
set by the loud channel, and a quiet sub-vector near $0.01$ collapses under $\operatorname{round}(w/\Delta)$
to all zeros — its entire magnitude lost — where a local scale $\Delta\approx0.0014$ would have kept it to
better than $1\%$ error, an order of magnitude or two recovered purely by not letting a distant outlier
dictate the step. This is why grouping exists and why finer groups help: a group of `group_size` columns is
a smaller hostage set, so its max pollutes fewer neighbors. And the local max over 64 columns can be no
larger than over the containing 128, no larger than over the whole row, so the error should order
$\text{group64}\le\text{group128}\le\text{per-channel}$, with the gap largest precisely when the magnitude
varies a lot along the row. This is the one lever RTN genuinely responds to, and it responds in the
direction the arithmetic predicts.

That said, grid resolution is not the real indictment of RTN; the objective is. What I care about is not
the weights — it is the layer's behavior. For a linear layer with weights $\mathbf W$ and calibration
inputs stacked as columns of $\mathbf X$, the quantity that propagates to the next block is
$\mathbf W\mathbf X$, so the honest objective is to keep
$\lVert\mathbf W\mathbf X-\widehat{\mathbf W}\mathbf X\rVert^2$ small. RTN minimizes a *different*
objective — the per-weight error $\lVert\mathbf W-\widehat{\mathbf W}\rVert^2$, and not even that
optimally across a group, just element-wise nearest. The two diverge exactly when some input directions
are excited far more than others by real text: an error on a weight that multiplies a high-variance input
feature costs the output a lot, an equal error on a weight multiplying a near-dead input costs almost
nothing, and RTN treats them identically. Two distinct leaks. One, the rounding errors across a row are
uncorrelated with which input directions $\mathbf X$ actually points along, so I spend equal care on
weights the data rarely touches. Two, the per-weight errors in a row add up *coherently* into the output,
and a scheme that quantized columns in sequence and pushed each column's leftover error into the
not-yet-quantized columns could cancel that accumulation, where RTN just lets it stand. The whole
calibration apparatus the harness hands me — the `add_batch` hook that streams 128 real sequences through
each layer — exists to measure precisely this input structure, and RTN throws it away.

So RTN's `add_batch` does the minimum the interface demands: reshape the input to 2-D and count samples,
keeping an `H` buffer allocated only so the class signature matches what the calibration-using methods will
fill; nothing in `quantize()` reads it, and `free()` releases it. Its non-use of data is deliberate, not an
oversight — weight ranges are static, sitting in the tensor the instant training ends, so $\max|\mathbf w|$
needs no forward pass. RTN is the scheme whose defining feature is zero data and zero overhead: read the
weights, compute per-group scales, round, done. My step-1 edit is therefore the trivial one — leave the
region at its default: `quantize()` clones the weight to float, calls `find_scale_zero` with the eval-set
`num_bits` and `group_size` and `symmetric=True`, rounds, dequantizes, casts back. No calibration consumed,
no weight interacting with another, no output error considered. The full module is in the answer.

Now what this floor must do, because that is the reason to run it. Group size is the one lever RTN responds
to, so I expect the three settings to order cleanly by grid resolution. INT4-g128 is the standard case and
should land within a small perplexity gap of FP16 — 16 levels per group is coarse but survivable when the
model is this redundant. INT4-g64 should be *slightly* better (finer granularity, fewer weights polluted
per outlier), a small improvement since at INT4 the bottleneck is the level count more than the group
extent. INT3-g128 is where I expect RTN to fall apart: the $5.44\times$ per-weight noise, the
outlier-inflation biting harder with only 8 levels to spend, and the errors compounding with depth — the
driver re-runs the calibration inputs through the *quantized* block to feed the next, so an output error
introduced in block $k$ becomes part of the input distribution later blocks must themselves quantize
against. So the $5.44\times$ does not add $5.44\times$ to final perplexity; it feeds a chain in which each
block's degraded output is the next block's substrate, and the end metric can move far more. I can predict
the direction and ordering from the arithmetic but must leave the magnitude to the table: degradation a
fraction of a perplexity point at INT4-g128, slightly smaller at INT4-g64, several times larger at
INT3-g128.

And the two leaks already point at where to go next. The diagnosis is not that rounding is implemented
wrong — it is that RTN optimizes the wrong objective, per-weight error instead of per-output error, with a
scale set by outliers and no use of the calibration data streaming through `add_batch` for free. Bring that
activation structure into the rounding decision, either by compensating each column's residual onto the
still-free columns through the input second moment $\mathbf X\mathbf X^\top$, or by scaling up the weights
on the input channels real text excites hardest so they get finer effective resolution. Whatever the
precise numbers RTN posts, INT3 is the metric I am watching, because that is where the floor is lowest and
where there is the most room to climb.
