The algorithm is the whole point, but it has to bolt onto a fixed pipeline, and the simplest fill of
that pipeline is the floor I should start from before I spend any cleverness — so the pain to begin with
is just: take Mistral-7B's weights and put them on an integer grid, once, with nothing but the grid.
Everything downstream — the layer-by-layer driver, the calibration hooks, the perplexity eval — is
frozen, and the only region I own is `LayerQuantizer` plus the three primitives. The cheapest thing that
region can do is round each weight independently to its nearest grid point, and that is exactly what I
want as the baseline: it costs nothing, it depends on no calibration data, and it tells me how far plain
rounding gets at 7B before I add any structure.

Let me write down what rounding actually is here, because the grid is the substrate every later rung also
lives on. Symmetric quantization with $b$ bits: the integer range is $q_{\min}=-2^{b-1}$,
$q_{\max}=2^{b-1}-1$, so INT4 gives 16 levels ($-8..7$) and INT3 gives 8 levels ($-4..3$). For a vector
of weights $\mathbf w$ the step is $\Delta=\max(|\mathbf w|)/q_{\max}$ — one scale stretched so the
largest-magnitude weight in the group lands at the top of the grid — and the quantized-dequantized value
is $\widehat w=\Delta\cdot\operatorname{clamp}(\operatorname{round}(w/\Delta),q_{\min},q_{\max})$.
`find_scale_zero` builds this: with `group_size>0` it reshapes each output row into groups of
`group_size` consecutive input columns and computes one $\max|\cdot|/q_{\max}$ per group, then
broadcasts the scale back over the columns of that group; with `group_size=-1` it does one scale per row.
The zero point is zero — symmetric, no offset — which is the right default for weight distributions that
are roughly centered at zero. So the only free choice RTN makes is the per-group scale, and it makes it
by the crudest possible rule: stretch to the absolute max.

Why is that the floor and not something better? Because the per-element residual is irreducible under
this rule and it is large at low bit-width. Rounding $w/\Delta$ to the nearest integer leaves a residual
uniform on $[-\tfrac12,\tfrac12]$ in grid units, i.e. an absolute weight error around $0.25\,\Delta$ on
average, $0.5\,\Delta$ in the worst case. At INT4, $q_{\max}=7$, so $\Delta$ is the group's max magnitude
divided by 7 — already a coarse grid; at INT3, $q_{\max}=3$, the grid is more than twice as coarse and
the same fractional residual maps to more than twice the absolute weight error. And here is the part that
makes RTN genuinely the floor rather than merely crude: the scale is set by the single largest weight in
the group. One outlier weight inflates $\Delta$ for *every* other weight in that group, so the typical
small weights — which are the overwhelming majority and which carry most of the layer's behavior in
aggregate — get rounded on a grid that was stretched to accommodate a value they have nothing to do with.
RTN spends its resolution on the extremes and starves the bulk. There is no mechanism in it to notice or
fix this, by construction; it never looks at activations and never looks at what the *output* does.

That last point is the real indictment, and it is worth stating precisely because it is what every later
rung attacks. What I care about is not the weights — it is the layer's behavior. For a linear layer with
weights $\mathbf W$ and calibration inputs stacked as columns of $\mathbf X$, the quantity that
propagates to the next block is $\mathbf W\mathbf X$, so the honest objective is to keep
$\lVert\mathbf W\mathbf X-\widehat{\mathbf W}\mathbf X\rVert^2$ small. RTN minimizes a *different*
objective — the per-weight error $\lVert\mathbf W-\widehat{\mathbf W}\rVert^2$, and not even that
optimally across a group, just element-wise nearest. The two objectives diverge exactly when some input
directions are excited far more than others by real text: an error on a weight that multiplies a
high-variance input feature costs the output a lot, an equal error on a weight multiplying a near-dead
input costs almost nothing, and RTN treats them identically. The whole calibration apparatus the
scaffold hands me — the `add_batch` hook that streams 128 real sequences through each layer — exists to
measure precisely this input structure, and RTN throws it away. So in the scaffold, RTN's `add_batch`
does the minimum the interface demands: it counts samples and keeps an `H` buffer allocated only so the
class signature matches what later Hessian-based methods will fill; nothing in `quantize()` reads it.

Why keep `H` at all if RTN ignores it? Because the interface is shared across the ladder. The driver
constructs `LayerQuantizer(layer, num_bits, group_size)`, calls `add_batch` under a forward hook for
every calibration sequence, then calls `quantize()`. GPTQ will use that hook to accumulate
$\mathbf X\mathbf X^\top$; AWQ will use it to accumulate per-channel activation magnitudes and a reservoir
of raw inputs. RTN keeps the buffer and the sample count so it is a clean, minimal *instance* of the same
contract — the floor that the later fills are deltas against — and `free()` releases it. This keeps the
comparison honest: every rung is the same class with the same entry points, differing only in what
`add_batch` collects and what `quantize()` does with it.

There is one design decision inside RTN that is not forced and that I should make deliberately, because it
sets the convention the whole ladder inherits: symmetric versus asymmetric. `find_scale_zero` supports
both — asymmetric would fit $(\max-\min)/(q_{\max}-q_{\min})$ with a nonzero integer zero point, spending
the grid on the actual $[\min,\max]$ range rather than the symmetric $[-\max|\cdot|,\max|\cdot|]$. For a
distribution skewed off zero, asymmetric uses the levels better. But LLM weight groups are close to
zero-mean and roughly symmetric, asymmetric costs a stored zero point per group, and — more important for
a baseline — the symmetric grid is the cleaner reference point against which scaling and error
compensation can be measured later, since it has no zero-point degree of freedom to confound the
comparison. So I take `symmetric=True`: zero point fixed at zero, scale $=\max|\cdot|/q_{\max}$. That is
the default the template already encodes, and I keep it.

So my step-1 edit is the trivial one: leave the region at its default. `add_batch` reshapes the input to
2-D and increments `nsamples` (the `H` accumulation in the template's default is harmless overhead, but
the literal floor is "collect nothing useful and round"); `quantize()` clones the weight to float, calls
`find_scale_zero` with the eval-set `num_bits` and `group_size` and `symmetric=True`, rounds onto the
grid, dequantizes, and casts back to the layer's dtype. No calibration is consumed, no weight interacts
with another, no output error is considered. The full module is in the answer.

Now reason about what this floor must do, because that is the entire reason to run it. Group size is the
one lever RTN does respond to, and it responds in the obvious direction: a smaller group means $\Delta$
is set by the local max over fewer columns, so a single outlier pollutes fewer neighbors and the typical
weight gets a tighter grid. So I expect the three settings to order cleanly by how much grid resolution
RTN has. INT4 with group 128 is the standard case and should land within a small perplexity gap of FP16 —
16 levels per group is coarse but survivable when the model is this redundant. INT4 with group 64 should
be *better* than group 128 (finer granularity, fewer weights polluted per outlier) — a small improvement,
not a large one, since the bottleneck at INT4 is the level count more than the group extent. INT3 with
group 128 is where I expect RTN to fall apart: halving the grid more than doubles the absolute residual,
the outlier-inflation problem bites harder when there are only 8 levels to spend, and across 32 blocks
the per-layer output errors compound into a large perplexity jump. INT3 is the setting that will most
loudly demand something smarter than nearest-rounding.

And that demand is already pointed at the next rung. The diagnosis is not that rounding is implemented
wrong — it is that RTN optimizes the wrong objective: per-weight error instead of per-output error, with
a scale set by outliers and no use of the calibration data the scaffold is streaming through `add_batch`
for free. The two ways to fix that are the two later baselines. One: stop rounding each column in
isolation — quantize column by column and *compensate* the residual onto the still-free columns using the
input second moment $\mathbf X\mathbf X^\top$, so the layer output is preserved even though individual
weights drift further from their originals (GPTQ). Two: stop letting outlier *input channels* dictate
the grid — measure which input channels are activated hardest, scale the corresponding weights up before
rounding so they get finer effective resolution, and undo the scale afterward (AWQ). Either way the move
is the same in spirit: bring the activation structure that RTN ignored back into the rounding decision.
Whatever the precise numbers RTN posts, INT3 is the metric I am watching, because that is where the floor
is lowest and where the next rung has the most room to climb.
