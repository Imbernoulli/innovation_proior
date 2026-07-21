I am starting the ladder, so there is no prior result to diagnose — only the bare contract. The harness
hands me a window `x_enc` of shape `[batch, 100, C]`, trains me with MSE to reproduce normal windows,
and reads the anomaly score off the per-point squared error. The default fill is identity, which is
degenerate: return the window unchanged and the reconstruction error is zero everywhere, the score is
flat, and the threshold has nothing to bite on. So the first real question is which reconstruction
backbone to start from, and I want the one whose *input representation* is least obviously wrong for
this data, because everything downstream inherits that choice.

Before fixating on the Transformer family, the other three backbone classes each owe a reason to pass. A
dense autoencoder flattens the `[100, C]` window into one vector and squeezes it through a bottleneck;
flattening throws away temporal ordering — step 3 and step 97 become two coordinates with no notion that
time flows between them — so adjacency has to be re-learned through a dense matrix, and the bottleneck
width is a blunt knob: too wide and it copies the window verbatim (anomalies included), too narrow and it
blurs normal structure. An LSTM autoencoder restores ordering but pays with a sequential path of length
100: the gradient tying step 100 back to step 1 survives a hundred recurrent hops, and forgetting means
the long-range structure I most need — the daily cycle — arrives attenuated at the far end. The
frequency-/decomposition-aware models aggregate same-phase points, appealing for periodic telemetry, but
each bakes a *particular* period-extraction machinery in; for a first rung I do not want to commit to a
periodicity assumption before establishing that a generic sequence reconstructor works — if the most
structured model underperforms I cannot tell whether the structure or the fit was at fault. So the
disciplined first move is the least-committed backbone that still respects temporal locality: the
Transformer family — where the decision that dominates everything is what counts as a token.

The point-wise Transformer reconstructors treat a single time step as a token — one scalar at time `t`,
or the full `C`-vector at `t`. In language a token is a word with standalone meaning; attention between
two words compares two semantic units. What is the meaning of a sensor reading at one instant? Nothing in
isolation. The information in these streams lives in *shapes over short stretches*: a rising edge, a dip,
a local oscillation, the slope of a ramp. Point-wise attention asks how the scalar at one step relates to
the scalar at another, and the answer is mostly noise because neither scalar means anything alone. There
is a sharper failure for reconstruction-for-anomaly: a window is almost entirely normal points, so
pairwise attention is dominated by them and the rare abnormal pattern — exactly what the score must keep
crisp — gets averaged into the normal mass. The attention map is being computed over the wrong objects.

A second symptom points the same way. When the token is one step, the token count `N` equals the window
length `L`, the attention map is `L×L`, and lengthening the window is punished quadratically. Yet the
structure I need — a daily cycle, slow telemetry rhythms — lives across the whole window, and neighboring
steps in these streams are redundant, carrying overlapping compressible structure rather than independent
information. Both observations push the same way: do not tokenize per step; group a local stretch of steps
into one token.

Vision hit this exact wall. An image has `H×W` pixels; one pixel is meaningless and per-pixel attention
hopeless, so the Vision Transformer cut the image into patches and called each a token. The analogy is
immediate: cut the length-`L` window into contiguous sub-series patches, each a little shape — a ramp, a
bump, a level — the kind of object attention can compare. Pick a patch length `P` and stride `S`, slide a
width-`P` window in steps of `S`, each placement one patch in `R^P`, giving `N = floor((L−P)/S)+1`. At
the boundary, if `(L−P)` is not a multiple of `S` the last window does not reach the final step, and for
reconstruction I cannot drop the tail — those points need a reconstruction or the score is undefined
there. So pad the end by repeating the last value `S` times; this slides one more full window into
existence covering the true end, so `N = floor((L−P)/S)+2`. With `L=100, P=16, S=8`: `floor(84/8)+2 =
12` patches. The payoff is quantitative: attention cost is quadratic in `N`, and patching drops `N` from
100 to ~12, roughly a 64× reduction — but the representational point is that the tokens are now local
shapes. I want `S` below `P` so consecutive patches overlap and no edge shape is split cleanly; `S=8,
P=16` overlaps each patch with its neighbor by half.

`P=16, S=8` is not handed to me, so price the neighbors. Shrink to `P=8, S=4` and the count jumps to 25:
each token spans only eight steps, barely more than the point-wise case I am fleeing, and the map grows to
625 — over four times the 144 of my choice — for tokens too short to carry a recognizable shape. Grow to
`P=32, S=16` and the count collapses to 6, each token smearing nearly a third of the window into one
vector, so an anomaly localized to a few steps is diluted across a token that mostly reconstructs fine and
the score loses spatial resolution. Sixteen at stride eight sits in the middle: long enough that a ramp or
bump is a whole token, short enough that a local glitch dominates its own token, with a cheap 12×12 map.

Now the channels, independent of patching. The standard multivariate Transformer mixes them — at each step
it projects the whole `C`-vector into one token, fusing cross-channel information at the input under one
shared attention pattern. I want the opposite. First, adaptability: one attention map across channels is a
compromise, but these channels have completely different temporal behavior — one a slow trend, another a
sharp cycle, another near-noise — and each should be free to attend to whatever lag structure suits it.
Second, data efficiency: learning cross-channel correlation jointly with temporal structure is a far
larger hypothesis space, and these datasets are not huge; a channel-independent model only learns
structure along time. Third, overfitting: a channel-mixing model fits spurious cross-channel coincidences
that hold in the normal training window and do not generalize — poison for a backbone whose whole job is to
model *only* normal structure cleanly. So process each channel independently — but with one set of weights
run on each univariate series, not `C` separate models: channel-independent in the forward pass,
weight-shared in the parameters. This also means the weights do not know how many channels there are, so
the same backbone serves PSM's 25, MSL's 55, and SMAP's 25 unchanged. The cost is nothing but a reshape:
permute `[B,L,C]` to `[B,C,L]`, patch to `[B,C,N,P]`, fold channels into the batch to `[B·C,N,P]` so the
encoder sees a batch of `B·C` independent length-`N` token sequences, run it, reshape back. And the
parameter budget is then independent of `C` — had I mixed channels the first projection would have been
`C·L→D`, growing with every dataset and forcing a `C`-specific input map; channel-independence buys one
backbone for all three at a fixed count of a few million weights.

The encoder itself I keep deliberately *vanilla* — the thesis is that the input representation, not the
attention kernel, was the problem, so I must not sneak in a fancy kernel. Linear-project each patch to `D`
with no bias (an instance-normalized patch has its level removed, so a per-patch offset buys nothing), add
a learnable positional embedding since patches are otherwise an unordered set while temporal order is
everything, then standard multi-head attention over the `N` tokens, a position-wise feed-forward
`D→d_ff→D`, residuals around both. The one choice I will not make on autopilot is normalization. LayerNorm
normalizes within each token, which is risky here: a spike landing in a patch skews that token's own
statistics and drags LayerNorm around, corrupting 100% of the statistics used to normalize the very token
whose error I need to preserve. BatchNorm instead normalizes each feature across the batch of patch
positions — for MSL that is `1760` folded sequences × 12 patches ≈ 21k samples, so one anomalous patch
shifts the mean by ≈1/21k, negligible. For reconstruction-for-anomaly, where the point is to keep the
abnormal token's error intact rather than normalize it away, BatchNorm is the safe choice (transpose the
feature axis into place, BatchNorm1d, transpose back).

The head is cleaner here than in forecasting: I reconstruct the window I was given, not a future horizon,
so the target length is `seq_len` itself. Flatten the per-series `[D,N]` representation into a `D·N` vector
and linear-project to `seq_len`, then permute back to `[B,seq_len,C]`; head input width `head_nf = D·N`.
No decoder, no learned temporal extension. I considered a more locality-respecting head — each patch its
own small linear `D→P` back to its own steps, overlap-added — but it must average the doubly-covered
stride-8 overlaps, and, more importantly, each patch would be reconstructed only from its own encoded
vector, blind to the rest of the window; the flatten head sees all `N·D` features at once so a step's
reconstruction can draw on the whole window's context. On a window this short the flatten head is cheap, so
its global view wins. The tail-padding earns its keep exactly at the head: without the extra patch the last
patch covers only through step 95, leaving 96–99 in no patch and their reconstruction undefined; the
twelfth patch, starting at step 88, covers them, so every position gets a reconstruction.

One last thing the data forces: distribution shift. The channels and datasets span very different
magnitudes and the level wanders. Feeding raw values makes the backbone burn capacity on each window's
offset and scale instead of shape. The fix is reversible per-instance normalization: subtract the window's
temporal mean, divide by its std (biased variance plus `1e-5`), detach both as constants, run the model,
and at the output multiply the std back and add the mean, broadcast over length. There is a subtlety —
per-window centering can partly absorb an anomaly that is purely a level shift — but I can bound it: a
level shift `δ` over `m` of 100 steps moves the mean by `δ·m/100`, so centering leaves `δ·(1−m/100)` in
the residual; for a short anomaly (`m=5`) 95% survives to spike the error. Centering would only erase a
level shift covering most of the window, which is not the regime here — the anomalies that matter are
violations of shape, which survive centering because they are deviations from the structure the model
reconstructs.

That settles the first rung: reversible instance norm, channel-independent patches into local shape tokens
(`N=12` at `P=16, S=8`), a vanilla BatchNorm Transformer encoder, a flatten-and-project head back to the
window. The full module is in the answer.

The patch tokens should reconstruct the smooth, near-periodic streams best, so I expect PSM — strongly
daily, mostly normal — highest and the noisier MSL/SMAP telemetry weaker. If the over-flexibility risk is
real it shows as a recall shortfall, not a precision one: a reconstructor flexible enough to partly fit an
anomaly lets its error slip under the threshold, so what it flags stays right while it misses the rest.
Precision collapsing instead would invert the reading — too rigid a map smearing normal structure into
false spikes — and push capacity the other way. The worry I carry is that a patch Transformer is a powerful
but generic sequence reconstructor with no explicit model of each window's period, so if the shortfall
lands on the smooth periodic streams the answer is a representation organized around the periods a window
carries rather than a bigger backbone.
