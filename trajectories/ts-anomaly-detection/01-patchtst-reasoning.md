I am starting the ladder, so there is no prior result to diagnose — only the bare contract. The harness
hands me a window `x_enc` of shape `[batch, 100, C]`, trains me with MSE to reproduce normal windows,
and reads the anomaly score off the per-point squared error between my reconstruction and the input.
The default fill is identity, which is degenerate: if I return the window unchanged the reconstruction
error is zero everywhere, the score is flat, and the threshold has nothing to bite on. So the first
real question is which reconstruction backbone to start from, and I want to start from the one whose
*input representation* is least obviously wrong for this data, because everything downstream inherits
that choice.

Let me think about what a token even is in the reconstruction backbones on the shelf, because I suspect
the token is where these models go wrong. The point-wise Transformer reconstructors treat a single time
step as a token — either one scalar at time `t`, or the full `C`-channel vector at `t`. Sit with that.
In language a token is a word with standalone meaning; comparing two words by attention compares two
semantic units. What is the meaning of a sensor reading at one instant? Nothing in isolation. The
information in a server-metric or telemetry stream lives in *shapes over short stretches*: a rising
edge, a dip, a local oscillation, the slope of a ramp. Point-wise attention asks "how does the scalar
at this step relate to the scalar at that step?" and the answer is mostly noise, because neither scalar
means anything alone. And for reconstruction-for-anomaly there is a sharper failure on top: a window is
almost entirely normal points, so the pairwise attention is dominated by them and the rare abnormal
pattern — exactly the thing the score must keep crisp — gets averaged into the normal mass. That is why,
on this very task, the plain attention encoder is the worst reconstruction backbone I could pick. The
attention map is being computed over the wrong objects.

There is a second symptom pointing the same way. When the token is one step, the token count `N` equals
the window length `L`, the attention map is `L×L`, and lengthening the window is punished quadratically.
Yet the structure I need to reconstruct — a daily cycle on the server load, the slow rhythms in
telemetry — lives across the whole window, so I want the model to read the window at a scale where
neighboring steps are not treated as independent. Neighboring steps in these streams are redundant:
they carry overlapping, compressible structure rather than independent information. Both observations —
a single step is meaningless, and neighboring steps are redundant — push toward the same move: do not
tokenize per step. Group a local stretch of steps into one token.

This is exactly what vision did at the same wall. An image has `H×W` pixels; one pixel is meaningless
and per-pixel attention is hopeless, so the Vision Transformer cut the image into patches and called
each patch a token, collapsing the token count and giving each token a local visual concept. The
analogy to a series is immediate: cut the length-`L` window into contiguous sub-series patches, and let
each patch be a token. A patch of, say, sixteen consecutive steps is a little shape — a ramp, a bump, a
level — which is the kind of object attention can actually compare. Concretely: pick a patch length `P`
and a stride `S` (the hop between consecutive patch starts), slide a width-`P` window along the series
in steps of `S`, and each placement is one patch in `R^P`. The number of patches is `N = floor((L −
P)/S) + 1` by the usual count of how many length-`P` windows of stride `S` fit in length `L`.

I have to be careful at the boundary. If `(L − P)` is not a multiple of `S`, the last window does not
reach the final step, and for reconstruction I cannot afford to silently drop the tail of the window —
those points need a reconstruction too, or the score is undefined there. So before patching I pad the
end by repeating the last value `S` times. Padding by exactly the stride guarantees one more full
window slides into existence that covers the true end, adding exactly one patch, so the count becomes
`N = floor((L − P)/S) + 2`. With `L = 100`, `P = 16`, `S = 8`: `floor((100 − 16)/8) + 2 = floor(84/8) +
2 = 10 + 2 = 12` patches. Clean. And the payoff is quantitative: the attention map is `N×N` at cost
quadratic in `N`; without patching `N = L = 100`, with patching `N ≈ L/S ≈ 12`, so the token count
drops by a factor of `S` and the attention cost by about `S²`. With `S = 8` that is roughly a 64×
reduction — but more importantly the tokens are now local shapes, which is the representational point.
I want `S` a bit smaller than `P` so consecutive patches overlap and no edge shape gets split cleanly
down the middle; `S = 8` with `P = 16` overlaps each patch with its neighbor by half, keeping local
continuity.

Now the second decision, independent of patching: how do I handle the `C` channels? The standard
multivariate Transformer mixes them — at each step it projects the whole `C`-vector into one token, so
cross-channel information is fused at the input and every channel ends up under one shared attention
pattern. I want to do the opposite, and I can reason out why. First, adaptability: if all channels live
under one attention map, that single pattern is a compromise across channels that can have completely
different temporal behavior — one a slow trend, another a sharp cycle, another near-noise. If instead
each channel passes through the backbone separately, each produces its own attention map, free to attend
to whatever lag structure suits it. Second, data efficiency: learning genuine cross-channel correlation
jointly with temporal structure is a much larger hypothesis space, and these datasets are not huge; a
channel-independent model only has to learn structure along time, a far smaller space, so it fits with
less data. Third, overfitting: a channel-mixing model can fit spurious cross-channel coincidences that
hold in the (normal) training window and do not generalize, which is poison for a reconstruction
backbone whose whole job is to model *only* normal structure cleanly. So I process each channel
independently — but not with `C` separate models, which would be `C×` the parameters and would share
nothing across channels. The right design is one backbone with one set of weights, run independently on
each univariate series: channel-independent in the forward pass, weight-shared in the parameters. This
also has a structural gift — the weights do not know how many channels there are, so the same backbone
serves PSM's 25, MSL's 55, and SMAP's 25 without change.

The implementation cost of channel-independence is nothing but a reshape. Start from `x_enc` of shape
`[B, L, C]`, permute to `[B, C, L]`, patch each series to `[B, C, N, P]`, and fold the channel axis into
the batch axis to get `[B·C, N, P]` — from the encoder's point of view simply a batch of `B·C`
independent length-`N` token sequences. Run the encoder, get `[B·C, N, D]`, reshape back to `[B, C, N,
D]` at the end. The `C` copies of the weights are conceptual; the actual tensor is shared and the
channels ride in the batch dimension.

The backbone itself I keep deliberately *vanilla* — the whole thesis is that the input representation,
not the attention kernel, was the problem, so I must not sneak in a fancy kernel. Linear-project each
patch to `D` with no bias (an instance-normalized patch already has its level removed, so a per-patch
additive offset buys nothing), add a learnable positional embedding because patches are otherwise an
unordered set while temporal order is everything, then standard multi-head scaled dot-product attention
over the `N` patch tokens, a position-wise feed-forward `D → d_ff → D`, residual connections around both
sublayers. One choice inside the encoder I will not make on autopilot: the normalization. The default
Transformer uses LayerNorm, which normalizes within each token — risky here, because a spike or glitch
landing in a patch will skew that token's own statistics and drag LayerNorm around. BatchNorm instead
normalizes each feature across the batch of patch positions, so a single outlier patch is diluted by
all the others rather than corrupting its own normalization. For time-series tokens that is the safer
choice, so I use BatchNorm in the encoder (transpose the feature axis into place, BatchNorm1d, transpose
back).

Now the head — and here the anomaly-detection setting is cleaner than forecasting. I am not predicting a
future horizon; I am reconstructing the window I was given. So the target window for the head is
`seq_len` itself: flatten the per-series `[D, N]` representation into a `D·N` vector and linear-project
it to `seq_len`, then permute back so the output is `[B, seq_len, C]`. The head input width is `head_nf
= D · N = D · (floor((L − P)/S) + 2)`. There is no decoder, no learned temporal extension — the window
comes in and the reconstruction of that same window comes out.

One last thing the data forces before this is done: distribution shift. The channels and datasets span
very different magnitudes and the level wanders over time. If I feed raw values the backbone burns
capacity absorbing each window's offset and scale instead of modeling shape. The fix is reversible
per-instance normalization: subtract the window's temporal mean, divide by its standard deviation (the
biased variance plus a small `1e-5` for a flat series), detach both as constants, run the model on the
normalized window, and at the output multiply the std back and add the mean back, broadcast over the
window length. There is a subtlety for anomaly detection — per-window centering can partly absorb an
anomaly that is purely a level shift — but the data is already globally Z-scored per dataset, the
windows are short, and the anomalies that matter here are violations of the *shape* (the cycle breaks,
the local dynamics go wrong), which survive per-window centering precisely because they are deviations
from the structure the model has learned to reconstruct.

That settles the first rung: reversible instance norm, patch the channel-independent windows into local
shape tokens (padded so the tail is covered, `N = 12` patches at `P=16, S=8`), a vanilla BatchNorm
Transformer encoder over the patches, and a flatten-and-project head back to the `seq_len` window. The
full scaffold module is in the answer.

What do I expect, and where am I unsure — this sets the bar the next rung must beat. The patch
tokenization should already give a respectable reconstruction on the near-periodic, smooth streams: PSM
is server metrics with strong daily structure and many normal points, so I expect F1 to land high
there. MSL and SMAP are noisier telemetry with sharper, less regular events; channel-independence helps
because the 55 MSL channels behave very differently, but the flatten head reconstructs the window
through a single fixed linear projection of patch features, which has no explicit handle on *which*
periodicity each window carries — and SMAP in particular has the hardest anomalies to separate by
reconstruction error, so I expect it to be the weakest of the three. The open worry I will carry to the
next step is exactly this: a patch Transformer is a powerful but *generic* sequence reconstructor; it
does not model the period structure of each window explicitly, and if the next backbone's failure is on
the smooth, strongly periodic streams, the diagnosis is already written — I would want a reconstructor
whose representation is *itself* organized around the periods present in the window.
