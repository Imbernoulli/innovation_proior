I am starting the ladder, so there is no prior result to diagnose — only the bare contract. The harness
hands me a window `x_enc` of shape `[batch, 100, C]`, trains me with MSE to reproduce normal windows,
and reads the anomaly score off the per-point squared error between my reconstruction and the input.
The default fill is identity, which is degenerate: if I return the window unchanged the reconstruction
error is zero everywhere, the score is flat, and the threshold has nothing to bite on. So the first
real question is which reconstruction backbone to start from, and I want to start from the one whose
*input representation* is least obviously wrong for this data, because everything downstream inherits
that choice.

Before I fixate on the Transformer family, let me actually walk the shelf, because three other
backbone classes are sitting right there and I owe each a concrete reason to pass. A dense autoencoder
flattens the whole `[100, C]` window into one long vector and squeezes it through a bottleneck; the
trouble is that flattening throws away the temporal ordering entirely — step 3 and step 97 become just
two coordinates of a vector with no notion that time flows between them — so the model has to re-learn
adjacency from scratch through a dense weight matrix, and worse, a tight bottleneck is a blunt,
unprincipled knob: too wide and it copies the window verbatim (anomalies included), too narrow and it
blurs the normal structure too. An LSTM autoencoder fixes the ordering by rolling an encoder-decoder
along time, but it pays for it with a sequential signal path of length `L = 100`: the gradient tying
step 100's reconstruction back to step 1 has to survive a hundred recurrent hops, and the well-known
forgetting means the long-range structure I most need to reconstruct — the daily cycle — arrives
attenuated at the far end of the window. The frequency- and decomposition-aware models estimate period
lengths from the spectrum or autocorrelation and aggregate same-phase points; that is genuinely
appealing for periodic telemetry, but each bakes a *particular* period-extraction machinery into the
architecture, and for a first rung I do not want to commit the backbone to a periodicity assumption
before I have even established that a generic sequence reconstructor works — if I start with the most
structured model and it underperforms I cannot tell whether the structure or the fit was at fault. So
the disciplined first move is the least-committed backbone that still respects temporal locality, which
points me at the Transformer family — where the one decision that dominates everything is what I call a
token.

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

I should not treat `P = 16, S = 8` as handed to me, so let me price the neighbors. If I shrink to `P =
8, S = 4` the count jumps to `floor((100 − 8)/4) + 2 = 23 + 2 = 25` patches: each token now spans only
eight steps, barely more than the point-wise case I am fleeing, and the attention map grows to `25 × 25
= 625` — over four times the `12 × 12 = 144` of my choice — for tokens too short to carry a
recognizable shape. If instead I grow to `P = 32, S = 16` the count collapses to `floor((100 − 32)/16)
+ 2 = 4 + 2 = 6` patches, each spanning nearly a third of the window; now attention has only six objects
to relate and each token smears a third of the window into one vector, so an anomaly localized to a few
steps is diluted across a token that mostly reconstructs fine, and the score loses its spatial
resolution. Sixteen steps at stride eight sits in the middle: long enough that a ramp or a bump is a
whole token, short enough that a local glitch dominates its own token rather than being averaged away,
with a cheap `12 × 12` map. I can sanity-check the extreme: push `P` all the way to `L = 100` and the
count is `floor(0/S) + 2 = 2` — one real patch plus the padding patch — so attention has essentially
nothing to relate and the head degenerates into a single global linear map of the whole window, exactly
the un-patched flatten I was trying to improve on. Patching only buys anything while `P` stays well
below `L`, which confirms the regime `P = 16 ≪ 100`.

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

It is worth pricing this backbone before I build it, because the weight-sharing is what keeps it
affordable. Each encoder layer carries the four attention projections `Q, K, V, O`, each `D · D = 512 ·
512 ≈ 0.26M`, so `≈ 1.05M` for attention, plus the position-wise feed-forward `D → d_ff → D = 2 · 512 ·
512 ≈ 0.52M`, about `1.57M` per layer and `≈ 3.1M` over the two layers. The patch projection is a slim
`P → D = 16 · 512 ≈ 8K`, and the reconstruction head is one linear of width `head_nf = D · N = 512 · 12
= 6144` into `seq_len = 100`, i.e. `≈ 0.61M`. The whole module is `≈ 3.7M` parameters — and, crucially,
that count is *independent of `C`*: because the channels ride in the batch axis under shared weights,
MSL's 55 channels and PSM's 25 use the identical parameter tensor. Had I mixed channels at the input
instead, the very first projection would have been `C · L → D`, growing with every dataset and forcing
a `C`-specific input map; channel-independence buys me one backbone for all three datasets at a fixed
budget.

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
back). The dilution is quantitative, not a vibe: BatchNorm computes each feature's statistics over the
whole batch of patch positions, which for MSL is `1760` folded sequences `× 12` patches `= 21120`
samples, so one anomalous patch shifts the mean by `≈ 1/21120` — negligible. LayerNorm would instead
normalize a single token over its own `D = 512` features, so a spike landing in that patch corrupts
`100%` of the statistics used to normalize it, dragging the whole token's representation. For
reconstruction-for-anomaly, where the entire point is to keep the abnormal token's error intact rather
than normalize it away, BatchNorm is the safe choice.

Now the head — and here the anomaly-detection setting is cleaner than forecasting. I am not predicting a
future horizon; I am reconstructing the window I was given. So the target window for the head is
`seq_len` itself: flatten the per-series `[D, N]` representation into a `D·N` vector and linear-project
it to `seq_len`, then permute back so the output is `[B, seq_len, C]`. The head input width is `head_nf
= D · N = D · (floor((L − P)/S) + 2)`. There is no decoder, no learned temporal extension — the window
comes in and the reconstruction of that same window comes out.

I considered a more locality-respecting head — give each patch its own small linear `D → P` back to its
own `P` steps and overlap-add the stride-8 overlaps — but it has two costs the flatten head avoids. It
must average the doubly-covered overlap regions (stride below patch means each interior step is written
by two patches), and, more importantly, each patch would be reconstructed only from its own encoded
vector, blind to the rest of the window; whereas the flatten head sees all `N · D = 6144` features at
once, so a step's reconstruction can draw on the whole window's encoded context. On a window this short
`head_nf = 6144` is cheap, so the flatten head's global view wins over the local head's tidiness.

Let me trace the shapes end to end on the largest case, MSL with `C = 55` and `B = 32`, to be sure
nothing is off by a transpose. The window enters as `[32, 100, 55]`; instance-norm leaves it `[32, 100,
55]`; permute to `[32, 55, 100]`; the patch embedding folds channels into the batch and cuts patches to
`[32 · 55, 12, 512] = [1760, 12, 512]`, so the encoder sees 1760 independent 12-token sequences and its
attention maps are `[1760, 8, 12, 12]`. Out comes `[1760, 12, 512]`, reshaped to `[32, 55, 12, 512]`
and permuted to `[32, 55, 512, 12]`; the head flattens the trailing `512 · 12 = 6144` and projects to
`[32, 55, 100]`; permute back to `[32, 100, 55]` — identical to the input, which is the contract for a
reconstruction, and the per-point squared error the harness scores is then well-defined at every one of
the 100 positions. The tail-padding earns its keep exactly here: without the extra patch `N` would be
`floor(84/8) + 1 = 11`, whose patches start at `0, 8, …, 80` and whose last one covers steps `80`
through `95`, leaving steps `96–99` in no patch at all and their reconstruction undefined; padding eight
copies of the last value creates the twelfth patch starting at step `88`, which covers `96–99` (with
harmless repeats past the true end that the head learns to map back), so every position gets a
reconstruction.

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
from the structure the model has learned to reconstruct. I can bound the worst case. A pure level-shift
anomaly of magnitude `δ` added to `m` of the 100 steps moves the window mean by `δ · m/100`, so
per-window centering subtracts that much and leaves `δ · (1 − m/100)` of the shift in the residual. For
a short anomaly — say `m = 5` steps — only 5% is absorbed and 95% survives to spike the reconstruction
error; centering would only erase a level shift covering most of the window, which is not the anomaly
regime I face. So the normalization is nearly free of the failure mode it superficially risks.

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
reconstruction error, so I expect it to be the weakest of the three. To make that falsifiable in the
task's own metrics: if the diagnosis is right, the failure will show as a recall shortfall rather than a
precision one — an over-flexible reconstructor that partly fits anomalies lets their error slip under
the `anomaly_ratio` threshold, so I would see precision hold up (what it flags is right) while recall
sags on MSL and SMAP, with PSM's F1 highest and SMAP's lowest of the three. If instead precision were
the one to collapse, the story would be the opposite — too *rigid* a reconstructor smearing normal
structure into false spikes — and I would rethink the capacity direction entirely. The open worry I
will carry to the next step is exactly this: a patch Transformer is a powerful but *generic* sequence reconstructor; it
does not model the period structure of each window explicitly, and if the next backbone's failure is on
the smooth, strongly periodic streams, the diagnosis is already written — I would want a reconstructor
whose representation is *itself* organized around the periods present in the window.
