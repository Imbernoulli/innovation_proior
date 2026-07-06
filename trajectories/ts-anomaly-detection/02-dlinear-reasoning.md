The patch Transformer's numbers tell a sharper story than its mean F1 (0.8135) admits. On PSM it is
fine — 0.9617 F1, precision 0.9888, recall 0.9361 — the server metrics are smooth and strongly periodic
and the patch tokens reconstruct them cleanly. But MSL came in at only 0.7904 (precision 0.8867, recall
0.7130) and SMAP at 0.6883 (precision 0.9040, recall a poor 0.5557). Look at the recall: on the two
telemetry datasets the model is *missing* anomalies — precision stays high (when it flags, it is usually
right) but recall sags, which for a reconstruction detector means the model is reconstructing the
abnormal points almost as well as the normal ones, so their error does not clear the threshold. That is
the signature of an over-flexible reconstructor: a 512-wide, two-layer attention encoder with a
flatten-and-project head over `D·N = 512·12` features is powerful enough to partly fit the anomalies too,
which smooths the very error spikes the score depends on. The patch backbone reconstructs *too well*,
and indiscriminately. So the move I want now is not more capacity — it is *less*, with the capacity
spent only on the structure that genuinely characterizes a normal window: its trend and its periodicity.

Let me put numbers on "less capacity, spent well," by reading the gap between precision and recall on
each dataset, because that gap is the mechanism. On PSM precision and recall sit at 0.9888 and 0.9361, a
gap of 0.053 — essentially balanced, the detector neither missing nor over-flagging. On MSL the gap
opens to `0.8867 − 0.7130 = 0.174`, and on SMAP it yawns to `0.9040 − 0.5557 = 0.348`. High precision
with collapsing recall is a specific signature: when the model does flag a point it is almost always
right, but it is *missing* most anomalies, which for a reconstruction detector means their squared error
never clears the `anomaly_ratio` percentile threshold — the model reconstructed them nearly as
faithfully as the normal points. And I can check that recall, not precision, is the lever worth pulling.
SMAP's F1 is `2 · 0.9040 · 0.5557 / (0.9040 + 0.5557) = 1.0047 / 1.4597 = 0.6883`, which reproduces the
reported number, so the arithmetic is sound. Hold precision fixed and lift recall to 0.70 and it becomes
`2 · 0.9040 · 0.70 / (0.9040 + 0.70) = 1.2656 / 1.604 = 0.789` — a `0.10` jump from recall alone. So the
entire deficit is that an over-flexible reconstructor is fitting the anomalies; every unit of capacity I
remove that still lets normal structure through converts almost directly into recall, and recall is
where the mean F1 of 0.8135 is bleeding.

Let me reason from what actually carries the reconstructable signal in these streams. A normal window is,
to first order, a slow trend-cyclical component plus an oscillation riding on top — the server load
drifts and cycles daily, the telemetry has a slow baseline with periodic structure. Both pieces are, at
heart, "the value at a step is a fixed linear functional of the rest of the window." A trend is well
captured by extrapolating a slow drift, which is a weighted combination of the window's values; a
periodicity is well captured by reading off the same-phase values, which is again a weighted
combination. So the most minimal reconstructor that can represent normal structure is a single linear
map along the time axis: for one channel, reconstruct the window as `X̂ = W X` with `W` a `seq_len ×
seq_len` matrix. Every input step connects to every output step with a learned weight — signal path
length one, no recurrence to forget through, no attention to be distracted by the dominant normal mass.
And crucially: a linear map has *exactly* the capacity to reproduce trend-plus-periodicity and almost no
capacity to fit the idiosyncratic shape of an anomaly. That is the property I want. Where the patch
Transformer reconstructed the abnormal points too faithfully and killed recall, a linear reconstructor
should reproduce the normal periodic structure and leave the anomalies as residual — a clean error
spike. The cure for the over-flexible backbone is a deliberately under-flexible one.

There is more than one way to spend less capacity, so let me weigh the honest alternatives before
committing to a bare linear map. I could keep the patch Transformer and simply shrink it — drop
`d_model` below 512 or cut to one layer — but that is a continuous knob with no principled stopping
point: attention over patches stays a universal-enough approximator that a narrower one keeps some
ability to trace an anomaly's shape, and I would be tuning a dial hoping to land between "still fits
anomalies" and "cannot fit normal structure." I could use a bottleneck autoencoder, forcing compression
through a narrow latent — but the bottleneck width is the same unprincipled dial, and a nonlinear
decoder downstream of even a modest bottleneck can resynthesize a sharp anomaly. What I actually want is
a capacity ceiling that is *structural* rather than dialed: a function class that provably cannot
represent an idiosyncratic anomaly shape no matter how it is trained. An affine map is exactly that. For
one channel it is `X̂ = W X` with `W` a `seq_len × seq_len = 100 × 100` matrix — `10^4` weights plus a
`100`-vector bias, and with a seasonal and a trend map that is about `2 · 10100 ≈ 20200` parameters,
shared across all channels. Against the patch backbone's `≈ 3.7M` that is roughly a `180×` cut, and the
cut is qualitative: an affine map can reproduce trend-plus-periodicity — both linear functionals of the
window — and essentially nothing that is not a fixed linear readout of the window's positions.

I want to see the anomaly-survival concretely, not just assert it. Suppose training has fitted `W_eff`
(the model's single effective matrix) so a normal near-periodic window reconstructs at tiny error. Now
add an isolated spike of height `h` at step `j`. Because the map is fixed and linear, the reconstruction
changes by exactly `W_eff[:, j] · h` — the spike's energy is spread across all output steps according to
column `j` of `W_eff`, which for a smooth periodic reconstructor is small and diffuse near the spike's
own location. So the reconstruction at step `j` barely rises to meet the spike, and the residual `h · (1
− W_eff[j, j])` there stays large — a clean error spike, exactly what the threshold needs. A
high-capacity model could instead have learned a `W` with `W[j, j] ≈ 1` locally and cancelled the spike;
the linear map, tied to one global set of weights that must also reconstruct every *normal* window,
cannot afford to. That is the property I am buying.

But a single linear map fights itself when a window has a strong trend *and* a seasonal oscillation,
which is the common case in these streams. One `W` now has to fit both at once, and they want very
different weight patterns. The trend wants broadly smooth weights that extrapolate a slow drift; the
seasonality wants weights sharply concentrated at the periodic lags. Worse, the trend component is large
in magnitude and the seasonal component is small, so in a squared-error fit the trend dominates the
gradient — `W` spends itself getting the big ramp roughly right and under-fits the small oscillation
that carries the fine structure. And the fine oscillation is exactly where anomalies show: an anomaly is
usually a break in the *shape*, the small structure, not a gross level move. If my one matrix
under-fits the oscillation, it under-fits the very thing whose violation I am trying to detect. So I need
to let the two components specialize.

The classical move is seasonal-trend decomposition — write the window additively as a smooth trend plus
a residual, because each piece on its own is more regular and more predictable than the sum. I do not
need to invent it; the harness already exposes the moving-average decomposition block
(`layers.Autoformer_EncDec.series_decomp`) as a parameter-free preprocessing. Decompose the window once:
`trend = MovingAvg(x)`, `seasonal = x − trend`. Now the trend stream is the smooth baseline and the
seasonal stream is the oscillation around it, cleanly separated and on comparable footing within each
stream. Give each its own linear map — `Linear_Seasonal` and `Linear_Trend`, both `seq_len → seq_len` —
and let each specialize: the trend linear learns to reproduce a slow baseline, the seasonal linear
learns to read off the periodic lags, neither contaminated by the other and neither's gradient swamped
by the other's magnitude. Then sum the two reconstructed streams to recombine.

I should be honest that this looks like I added depth, which would betray the whole point of going
*simpler* than the Transformer. Is this still a linear model? The two linear maps plus a sum are affine,
and a moving average is itself linear, so decomposition followed by two linear maps and a sum is, end to
end, a single affine map from window to reconstruction. I have not added representational capacity in the
function-class sense at all. What I added is *conditioning*: the decomposition is a fixed, parameter-free
reparameterization that separates the loud trend from the quiet seasonality so that gradient descent on
the squared error can fit each well instead of letting the trend's magnitude dominate. It is
preconditioning, not depth — same solution set, far better-behaved learning — and it pays off precisely
when there is a clear trend, which is the case the bare linear map handled worst. Let me actually prove
the no-added-capacity claim to myself, because if it secretly failed I would be betraying the whole
point. The moving average is a fixed linear operator — call its matrix `A`, so `trend = A X` and
`seasonal = X − A X = (I − A) X`. The two linear maps then produce `W_s (I − A) X + W_t A X = [W_s (I −
A) + W_t A] X = W_eff X`, a single `100 × 100` matrix — the exact same function class as the bare linear
reconstructor, identical solution set. What the decomposition changes is not *what* can be represented
but *how the loss is conditioned*: `A` is a fixed, well-chosen basis change handing the loud
low-frequency trend to `W_t` and the quiet high-frequency residual to `W_s` separately. Concretely, if a
window's trend swings over a range roughly `10×` the seasonal oscillation, the squared-error gradient
scales with the reconstruction residual and the trend contributes on the order of `100×` the gradient
energy of the seasonality; one shared matrix then spends almost all its update budget on the ramp and
barely moves on the oscillation — precisely the fine structure whose violation is the anomaly. Splitting
the streams equalizes their footing so each fits. So the model stays as
weak in capacity as I want it (it cannot fit an anomaly's idiosyncratic shape), while reconstructing the
normal trend-plus-seasonality cleanly.

There is one decision the patch backbone got right that I will keep, because it is independent of the
backbone: channels. The TS-Lib DLinear shares the linear weights across channels by default — one `W`
applied identically to every channel — rather than learning a separate map per channel. Within a single
dataset the channels usually share temporal dynamics (the daily rhythm of all server metrics, the
diurnal cycle of all telemetry sensors), so a shared map encodes that prior and slashes the parameter
count, which matters for not overfitting on the smaller datasets and keeps the model from fitting
spurious cross-channel coincidences in the normal training data. So I keep channel-shared weights and
model no cross-variate correlation at all — the same bet the patch backbone's channel-independence made,
expressed here as one shared map.

Trace the shapes to be sure the shared map really does act along time. A PSM batch enters as `[32, 100,
25]`; `series_decomp` returns `seasonal` and `trend`, each `[32, 100, 25]`; I permute both to `[32, 25,
100]` so the last axis is time; `Linear_Seasonal` and `Linear_Trend`, each a single `nn.Linear(100,
100)`, broadcast over the leading `[32, 25]` and map the length-100 time axis to length-100 — the *same*
`100 × 100` weight applied to every one of the `32 · 25 = 800` series, which is what channel-sharing
means concretely. Sum the two `[32, 25, 100]` outputs and permute back to `[32, 100, 25]`, matching the
input. No axis ever mixes channels, so the model carries zero cross-variate parameters — the same bet
the patch backbone made, here reduced to its barest expression, and the `100 × 100` weight is oblivious
to whether it is fed PSM's 25 series or MSL's 55.

Let me pin down the decomposition concretely, because the boundary behavior decides whether it helps. I
want `trend = AvgPool` over a sliding window of odd kernel size `k`, stride 1, length-preserving. A
stride-1 average pool over a length-`L` signal with kernel `k` produces `L − k + 1` outputs, shorter
than the input, so I must pad — and zero-padding would drag the trend toward zero at the two ends,
creating spurious dips exactly where I have least information, which would then show up as spurious
reconstruction error at the window edges (false anomalies at every window boundary). Instead replicate
the endpoints: pad the front with `(k−1)/2` copies of the first value and the back with `(k−1)/2` copies
of the last value, then average-pool, so the trend stays flat-but-faithful at the edges. With `k` odd,
`(k−1)/2` on each side restores the length exactly. The harness's `series_decomp` takes the kernel as
`configs.moving_avg`; I keep the established smoothing scale (kernel 25), which averages out sub-cycle
wiggles while preserving the cycle-and-slower trend, and keeping it identical to the established block
keeps the comparison clean — I am reusing the decomposition, not tuning a new knob. Let me verify the
length bookkeeping so no spurious edge error creeps in. Kernel `k = 25` is odd, so I replicate `(k −
1)/2 = 12` copies at each end, growing the length to `100 + 24 = 124`; a stride-1 average pool with
kernel 25 returns `124 − 25 + 1 = 100` outputs — exactly the input length, so trend and seasonal are
both `[B, 100, C]` and the two linears stay `100 → 100`. Replicate padding rather than zeros is what
avoids the false-anomaly trap: a zero-pad would drag the 12-wide average toward zero at the first and
last steps, manufacturing a dip in the trend right at the window edges and hence a spurious
reconstruction error — a false anomaly at every window boundary — whereas replicating the endpoint keeps
the trend flat-but-faithful there. As for the cutoff, a 25-wide box average attenuates variation faster
than about 25 steps and passes slower structure, so on a length-100 window it cleanly separates a slow
baseline from oscillations of period well under 25, which is the trend/seasonal split I want.

One TS-Lib-specific wrinkle I must get right against the scaffold. For anomaly detection the harness sets
`pred_len = seq_len` (reconstruction, same length in and out), so the two `nn.Linear(seq_len, pred_len)`
maps are `seq_len → seq_len`. The DLinear model routes `anomaly_detection` straight through its
`encoder` method — decompose, permute time to the last axis so the linears act along time (`[B, L, C] →
[B, C, L]`), apply `Linear_Seasonal` and `Linear_Trend`, sum, permute back to `[B, L, C]`. There is one
thing DLinear does *not* do here that the patch backbone did: it applies **no reversible instance
normalization**. The TS-Lib DLinear anomaly path feeds raw (already per-dataset Z-scored) windows
straight into the decomposition, because the moving-average split itself handles the level — the trend
linear absorbs the window's baseline directly. That is a real difference from a normalized-linear
variant and from the patch backbone, and I am taking the scaffold's version: no per-window mean/std,
just decompose-linear-sum. The weights are also initialized to the uniform
`1/seq_len` average (every entry equal) rather than randomly, so each linear starts as an identity-ish
averaging map and learns away from it — a sensible warm start for a reconstructor. I can check that the
uniform init is a *sensible* starting reconstruction rather than an arbitrary one. At init each weight
row is all `1/100`, so `Linear(v)` returns the *mean* of `v` broadcast to every output step. Then
`trend_output = mean(A X)` is the window's level, `seasonal_output = mean((I − A) X) ≈ 0` because the
residual is mean-zero by construction, and their sum is a flat line at the window's average — the
model's very first reconstruction is "predict the mean everywhere," a stable floor from which training
bends the rows toward the actual trend and periodic lags. This also shows why skipping instance
normalization is safe: a constant level `c` added to the window becomes pure trend (`A` of a constant is
that constant, its residual zero), and the uniform-init trend row reproduces `c` exactly, so the level
passes straight through the trend linear without any separate normalization wrapper — the decomposition
*is* the level handling that PatchTST needed a normalize-denormalize wrapper to accomplish.

So the delta from the patch backbone is concrete and in the *opposite* direction of more capacity: where
the patch Transformer reconstructed the window through a 512-wide attention encoder and a `512·12`
flatten head — powerful enough to partly reconstruct anomalies and kill recall — I now reconstruct it
through a parameter-free trend/seasonal split and two shared linear maps along time, summed. The model
can reproduce normal trend-plus-periodicity and almost nothing else, so anomalies should remain as
residual error.

Here is what I expect against the patch backbone's measured numbers, falsifiably. On MSL, where the
patch backbone's recall sagged to 0.7130 and F1 to 0.7904, the under-flexible linear reconstructor
should leave anomalies as cleaner residual and lift F1 — I expect MSL to be the dataset where DLinear
most clearly beats the patch backbone, the difference that should pull its mean F1 above 0.8135. I can
size the arithmetic: the mean is `(PSM + MSL + SMAP)/3 = 0.8135` now; if MSL alone climbs from 0.7904 to
around 0.82 while PSM and SMAP hold, the mean rises by `(0.82 − 0.7904)/3 ≈ 0.010` to roughly 0.824 —
enough to clear the bar even if the other two do nothing, which tells me MSL is where the whole case for
going simpler has to be won. On SMAP, the hardest, the gain is the open question: SMAP's recall was the floor (0.5557) and its anomalies
are the least separable by reconstruction error of any backbone, so a linear reconstructor may help only
marginally or not at all there — I would not be surprised to see SMAP roughly flat, and I should take
seriously that it could even regress: if SMAP's normal rhythm shifts window to window, one fixed `100 ×
100` map is *wrong* for most windows and will smear the periodic structure rather than reproduce it
cleanly, pushing more anomaly energy into the reconstruction and dropping recall below the patch
backbone's 0.5557. That would not refute the capacity argument — it would localize it: less flexibility
helps exactly when a single fixed readout suits the data, and hurts when the readout must change per
window. On PSM the patch backbone was already near saturation (0.9617); a linear reconstructor should land in the same
neighborhood, perhaps a hair lower precision since it cannot capture the finest non-periodic structure.
If DLinear's mean F1 clears the patch backbone's mainly through MSL while SMAP stays stuck, then the next
diagnosis is already written: the linear map captures one flat, position-indexed function of the window
with no explicit handle on *which* periodicities a given window carries — and the dataset it cannot
crack (SMAP) is the one whose periods vary window to window. That would point straight at a
reconstructor whose representation is organized around the discovered periods of each window.
