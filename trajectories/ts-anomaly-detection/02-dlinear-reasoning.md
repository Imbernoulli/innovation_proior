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
when there is a clear trend, which is the case the bare linear map handled worst. So the model stays as
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
keeps the comparison clean — I am reusing the decomposition, not tuning a new knob.

One TS-Lib-specific wrinkle I must get right against the scaffold. For anomaly detection the harness sets
`pred_len = seq_len` (reconstruction, same length in and out), so the two `nn.Linear(seq_len, pred_len)`
maps are `seq_len → seq_len`. The DLinear model routes `anomaly_detection` straight through its
`encoder` method — decompose, permute time to the last axis so the linears act along time (`[B, L, C] →
[B, C, L]`), apply `Linear_Seasonal` and `Linear_Trend`, sum, permute back to `[B, L, C]`. There is one
thing DLinear does *not* do here that both the patch backbone and the next rung will: it applies **no
reversible instance normalization**. The TS-Lib DLinear anomaly path feeds raw (already per-dataset
Z-scored) windows straight into the decomposition, because the moving-average split itself handles the
level — the trend linear absorbs the window's baseline directly. That is a real difference from the
paper's normalized-linear variant and from the patch backbone, and I am taking the scaffold's version:
no per-window mean/std, just decompose-linear-sum. The weights are also initialized to the uniform
`1/seq_len` average (every entry equal) rather than randomly, so each linear starts as an identity-ish
averaging map and learns away from it — a sensible warm start for a reconstructor.

So the delta from the patch backbone is concrete and in the *opposite* direction of more capacity: where
the patch Transformer reconstructed the window through a 512-wide attention encoder and a `512·12`
flatten head — powerful enough to partly reconstruct anomalies and kill recall — I now reconstruct it
through a parameter-free trend/seasonal split and two shared linear maps along time, summed. The model
can reproduce normal trend-plus-periodicity and almost nothing else, so anomalies should remain as
residual error.

Here is what I expect against the patch backbone's measured numbers, falsifiably. On MSL, where the
patch backbone's recall sagged to 0.7130 and F1 to 0.7904, the under-flexible linear reconstructor
should leave anomalies as cleaner residual and lift F1 — I expect MSL to be the dataset where DLinear
most clearly beats the patch backbone, the difference that should pull its mean F1 above 0.8135. On
SMAP, the hardest, the gain is the open question: SMAP's recall was the floor (0.5557) and its anomalies
are the least separable by reconstruction error of any backbone, so a linear reconstructor may help only
marginally or not at all there — I would not be surprised to see SMAP roughly flat. On PSM the patch
backbone was already near saturation (0.9617); a linear reconstructor should land in the same
neighborhood, perhaps a hair lower precision since it cannot capture the finest non-periodic structure.
If DLinear's mean F1 clears the patch backbone's mainly through MSL while SMAP stays stuck, then the next
diagnosis is already written: the linear map captures one flat, position-indexed function of the window
with no explicit handle on *which* periodicities a given window carries — and the dataset it cannot
crack (SMAP) is the one whose periods vary window to window. That would point straight at a
reconstructor whose representation is organized around the discovered periods of each window.
