The patch Transformer's numbers tell a sharper story than its mean F1 (0.8135) admits. PSM is fine —
0.9617 F1, precision 0.9888, recall 0.9361 — the server metrics are smooth and strongly periodic and the
patch tokens reconstruct them cleanly. But MSL came in at 0.7904 (precision 0.8867, recall 0.7130) and
SMAP at 0.6883 (precision 0.9040, recall a poor 0.5557). The recall is the tell: on the two telemetry
datasets the model is *missing* anomalies — precision stays high, so when it flags it is usually right,
but recall sags, which for a reconstruction detector means the model reconstructs the abnormal points
almost as well as the normal ones, so their error does not clear the threshold. That is the signature of
an over-flexible reconstructor: a 512-wide two-layer attention encoder with a flatten head over `D·N =
512·12` features is powerful enough to partly fit the anomalies, smoothing the very error spikes the score
depends on. So the move now is not more capacity — it is *less*, spent only on the structure that
characterizes a normal window: its trend and its periodicity.

The precision–recall gap makes the mechanism quantitative. On PSM the gap is `0.9888−0.9361 = 0.053`,
essentially balanced. On MSL it opens to `0.174`, on SMAP it yawns to `0.348`. High precision with
collapsing recall means the model reconstructed the anomalies nearly as faithfully as the normal points.
And recall is the lever: hold SMAP's precision at 0.9040 and lift recall from 0.5557 to 0.70 and F1 goes
`2·0.9040·0.70/(0.9040+0.70) = 0.789` — a 0.10 jump from recall alone. Every unit of capacity I remove
that still lets normal structure through converts almost directly into recall, and recall is where the
mean is bleeding.

Reason from what carries the reconstructable signal. A normal window is, to first order, a slow
trend-cyclical component plus an oscillation riding on top — server load drifts and cycles daily,
telemetry has a slow baseline with periodic structure. Both pieces are at heart "the value at a step is a
fixed linear functional of the rest of the window": a trend is a weighted combination extrapolating a slow
drift, a periodicity a weighted combination reading off same-phase values. So the most minimal
reconstructor is a single linear map along time: for one channel, `X̂ = W X` with `W` a `seq_len × seq_len`
matrix. Signal path length one, no recurrence to forget through, no attention distracted by the dominant
normal mass. Crucially, a linear map has *exactly* the capacity to reproduce trend-plus-periodicity and
almost none to fit the idiosyncratic shape of an anomaly. Where the patch Transformer reconstructed
abnormal points too faithfully and killed recall, a linear reconstructor should reproduce normal periodic
structure and leave anomalies as residual — a clean error spike. The cure for the over-flexible backbone
is a deliberately under-flexible one.

There is more than one way to spend less capacity. I could keep the patch Transformer and shrink it — drop
`d_model` or cut to one layer — but that is a continuous knob with no principled stop: attention over
patches stays universal enough to trace some anomaly shape, and I would be tuning a dial hoping to land
between "still fits anomalies" and "cannot fit normal structure." A bottleneck autoencoder has the same
unprincipled width knob, and a nonlinear decoder downstream of even a modest bottleneck can resynthesize a
sharp anomaly. What I want is a capacity ceiling that is *structural*, not dialed: a function class that
provably cannot represent an idiosyncratic anomaly shape no matter how it is trained. An affine map is
exactly that — for one channel `X̂ = W X`, a `100×100` matrix plus bias, a few tens of thousands of weights
shared across channels against the patch backbone's few million, and the cut is qualitative: it can
reproduce trend-plus-periodicity — both linear functionals of the window — and essentially nothing that is
not a fixed linear readout of positions.

The anomaly-survival is concrete. Suppose training fitted `W_eff` so a normal near-periodic window
reconstructs at tiny error. Add an isolated spike of height `h` at step `j`. Because the map is fixed and
linear, the reconstruction changes by exactly `W_eff[:,j]·h` — the spike's energy spreads across all
outputs by column `j`, which for a smooth periodic reconstructor is small and diffuse near the spike's own
location. So the reconstruction at `j` barely rises to meet the spike, and the residual `h·(1−W_eff[j,j])`
stays large — the clean spike the threshold needs. A high-capacity model could instead have learned
`W[j,j]≈1` locally and cancelled the spike; the linear map, tied to one global set of weights that must
also reconstruct every *normal* window, cannot afford to. That is the property I am buying.

But a single linear map fights itself when a window has a strong trend *and* a seasonal oscillation, the
common case here. One `W` must fit both, and they want different weight patterns — the trend wants broadly
smooth weights extrapolating a drift, the seasonality wants weights concentrated at the periodic lags.
Worse, the trend is large in magnitude and the oscillation small, so in a squared-error fit the trend
dominates the gradient: `W` spends itself getting the ramp roughly right and under-fits the small
oscillation that carries the fine structure — which is exactly where anomalies show, as breaks in shape,
not gross level moves. So I let the two components specialize.

The classical move is seasonal-trend decomposition: write the window as a smooth trend plus a residual,
each piece more regular than the sum. The harness already exposes the moving-average decomposition
(`layers.Autoformer_EncDec.series_decomp`) as a parameter-free preprocessing. Decompose once — `trend =
MovingAvg(x)`, `seasonal = x − trend` — then give each its own linear map, `Linear_Seasonal` and
`Linear_Trend`, both `seq_len → seq_len`, and sum the reconstructed streams. The trend linear learns a
slow baseline, the seasonal linear reads off the periodic lags, neither contaminated by the other nor
swamped by the other's magnitude.

This looks like added depth, which would betray the point of going simpler — so is it still linear? The
moving average is itself a fixed linear operator `A`, so `trend = A X`, `seasonal = (I−A) X`, and the two
maps produce `W_s(I−A)X + W_t A X = [W_s(I−A) + W_t A]X = W_eff X` — a single `100×100` matrix, the exact
same function class as the bare linear reconstructor, identical solution set. What decomposition changes is
not *what* can be represented but *how the loss is conditioned*: `A` is a fixed, well-chosen basis change
handing the loud low-frequency trend to `W_t` and the quiet high-frequency residual to `W_s` separately.
If a window's trend swings roughly `10×` the seasonal oscillation, the trend contributes on the order of
`100×` the gradient energy, and one shared matrix spends almost all its update budget on the ramp and
barely moves on the oscillation — precisely the fine structure whose violation is the anomaly; splitting
the streams equalizes their footing so each fits. It is preconditioning, not depth — same solution set, far
better-behaved learning — and it pays off precisely when there is a clear trend, the case the bare linear
map handled worst.

One decision the patch backbone got right, independent of backbone: channels. The TS-Lib DLinear shares
linear weights across channels by default — one `W` applied identically to every channel. Within a dataset
the channels usually share temporal dynamics (the daily rhythm of all server metrics, the diurnal cycle of
all telemetry sensors), so a shared map encodes that prior, slashes the parameter count against overfitting
on the smaller datasets, and keeps the model from fitting spurious cross-channel coincidences in the normal
data. So I keep channel-shared weights and model no cross-variate correlation — the same bet the patch
backbone's channel-independence made, here expressed as one shared map. Concretely a PSM batch `[32,100,25]`
decomposes into seasonal and trend, each permuted to `[32,25,100]` so time is the last axis; each
`nn.Linear(100,100)` broadcasts over the leading `[32,25]`, applying the *same* `100×100` weight to every
one of the `800` series; sum and permute back. No axis ever mixes channels, and the weight is oblivious to
whether it is fed 25 series or 55.

The decomposition's boundary behavior decides whether it helps. `trend` is a stride-1 average pool of odd
kernel `k`, which returns `L−k+1` outputs and so must pad. Zero-padding would drag the trend toward zero at
the two ends, manufacturing spurious dips exactly where I have least information — false anomalies at every
window boundary. Instead replicate the endpoints: `(k−1)/2` copies of the first value at the front, the
last value at the back, then average-pool, restoring the length exactly (`k=25`: 12 copies each side,
`100+24=124`, pool back to `100`) and keeping the trend flat-but-faithful at the edges. I keep the
established kernel 25, which averages out sub-cycle wiggles while preserving the cycle-and-slower trend, and
keeping it identical to the established block keeps the comparison clean — I am reusing the decomposition,
not tuning a new knob.

One scaffold-specific wrinkle: the TS-Lib DLinear anomaly path applies **no reversible instance
normalization**. It feeds the raw (already per-dataset Z-scored) window straight into the decomposition,
because the moving-average split itself handles the level — the trend linear absorbs the baseline directly.
That is a real difference from the patch backbone, and I take the scaffold's version. The weights are
initialized to the uniform `1/seq_len` average rather than randomly: at init each row is all `1/100`, so
`Linear(v)` returns the mean of `v` broadcast to every step; `trend_output` is the window's level,
`seasonal_output ≈ 0` (the residual is mean-zero by construction), and their sum is a flat line at the
window average — a stable "predict the mean everywhere" floor from which training bends the rows toward the
actual trend and lags. This also shows skipping instance norm is safe: a constant level `c` is pure trend
(`A` of a constant is that constant, residual zero), and the uniform-init trend row reproduces `c` exactly,
so the level passes straight through the trend linear — the decomposition *is* the level handling that
PatchTST needed a normalize-denormalize wrapper for.

The bet inverts the patch backbone's: a parameter-free trend/seasonal split and two shared linear maps in
place of a 512-wide encoder, reproducing normal trend-plus-periodicity and little else so anomalies remain
as residual error. Against the patch backbone's numbers, MSL is where the case has to be won — its recall
sagged to 0.7130, and the under-flexible map should leave those anomalies as cleaner residual and pull the
mean above 0.8135. PSM was near saturation (0.9617), so I expect it to hold in that neighborhood. SMAP is
the open risk: its rhythm looks least stable, and if it shifts window to window one fixed `100×100` map is
wrong for most windows and smears the periodic structure rather than reproducing it — a linear
reconstructor could then help only marginally there, or even regress below the patch backbone's 0.5557
recall. That would not refute the capacity argument but localize it: less flexibility helps when a single
fixed readout suits the data and hurts when the readout must change per window — which is the handle the
next backbone would need, a representation that adapts its lag structure to each window.
