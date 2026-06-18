The forecaster is the whole point, and with the scaffold returning zeros there is no forecaster at
all — so the floor I should start from is the simplest thing that can possibly fit M4, the one whose
failure modes I can read off without ambiguity. I have a short look-back (Monthly `seq_len=36`,
Quarterly `16`, Yearly `12`), one channel, a direct horizon to fill (`18`, `8`, `6`), and a fixed
SMAPE loss. The lineage that precedes this rung — iterated RNNs, then the heavy attention forecasters
— spent its capacity on long-range dependency modeling. But I should ask whether any of that is
*needed* before I pay for it, because on M4 the series are short and the dominant structure is plain:
a slow level drift (trend) plus a repeating shape (seasonality). Both of those are *linear* in the
look-back. Extrapolating a drift is a linear extrapolation; reading off the value one period back is
a linear read of a past coordinate. So the minimal hypothesis is that a single linear map from the
look-back window to the horizon captures most of what is forecastable here, and everything heavier is
either unnecessary or actively harmful on these tiny windows.

Let me make that map concrete. The cleanest direct-multi-step forecaster is one linear layer along
time: with the look-back `x ∈ R^{L}` (per series, per channel), produce `x̂ = W x` with
`W ∈ R^{T×L}`, so every past step connects to every future step through one learned weight and the
signal path length is one. There is no recurrence to compound error over the horizon and no attention
to tokenize single steps that have no standalone meaning. Direct-multi-step matters specifically
because the alternative — predict one step, feed it back — accumulates error, and on Yearly the
horizon is six steps of pure trend where a compounding error would be most visible.

But a single `W` fit by the loss has a known pathology: when the series carries a strong trend, the
large-magnitude trend dominates the error budget, the linear map spends its weights tracking the
level, and the smaller-magnitude seasonal shape gets under-fit. The remedy is to *separate the loud
component from the quiet one before fitting*. This is exactly what the decomposition primitive from
the attention-forecaster lineage gives me for free: a length-preserving moving average extracts the
trend, `trend = MovingAvg(x)`, and the residual is the seasonal part, `seasonal = x − trend`. I then
give each its own linear map and sum the outputs:
`x̂ = Linear_T(trend) + Linear_S(seasonal)`. Crucially this adds *no* representational capacity — a
moving average followed by two linear maps and a sum is still affine end to end — so it is not a more
powerful model, it is *preconditioning* that lets each linear layer specialize on a component of
roughly uniform scale. That preconditioning is precisely what should help most on the regimes with a
clear trend (Yearly, and the trending Monthly series) and cost nothing on the rest.

I have to fit the decomposition to *this* harness, not to the long-horizon defaults. The canonical
moving-average kernel is 25, the same smoothing scale the decomposition block uses elsewhere, with
replicate-padding at both ends (`(k−1)//2` copies of the first and last value) so the trend stays
faithful at the window edges instead of being pulled toward zero. But here the Monthly window is only
36 steps and the Yearly window only 12 — a kernel of 25 is wider than the Yearly window. With
replicate padding the moving average still returns a valid length-`L` output (it just smooths very
aggressively, the trend collapsing toward the window's mean and almost all variation flowing into the
seasonal residual), so it does not crash; but it means on the short regimes the decomposition is
effectively "trend ≈ window mean, seasonal ≈ everything else," which is a perfectly reasonable split
for series this short. I keep the kernel read from `configs.moving_avg` (default 25) so the behavior
matches the reference exactly under the fixed Custom protocol, where `moving_avg` is not overridden
and so stays 25.

Channel handling is trivial here because M4 is univariate (`enc_in = c_out = 1`), but the design
choice still matters in principle: the linear maps are *channel-shared* (one `Linear_S`, one
`Linear_T`), not per-channel, which on a multivariate dataset cuts parameters from `C·T·L` to `2·T·L`
and avoids fitting spurious cross-channel coupling. With one channel this collapses to a single pair
of maps of size `T×L` each — for Monthly that is `18×36`, a few hundred parameters, which is the
entire trainable footprint of this rung. That tiny footprint is the point: under the fixed protocol
(`lr=1e-3`, batch 16, 10 epochs, patience 3) a model this small cannot overfit M4's short series, and
it trains to convergence well inside ten epochs.

One subtlety about the fixed protocol I must respect: the harness passes `d_model=512`, `e_layers=2`,
`n_heads=8` on the command line, but DLinear *ignores all of them* — there is no embedding, no
attention, no notion of model width. The linear maps are sized purely by `seq_len` and `pred_len`.
So I simply do not read those configs; the rung is deliberately blind to the capacity knobs the
harness offers, which is the cleanest statement of the hypothesis that capacity is not what M4 needs.
I also do not touch the instance-normalization question yet: DLinear in its plain decomposition form
does not re-center the window (its NLinear sibling does, by subtracting the last value and adding it
back). I leave that out at the floor on purpose, because I want to see whether the *raw* decomposition
linear is already enough, and because adding normalization now would confound the comparison with the
rungs above that build normalization in deliberately.

There is one more reason to start exactly here rather than with anything fancier: the SMAPE loss the
harness optimizes is a *percentage* error, `(200/T)·Σ_t |y_t − ŷ_t| / (|y_t| + |ŷ_t|)`, which is
already scale-robust across series — it does not let a large-magnitude series dominate the gradient
the way a raw MSE would. That partly masks the missing instance normalization at training time (the
loss itself normalizes the residual by magnitude), which means a plain decomposition-linear is a
*fair* floor under this protocol: it is not handicapped by an unlucky choice of loss, so whatever
SMAPE it reaches is genuinely the best a purely affine, un-normalized direct-multi-step model can do
here. If a richer rung beats it, the gain is attributable to capacity or normalization, not to the
linear model having been set up to fail. Keeping the floor honest in this way is the reason I resist
the temptation to bolt on the NLinear last-value re-centering now — that would already be conceding
the distribution-shift point I want the *data* to make for me.

So the step-1 edit is the literal minimum: a `series_decomp` block (replicate-padded moving average,
kernel 25), two channel-shared `nn.Linear(seq_len, pred_len)` maps applied along the time axis to the
permuted seasonal and trend tensors, their sum permuted back to `[B, pred_len, 1]`, and the forward
dispatch returning the last `pred_len` steps (the distilled module is in the answer). No embedding, no
nonlinearity beyond the parameter-free decomposition, no marks (the harness passes `x_mark_enc=None`
and DLinear never wanted them).

Now reason about what this floor must do, because reading its failure is the entire reason to run it
first. The model is affine in the look-back, so it can capture exactly two things: a linear
extrapolation of the trend, and a fixed linear combination of past values that reproduces a periodic
shape. On **Yearly**, where the structure is almost pure trend over a six-step horizon and the window
is only twelve steps, an affine map should do *relatively* well in absolute terms but is the regime
where a tiny look-back hurts most — there is barely any history to extrapolate from, and SMAPE on
yearly series is intrinsically the hardest of the three because the values are large and the horizon
is short and trend-dominated. On **Monthly**, where the 12-step seasonality is strong and the window
(36) spans three full cycles, the seasonal linear map has enough periods to lock onto the repeating
shape, so this is where a linear model should be most competitive. On **Quarterly** (window 16, two
full 4-step cycles, horizon 8) it sits in between. So I expect the SMAPE to come out *worst on Yearly*
and best on the regime where the linear seasonal read has the most cycles to fit.

What this floor *cannot* do is the diagnosis I am setting up for the rung above. It has no nonlinearity
and no instance normalization, so two failure modes are baked in. First, every M4 series sits at a
different level and scale; a single shared linear map has to serve all of them at once, and without
per-window normalization it cannot decouple "what shape" from "what level," so series far from the
training-set average level will be systematically off — a distribution-shift failure that
normalization is the standard cure for. Second, an affine map cannot represent any interaction between
trend and season or any shape that is not a fixed linear function of the window; where the dynamics are
even mildly nonlinear, it will leave structure on the table. Both of those are exactly what a
higher-capacity rung — one that normalizes each window and adds a learned nonlinear representation —
should be able to claw back. So I expect this rung to be beatable, and I expect the gap to open
*widest on the regime where the linear model is most strained*. Running it tells me the SMAPE floor per
regime and which regime is the binding constraint; that floor is the number every later rung has to
beat, and the per-regime spread is the first clue to what the next architecture must add.
