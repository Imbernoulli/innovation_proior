The forecaster is the whole point, and with the scaffold returning zeros there is no forecaster at
all — so the floor I should start from is the simplest thing that can possibly fit M4, the one whose
failure modes I can read off without ambiguity. I have a short look-back (Monthly `seq_len=36`,
Quarterly `16`, Yearly `12`), one channel, a direct horizon to fill (`18`, `8`, `6`), and a fixed
SMAPE loss. The lineage that precedes this — iterated RNNs, then the heavy attention forecasters —
spent its capacity on long-range dependency modeling. But I should ask whether any of that is
*needed* before I pay for it, because on M4 the series are short and the dominant structure is plain:
a slow level drift (trend) plus a repeating shape (seasonality). Both of those are *linear* in the
look-back. Extrapolating a drift is a linear extrapolation; reading off the value one period back is
a linear read of a past coordinate. So the minimal hypothesis is that a single linear map from the
look-back window to the horizon captures most of what is forecastable here, and everything heavier is
either unnecessary or actively harmful on these tiny windows.

Make that map concrete. The cleanest direct-multi-step forecaster is one linear layer along time:
with the look-back `x ∈ R^{L}` produce `x̂ = W x` with `W ∈ R^{T×L}`, so every past step connects to
every future step through one learned weight and the signal path length is one. There is no
recurrence to compound error over the horizon and no attention to tokenize single steps that have no
standalone meaning. The direct map matters specifically because the alternative — predict one step,
feed it back — accumulates error: a one-step model with per-step error `ε` conditions its horizon-`T`
prediction on `T−1` already-noisy inputs, so the error grows at least linearly and in a trending
regime compounds, whereas the direct map pays `ε` once per horizon slot from the clean observed
window. On Yearly the horizon is six steps of pure trend where compounding would be most visible, so
I fix the output strategy to direct-multi-step now.

Before committing to a *bare* linear map I walk the two tempting elaborations, because "simplest
possible floor" is a claim I have to earn. The first is to give the map a hidden layer — an MLP
`L → h → T` with a nonlinearity. Count what that buys on Monthly: a width-`h=512` hidden layer is
`36·512 + 512·18 = 27648` weights, against the pure linear's `18·36 = 648`, a forty-fold jump in
capacity to fit series that are individually thirty-six points long. The hidden layer has enough
freedom to memorize idiosyncratic window noise, and on short series under a 10-epoch budget that is
exactly the overfitting that historically sank naive MLP entrants on M4. So the MLP is not a safer
floor, it is a worse one — it would confound "linear is enough" with "I picked a model that
overfits." I drop it. The second temptation is the NLinear trick: subtract the last observed value
`x[:,-1]` before the linear map and add it back after, a free per-window re-centering that handles
level shift. It is genuinely attractive and I will want something like it later. But adding it *now*
would pre-concede the distribution-shift point I want the data to make on its own, and confound any
later gain from a principled instance normalization with a gain I could have grabbed at the floor. So
I hold it out deliberately; the floor stays un-normalized on purpose.

That leaves a single `W` fit by the loss, which has a known pathology worth fixing because it is free
to fix. When the series carries a strong trend, the large-magnitude trend dominates the error budget,
the linear map spends its weights tracking the level, and the smaller-magnitude seasonal shape gets
under-fit. The remedy is to separate the loud component from the quiet one before fitting — exactly
what the decomposition primitive from the attention-forecaster lineage gives me for free: a
length-preserving moving average extracts the trend, `trend = MovingAvg(x)`, and the residual is the
seasonal part, `seasonal = x − trend`. I give each its own linear map and sum the outputs,
`x̂ = Linear_T(trend) + Linear_S(seasonal)`. Crucially this adds *no* representational capacity — a
moving average `A` followed by two linear maps and a sum is still affine end to end,
`x̂ = W_S(I−A)x + W_T A x = (W_S(I−A) + W_T A) x`, a single effective matrix. So it is not a more
powerful model, it is preconditioning. The reason it aids the fit despite adding no expressivity: `Ax`
is a smooth, large-magnitude signal (level and drift) while `(I−A)x` is a small-magnitude, roughly
zero-mean residual (the seasonal wiggle). Fed a single `W`, the loss gradient is dominated by the
trend and the seasonal directions in weight space get tiny gradients and settle slowly — the loud
component starves the quiet one during optimization, not just at the optimum. Routing the two through
separate maps lets each see an input of roughly uniform scale, so `W_S` grows to whatever the seasonal
residual needs without fighting the trend's gradient, and both branches reach a good fit inside ten
epochs. Summing the branch forecasts is exactly right because the series *is* trend plus seasonal by
construction (`x = Ax + (I−A)x`). The cost is doubling the trainable count to `2·18·36 = 1296` weights
plus `36` biases on Monthly (`272` on Quarterly, `156` on Yearly), negligible against tens of thousands
of training windows per regime — pure upside, and it should help most on the regimes with a clear
trend.

The kernel size is where the short windows first bite. The moving-average kernel is 25 with
replicate-padding at both ends (`(k−1)//2 = 12` copies of the first and last value) so the trend stays
faithful at the window edges instead of being pulled toward zero. A kernel of 25 is wider than the
Yearly window of 12, so does the average even return the right length? For odd `k` the identity
`L + 2·((k−1)//2) − k + 1 = L` holds, so length is always preserved: Yearly `12 → 36 → 12`, Monthly
`36 → 60 → 36`, Quarterly `16 → 40 → 16`. What the wide kernel *does* on the short regimes is smooth
very aggressively — on Yearly each output averages 25 values from a padded array that is mostly copies
of the endpoints, so the trend collapses toward the window mean and almost all variation flows into
the seasonal residual. That is not a bug; "trend ≈ window mean, seasonal ≈ everything else" is a
reasonable split for a twelve-point series, and it degrades gracefully. I read the kernel from
`configs.moving_avg` with default 25, which the fixed protocol does not override.

The linear maps are channel-shared (one `Linear_S`, one `Linear_T`), not per-channel — on a
multivariate dataset that cuts parameters from `C·T·L` to `2·T·L` and avoids fitting spurious
cross-channel coupling; with M4's single channel it collapses to one pair of `T×L` maps, the few
hundred parameters counted above. That tiny footprint is the point: under the fixed protocol a model
with ~1300 parameters cannot overfit M4's short series and trains to convergence well inside ten
epochs, so early stopping will rarely trigger.

One subtlety about the fixed protocol: the harness passes `d_model=512`, `e_layers=2`, `n_heads=8`,
but a decomposition-linear *ignores all of them* — there is no embedding, no attention, no notion of
model width. So I simply do not read those configs; the model is deliberately blind to the capacity
knobs the harness offers, which is the cleanest statement of the hypothesis that capacity is not what
M4 needs. The scale of what I am declining is worth naming: a 512-wide two-layer attention encoder is
on the order of a few million parameters, so by not reading `d_model` I fit Monthly with roughly three
orders of magnitude fewer weights than the harness stands ready to hand me. If a linear map at that
footprint is competitive, that ratio *is* the argument. I also do not read the marks: the harness
passes `x_mark_enc=None`, and this model never wanted calendar features.

There is a design fork on *how* to fit `W` I should settle rather than default into. A linear map has
a closed-form option the richer rungs do not: solve the two maps by ridge least squares in one shot,
no epochs. Tempting for a model this small. But the harness optimizes and scores SMAPE, a percentage
error, and the L2-optimal solution is not the SMAPE-optimal one — least squares weights each residual
by magnitude, over-serving the large-valued series exactly the way I argued the loss should not.
Fitting under the harness's SMAPE loss aligns training with evaluation, which matters most on the
trending large-magnitude Yearly series where the L2/SMAPE gap is widest. So I do not shortcut to the
closed form; I let the harness train the maps under its own loss, which is also the only path the
fixed protocol offers.

That percentage form also makes this a *fair* floor rather than one handicapped by an unlucky loss.
SMAPE, `(200/T)·Σ_t |y_t − ŷ_t| / (|y_t| + |ŷ_t|)`, is scale-robust — it does not let a large series
dominate the gradient the way raw MSE would — which partly masks the missing instance normalization
at training time. So whatever SMAPE a plain decomposition-linear reaches is genuinely the best a
purely affine, un-normalized direct-multi-step model can do here; if a richer model beats it, the gain
is attributable to capacity or normalization, not to the linear model having been set up to fail. And
because M4 series are counts and prices that stay strictly positive and well away from zero, the SMAPE
denominator is bounded below and the per-step contribution is bounded in `[0, 200]` — no degenerate
near-zero term to clamp. Keeping the floor honest this way is the second reason I resisted NLinear.

So the edit is the literal minimum: a `series_decomp` block (replicate-padded moving average, kernel
25), two channel-shared `nn.Linear(seq_len, pred_len)` maps applied along time to the permuted
seasonal and trend tensors, their sum permuted back to `[B, pred_len, 1]`, and the forward dispatch
returning the last `pred_len` steps (the full module is in the answer). Input `[B, 36, 1]` on Monthly
decomposes to two `[B, 36, 1]` tensors, permute each to `[B, 1, 36]`, apply `Linear(36 → 18)`, sum,
permute back to `[B, 18, 1]`; the trailing slice is a no-op here but keeps the contract identical to
richer rungs. Only `L` and `T` change across regimes, both read from `configs`, so the shapes compose
on all three.

I want to confirm the two structures I claim are forecastable actually lie inside the affine class, by
checking the two limiting cases where I know the right answer. A pure linear ramp `x_t = a + b·t` (the
caricature of Yearly): the correct six-step continuation is `a + b·(L+h)`, and the least-squares line
fit to a window is a linear function of the window's values, so there exists a `W` whose rows compute
"fit a line, extend it `h` steps" and reproduce the ramp exactly. The trend branch can be perfect on a
noise-free trend. A pure period-`p` signal `x_t = s_{t mod p}` (the caricature of Monthly): "copy the
value one period back," `x̂_{L+h} = x_{L+h−p}`, is a single shifted-identity row in `W`, hence linear,
so the seasonal branch can represent exact periodic copy whenever the window spans at least one full
period — true for all three regimes. Both components are individually expressible, so any error this
floor shows is *not* a failure of the hypothesis class on clean signal; it is the compromise that one
*shared* `W` cannot hold the right `a,b` for every series at once, plus whatever nonlinear residual the
affine map structurally cannot touch. That is precisely the diagnosis I want the floor to isolate.

Now what this floor must do, since reading its failure is the reason to run it first. Being affine in
the look-back, it captures exactly two things: a linear extrapolation of trend and a fixed linear
combination reproducing a periodic shape. The count of full cycles in each window — three for Monthly,
two for Quarterly, effectively zero clean periods for Yearly — is the cleanest predictor of where the
linear seasonal read has enough data to lock on. So I expect SMAPE *worst on Yearly*, where the
structure is almost pure trend over a six-step horizon extrapolated from twelve points, and best where
the cycle count is highest. What it *cannot* do is the diagnosis I am setting up for whatever comes next.
Two failure modes are baked in. First, every M4 series sits at a different level and scale; a single
shared linear map with no per-window normalization cannot decouple "what shape" from "what level," so
series far from the training-set average level are systematically off — the distribution-shift failure
normalization is the standard cure for, and the exact failure I declined to patch with NLinear.
Second, an affine map cannot represent any interaction between trend and season or any shape that is
not a fixed linear function of the window. Both are exactly what a higher-capacity model that
normalizes each window and adds a learned nonlinearity should claw back. So I expect this floor to be
beatable, and the gap to open widest on Yearly, the regime where the linear model is most strained.
The per-regime spread — how far Yearly sits above Monthly and Quarterly — is the first quantitative
clue to what the next architecture must add.
