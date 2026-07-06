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
because the alternative — predict one step, feed it back — accumulates error, and I can size that
problem before running anything: if a one-step model carries per-step error `ε` and I roll it out, the
horizon-`T` prediction is conditioned on `T−1` already-noisy inputs, so the error grows at best
linearly and in a trending regime compounds, whereas the direct map pays `ε` once per horizon slot from
the clean observed window. On Yearly the horizon is six steps of pure trend where a compounding error
would be most visible, so I fix the output strategy to direct-multi-step now and never revisit it.

Before I commit to a *bare* linear map I should walk the two tempting elaborations and kill them with
arithmetic rather than taste, because "simplest possible floor" is a claim I have to earn. The first
temptation is to give the map a hidden layer — an MLP `L → h → T` with a nonlinearity — since surely a
nonlinear function fits M4 better. Count what that buys on Monthly: a width-`h=512` hidden layer is
`36·512 + 512·18 = 18432 + 9216 = 27648` weights, against the pure linear's `18·36 = 648`. That is a
forty-fold jump in capacity to fit series that are, individually, thirty-six points long. The hidden
layer has enough freedom to memorize idiosyncratic window noise, and on short series under a 10-epoch
budget that is exactly the overfitting that historically sank naive MLP entrants on M4. So the MLP is
not a safer floor, it is a *worse* one — it would confound "linear is enough" with "I picked a model
that overfits." I drop it. The second temptation is the NLinear trick: subtract the last observed value
`x[:,-1]` before the linear map and add it back after, a free per-window re-centering that handles level
shift. It is genuinely attractive and I will want something like it later. But adding it *now* would
pre-concede the distribution-shift point I want the data to make on its own, and it would confound any
later gain from a principled instance normalization with a gain I could have grabbed at the floor. So I
hold it out deliberately; the floor stays un-normalized on purpose.

That leaves a single `W` fit by the loss, which has a known pathology I do want to fix, because it is
free to fix. When the series carries a strong trend, the large-magnitude trend dominates the error
budget, the linear map spends its weights tracking the level, and the smaller-magnitude seasonal shape
gets under-fit. The remedy is to *separate the loud component from the quiet one before fitting*. This
is exactly what the decomposition primitive from the attention-forecaster lineage gives me for free: a
length-preserving moving average extracts the trend, `trend = MovingAvg(x)`, and the residual is the
seasonal part, `seasonal = x − trend`. I then give each its own linear map and sum the outputs:
`x̂ = Linear_T(trend) + Linear_S(seasonal)`. Crucially this adds *no* representational capacity — a
moving average `A` followed by two linear maps and a sum is still affine end to end,
`x̂ = W_S(I−A)x + W_T A x = (W_S(I−A) + W_T A) x`, a single effective matrix — so it is not a more
powerful model, it is *preconditioning* that lets each linear layer specialize on a component of roughly
uniform scale. It does double the trainable count to `2·18·36 = 1296` weights plus `2·18 = 36` biases,
`1332` parameters on Monthly (`272` on Quarterly, `156` on Yearly), but that is still negligible against
tens of thousands of training windows per regime, so the preconditioning is pure upside. That
preconditioning is precisely what should help most on the regimes with a clear trend (Yearly, and the
trending Monthly series) and cost nothing on the rest.

It helps to see *why* the preconditioning aids optimization even though it adds no expressivity, because
that is the whole justification for spending the second linear map. The moving average `A` is a low-pass
filter, so `Ax` is a smooth, large-magnitude signal — the level and drift — while `(I−A)x` is a high-pass
residual, small-magnitude and roughly zero-mean, carrying the seasonal wiggle. Fed a *single* `W`, the
loss gradient is dominated by the large-magnitude trend, and the seasonal directions in weight space get
tiny gradients and settle slowly; the loud component starves the quiet one during optimization, not just
at the optimum. By routing the two through separate maps `W_T A` and `W_S(I−A)`, each map sees an input
of roughly uniform scale, so `W_S` can grow to whatever magnitude the seasonal residual needs without
fighting the trend's gradient, and both branches reach a good fit inside the 10-epoch budget. The
recombination is a plain sum of the two horizon predictions, which is exactly right because the original
series *is* trend plus seasonal by construction (`x = Ax + (I−A)x`), so summing the branch forecasts
reconstructs a forecast of the whole. And because M4 is univariate, the "channel-shared" maps operate on
a `[B, 1, L]` tensor with the single channel folded against the batch, so there is no cross-channel
weight to learn at all here — the shared-map machinery is present for generality but collapses to the
one pair of matrices I counted.

I have to fit the decomposition to *this* harness, not to the long-horizon defaults, and the kernel size
is where the short windows first bite, so let me trace it rather than assume it works. The canonical
moving-average kernel is 25, the same smoothing scale the decomposition block uses elsewhere, with
replicate-padding at both ends (`(k−1)//2 = 12` copies of the first and last value) so the trend stays
faithful at the window edges instead of being pulled toward zero. The obvious worry is that a kernel of
25 is wider than the Yearly window of 12 — does the average even return the right length? I walk the
shapes: with `L=12` I pad `12` on each side to length `12+24 = 36`, then `AvgPool1d(kernel=25, stride=1)`
returns `36 − 25 + 1 = 12`. Length preserved. The same check on Monthly (`36 → 60 → 36`) and Quarterly
(`16 → 40 → 16`) both preserve length, so the block never crashes on any regime — the identity
`L + 2·((k−1)//2) − k + 1 = L` holds whenever `k` is odd, which `25` is. What the wide kernel *does* on
the short regimes is smooth very aggressively: on Yearly each of the 12 outputs averages 25 values from
a padded array that is 12 copies of `x[0]`, the 12 real points, and 12 copies of `x[11]`, so the trend
collapses toward the window's mean and almost all variation flows into the seasonal residual. That is
not a bug — "trend ≈ window mean, seasonal ≈ everything else" is a perfectly reasonable split for a
twelve-point series, and it degrades gracefully rather than failing. I keep the kernel read from
`configs.moving_avg` (default 25) so the behavior matches the reference exactly under the fixed Custom
protocol, where `moving_avg` is not overridden and so stays 25.

Channel handling is trivial here because M4 is univariate (`enc_in = c_out = 1`), but the design choice
still matters in principle: the linear maps are *channel-shared* (one `Linear_S`, one `Linear_T`), not
per-channel, which on a multivariate dataset cuts parameters from `C·T·L` to `2·T·L` and avoids fitting
spurious cross-channel coupling. With one channel this collapses to a single pair of maps of size `T×L`
each — for Monthly that is `18×36`, the few hundred parameters I counted above, which is the entire
trainable footprint of this rung. That tiny footprint is the point: under the fixed protocol (`lr=1e-3`,
batch 16, 10 epochs, patience 3) a model with roughly `1300` parameters cannot overfit M4's short
series, and it trains to convergence well inside ten epochs, so early stopping will rarely even trigger.

One subtlety about the fixed protocol I must respect: the harness passes `d_model=512`, `e_layers=2`,
`n_heads=8` on the command line, but DLinear *ignores all of them* — there is no embedding, no attention,
no notion of model width. The linear maps are sized purely by `seq_len` and `pred_len`. So I simply do
not read those configs; the rung is deliberately blind to the capacity knobs the harness offers, which
is the cleanest statement of the hypothesis that capacity is not what M4 needs. It is worth naming the
scale of what I am declining: a 512-wide two-layer attention encoder is on the order of a few million
parameters, so by not reading `d_model` I am fitting Monthly with roughly three orders of magnitude
fewer weights than the harness stands ready to hand me. If a linear map at that footprint is
competitive, that ratio *is* the argument. I also do not read the marks: the harness passes
`x_mark_enc=None`, and DLinear never wanted calendar features, so the `forecast` signature accepts and
ignores them.

There is one more reason to start exactly here rather than with anything fancier: the SMAPE loss the
harness optimizes is a *percentage* error, `(200/T)·Σ_t |y_t − ŷ_t| / (|y_t| + |ŷ_t|)`, which is
already scale-robust across series — it does not let a large-magnitude series dominate the gradient
the way a raw MSE would. That partly masks the missing instance normalization at training time (the
loss itself normalizes the residual by magnitude), which means a plain decomposition-linear is a
*fair* floor under this protocol: it is not handicapped by an unlucky choice of loss, so whatever
SMAPE it reaches is genuinely the best a purely affine, un-normalized direct-multi-step model can do
here. If a richer rung beats it, the gain is attributable to capacity or normalization, not to the
linear model having been set up to fail. Keeping the floor honest in this way is the second reason I
resisted the NLinear last-value re-centering above — that would already be conceding the
distribution-shift point I want the *data* to make for me.

There is a design fork on *how* to fit `W` that I should settle explicitly rather than default into,
because a linear map has a closed-form option the richer rungs do not. I could solve the two linear maps
by ridge least squares in one shot — the normal equations give the L2-optimal `W` without any epochs.
It is tempting for a model this small. But the harness optimizes and scores **SMAPE**, a percentage
error, and the L2-optimal solution is *not* the SMAPE-optimal one: least squares weights each residual
by magnitude, so it would over-serve the large-valued series exactly the way I argued the loss should
not. Fitting the same maps by Adam under the harness's SMAPE loss aligns the training objective with the
evaluation metric, which matters most on the trending large-magnitude Yearly series where the L2/SMAPE
gap is widest. So I do not shortcut to the closed form; I let the harness train the maps under its own
loss, which is also the only path the fixed protocol actually offers me. This is a small decision but it
is the difference between fitting the metric I am judged on and fitting a proxy.

So the step-1 edit is the literal minimum: a `series_decomp` block (replicate-padded moving average,
kernel 25), two channel-shared `nn.Linear(seq_len, pred_len)` maps applied along the time axis to the
permuted seasonal and trend tensors, their sum permuted back to `[B, pred_len, 1]`, and the forward
dispatch returning the last `pred_len` steps (the distilled module is in the answer). No embedding, no
nonlinearity beyond the parameter-free decomposition, no marks. Let me trace one forward pass end to end
to be sure the shapes land, since a silent permute error here would poison every reading I take off this
floor. Input `x_enc` is `[B, 36, 1]` on Monthly. The decomposition returns `seasonal, trend` each
`[B, 36, 1]`. I permute each to `[B, 1, 36]` so time is the last axis the linear map consumes, apply
`Linear(36 → 18)` to get `[B, 1, 18]` for each, sum to `[B, 1, 18]`, and permute back to `[B, 18, 1]`.
The forward slice `[:, -pred_len:, :]` is a no-op here since the output is already length 18, but it
keeps the contract identical to the richer rungs that will emit longer sequences. Shapes check on all
three regimes because only `L` and `T` change and both are read from `configs`.

Before I read off predictions I want to *verify* that the two structures I claim are forecastable are
actually inside the affine hypothesis class, by checking the two limiting cases where I know the right
answer. Take a pure linear ramp `x_t = a + b·t` with no noise, the caricature of a Yearly series. The
correct six-step continuation is `x_{L+h} = a + b·(L+h)`, and the least-squares line fit to a window is
a *linear* function of the window's values — intercept and slope are each fixed linear combinations of
`x` — so there exists a `W` whose rows compute "fit a line, extend it `h` steps" and reproduce the ramp
with exactly zero error. The trend branch can therefore be perfect on a noise-free trend, which tells me
the affine class is not the bottleneck on clean trend; the bottleneck will be that one *shared* `W`
cannot hold the right `a,b` for every series at once. Now take a pure period-`p` signal
`x_t = s_{t mod p}` with no trend, the caricature of Monthly seasonality. The map "copy the value one
period back," `x̂_{L+h} = x_{L+h−p}`, is a single shifted-identity row in `W`, hence linear, so the
seasonal branch can represent exact periodic copy whenever the window contains the phase `L+h−p` it needs
— which it does when the window spans at least one full period, true for all three regimes. Both
components are individually expressible, so any error this floor shows is *not* a failure of the
hypothesis class on the clean signal; it is the shared-across-series and shared-across-level compromise,
plus whatever nonlinear residual the affine map structurally cannot touch. That is precisely the
diagnosis I want the floor to isolate.

One more property of the loss makes this floor safe to read without special-casing: the SMAPE
denominator `|y_t| + |ŷ_t|` could in principle blow the loss up near zero, but M4 series are counts and
prices that stay strictly positive and well away from zero, so the denominator is bounded below and the
per-step contribution is bounded in `[0, 200]`. There is no degenerate near-zero term I need to clamp,
which is another reason the bare affine map is a *fair* floor rather than one handicapped by a loss
pathology — I can trust the SMAPE it reports as the honest affine ceiling.

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
full 4-step cycles, horizon 8) it sits in between. The count of full cycles in each window — three for
Monthly, two for Quarterly, effectively zero clean periods for Yearly — is the cleanest predictor I
have of where the linear seasonal read has enough data to fit, so I expect the SMAPE to come out
*worst on Yearly* and best where that cycle count is highest.

What this floor *cannot* do is the diagnosis I am setting up for the rung above. It has no nonlinearity
and no instance normalization, so two failure modes are baked in. First, every M4 series sits at a
different level and scale; a single shared linear map has to serve all of them at once, and without
per-window normalization it cannot decouple "what shape" from "what level," so series far from the
training-set average level will be systematically off — a distribution-shift failure that
normalization is the standard cure for, and the exact failure I declined to patch with NLinear so that
the next rung's normalization has something real to fix. Second, an affine map cannot represent any
interaction between trend and season or any shape that is not a fixed linear function of the window;
where the dynamics are even mildly nonlinear, it will leave structure on the table. Both of those are
exactly what a higher-capacity rung — one that normalizes each window and adds a learned nonlinear
representation — should be able to claw back. So I expect this rung to be beatable, and I expect the
gap to open *widest on the regime where the linear model is most strained*, which by the cycle-count
argument is Yearly. Running it tells me the SMAPE floor per regime and which regime is the binding
constraint; that floor is the number every later rung has to beat, and the per-regime spread — how far
Yearly sits above Monthly and Quarterly — is the first quantitative clue to what the next architecture
must add.
