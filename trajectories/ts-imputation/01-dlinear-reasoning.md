The identity fill is the floor, and staring at it tells me exactly what the metric asks for and why
doing nothing is wrong. The loop zeroes a quarter of the entries, hands me the punched-out window, and
scores me only at the holes. Identity returns the window untouched, predicting zero at every masked
position; after standardisation zero is the channel mean, so identity is really "guess the mean everywhere
a value is missing" — the dumbest non-degenerate imputer, ignoring the timestep, the neighbours, and the
other channels. Anything that learns even the crudest temporal regularity should beat it. So the question
for the first real rung is not "what is the most powerful model" but "what is the simplest thing that
actually uses the context the metric rewards" — a floor-setting baseline I understand completely before I
reach for anything intricate.

What context is even available at a masked entry decides what model is appropriate. The window is
`[B, 96, C]` — batch, 96 timesteps, C channels — with entries deleted independently at probability 0.25,
so per channel I lose in expectation 0.25 × 96 = 24 of the 96 timesteps and keep 72. What matters is not
the count of holes but their arrangement. A given masked entry has both immediate temporal neighbours
observed with probability 0.75 × 0.75 = 0.5625 — more than half of all holes are isolated islands of one
missing value in a sea of observed ones — and runs of consecutive holes are short: a maximal masked run is
geometric with continuation probability 0.25, so its expected length is 1/(1 − 0.25) = 1.33 timesteps and
a run of three or more has probability only 0.25² = 0.0625. So the typical thing I reconstruct is a value
sitting between observed values one and two steps away, not a long blackout I must hallucinate across.
Almost every gap is *interior* with observed values on both flanks — which tells me the right first model
is an interpolator, one with a direct, short path from the surrounding observed timesteps to the hole.

To fill a hole at channel *c*, time *t* I have two reservoirs: the temporal one — channel *c*'s other 95
timesteps, most observed — and the cross-channel one — the other channels at the same time *t*. The metric
rewards using either; the cheapest, highest-leverage one is temporal, and the run-length arithmetic is
exactly why: with each gap surrounded by observed neighbours of its own channel, the within-channel record
already pins the missing value tightly. A daily cycle makes the value at *t* well predicted by values one
period earlier and later; a slow trend makes it well predicted by local extrapolation — both *linear*
functionals of the rest of the window along time. So the minimal model that uses the rewarded context is a
single linear map along the time axis, applied per channel: read all 96 timesteps of a channel, emit a
reconstructed 96. That is the whole hypothesis I want to test first.

The obvious move would be to reach straight for a Transformer, so I should be honest about why I resist. In
the forecasting literature next door, an embarrassingly simple linear map along time — with one trick —
matches or beats a whole stack of attention-based models. Attention's core operation is
permutation-invariant: shuffle the input tokens and you get the same multiset of pairwise scores. That is
fine for language, where a word carries standalone meaning, but a single timestep of a numeric series
carries almost none — its entire content is its position in the order. An architecture whose core throws
away order, re-injecting it only as an additive positional code, works against the grain of data whose
signal *is* the order. A linear map along time reads the whole window through learned weights, sees the
shape directly, and gives a length-one signal path from any observed timestep to any reconstructed one — no
recurrence to forget through, no attention bottleneck. For imputation the argument is stronger than for
forecasting, because the masked positions are interior with observed values on both sides: I am
interpolating, not extrapolating, and interpolation is exactly what a linear functional of the surrounding
points does well. So rung one commits to the linear map and finds out how much of the accuracy the
elaborate machinery is actually responsible for.

Now the one trick. A bare single linear map `W·x`, `W` a 96×96 matrix, breaks the moment a window carries a
strong trend together with its seasonality — the common case on the hourly electricity and temperature
series. Picture a window as a big slow ramp with a small daily oscillation riding on it. A single `W` has
to fit both at once, and they want opposite weight patterns: the trend wants smooth weights that extrapolate
a drift, the seasonality wants weights concentrated at the periodic lags. Worse, the trend is large and the
seasonal small, so in a squared-error fit the trend dominates the gradient. Concretely, if the trend
amplitude exceeds the seasonal by a factor *r* — comfortably more than 2 on these series — squared error
weights each component by its *squared* amplitude, so the trend claims on the order of *r²* of the gradient
budget, a 4-to-1-or-worse split toward getting the ramp right. That is exactly backwards for imputation: at
an interior hole the *fine* seasonal shape is what distinguishes the true value from the local average, and
one `W` under this loss polishes the small seasonal term with whatever gradient is left over.

The fix is the oldest move in time-series analysis: seasonal-trend decomposition. Split the window
additively into a slow trend part and a residual seasonal part, each more regular than the sum. The library
already ships a moving-average decomposition block, `series_decomp`: estimate the trend by a
length-preserving moving average, take the residual as the seasonal part. I use it as fixed, parameter-free
preprocessing in front of the linear maps: `trend = MovingAvg(x)`, `seasonal = x − trend`. Give each stream
its own linear layer — `W_trend` and `W_seasonal`, both 96→96 along time — let each specialise, neither
contaminated by nor gradient-swamped by the other, then sum the two predicted streams to recombine.

I should not fool myself that this adds capacity. Two linear maps plus a sum is affine, the moving average
is itself linear, so end to end this is still a single affine map: with `M` the moving-average matrix, the
output is `W_s(I − M)x + W_t Mx = (W_s(I − M) + W_t M)x`, one 96×96 matrix. The function class is unchanged.
What I have added is *conditioning* — a fixed reparameterisation that separates the loud trend from the
quiet seasonality so gradient descent on the masked MSE fits each well instead of letting the trend's
magnitude dominate. Preconditioning, not depth — the same minimal-capacity spirit as the rung — and it pays
off precisely on the trended hourly series where a bare map would under-fit the oscillation.

There is a design decision inside "a linear layer per stream": should the maps be *shared* across channels
or should each channel get its own pair? The `individual` flag exposes both, and parameter counts settle
it. Shared: two 96→96 layers with bias, 2 × (96 × 96 + 96) = 18,624 weights, regardless of channel count.
Individual: that times the channel count — 7 × 18,624 ≈ 130k on ETTh1, tolerable, but 321 × 18,624 ≈ 5.98M
on ECL, each per-channel matrix trained on the gradient from a single client only. That is the wrong trade
twice: 5.98M parameters fit from windows of one 321-channel series overfits the training windows, and more
fundamentally "fill an interior gap from surrounding observed values" is *the same operator* for every
channel — a daily cycle is a daily cycle whether client 12 or client 300 — so tying weights across channels
lets all 321 clients pour their gradient into one shared map, both regularising and sharpening it. So I set
`individual=False`: channel-shared maps, 18,624 parameters. Whatever accuracy this buys is bought by the
*idea* — decompose then linearly interpolate — not by a parameter count that scales with the dataset.

The initialisation connects the rung back to the floor I am trying to beat. I set both weight matrices to a
uniform 1/96, so at initialisation each output timestep is the average of all 96 input timesteps of its
stream: the trend output starts as the mean of the trend, the seasonal as the mean of the seasonal (near
zero), so the model's first prediction is essentially "the window's own average," close to the channel-mean
guess the identity floor makes. That is deliberate — gradient descent starts *at* the mean-predictor and
walks toward a real interpolator, so the trajectory begins from the known floor and the first gradient step
refines a sensible estimate rather than recovering from noise.

The moving average's boundary behaviour matters at the edges, where I have least information. `series_decomp`
uses an odd kernel (25, the library default and a sensible sub-daily smoothing scale on hourly data) with
stride 1 and replicate-pads the endpoints — front with the first value, back with the last — so the trend
stays flat-but-faithful at the edges instead of being dragged toward zero by zero-padding, and (25−1)/2 = 12
on each side restores the length to 96 exactly. I reuse the block as-is.

One point specific to *this* task, not the forecasting version, so I derive it rather than copy it: the
moving average is computed on a window that already has zeros punched in. A 25-wide averaging window near a
hole contains in expectation 25 × 0.25 = 6.25 fake zeros, so its trend estimate is roughly 18.75 observed
values summed plus 6.25 zeros over 25 — about 0.75 times the true local level. The mask *does* shrink the
trend toward zero by about a quarter near holes. But the seasonal residual is `x − trend`, so wherever the
trend is shrunk by δ the seasonal picks up exactly δ; the two streams still sum entry-by-entry back to `x`,
and the two linear maps, summed at the end, can jointly represent the true reconstruction regardless of how
the low-frequency energy was apportioned. With 25% missing and a 25-wide kernel each window still contains
mostly observed values, so the shrinkage is mild and, crucially, absorbed rather than lost — the
parameter-free decomposition is safe to run directly on the masked input.

The other task-specific point is what I deliberately do *not* do. In forecasting the normalisation trick is
"subtract the last observed value, predict the de-levelled continuation, add it back." That is wrong here:
this is imputation with `pred_len = seq_len`, the masked positions are interior, there is no single "current
level" to subtract, and I must never let the fill depend on the punched-out values as if they were real —
the de-levelling step would pick a reference off the sequence and there is no principled interior reference.
The standardised, mean-zero input is already on a common baseline, and the affine maps can represent any
residual offset through their biases. So the imputation path is just decompose → two linear maps along time
→ sum, no normalisation wrapper: `x_enc` arrives `[B, 96, C]`, `series_decomp` returns seasonal and trend
each `[B, 96, C]`, I permute each to `[B, C, 96]` so the map acts along the time axis, apply `Linear(96→96)`
to each stream, sum, and permute back to `[B, 96, C]` — same length in as out, mask never consulted inside
the model since the masked-MSE loss scores it outside.

One structural fact reassures me the rung can only improve on the baseline: the model contains the identity
floor as a single point in its parameter space — set both `W` to the identity and the sum is
`seasonal + trend = x` exactly, reproducing the masked input — so training can only walk away from the floor
toward a genuine interpolator, never do worse in the limit.

One honest limitation names what the next rung must fix. This model is deliberately channel-blind: the maps
are shared across channels and act only along time, so channel *c* is reconstructed from its own observed
timesteps and nothing from the correlated channels at the same instant. On ECL, with 321 co-moving clients,
that leaves real information on the table — a masked client's simultaneous neighbours are a strong predictor
a per-channel temporal map cannot see. I am betting that within-channel temporal interpolation alone already
crushes the mean-guessing identity, and that cross-channel coupling is the *next* lever, not this one.

So rung one is settled: a parameter-free seasonal-trend split feeding two channel-shared linear maps along
time, summed, trained by the masked MSE. A priori it should beat the identity decisively everywhere, and
struggle most where ignoring cross-channel structure costs the most (ECL) and where an interior miss is
genuinely nonlinear in its neighbours (sharp local transients an affine map of the window smears).
Whichever way the numbers fall, the blind spot is already named: the same-time cross-channel context this
map cannot use.
