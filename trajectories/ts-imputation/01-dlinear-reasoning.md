The identity fill is the floor, and it is a useful floor to stare at, because it tells me exactly what
the metric is asking for and exactly why doing nothing is wrong. The loop zeroes a quarter of the
entries, hands me the punched-out window, and scores me only at the holes. The identity model returns
the window untouched, which means at every masked position it predicts zero. After standardisation zero
is the channel mean, so the identity is really "guess the mean everywhere a value is missing." That is
the dumbest non-degenerate imputer: it ignores the timestep, it ignores the neighbours, it ignores the
other channels. Anything that learns even the crudest temporal regularity should beat it. So the
question for the first real rung is not "what is the most powerful model" but "what is the simplest
thing that actually uses the context the metric is rewarding," because I want a floor-setting baseline I
understand completely before I reach for anything intricate.

Let me reason about what context is even available at a masked entry, and let me put numbers on it,
because the geometry of the mask decides what kind of model is even appropriate. The masked window is
`[B, 96, C]` — batch, 96 timesteps, C channels — with zeros punched in at random (timestep, channel)
positions, plus the binary mask telling me which is which. Each entry is deleted independently with
probability 0.25, so per channel I lose in expectation 0.25 × 96 = 24 of the 96 timesteps and keep 72.
The question that matters for imputation is not how many holes there are but how they are *arranged*.
If deletions are independent per timestep, then a given masked entry has both immediate temporal
neighbours observed with probability 0.75 × 0.75 = 0.5625 — more than half of all holes are completely
isolated, an island of one missing value in a sea of observed ones. And the runs of consecutive holes
are short: the length of a maximal masked run is geometric with continuation probability 0.25, so its
expected length is 1/(1 − 0.25) = 1.33 timesteps, and a run of length three or more has probability only
0.25² = 0.0625. So the typical thing I am asked to reconstruct is a value sitting between two observed
values one and two steps away, not a long blackout I have to hallucinate across. That single fact —
almost every gap is *interior* with observed values on both flanks — is what tells me the right first
model is an interpolator, and specifically that a model with a direct, short path from the surrounding
observed timesteps to the hole will do most of the work.

To fill a hole at channel *c*, time *t*, I have two reservoirs of information: the *temporal* one — the
other 95 timesteps of channel *c*, most of them observed — and the *cross-channel* one — the other
channels at the same time *t*. The metric rewards using either. The cheapest, highest-leverage one is
temporal, and the run-length arithmetic above is exactly why: with each gap surrounded by observed
neighbours of its own channel, the within-channel record already pins the missing value tightly. If a
channel has a daily cycle, the value at *t* is well predicted by values one period earlier and later in
the same window; if it has a slow trend, it is well predicted by a local extrapolation. Both of those
are *linear* functionals of the rest of the window along time. So the minimal model that uses the
rewarded context is a single linear map along the time axis, applied per channel: read all 96 timesteps
of a channel, emit a reconstructed 96 timesteps. That is the whole hypothesis I want to test first.

I should be honest about why I am even tempted by something this small, because the obvious move would be
to reach straight for a Transformer. The reason is a tension I cannot ignore: in the forecasting
literature next door, an embarrassingly simple linear map along time — with one trick — matches or beats
a whole stack of attention-based models. Attention's core operation is permutation-invariant: shuffle the
input tokens and you get the same multiset of pairwise scores. That is fine for language, where a token
(a word) carries standalone meaning, but a single timestep of a numeric series carries almost none — its
entire content is its position in the order. So an architecture whose core throws away order, re-injecting
it only as an additive positional code, is working against the grain of data whose signal *is* the order.
A linear map along time does the opposite: it reads the whole window through learned weights, so it sees
the shape directly, and the maximum signal path from any observed timestep to any reconstructed one is
length one — no recurrence to forget through, no attention bottleneck. Let me quantify the contrast that
matters for cost as well: a per-channel linear map along time costs on the order of C × 96² multiply-adds
per window, whereas point-wise self-attention over the 96 timesteps costs 96² × d for the score matrix
plus the value projections — the same quadratic 96² in the sequence length, but multiplied by an
embedding width d and stacked over layers, for a model whose core operation is fighting the ordering I
most need to keep. For imputation the argument for the linear map is, if anything, stronger than for
forecasting, because the masked positions are *interior* to the window with observed values on both
sides; I am interpolating, not extrapolating, and interpolation is exactly what a linear functional of the
surrounding points does well. So I will commit to the linear map as rung one and find out how much of the
imputation accuracy the elaborate machinery is actually responsible for.

Now the one trick. A bare single linear map `W·x`, `W` a 96×96 matrix per channel (or shared), has a
problem the moment a window carries a strong trend together with its seasonality — which is the common
case here, especially on the hourly electricity and temperature series. Picture a window as a big slow
ramp with a small daily oscillation riding on it. A single `W` has to fit both at once, and these two
structures want very different weight patterns: the trend wants smooth weights that extrapolate a drift,
the seasonality wants weights sharply concentrated at the periodic lags. Worse, the trend component is
large in magnitude and the seasonal component is small, so in a squared-error fit the trend dominates the
gradient — `W` spends itself getting the big ramp roughly right and under-fits the small oscillation that
carries the fine structure I actually need at the masked positions. One weight matrix is being asked to be
two different filters, and the loss makes it prioritise the loud one.

Let me make the magnitude imbalance quantitative, because "the trend dominates the gradient" is the kind
of claim I should be able to size. On an hourly series the trend component of a 96-hour window — the slow
drift over four days — routinely swings across a range several times larger than the daily oscillation
riding on it; call the ratio of their typical amplitudes *r*, and on the electricity and temperature
series *r* is comfortably more than 2 and often much more. Squared error weights each component by its
*squared* amplitude, so the trend contributes on the order of *r²* times the loss that the seasonal part
does — a 4-to-1 or worse split of the gradient budget toward getting the ramp right. A single `W` fit
under that loss will spend its capacity flattening the trend residual first and only polish the small
seasonal term with whatever gradient is left, which is exactly backwards for imputation: at an interior
hole the *fine* seasonal shape is what distinguishes the true value from the local average, and that is the
signal being starved. That is the concrete mechanism the decomposition defuses — separate the streams and
each linear map fits its own component with its own full gradient, so the seasonal map is no longer
competing against a term *r²* times louder.

The fix is the oldest move in time-series analysis: seasonal-trend decomposition. Split the window
additively into a slow trend part and a residual seasonal part, because each piece on its own is more
regular and more predictable than the sum. I do not have to invent this — the library already ships a
moving-average decomposition block, `series_decomp`: estimate the trend by a length-preserving moving
average of the window, take the residual as the seasonal part. I use it as a fixed, parameter-free
preprocessing in front of the linear maps. Decompose once: `trend = MovingAvg(x)`, `seasonal = x - trend`.
Now the trend stream is the smooth ramp and the seasonal stream is the oscillation around it, cleanly
separated and on comparable footing within each stream. Give each its own linear layer — `W_trend` and
`W_seasonal`, both 96→96 along time — and let each specialise, neither contaminated by the other, neither
gradient swamped by the other's magnitude. Then sum the two predicted streams to recombine.

I want to be careful not to fool myself that I have added capacity. Two linear maps plus a sum is affine;
the moving average is itself linear; so decomposition-then-two-linears-then-sum is, end to end, still a
single affine map from input to output — concretely, if `M` is the moving-average matrix then the output
is `W_s(I − M)x + W_t M x = (W_s(I − M) + W_t M) x`, one 96×96 matrix. I have not enlarged the function
class at all. What I have added is *conditioning*: a fixed reparameterisation that separates the loud
trend from the quiet seasonality so gradient descent on the masked MSE can fit each well instead of
letting the trend's magnitude dominate. It is preconditioning, not depth — the same spirit as the rung
itself, minimal capacity, better-posed learning. And it pays off precisely on the trended hourly series
where a bare linear map would under-fit the oscillation.

There is a design decision hiding inside "a linear layer per stream," and I want to resolve it by
arithmetic rather than taste: should the two maps be *shared* across channels, or should each channel get
its own pair? The scaffold exposes both through an `individual` flag. Count the parameters. Shared: two
96→96 layers with bias, 2 × (96 × 96 + 96) = 2 × 9312 = 18,624 weights total, regardless of how many
channels there are. Individual: multiply that by the channel count. On ETTh1 with 7 channels that is
7 × 18,624 ≈ 130k, tolerable; but on ECL with 321 channels it is 321 × 18,624 ≈ 5.98M weights, and each
of those per-channel matrices is trained on the gradient from a single client only. That is the wrong
trade twice over. First, 5.98M parameters fit from windows of a 321-channel series is a recipe for
overfitting the training windows rather than learning the interpolation operator. Second, and more
fundamental, the whole reason a linear temporal interpolator should generalise is that "fill an interior
gap from surrounding observed values" is *the same operator* for every channel — a daily cycle is a daily
cycle whether it is client 12 or client 300 — so tying the weights across channels lets all 321 clients'
windows pour their gradient into one shared map, which both regularises it and sharpens it. Individual
maps throw that sharing away for a per-channel idiosyncrasy that a bare affine map cannot really capture
anyway. So I set `individual=False`: channel-shared maps, 18,624 parameters, the same interpolation
operator applied independently along time to every channel. That keeps the rung genuinely minimal and
keeps the comparison honest — whatever accuracy this buys is bought by the *idea* (decompose then linearly
interpolate), not by a parameter count that scales with the dataset.

The initialisation is worth pinning down too, because it connects the rung directly back to the floor I am
trying to beat. I initialise both weight matrices to `(1/seq_len)` in every entry — a uniform 1/96. At
initialisation, then, each output timestep is the *average* of all 96 input timesteps of its stream:
`trend` output starts as the mean of the trend, `seasonal` output as the mean of the seasonal (which is
near zero), so the model's first prediction is essentially "the window's own average," which after
standardisation is close to the channel-mean guess that the identity floor makes. That is deliberate: I
am starting gradient descent *at* the mean-predictor and letting it walk away toward a real interpolator,
so the training trajectory begins from the known floor and can only improve on it. It also means the very
first gradient step is informative rather than random — the model is refining a sensible estimate, not
recovering from noise.

The moving average needs its boundary behaviour pinned down, because the window edges are where I have the
least information and the decomposition can misbehave there. `series_decomp` uses an odd kernel (here 25,
the library default and a sensible sub-daily smoothing scale on hourly data) with stride 1, and it
replicate-pads the endpoints — front padded with copies of the first value, back with copies of the last —
before average pooling, so the trend stays flat-but-faithful at the edges instead of being dragged toward
zero by zero-padding. With kernel 25 and (25−1)/2 = 12 on each side, the length is restored exactly to 96.
I reuse the block as-is; keeping the kernel identical to the established decomposition keeps the comparison
clean.

Now the part that is specific to *this* task and is not in the forecasting version of the method, so I
have to derive it rather than copy it — and I want to check whether the mask corrupts the decomposition,
because the moving average is computed on a window that already has zeros punched into it. Consider a
25-wide averaging window sitting over the series near a hole. In expectation 25 × 0.25 = 6.25 of those 25
values are fake zeros rather than real observations, so the trend estimate at that point is roughly
(18.75 observed values, summed, plus 6.25 zeros) divided by 25 — that is, about 0.75 times the true local
level. So the mask *does* shrink the trend toward zero by about a quarter near any region of holes. Is
that fatal? No, and here is the check: the seasonal residual is `x − trend`, so wherever the trend is
shrunk by δ, the seasonal picks up exactly that δ. The two streams still sum, entry by entry, back to the
same masked input `x`, so no information is destroyed by the decomposition — the split just moves some
low-frequency energy from the trend stream into the seasonal stream near holes, and the two linear maps,
which are summed at the end, can jointly represent whatever the true reconstruction is regardless of how
the energy was apportioned. With only 25% missing and a 25-wide kernel, each averaging window still
contains mostly observed values, so the shrinkage is mild and, crucially, it is *absorbed*, not lost.
That reassures me the parameter-free decomposition is safe to run directly on the masked input.

The other task-specific point is what I deliberately do *not* do. In forecasting the normalisation trick
is "subtract the last observed value, predict the de-levelled continuation, add it back." That is wrong
here for two reasons. First, this is imputation: `pred_len = seq_len`, the masked positions are interior,
and there is no single "current level" to subtract — the masked window already has its statistics handled
by the pipeline's standardisation, and the masked entries are already zero. Second, and more importantly,
I must never let the imputation depend on the punched-out values as if they were real; the de-levelling
step would pick a reference off the sequence and there is no principled interior reference here. The
standardised, mean-zero input is already on a common baseline, and the affine linear maps can represent
any residual offset they need through their biases. So the imputation path is just `decompose → two linear
maps along time → sum`, with no normalisation wrapper — the literal scaffold edit, distilled in the
answer. Let me trace the shapes once to be sure the wiring is right: `x_enc` arrives `[B, 96, C]`;
`series_decomp` returns seasonal and trend each `[B, 96, C]`; I permute each to `[B, C, 96]` so the linear
map acts along the last (time) axis, apply `Linear(96 → 96)` to each stream to get `[B, C, 96]`, sum them,
and permute back to `[B, 96, C]` — the dense reconstruction the contract asks for, same length in as out,
mask never consulted inside the model because the masked-MSE loss outside will do the scoring.

Before I commit, I want one concrete verification that the summed-affine construction can actually do the
job I am claiming, rather than trusting the intuition. Take the degenerate limit first: what is the
smallest thing this model must be able to represent? It must contain the identity, because the identity is
the floor and a model that could not even reproduce its input would be a regression. Set `W_seasonal` and
`W_trend` both to the identity matrix; then `seasonal` maps to `seasonal`, `trend` maps to `trend`, and
their sum is `seasonal + trend = x` exactly — the model reproduces the masked input, which at observed
positions is the truth and at holes is zero, i.e. the identity floor. So the floor is a single point in
this model's parameter space, and everything the training does is walk away from it toward a genuine
interpolator; the rung can only improve on the baseline it is meant to beat, never do worse in the limit.
Now a positive micro-trace to see the interpolation actually happen. Imagine a channel whose window is a
pure ramp `x_t = t` for t = 0..95 with the single interior entry t = 48 masked (set to 0). A moving
average of a ramp is again a ramp (up to edge effects), so `trend ≈ x` and `seasonal ≈ 0` everywhere
except a downward spike at t = 48 where the hole makes the raw input dip to 0. If `W_trend` learns the
identity and `W_seasonal` learns a smoother that pulls each point toward the average of its neighbours,
then at t = 48 the trend contributes ≈ 48 and the seasonal contributes ≈ 0 (its spike is averaged out
against the flanking zeros of the residual), so the reconstruction lands near 48 — the true value — even
though the input there was 0. A single undecomposed map would have to fit the ramp and undo the spike with
one matrix; splitting lets the trend carry the ramp untouched while the seasonal map does the local
smoothing that fills the dip. The trace confirms the mechanism: interior holes get filled by the linear
combination of their observed neighbours, and the decomposition keeps the loud ramp from stealing the
capacity the fill actually needs.

One more honest limitation to name, because it sets up what the next rung will have to fix. This model is
deliberately *channel-blind*: the linear maps are shared across channels and act only along time, so the
reconstruction of channel *c* uses only channel *c*'s own observed timesteps and nothing from the
correlated channels at the same instant. On ECL, with 321 strongly co-moving electricity clients, that
is leaving real information on the table — if a client's value is masked at time *t*, the simultaneous
values of the hundreds of correlated clients are a strong predictor that a per-channel temporal map cannot
see. I am betting that within-channel temporal interpolation alone already crushes the mean-guessing
identity, and that the cross-channel coupling is the *next* lever, not this one. That bet is exactly what
the measured numbers will adjudicate.

So rung one is settled: a parameter-free seasonal-trend split feeding two channel-shared linear maps along
time, summed, trained by the loop's masked MSE — the full scaffold module is in the answer. What must this
floor do? It must beat the identity decisively everywhere, because mean-guessing ignores temporal context
entirely and this model uses it directly through a length-one signal path. Where it should *struggle* is
precisely the high-channel-count, strongly-correlated regime — ECL — where ignoring cross-channel structure
costs the most, and on any dataset where the missing-at-interior structure is genuinely nonlinear in the
neighbours (sharp local transients that an affine map of the window smears). I expect a clean, large gap
over the identity on all three datasets and the *largest residual error on ECL relative to what a
cross-channel model could reach* — and that residual is the signal that the next rung must stop being
channel-blind. Whatever the precise numbers, the diagnosis is already pointed: a linear temporal
interpolator is the right floor, and its blind spot is the same-time cross-channel context it cannot use.
