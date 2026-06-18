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

Let me reason about what context is even available at a masked entry. The masked window is
`[B, 96, C]` — batch, 96 timesteps, C channels — with zeros punched in at random (timestep, channel)
positions, plus the binary mask telling me which is which. To fill a hole at channel *c*, time *t*, I
have two reservoirs of information: the *temporal* one — the other 95 timesteps of channel *c*, most of
them observed — and the *cross-channel* one — the other channels at the same time *t*. The metric
rewards using either. The cheapest, highest-leverage one is temporal: with only 25% of entries deleted
uniformly at random, every channel still has roughly three out of four of its timesteps intact, so each
gap is surrounded by observed neighbours of its own channel. If a channel has a daily cycle, the value
at *t* is well predicted by values one period earlier and later in the same window; if it has a slow
trend, it is well predicted by a local extrapolation. Both of those are *linear* functionals of the
rest of the window along time. So the minimal model that uses the rewarded context is a single linear
map along the time axis, applied per channel: read all 96 timesteps of a channel, emit a reconstructed
96 timesteps. That is the whole hypothesis I want to test first.

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
length one — no recurrence to forget through, no attention bottleneck. For imputation the argument is, if
anything, stronger than for forecasting, because the masked positions are *interior* to the window with
observed values on both sides; I am interpolating, not extrapolating, and interpolation is exactly what a
linear functional of the surrounding points does well. So I will commit to the linear map as rung one and
find out how much of the imputation accuracy the elaborate machinery is actually responsible for.

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
single affine map from input to output. I have not enlarged the function class at all. What I have added
is *conditioning*: a fixed reparameterisation that separates the loud trend from the quiet seasonality so
gradient descent on the masked MSE can fit each well instead of letting the trend's magnitude dominate. It
is preconditioning, not depth — the same spirit as the rung itself, minimal capacity, better-posed
learning. And it pays off precisely on the trended hourly series where a bare linear map would under-fit
the oscillation.

The moving average needs its boundary behaviour pinned down, because the window edges are where I have the
least information and the decomposition can misbehave there. `series_decomp` uses an odd kernel (here 25,
the library default and a sensible sub-daily smoothing scale on hourly data) with stride 1, and it
replicate-pads the endpoints — front padded with copies of the first value, back with copies of the last —
before average pooling, so the trend stays flat-but-faithful at the edges instead of being dragged toward
zero by zero-padding. With kernel 25 and (25−1)/2 = 12 on each side, the length is restored exactly to 96.
I reuse the block as-is; keeping the kernel identical to the established decomposition keeps the comparison
clean.

Now the part that is specific to *this* task and is not in the forecasting version of the method, so I have
to derive it rather than copy it. In forecasting the normalisation trick is "subtract the last observed
value, predict the de-levelled continuation, add it back." That is wrong here for two reasons. First, this
is imputation: `pred_len = seq_len`, the masked positions are interior, and there is no single "current
level" to subtract — the masked window already has its statistics handled by the pipeline's
standardisation, and the masked entries are already zero. Second, and more importantly, I must never let
the imputation depend on the punched-out values as if they were real. The input already has zeros at the
holes; the decomposition's moving average will average those zeros in along with the real neighbours, which
slightly biases the trend toward zero near a gap — but with only 25% missing and a 25-wide kernel, each
window of the average still contains mostly observed values, so the bias is small and the seasonal residual
absorbs the rest. I deliberately do *not* add a de-levelling step: the standardised, mean-zero input is
already on a common baseline, and the affine linear maps can represent any residual offset they need. So
the imputation path is just `decompose → two linear maps along time → sum`, with no normalisation wrapper —
the literal scaffold edit, distilled in the answer.

One more honest limitation to name, because it sets up what the next rung will have to fix. This model is
deliberately *channel-blind* by default: the linear maps are shared across channels and act only along
time, so the reconstruction of channel *c* uses only channel *c*'s own observed timesteps and nothing from
the correlated channels at the same instant. On ECL, with 321 strongly co-moving electricity clients, that
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
