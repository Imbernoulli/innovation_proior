I am starting the ladder, so there is no prior result to react to — only the scaffold and the lineage
it sits in. The scaffold hands me a frozen direct-multi-step loop and one empty class to fill, and the
question I am actually being asked is not "what is the best forecaster" but "what is the right place to
begin." I want the first rung to be the most honest possible control: the simplest architecture that
could plausibly forecast at all, built precisely so that whatever the later, heavier models add can be
measured against something whose every part I understand. So I begin from a doubt rather than from an
architecture.

The doubt is about the whole attention-based forecasting lineage that the initial context lays out —
Informer, Autoformer, FEDformer. Each is a clever surgery on the attention kernel, each reports beating
the last on the long-horizon benchmarks, and the accuracy keeps creeping up. And yet I know what
attention *is* and I know what a numeric time series *is*, and the two do not obviously fit. Attention's
whole job is to score pairs of tokens by content and aggregate them by a softmax over those scores:
softmax(QKᵀ/√d)V. That operation is permutation-invariant — shuffle the input tokens and you get the
same multiset of pairwise scores and the same outputs up to the matching permutation. For language and
vision that is a feature, because a word embedding or an image patch carries standalone semantic
content and order can be re-injected as a positional side channel. But stare at a single number in a
load trace: 0.42. It means nothing by itself. There is no point-wise semantics in 0.42 the way there is
in the word "cat." The only thing carrying the signal is the order — that this value sits between the
previous hour and the next, on the rising shoulder of the evening peak, the same shape that recurred 24
and 168 hours ago. So the lineage is asking me to believe that an operator whose core throws away order,
and re-injects it only as an additive encoding, is the right tool for data whose entire content *is*
order. That is working against the grain.

Two observations about the deployed Transformer forecasters turn this hunch into a real doubt. First:
feed them a *longer* look-back window — more context — and their error does not drop, and in several
cases it gets worse. That is backwards. A model that genuinely extracts temporal structure should
improve when handed more of it; error that stalls or climbs as the window grows is the fingerprint of a
model latching onto something local and being confused by the rest. Second, there is a confound in how
those wins were obtained. In each of those papers the non-Transformer baselines they beat were iterated
multi-step forecasters — learn a one-step model, feed its own prediction back in, roll it forward T
times. Iterated prediction has a known disease over long horizons: each step is fed the previous step's
error and those errors compound, so by step 96 the input is largely the model's own accumulated
mistakes. The Transformers, meanwhile, use direct multi-step — one map straight from history to the
whole horizon, no feedback loop, no compounding. So when a direct Transformer beats an iterated RNN or
ARIMA, I cannot tell how much of the win is the attention and how much is just that one side was
handicapped by error accumulation and the other was not. The comparison conflates "Transformer" with
"direct multi-step," which is exactly the strategy the scaffold loop already fixes for everyone.

If that is the real situation, the way to find out is not to build a better Transformer but to build the
simplest thing that *removes the confound* and see how much of the reported gap survives. Strip away the
attention entirely and keep only the part I suspect is doing the work — the direct multi-step strategy,
which the loop hands me for free. What is the most minimal direct multi-step forecaster I can write
down? It takes a length-L history and emits a length-T horizon in one shot, no recursion. The floor of
that is a single affine map along the time axis. For one variate, predict the horizon as a weighted sum
of the look-back: X̂ = W·X, with W a T×L matrix. Row t of W says how to combine the L past values into
the t-th future value. One linear layer, applied along time. That is it.

Before I even write it down I want to be sure it is the right floor, so I line up the three genuinely
minimal direct-multi-step forecasters I could pick and knock two out on their own terms. The first is a
small recurrent net — a GRU that ingests the window and rolls the horizon out — but rolling out means
feeding its own step-t prediction back to produce step t+1, and over ninety-six steps that is exactly the
compounding-error disease I already indicted the iterated baselines for; even trained teacher-forced it
carries the wrong inductive bias for a one-shot horizon, so it is out. The second is a shallow
self-attention over the ninety-six step-tokens, the lineage's reflex shrunk to minimal size — but it
costs ninety-six squared pairwise scores, needs a positional encoding bolted on to undo the permutation
invariance it should not have on ordered data, and by the confound argument its apparent edge might be
nothing but the direct strategy it shares with everything under this loop; it is neither minimal nor
free of the against-the-grain problem, so it is out too. The third is a single affine map read straight
across the window, and it survives both tests: no feedback loop, no permutation pathology, the fewest
parameters of the three. So the floor is the linear map, and I build it.

Before I take that seriously I have to check it is not trivially crippled. The reflex objection is "it
is linear, it cannot model anything." But consider what a single linear row can express. If the series
has a period p — 24 for daily, 168 for weekly — then a future value is, to first order, well predicted
by the value one period back in the look-back, and a row of W can place its weight on exactly that lag
and read it off. If the series has a slow trend, a future value is well predicted by an extrapolation of
recent values, which is a weighted difference — also a linear combination of the look-back. Periodicity
and trend are exactly the two kinds of structure a linear temporal map captures naturally, because both
are at heart "the future is a fixed linear functional of the recent past." And here is the hypothesis
that makes me think this is sufficient rather than a strawman: long-term forecasting is only feasible at
all for series with relatively clear trend and periodicity — a genuinely chaotic series is not
long-horizon-predictable by anything. So if the only forecastable structure in the long-horizon regime
is trend plus periodicity, and a linear map captures trend plus periodicity, then a linear map should
already capture most of what is forecastable, while a high-capacity nonlinear model would mostly be
fitting noise that does not cross the train/test boundary. There is also a structural nicety: in this
map every input step connects directly to every output step with a learned weight, so the longest
signal path from any past observation to any future prediction is length one. No recurrence to forget
through, no attention bottleneck — the long-range dependency lives in the weight matrix itself.

I can make the not-crippled claim fully concrete rather than leave it as a hunch. Put the daily period at
twenty-four on hourly data and the look-back at ninety-six — four days. To predict future step t,
counting one to ninety-six past the window end, the sharpest periodic estimate is the value one period
earlier, at absolute index (96 + t) − 24 = 72 + t. For every t from one to twenty-four that index lands
in [73, 96], squarely inside the window, so row t of W simply places its weight on column 72 + t and
copies the value off — one-period-back prediction is literally one nonzero entry per row. For t from
twenty-five to forty-eight the one-period-back index runs past the window end, but the two-periods-back
index 48 + t sits in [73, 96] and the row reads that instead; the map degrades gracefully to the nearest
in-window period rather than failing. So a single linear row expresses "copy the value k periods back"
exactly, for whatever k keeps the lag in range, and superposing such rows over the L columns expresses
any fixed linear combination of periodic lags plus a slope extrapolation for the trend. That is not a
crippled hypothesis class for a signal whose forecastable content is trend and periodicity; it is a tight
fit to it.

So I commit to the minimal model: one linear layer L→T along time, trained by the loop's direct MSE over
the whole horizon. One decision falls out immediately. There are C channels; do I learn a W per channel
or share one W? Within a single dataset the channels usually share temporal dynamics — every electricity
client has the same daily and weekly rhythm, every weather sensor shares the same diurnal cycle — so a
shared W applied identically to each channel both encodes that prior and slashes the parameter count
from C·T·L to T·L, which matters for overfitting on the smaller datasets. I share weights across
channels by default and deliberately model no cross-channel coupling at all. That is a real bet: the
heavy models spend capacity learning which channels influence which, and I am wagering that on these
benchmarks that coupling is weak or spurious and that ignoring it avoids overfitting. The point is a
clean control, so I want the fewest moving parts that can still represent the forecastable structure.

The parameter arithmetic makes the stakes of that bet exact. One shared W is pred_len × seq_len =
96 × 96 = 9216 weights plus a length-96 bias, and two of them — one per decomposed stream — is roughly
eighteen thousand parameters total, a count that does not grow with the channel dimension at all. The
per-channel alternative multiplies it by C: about a hundred and thirty thousand parameters on the
seven-channel data, four hundred thousand on the twenty-one-channel data, and close to six million on
the three-hundred-and-twenty-one-channel data. Six million weights fit against a single standardized
training split is exactly the regime where a model memorizes the training era and generalizes worse, and
the channel dimension where the per-channel map balloons most is precisely the one whose coupling I trust
least. So sharing W is not only the cleaner prior about a shared diurnal rhythm; it is the decisive
regularizer on the wide datasets. I default to shared and treat individual as an option I do not
exercise.

Now I run the bare linear map against the dataset feature I know will bite it: distribution shift. On
the hourly transformer-temperature data the *level* of the series drifts between the training span and
the test span — the mean is simply higher or lower later in time. A single W fit on training-era levels
will faithfully map a shifted test window to a shifted-and-wrong horizon, with no way to recenter. The
fix is cheap and I can see it without agonizing, but I also notice the scaffold loop already standardizes
each channel using statistics fit on the train split, which absorbs the *global* level but not the
within-window drift; for the parts the loader does not absorb, subtracting the look-back's own mean
before the map and adding it back after would recenter each window. I keep that in mind but do not bolt
it on, because the loader's standardization already covers most of it and I want the rung minimal.

It is worth being precise about which part of the drift is already handled, because it decides how much I
must add. The loader standardizes each channel by a mean and standard deviation fit once on the training
span, so it removes the constant offset between, say, a low-consumption client and a high-consumption one
and puts every channel on a comparable scale before the map ever sees it. What it cannot remove is drift
*within* the test era relative to the training era: if later windows sit systematically above the
training mean the standardization was fit on, the map receives an input shifted by that residual and,
being a fixed linear operator, returns a horizon shifted by W times that residual — wrong, with no
mechanism to recenter. Subtracting each window's own look-back mean before the map and adding it back
after would zero out exactly that residual for the price of two cheap operations, and I file it as the
obvious next handle. But since the loader absorbs the bulk of the level problem and the rung's whole
value is being the minimal understood control, I leave the model at two linear maps over a decomposed
window and let the heavier rungs decide whether per-window recentering earns its keep.

The more interesting weakness is trend *together with* seasonality, which is the common case. Picture
the signal as a big slow ramp with a small daily oscillation riding on top. A single W must fit both at
once, and the two want very different weight patterns: the trend wants weights that extrapolate a slow
drift — broadly smooth across the look-back — while the seasonality wants weights sharply concentrated
at the periodic lags. Worse, the trend is large in magnitude and the seasonal part is small, so in a
squared-error fit the trend dominates the gradient: W spends itself getting the big ramp roughly right
and under-fits the small oscillation that carries the fine structure. One matrix is being asked to be
two different filters and the loss makes it prioritize the loud one.

I can put a number on how badly. Write the window as trend plus season with amplitudes A_t and A_s, and
suppose the trend is a full order of magnitude larger, A_t ≈ 10·A_s — routine on the hourly
transformer-temperature data where a slow ramp of several units rides over a daily wiggle of a few
tenths. Squared error weights each part by its magnitude squared, so the trend contributes on the order
of a hundred times the seasonal part to the loss and to the gradient that trains the single W. Descent
therefore pours essentially all of its budget into getting the ramp roughly right and treats the
oscillation as rounding error, under-fitting the very structure that carries the daily rhythm. Split the
streams first and the arithmetic inverts: the seasonal map sees a target of amplitude A_s everywhere and
a gradient scaled to A_s, no longer swamped, while the trend map fits its own single-scale target — two
well-conditioned regressions in place of one that the loud component captured. So how do I let them
specialize?

Here a tool already in this literature becomes the obvious move. Seasonal-trend decomposition is the
oldest idea in time-series analysis: write the series additively as a slow trend-cyclical part plus a
seasonal/remainder part, because each piece on its own is more regular and more predictable than the
sum. Autoformer already wired a moving-average decomposition block into its network — estimate the trend
by a moving average, take the residual as the seasonal part — and the scaffold exposes exactly that
block as `series_decomp` in `layers.Autoformer_EncDec`. I do not need to invent it; I use it as a fixed,
parameter-free preprocessing in front of my linear maps. Decompose the window once: trend =
MovingAvg(x), seasonal = x − trend. The trend stream is the smooth ramp and the seasonal stream is the
oscillation, cleanly separated and on comparable footing within each stream. Give each its own linear
map — Linear_Trend and Linear_Seasonal, both L→T — and let each specialize, neither contaminated by the
other and neither's gradient swamped by the other's magnitude. Then sum the predicted streams:
X̂ = W_seasonal·seasonal + W_trend·trend.

I pause on whether the split should be learned rather than fixed. I could make the trend extractor a
learnable low-pass filter, or reach for a classical seasonal-trend decomposition that iteratively refines
both parts. But a learnable decomposition adds parameters and, worse, adds a second thing that can overfit
inside a control whose entire purpose is to have no moving parts I do not understand; and an iterative
classical decomposition is neither differentiable end-to-end nor cheap to run inside the loop. A single
fixed moving average is parameter-free, differentiable, and exactly the reparameterization I argued for —
it conditions the optimization without enlarging the function class. So the fixed block is not a
compromise forced by laziness; it is the only choice consistent with the rung being a clean control, and
I take the version the scaffold already exposes so that I am reusing machinery rather than introducing a
knob.

I should be honest about one thing, because it looks like I just made the model "deeper" and that would
betray the point. Is this still linear? Two linear maps plus a sum are affine; the only added operation
is the moving-average split, and a moving average is itself linear, so decomposition followed by two
linear maps and a sum is end-to-end still a single affine map. I have added no representational capacity
in the function-class sense. What I have added is *conditioning*: a fixed, parameter-free
reparameterization that separates the loud trend from the quiet seasonality so gradient descent on the
squared error fits each well instead of letting the trend's magnitude dominate. It is the same trick as
preconditioning an optimization — same solution set, far better-behaved learning — and it pays off
exactly when there is a clear trend, the case the bare linear map handled worst.

I should verify the two claims I just leaned on — that nothing is lost and that nothing is added — rather
than assert them. Reconstruction first: season is defined as x minus its moving average and trend as the
moving average, so season plus trend is x identically, an exact split with no residual. Take a four-step
window x = [1, 2, 3, 4] and a length-three replicate-padded average: pad to [1, 1, 2, 3, 4, 4], slide
width-three means to get trend = [1.33, 2, 3, 3.67], and season = x − trend = [−0.33, 0, 0, 0.33]; season
+ trend returns [1, 2, 3, 4] exactly, the trend the smooth part and the season the zero-sum detail.
Function class second: the moving average is a fixed linear operator M, so trend = M·x and season =
(I − M)·x, and the full model W_s·(I − M)·x + W_t·M·x collapses to a single matrix
(W_s(I − M) + W_t·M) acting on x — one T×L affine map, exactly the hypothesis class of the bare linear
model. So I have added zero representational capacity and only reparameterized the optimization, which is
the whole point: the same solutions remain reachable, far better conditioned to reach the one that fits
both scales at once.

The moving-average details decide the edge behavior. I want the trend to have the same length as the
input, so I average-pool with an odd kernel of size k, stride 1, and pad. Zero-padding would drag the
trend toward zero at the two ends, creating spurious dips where I have least information — the wrong
boundary behavior. Replicate-padding instead — front with (k−1)/2 copies of the first value, back with
(k−1)/2 copies of the last — keeps the trend flat-but-faithful at the edges; with k odd, (k−1)/2 per
side restores the length exactly. I can see the failure the replicate choice avoids on that same
four-step window: zero-padding the front would make the first averaged value (0 + 0 + 1)/3 = 0.33 instead
of the faithful 1.33 — a trend that plunges toward zero exactly where the window begins and I have the
least data to correct it, manufacturing a downward slope out of nothing at the boundary that matters most
for the recent past. Replicate-padding copies the edge value into the pad, so the boundary average stays
pinned near the true edge level and the trend runs flat-but-honest where it runs out of context. With
k = 25 that is twelve copies on each side, and twelve plus the ninety-six interior plus twelve, pooled at
width twenty-five, returns ninety-six — the length is preserved with no off-by-one, which the shape
bookkeeping downstream depends on. The scaffold's `series_decomp` already implements precisely this. For
the kernel I take k = 25, the value the decomposition block uses (it smooths sub-daily wiggles on hourly
data while preserving the daily-and-slower trend), and keeping it identical keeps the comparison clean —
I am reusing the block, not tuning a new knob. Here the edit surface bites in a small but real way: the
loop's `configs.moving_avg` default is 25, so I can read it from the config, which is what the
reference model class does; the result is the same number, but I note that under this fixed config every
dataset gets the same kernel, where a per-dataset script could have tuned it.

Shapes, end to end, because the permute bookkeeping is where this silently breaks. The window is
[B, L, C]. Decompose along time to seasonal and trend, each [B, L, C]. The linear maps act along time
(L→T) so I move time last: permute to [B, C, L], apply the two Linears to get [B, C, T] each, sum to
[B, C, T], permute back to [B, T, C], slice the last T steps. That is the whole model: two linear layers
over a moving-average-decomposed, channel-shared window, trained by the loop's direct MSE. The full
scaffold module is in the answer.

What I expect, stated so the next rung can falsify it. This is the embarrassingly simple control whose
entire purpose is to measure how much of the long-horizon accuracy the elaborate attention machinery was
actually responsible for. On ETTh1 — small, strongly trend-and-seasonal, the dataset the decomposition
prior fits best — I expect the linear model to be genuinely competitive, landing an MSE in the high
0.30s; if a heavy model cannot clearly beat it there, that is the point being made. On Weather and ECL,
which have many channels with real cross-variate structure, I expect the deliberate refusal to model
channel interaction to leave the most room on the table: a respectable but clearly beatable MSE, with
ECL — 321 channels — the hardest because the shared-W, channel-blind bet is weakest there. So the rung
sets the bar this way: any later architecture earns its complexity only by beating this linear control,
and the gap it opens on Weather and ECL versus the gap on ETTh1 will tell me whether the missing
ingredient is richer temporal modeling or cross-channel modeling.
