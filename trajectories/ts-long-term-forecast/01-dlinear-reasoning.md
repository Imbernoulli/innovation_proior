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

Now I run the bare linear map against the dataset feature I know will bite it: distribution shift. On
the hourly transformer-temperature data the *level* of the series drifts between the training span and
the test span — the mean is simply higher or lower later in time. A single W fit on training-era levels
will faithfully map a shifted test window to a shifted-and-wrong horizon, with no way to recenter. The
fix is cheap and I can see it without agonizing, but I also notice the scaffold loop already standardizes
each channel using statistics fit on the train split, which absorbs the *global* level but not the
within-window drift; for the parts the loader does not absorb, subtracting the look-back's own mean
before the map and adding it back after would recenter each window. I keep that in mind but do not bolt
it on, because the loader's standardization already covers most of it and I want the rung minimal.

The more interesting weakness is trend *together with* seasonality, which is the common case. Picture
the signal as a big slow ramp with a small daily oscillation riding on top. A single W must fit both at
once, and the two want very different weight patterns: the trend wants weights that extrapolate a slow
drift — broadly smooth across the look-back — while the seasonality wants weights sharply concentrated
at the periodic lags. Worse, the trend is large in magnitude and the seasonal part is small, so in a
squared-error fit the trend dominates the gradient: W spends itself getting the big ramp roughly right
and under-fits the small oscillation that carries the fine structure. One matrix is being asked to be
two different filters and the loss makes it prioritize the loud one. So how do I let them specialize?

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

I should be honest about one thing, because it looks like I just made the model "deeper" and that would
betray the point. Is this still linear? Two linear maps plus a sum are affine; the only added operation
is the moving-average split, and a moving average is itself linear, so decomposition followed by two
linear maps and a sum is end-to-end still a single affine map. I have added no representational capacity
in the function-class sense. What I have added is *conditioning*: a fixed, parameter-free
reparameterization that separates the loud trend from the quiet seasonality so gradient descent on the
squared error fits each well instead of letting the trend's magnitude dominate. It is the same trick as
preconditioning an optimization — same solution set, far better-behaved learning — and it pays off
exactly when there is a clear trend, the case the bare linear map handled worst.

The moving-average details decide the edge behavior. I want the trend to have the same length as the
input, so I average-pool with an odd kernel of size k, stride 1, and pad. Zero-padding would drag the
trend toward zero at the two ends, creating spurious dips where I have least information — the wrong
boundary behavior. Replicate-padding instead — front with (k−1)/2 copies of the first value, back with
(k−1)/2 copies of the last — keeps the trend flat-but-faithful at the edges; with k odd, (k−1)/2 per
side restores the length exactly. The scaffold's `series_decomp` already implements precisely this. For
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
