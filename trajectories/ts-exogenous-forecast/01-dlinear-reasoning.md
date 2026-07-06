I am standing in front of an empty `Model` class — `forecast` returns zeros of the right shape and
scores nothing — and I have to put the first real thing in it. The temptation, given that the research
question is *exogenous fusion*, is to reach straight for a cross-channel attention machine. I am going
to resist that, and the reason is methodological, not lazy. The whole point of a ladder is that each
rung's number only means something against the rung below it. If I open with a 512-wide Transformer
that fuses channels, and it scores well, I will not know how much of that score is the fusion and how
much is just "a big nonlinear model fit this dataset." So I want the first rung to be the most
minimal honest forecaster I can write — the one that establishes how far you get *before* any
exogenous fusion at all — and then every later rung has to earn its complexity by beating it. The
weakest baseline is not a throwaway; it is the control that calibrates the entire climb.

So let me think about what the most minimal honest forecaster is, on its own terms, before I worry
about channels. The substrate hands me `x_enc` of shape $(B,96,\text{enc\_in})$ — ninety-six steps of
look-back across every channel — and wants $(B,96,\text{c\_out})$ back, the next ninety-six steps, with
only the last channel scored. The horizon is long and fixed, the look-back is fixed. The first design
fork is *direct vs iterated* multi-step. The iterated route learns a one-step map and rolls it forward
ninety-six times, feeding each prediction back as input. Let me make the failure quantitative rather
than hand-wave it. Suppose the one-step map has some irreducible per-step error with variance
$\sigma^2$, and the map has any nonzero sensitivity $g=\partial \hat x_{t+1}/\partial x_t$ to its most
recent input — which it must, that is the whole point of an autoregressive step. When I feed a noisy
prediction back in, that noise is amplified by $g$ into the next step's input, so the injected-error
variance at horizon $k$ behaves like $\sigma^2\sum_{j=0}^{k-1} g^{2j}$. If $g\ge 1$ — and for a
persistent, trending series the recent value is a strong predictor, so $g$ sits right around one — that
sum is at best *linear* in $k$ and at worst *geometric*, so by $k=96$ the accumulated feedback error
dwarfs the signal. By step ninety the model is mostly forecasting from its own accumulated mistakes.
The standard long-horizon forecasters that actually work — Informer's generative decoder, Autoformer,
FEDformer — all abandoned iteration for *direct* multi-step: one map straight from the whole history to
the whole horizon, no feedback loop, so the error at step ninety-six is whatever the map makes on the
*true* inputs, not on ninety-five rounds of its own error. That $g^{2j}$ term simply does not exist in
a direct map. I take that lesson as settled. The first rung emits all ninety-six future steps in a
single shot.

Now, what is the absolute floor of a direct multi-step map? It is a single affine map along the time
axis. For one channel, predict the horizon as a learned weighted sum of the look-back:
$\hat{x}_{i} = W x_{i}$, with $x_i$ the length-96 history of channel $i$, $\hat x_i$ its length-96
forecast, and $W$ a $96\times 96$ matrix. Row $t$ of $W$ says exactly how to combine the ninety-six
past values into the $t$-th future value. That is it — one linear layer applied along time. No
attention, no recurrence, no positional encoding bolted on. And I should sit with *why* this is not
obviously stupid, because it is going to be the surprising part of the first feedback. A single number
in a load trace — $0.42$ — means nothing in isolation; the entire signal of a time series lives in its
*order*, in the fact that this value sits on the rising shoulder of the evening peak and that the same
shape recurred twenty-four and one-hundred-sixty-eight hours ago. A linear map along the time axis
reads the whole window at once through learned weights, so it sees the shape directly. Row $t$ of $W$
can, in principle, learn to be a shifted copy of a seasonal kernel — put weight on the values one full
period back — and thereby express "the horizon looks like the last cycle." Attention's core operation,
by contrast, is permutation-invariant over tokens and re-injects order only as an additive side channel
— it is working against the grain of data whose whole content *is* order. There is real reason to
believe the linear floor is not a strawman but a genuinely strong control.

Let me actually verify the strong claim — that a single $96\times 96$ matrix can express seasonal
copying exactly — on the cleanest special case, a purely periodic series with period 24, where
$x_{s}=x_{s-24}$ for all $s$. Index the look-back $1,\dots,96$ and the horizon step $t=1,\dots,96$. The
true future value at horizon step $t$ is $x_{96+t}$, and by periodicity $x_{96+t}=x_{96+t-24\lceil
t/24\rceil}$ where I subtract enough whole periods to land inside the look-back. For $t=1$ that is
$x_{73}$; for $t=24$ it is $x_{96}$; for $t=25$ it is $x_{73}$ again; and so on — every horizon step maps
to exactly one look-back index. So the ideal $W$ is a $0/1$ matrix with a single one per row, a pure
shift-and-repeat operator, and it forecasts this series with *zero* error. That matrix is plainly inside
the hypothesis class $\{W\in\mathbb{R}^{96\times96}\}$, so a linear map along time can represent perfect
seasonal extrapolation, and the decomposition only makes it easier by handing the periodic part to its
own map. Real series are not exactly periodic, of course, but this confirms the linear map's ceiling is
not the problem: it can already express the single most important structure in these traces, which is
why it is a control worth beating rather than a punching bag.

There is one structural refinement I want, because it costs almost nothing and it is the right
inductive bias for these series. A load or temperature trace is a slow trend with a periodic component
riding on it. If I ask one linear map to capture both, it has to spend its capacity reconciling a
near-DC drift with a sharp daily oscillation, and the two have very different scales — the trend can
wander over a range far larger than the oscillation amplitude, so a single least-squares fit is
dominated by getting the level roughly right and under-serves the shape. The clean move is to
*decompose first*: split the look-back into a smooth trend and a remainder, forecast each with its own
linear map, and add them back. The trend is just a moving average — a fixed-kernel average pool along
time — and the remainder (the "seasonal" part) is what is left after subtracting it. Let me check the
one piece of plumbing that has to line up, the kernel geometry, because a length mismatch here would
silently corrupt the decomposition. `series_decomp` runs an `AvgPool1d` of width 25, stride 1; to keep
the trend the same length as the 96-step input it pads $(25-1)/2 = 12$ steps on each side by replicating
the first and last values. Output length is then $96 + 12 + 12 - 25 + 1 = 96$ — it lines up exactly, and
the replication padding means the trend does not droop toward zero at the window edges the way
zero-padding would. So the rung becomes: decompose, run `Linear_Seasonal` on the seasonal part and
`Linear_Trend` on the trend part — each a $96\to96$ map applied per channel across time — and sum the
two forecasts. Two linear layers for the whole model (shared across channels), a decomposition with a
fixed moving-average kernel of width 25, and nothing else.

It is worth being precise about *what* a width-25 average actually separates, because the whole value of
the decomposition rests on the split landing in the right place. A boxcar moving average of width $w$ is
a low-pass filter whose frequency response is a Dirichlet kernel with its first null at period $w$
samples: components with period much longer than $w$ pass through nearly untouched into the trend, and
components with period at or below $w$ are strongly attenuated and therefore land in the seasonal
remainder. ETTh1 is sampled hourly, so $w=25$ puts the first null right around a 25-hour period — which
sits essentially on top of the 24-hour daily cycle. So the daily oscillation is pushed almost entirely
into the seasonal part, where `Linear_Seasonal` can learn its shape, while the multi-day drift stays in
the trend for `Linear_Trend`. That is exactly the division of labor I wanted: the sharp periodic
component and the slow level get separate maps whose weight patterns can specialize instead of fighting.
The same kernel on Weather and ECL (also roughly hourly-scale sampling) does the analogous thing. I do
not need to tune $w$ per dataset for the control — 25 is a reasonable default that puts the dominant
daily period into the seasonal stream on all three panels.

I want to actually verify the initialization does something sane before I trust it, because it will set
what the first few epochs even start from. I initialize both maps' weights to $1/\text{seq\_len} = 1/96$
everywhere. At init, then, the seasonal output at every horizon step is $\frac{1}{96}\sum_{s}
\text{seasonal}_s$ — the mean of the seasonal part of the window — and likewise the trend output is the
mean of the trend part. But `series_decomp` returns a `(seasonal, trend)` pair that sums back to the
input exactly: $\text{seasonal}_s + \text{trend}_s = x_s$ for every $s$. So the summed forecast at every
horizon step is $\frac{1}{96}\sum_s \text{seasonal}_s + \frac{1}{96}\sum_s \text{trend}_s =
\frac{1}{96}\sum_s x_s$ — the plain mean of the look-back, emitted as a flat line across all ninety-six
future steps. That is the persistence-of-mean forecast, which is a genuinely reasonable prior to start
from and to learn deviations away from. Good: the init is not arbitrary, it is a sensible baseline the
gradient then improves on. That check also tells me the decomposition cannot *hurt* at init relative to
a single map with the same init — they start at the identical flat-mean forecast — so the decomposition
only ever buys the freedom for the two maps to diverge, which is exactly the freedom I wanted.

I should pin down the parameter budget too, because it is the argument for one specific choice I am about
to make. Each linear map is $96\times 96$ weights plus a 96-vector bias: $9216 + 96 = 9312$ parameters.
Two maps is $18{,}624$ parameters — and that is the *entire* model, for every dataset, because the maps
are shared across channels. That number does not grow when I go from ETTh1's 7 channels to ECL's 321.
Hold that thought against the alternative I now have to weigh honestly.

Here is the channel question, which is the one the whole ladder is about, and the place where this rung
makes its defining choice *by omission*. I have `enc_in` channels in and `c_out == enc_in` channels
out, but only the last is scored. The linear maps apply along the **time** axis; the channel axis is
just a batch dimension as far as the layers are concerned. There is a real design fork here that I want
to walk into rather than skip: should the two weight matrices be **shared** across channels
(`individual=False` — one $W_\text{seasonal}$ and one $W_\text{trend}$ for all channels) or
**per-channel** (`individual=True` — a separate pair for each of the `enc_in` channels)? The per-channel
variant is tempting, because these channels are heterogeneous — on ECL, 321 clients have wildly different
load shapes, and forcing one $96\to96$ pattern onto all of them is obviously a compromise. So let me
actually cost the tempting option. Per-channel on ECL is $321 \times 18{,}624 \approx 6.0$ million
parameters, and — this is the part that kills it — every one of those per-channel maps is trained *only*
on that channel's own history, so each map sees a single time series' worth of gradient signal. On a
panel where I score exactly one channel, spending six million parameters to give 320 unscored channels
their own private maps is capacity poured into places I never read, and the one map that matters (the
target's) trains on strictly less data than the shared map would, which pools the statistical strength
of all channels into two matrices. The shared map is both $320\times$ smaller *and* better-regularized
for the target. On a small homogeneous panel like ETTh1 the per-channel option is cheap ($7\times$) but
also nearly pointless — the channels are physically related, a shared temporal prior is right for them.
So the fork resolves the same way at both extremes: shared. I take the `individual=False` path.

That choice has a consequence that is the entire reason this rung sits first. Because the maps are
shared and act along time with the channel axis as a batch dimension, this rung is
**channel-independent**: each channel is forecast from *its own* history alone, and the exogenous
covariates never touch the target. The weather observations cannot inform the wet-bulb forecast; the
other 320 clients cannot inform the one client I score. This is exactly the thing the research question
asks me to improve, and I am deliberately not doing it yet — because I want to measure how much the
target's *own* history already buys, so that the first cross-channel rung has a real number to beat.
The decomposition-linear forecaster is the cleanest possible "no fusion" point.

Let me make it concrete in the edit surface, because the loop is fixed and I only get to fill the
`Model` class. In `__init__` I read `seq_len` and `pred_len` off `configs`, build
`series_decomp(configs.moving_avg)` (default kernel 25), and create the two shared linear maps
`Linear_Seasonal` and `Linear_Trend`, each `nn.Linear(seq_len, pred_len)`, weight-initialized to
$1/\text{seq\_len}$. In `forecast`, I decompose `x_enc` into `(seasonal_init, trend_init)`, each of
shape $[B, 96, C]$; permute each so the time axis is last, giving $[B, C, 96]$; apply the two linear
maps along that last axis, which sends $96\to 96$ and yields $[B, C, 96]$ each; sum them; and permute
back to $[B, 96, C]$. Let me trace the shapes once more to be sure the head lines up: input $[B,96,C]
\to$ decompose $\to$ two $[B,96,C]\to$ permute $\to$ two $[B,C,96]\to$ Linear $\to$ two $[B,C,96]\to$
sum $\to [B,C,96]\to$ permute $\to [B,96,C]$. Because `channels == enc_in == c_out` and the maps are
shared, the output already has the right shape; the harness slices the last channel for scoring.
`forward` just guards on `task_name` and slices the last `pred_len` steps. There is no normalization,
no embedding, no attention — and that minimalism is the point. The full scaffold module is in the
answer.

There is a subtlety in how "shared across channels" actually behaves at training time that I should be
clear-eyed about, because it sharpens what this control is really measuring. The harness scores only the
last channel, and in the `features=MS` convention the training MSE is taken on that same target column —
the loss slices `outputs[:, :, -1:]`. So even though the forward pass computes a forecast for all
$C$ channels through the shared maps, only the target channel's output ever produces a gradient; the
other $C-1$ channels are computed and then discarded, contributing nothing to the loss. That means the
two shared weight matrices are, in effect, trained *by the target channel alone* and merely applied to
the rest. The upshot is that this rung is an even purer "target's own history" control than the phrase
channel-independent suggests: it is a single-series forecaster of the target, wearing the shape of a
$C$-channel model only because the maps are broadcast across the channel axis. That is fine — it is
exactly the floor I want to measure — and it also tells me the extra channels cost only forward compute,
not any confounding supervision.

I briefly consider whether to bolt on a per-instance normalization — subtract the look-back mean and
divide by its standard deviation before the maps, add them back after — since these benchmarks have real
distribution shift between training and test windows. It would probably help. But I decide against it
here, deliberately, on two grounds. First, the uniform $1/96$ initialization already starts the model at
the window mean, so the linear maps can learn to track the level themselves; normalization is not
load-bearing for the model to function. Second, and more importantly, adding it would put a piece of
machinery into the control that I would then have to disentangle from every later comparison — I want
the very first rung to be the barest possible affine-plus-decomposition map, so that when a later rung
adds normalization *and* a stronger temporal model *and* fusion, I can reason about each addition against
a clean, un-normalized floor. Normalization is a lever I am consciously leaving on the table for a rung
that has earned the right to spend the complexity.

One thing I want to be honest about up front, because it shapes what I will read in the feedback.
There is no per-channel `individual` head here — the seasonal and trend maps are shared across all
channels. For a small panel like ETTh1 (seven channels) that is fine; the channels are physically
related and a shared map is a reasonable prior. For a large panel like ECL (321 clients), a shared map
forces a single $96\to96$ pattern onto wildly heterogeneous load profiles, which I expect to hurt — and
crucially, on Weather and ECL the *target channel specifically* may depend on the other channels in a
way that no amount of clever single-channel mapping can recover, because the information simply is not
in the target's own history. That is the failure mode I am setting up to diagnose. I am not building
the fusion that fixes it; I am building the control that proves it is needed.

So here is what I expect from this first rung, stated as falsifiable predictions against the numbers I
am about to see. On **ETTh1** the target (oil temperature) is smooth and strongly autocorrelated, and
the panel is small and homogeneous, so a decomposition-linear map on the target's own history should
already be respectable — I would not be surprised if the linear floor here is hard to beat by much, and
I expect the MSE to sit in the low-to-mid $0.06$s. On **Weather** the target is one of twenty-one
observations and the cross-channel coupling (pressure, humidity, wind all bearing on the target) is
real; here I expect the channel-independent floor to be *visibly* weak, because the exogenous signal it
throws away is exactly the signal that matters — this is the dataset where a later fusion rung should
open the biggest gap, and I would guess the MSE lands an order of magnitude above the small numbers a
fusing model could reach. On **ECL** the target is one of 321 clients; the shared linear map and the
discarded cross-client structure should both bite, and I expect this to be the worst rung relative to
what is achievable, MSE up in the high tenths. If the linear floor is already strong on ETTh1 but
clearly loose on Weather and ECL, that pattern *is* the case for exogenous fusion, written in the
metrics — and the next rung's job is to start closing the Weather/ECL gap while at least holding ETTh1.
If, instead, the linear floor is loose *everywhere* including ETTh1, then I have a different problem
(the temporal map itself is too weak) and the next move would be to strengthen the *per-channel*
temporal model before touching cross-channel fusion. Either way, I cannot know which without the
control, which is why this rung exists and why it is first.

I want to fix the *shape* of the pattern I am predicting so I can check it against the numbers rather
than rationalize whatever comes back. The claim is not just "Weather and ECL will be worse than ETTh1
in absolute terms" — they might be, on different scales, for reasons that have nothing to do with fusion.
The claim is about *relative looseness*: ETTh1's number should sit close to what a much stronger model
could reach (small residual, because the target's own history nearly suffices), while Weather's and
ECL's numbers should sit far above their reachable floors (large residual, because the missing signal is
cross-channel). So the diagnostic I will actually run on the feedback is not the raw MSE but how much
headroom each dataset appears to leave — and the prediction is that ETTh1 leaves little and Weather/ECL
leave a lot. If that ordering holds, the ladder has its mandate: spend the next rung's complexity where
the residual is, which the metrics will say is the cross-channel-dependent datasets, not the
self-predictable one.
