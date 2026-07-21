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
ninety-six times, feeding each prediction back as input — so each step is conditioned on the previous
step's error, and for a persistent, trending series where the recent value is a strong predictor that
feedback amplifies rather than damps the injected noise, compounding it over the horizon until by step
ninety the model is mostly forecasting from its own accumulated mistakes. The standard long-horizon
forecasters that actually work — Informer's generative decoder, Autoformer, FEDformer — all abandoned
iteration for *direct* multi-step: one map straight from the whole history to the whole horizon, no
feedback loop, so the error at step ninety-six is whatever the map makes on the *true* inputs, not on
ninety-five rounds of its own error. I take that lesson as settled. The first rung emits all ninety-six
future steps in a single shot.

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

A single $96\times 96$ matrix can in fact represent seasonal copying exactly: for a purely periodic
series each horizon step equals some look-back step a whole number of periods earlier, so the ideal $W$
is a $0/1$ shift-and-repeat operator — one one per row — that lives inside $\{W\in\mathbb{R}^{96\times
96}\}$ and forecasts it with zero error. Real series are not exactly periodic, but this says the linear
map's ceiling is not the problem: it already expresses the single most important structure in these
traces, which is why it is a control worth beating rather than a punching bag.

There is one structural refinement I want, because it costs almost nothing and it is the right
inductive bias for these series. A load or temperature trace is a slow trend with a periodic component
riding on it. If I ask one linear map to capture both, it has to spend its capacity reconciling a
near-DC drift with a sharp daily oscillation, and the two have very different scales — the trend can
wander over a range far larger than the oscillation amplitude, so a single least-squares fit is
dominated by getting the level roughly right and under-serves the shape. The clean move is to
*decompose first*: split the look-back into a smooth trend and a remainder, forecast each with its own
linear map, and add them back. The trend is just a moving average — a fixed-kernel average pool along
time — and the remainder (the "seasonal" part) is what is left after subtracting it. `series_decomp`
runs a width-25 `AvgPool1d` with replication padding so the trend stays length-96 and does not droop
toward zero at the window edges the way zero-padding would. So the rung becomes: decompose, run
`Linear_Seasonal` on the seasonal part and
`Linear_Trend` on the trend part — each a $96\to96$ map applied per channel across time — and sum the
two forecasts. Two linear layers for the whole model (shared across channels), a decomposition with a
fixed moving-average kernel of width 25, and nothing else.

The width $w=25$ is not arbitrary. A boxcar moving average is a low-pass filter whose first null sits at
period $w$ samples, so components much longer than $w$ pass into the trend and components at or below
$w$ land in the seasonal remainder. On hourly-sampled ETTh1 (and roughly hourly Weather and ECL) that
null falls right on the 24-hour daily cycle, so the daily oscillation goes to `Linear_Seasonal` and the
multi-day drift to `Linear_Trend` — the sharp periodic component and the slow level get separate maps
that can specialize instead of fighting. A reasonable default for the control; no per-dataset tuning.

I initialize both maps' weights to $1/\text{seq\_len} = 1/96$ everywhere, and that choice is what pins
down where training starts. At init each map outputs the mean of its part of the window, and since
`series_decomp` returns a `(seasonal, trend)` pair that sums back to the input exactly, the summed
forecast at every horizon step is $\frac{1}{96}\sum_s x_s$ — the plain mean of the look-back, a flat
line across all ninety-six future steps. That is the persistence-of-mean forecast, a reasonable prior to
learn deviations away from, and it means the decomposition cannot *hurt* at init relative to a single
map with the same init: they start at the identical flat-mean forecast, so the decomposition only ever
buys the two maps the freedom to diverge, which is exactly the freedom I wanted.

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
back to $[B, 96, C]$. Because `channels == enc_in == c_out` and the maps are shared, the output already
has the right shape; the harness slices the last channel for scoring. `forward` just guards on
`task_name` and slices the last `pred_len` steps. There is no normalization, no embedding, no attention
— and that minimalism is the point. The full scaffold module is in the answer.

One subtlety sharpens what this control measures. In `features=MS` the training MSE is taken on the
target column alone (`outputs[:, :, -1:]`), so even though the forward pass forecasts all $C$ channels,
only the target's output ever produces a gradient — the shared maps are in effect trained *by the target
channel alone* and merely applied to the rest. This rung is thus an even purer "target's own history"
control than channel-independent suggests: a single-series forecaster of the target wearing the shape of
a $C$-channel model, with the extra channels costing only forward compute, not confounding supervision.

I briefly consider bolting on a per-instance normalization — subtract the look-back mean and divide by
its std before the maps, add back after — since these benchmarks have real distribution shift between
train and test windows, and it would probably help. But I leave it off deliberately. The uniform $1/96$
init already starts the model at the window mean so the maps can track the level themselves, and more to
the point, adding it would put machinery into the control that I would then have to disentangle from
every later comparison. I want the first rung to be the barest affine-plus-decomposition map, so that
when a later rung adds normalization *and* a stronger temporal model *and* fusion, I can weigh each
addition against a clean, un-normalized floor. Normalization is a lever left on the table for a rung
that has earned the complexity.

So what I expect from this first rung, stated as a direction I can check rather than a number I could
not honestly know a priori. The claim is about *relative looseness*, not absolute MSE. On **ETTh1** the
target (oil temperature) is smooth and strongly autocorrelated on a small homogeneous panel, so a
decomposition-linear map on its own history should already sit close to what a much stronger model could
reach — little headroom, because the target's own history nearly suffices. On **Weather** and **ECL**
the target leans on its covariates (pressure, humidity, wind; or cross-client structure), and the
channel-independent floor throws exactly that signal away, so I expect these to sit far above their
reachable floors — large residual, because the missing signal is cross-channel and no single-channel
map can recover it. If that ordering holds — ETTh1 tight, Weather and ECL loose — the pattern *is* the
case for exogenous fusion, written in the metrics, and the next rung's job is to press on the
Weather/ECL residual while holding ETTh1. If instead the floor is loose *everywhere* including ETTh1,
the temporal map itself is too weak and the next move is to strengthen the per-channel model before
touching channels. Either way, I cannot know which without the control, which is why this rung is first.
