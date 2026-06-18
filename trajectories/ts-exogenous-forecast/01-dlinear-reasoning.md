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
ninety-six times, feeding each prediction back as input. I know what that does over a long horizon:
each step is conditioned on the previous step's error, and those errors compound, so by step ninety
the model is mostly forecasting from its own accumulated mistakes. The standard long-horizon
forecasters that actually work — Informer's generative decoder, Autoformer, FEDformer — all abandoned
iteration for *direct* multi-step: one map straight from the whole history to the whole horizon, no
feedback loop, no compounding. I take that lesson as settled. The first rung emits all ninety-six
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
reads the whole window at once through learned weights, so it sees the shape directly. Attention's
core operation, by contrast, is permutation-invariant over tokens and re-injects order only as an
additive side channel — it is working against the grain of data whose whole content *is* order. There
is real reason to believe the linear floor is not a strawman but a genuinely strong control.

There is one structural refinement I want, because it costs almost nothing and it is the right
inductive bias for these series. A load or temperature trace is a slow trend with a periodic component
riding on it. If I ask one linear map to capture both, it has to spend its capacity reconciling a
near-DC drift with a sharp daily oscillation, and the two have very different scales. The clean move
is to *decompose first*: split the look-back into a smooth trend and a remainder, forecast each with
its own linear map, and add them back. The trend is just a moving average — a fixed-kernel average pool
along time — and the remainder (the "seasonal" part) is what is left after subtracting it. The
Time-Series-Library gives me exactly this as `series_decomp`: a `moving_avg` block with reflect-style
end padding so the trend has the same length as the input, returning `(seasonal, trend)`. So the rung
becomes: decompose, run `Linear_Seasonal` on the seasonal part and `Linear_Trend` on the trend part —
each a $96\to96$ map applied per channel across time — and sum the two forecasts. Two linear layers per
the whole model (shared across channels), a decomposition with a fixed moving-average kernel of width
25, and nothing else.

Now the channel question, which is the one the whole ladder is about, and the place where this rung
makes its defining choice *by omission*. I have `enc_in` channels in and `c_out == enc_in` channels
out, but only the last is scored. The linear maps apply along the **time** axis; the channel axis is
just a batch dimension as far as the layers are concerned. The two linear weight matrices are
**shared** across channels (the `individual=False` path — one $W_\text{seasonal}$ and one
$W_\text{trend}$ for every channel, not a separate pair per channel). That means this rung is
**channel-independent**: each channel is forecast from *its own* history alone, and the exogenous
covariates never touch the target. The weather observations cannot inform the wet-bulb forecast; the
other 320 clients cannot inform the one client I score. This is exactly the thing the research question
asks me to improve, and I am deliberately not doing it yet — because I want to measure how much the
target's *own* history already buys, so that the first cross-channel rung has a real number to beat.
The decomposition-linear forecaster is the cleanest possible "no fusion" point.

Let me make it concrete in the edit surface, because the loop is fixed and I only get to fill the
`Model` class. In `__init__` I read `seq_len` and `pred_len` off `configs`, build
`series_decomp(configs.moving_avg)` (default kernel 25), and create the two shared linear maps
`Linear_Seasonal` and `Linear_Trend`, each `nn.Linear(seq_len, pred_len)`. I initialize their weights
to $1/\text{seq\_len}$ everywhere — a uniform-average start, which is the published reference's
initialization and a sensible prior (begin by predicting the mean of the window, then learn the
deviations). In `forecast`, I decompose `x_enc` into `(seasonal_init, trend_init)`, permute each so the
time axis is last (`[B, channels, seq_len]`), apply the two linear maps along that last axis, sum the
outputs, and permute back to `[B, pred_len, channels]`. Because `channels == enc_in == c_out` and the
maps are shared, the output already has the right shape; the harness slices the last channel for
scoring. `forward` just guards on `task_name` and slices the last `pred_len` steps. There is no
normalization, no embedding, no attention — and that minimalism is the point. The full scaffold module
is in the answer.

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
already be respectable — I would not be surprised if the linear floor here is hard to beat by much.
On **Weather** the target is one of twenty-one observations and the cross-channel coupling (pressure,
humidity, wind all bearing on the target) is real; here I expect the channel-independent floor to be
*visibly* weak, because the exogenous signal it throws away is exactly the signal that matters — this
is the dataset where a later fusion rung should open the biggest gap. On **ECL** the target is one of
321 clients; the shared linear map and the discarded cross-client structure should both bite, and I
expect this to be the worst rung relative to what is achievable. If the linear floor is already strong
on ETTh1 but clearly loose on Weather and ECL, that pattern *is* the case for exogenous fusion, written
in the metrics — and the next rung's job is to start closing the Weather/ECL gap while at least holding
ETTh1. If, instead, the linear floor is loose *everywhere* including ETTh1, then I have a different
problem (the temporal map itself is too weak) and the next move would be to strengthen the
*per-channel* temporal model before touching cross-channel fusion. Either way, I cannot know which
without the control, which is why this rung exists and why it is first.
