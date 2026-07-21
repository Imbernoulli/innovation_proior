The LightGBM run did what I expected on the most important axis and left a precise gap on another. On
csi300 the IR turned **positive** — 0.280, up from the transformer's −0.44 — and the IC climbed to 0.0399
from 0.0117, Rank IC 0.0492. That confirms the robustness diagnosis: a learner with no optimization
fragility formed real signal where the brittle transformer formed noise. csi300_recent tells the same
story, IR 0.266, return positive. But csi100 came back exactly as I worried: a respectable IC of 0.0363,
yet a *negative* portfolio return (−0.017) and IR (−0.334).

Reading across the two records tells me where to push. The IC jumps are real and uniform: csi300 `0.0117 →
0.0399` is a factor of `3.4`, csi100 `0.0204 → 0.0363` is `1.8`, csi300_recent `0.0014 → 0.0253`, from
essentially nothing to a genuine signal. So the tree formed signal everywhere. The IR, though, splits by
universe. On csi300 and csi300_recent the IR is positive and similar (0.280, 0.266) at similar IC (0.0399,
0.0253), and the active volatility backs out as return-over-IR to about 7% on both (`0.0203/0.280 ≈
0.072`, `0.0197/0.266 ≈ 0.074`). On csi100 the IC (0.0363) actually sits *between* those two, yet the IR
is −0.334 and the implied vol `0.0172/0.334 ≈ 0.052` is if anything lower. So it is not that csi100 is
noisier or riskier — it is that the *same quality of ranking* that earns a positive return on csi300 loses
money on csi100. That IC-to-IR decoupling, which I first flagged on the transformer (higher IC, worse IR),
has now survived a completely different model class — strong evidence it is a property of the *universe and
the backtest*, not of any one model: holding fifty of a hundred names is half the universe, so the
TopkDropout churn and constraint bite far harder than holding fifty of three hundred, and a decent ranking
cannot overcome it. I should not over-promise on csi100.

There is a second, quieter signal: the tree's IC lands in the high-0.03s on csi300 and mid-0.02s on
csi300_recent — decent but not the ceiling, and with the run deterministic (all three seeds identical)
there is no variance to blame. This is simply the ranking quality a per-row tabular view of the window can
extract. The tree threw away the temporal structure of the 60×6 window, and on csi300 especially, where
portfolio return *does* track IC, pushing the IC higher should push the IR higher too. That lost temporal
structure is leaving ranking quality on the table that a model *aware* of the time axis could recover.

So the directed move is back to a sequence model — but emphatically *not* the brittle transformer, whose
failure was never its inductive bias but its high-wire training. "Go back to a sequence model" has more
than one door. One is to reopen the transformer and finally give it the warmup schedule it wanted, now
that I have diagnosed the failure as optimization. But I argued against that already, and the lgbm result
only strengthens the case: what actually won was a learner with *no* delicate optimization, and a
hand-tuned warmup on a 94%-FFN, dropout-off, init-sensitive model is still the high-wire, just with a net
I have to sew myself. The better door is a sequence model whose *own* dynamics are forgiving — the temporal
inductive bias with the optimization robustness the tree showed me matters most here. A recurrent net
trained at a healthy learning rate with generous early-stop patience is that compromise: it reads the
window as a sequence but trains the way the tree was robust, no warmup, no delicate schedule. The question
is which recurrence, and the obvious one fails for a reason I can write down.

Take a plain recurrent net first. Its state is `h_t = f(h_{t-1}, x_t)`, and I want a feature from day 1 of
the window to influence the read-out at day 60. Follow a single error signal backward through the sixty
steps: telescoped, it is a *product* of `q` factors, each of the form `f'(net)·w`. With the logistic
squashing the derivative peaks at `0.25`, so with a well-behaved `|w| = 1` each backward factor is `0.25`,
and the gradient reaching day 1 from day 60 is scaled by `0.25⁶⁰ = 2⁻¹²⁰ ≈ 10⁻³⁶`. That is not
"attenuated," it is annihilated — thirty-six orders of magnitude below the day-60 gradient,
indistinguishable from zero in any finite-precision optimizer. Push `|w|` up to `4` to make the peak
factor `1` and two things go wrong: the product is only marginally non-vanishing at the exact knife-edge,
and any non-tiny input drives the logistic toward saturation where `f'` collapses faster than `w` grew. A
bigger learning rate does not help — it scales long-range and short-range credit identically, so the
*ratio* is unchanged and the recent days still dominate every update. The vanishing is structural: the lag
sits in the exponent, and no ordinary knob touches it. So a plain RNN over a 60-step window would learn
mostly from the last few days and be nearly as blind to day 1 as the tree was.

The cure has to make the product over the lag *not* shrink — ideally exactly 1 for any `q`. For a single
unit with a self-connection of weight `w_jj`, the per-step backward factor is `f'(net)·w_jj`, and demanding
it equal 1 forces `f'(net) = 1/w_jj`, a constant — the squashing must be *linear*. Take the identity with
`w_jj = 1`: the activation simply *persists* unchanged step to step, and the backpropagated error riding
through it is multiplied by exactly 1 each step, `1⁶⁰ = 1`, surviving the full window undamped — the exact
opposite of the `10⁻³⁶` a logistic recurrence gave. A linear self-loop of weight one, a constant error
carousel, is a channel down which gradient survives an arbitrarily long lag. That is the seed.

But a bare carousel cannot be wired to the rest of the net without two conflicts. One incoming weight has
to do two opposed jobs: at the moment the relevant day's information arrives it must let it *in* (write to
memory), but on all other days it must *not* let irrelevant inputs overwrite what is stored (protect) — the
input weight conflict. The mirror image on the output side: one outgoing weight must release the stored
content when needed and shield downstream units from it otherwise — the output weight conflict. A static
weight cannot resolve either, being one context-insensitive number. But another *unit* can be
context-sensitive, and a *multiplicative* gate in `[0,1]` can do what an additive bias cannot: zero an
irrelevant signal *completely* (perfect protection) or pass it through entirely. So wrap the carousel in
learned sigmoid gates reading the rest of the net — an input gate deciding when to write, an output gate
deciding when to read. A third issue, that on a stream never reset the linear state accumulates unbounded
and the output squashing saturates, is fixed by a forget gate multiplying the carried state: it recovers
the exact carousel when open (weight 1) and lets the cell reset itself when closed.

That is the gated memory cell — `i_t = σ(·)`, `f_t = σ(·)`, `g_t = tanh(·)`, `c_t = f_t ⊙ c_{t-1} + i_t ⊙
g_t` (the carousel, now gated), `o_t = σ(·)`, `h_t = o_t ⊙ tanh(c_t)`. The backward pass carries the state
error as `ε_s^t = o_t ⊙ tanh'(c_t) ⊙ ε_h^t + f_{t+1} ⊙ ε_s^{t+1}` — when the forget gate is near 1, the
state error flows back across the lag at unit gain, the constant carousel restated as a fact about
gradients. So over the 60-day window a feature from day 1 can influence the day-60 read-out *and* its
gradient can flow back to day 1 undamped, precisely the temporal credit the tree could not assign and the
plain RNN would have lost. I briefly consider the lighter gated recurrence that merges cell and hidden
states and drops the output gate — fewer parameters, often a wash on benchmarks — but the derivation leans
on an *explicit* cell state whose forget gate gives the unit-gain carousel as a clean, separable object; the
LSTM realizes that argument most directly and is the benchmark-supported choice here, so I take the LSTM.

Now the edit surface, because there is a same-named subtlety to respect. This uses qlib's **non-TS** LSTM,
wired to the ordinary `DatasetH` (not `TSDatasetH`). So each training sample is *still* the flat 360-dim
Alpha360 row — the same rows the tree saw — and the time axis is recovered *inside* the model by
`x.reshape(N, 6, 60).permute(0, 2, 1) -> [N, 60, 6]`, not by the dataset handing over overlapping
time-series windows. The LSTM here is reconstructing the 60-step sequence from the same flat vector the
tree treated as tabular; the difference is purely that it *interprets* the 360 numbers as a 60×6 sequence
and runs the gated recurrence over them. The shape trace: `[N, 360]`, `reshape(N, 6, 60)` reads
feature-major into `[stock, feature, day]`, `permute(0, 2, 1)` gives `[N, 60, 6]` so each timestep row is
the six ratios on that day; the `batch_first` LSTM returns `out` of shape `[N, 60, 64]`, I slice
`out[:, -1, :]` for the day-60 hidden state `[N, 64]` — the cell's summary of the window conditioned on the
most recent day — and `Linear(64, 1)` maps it to `[N]` scores. The `-1` is the last *day*, not the last
feature, because time is the middle axis after the permute.

Two LSTM layers of `hidden_size = 64` are stacked, and the parameter count contrasts instructively with
the transformer. Layer one has four gates, each with a `64×6` input matrix, a `64×64` recurrent matrix,
and two bias vectors of 64: `4·(64·6 + 64·64 + 64 + 64) = 18432`. Layer two takes the 64-dim hidden as
input: `4·(64·64 + 64·64 + 128) = 33280`. With the `Linear(64, 1) = 65` read-out, the whole model is about
`51.8k` parameters — roughly a *tenth* of the transformer's `~563k`. So the model that finally has both the
right temporal bias *and* forgiving training is also by far the smallest, entirely consistent with the
through-line: on a low-signal regression the danger is fitting noise, and a compact recurrent net is a
regularizer, the same logic that drove the narrow attention width and the heavily-penalized trees.

The second stacked layer earns its third of the parameters: the first LSTM consumes the raw 60×6 sequence
and emits a 60×64 sequence of hidden states — a re-encoding of each day in terms of the window's running
memory — and the second runs its own gated recurrence *over that sequence*, forming higher-order temporal
features (patterns in how the first layer's memory evolves, not in the raw ratios). Two layers is where
that compositional gain is real but capacity is still tightly bounded; a deeper stack would multiply the
fit-noise risk, the same reason I stopped the attention stack at two. One note on compute: unlike the
transformer, the LSTM pays for its temporal bias in *serial* time — each layer steps through all sixty
days in order and cannot parallelize across time the way attention's single matrix multiply can. At `T =
60` that serial chain is short enough to be a non-issue, but it is the structural price of the recurrence,
and exactly why the any-to-any attention was attractive — a trade I now reverse because on this data the
forgiving *training* the recurrence allows mattered far more than the parallelism.

The training loop is where this deliberately differs from the transformer, and it is the whole point. Adam
at `lr = 1e-3` — *ten times* the transformer's `1e-4` — because the LSTM is far less
initialization-sensitive (its gates start near the identity carousel rather than in the softmax's
saturation-prone regime) and can take a healthy step from the start, no warmup needed. Early-stop patience
is `20`, four times the transformer's 5, so the run is not cut off on a noisy validation dip. Batch size
800, MSE masked over the finite labels, gradient value-clipped at 3.0 — the clip is the one nod to the
*exploding* side of the error-flow analysis, capping an update without touching the cell equations. And
these settings buy more optimization steps, not just a bigger rate: on the same `~5·10⁵` training
stock-days, batch 800 gives about `620` steps per epoch versus the transformer's `~250` at batch 2048, and
patience 20 over up to 200 epochs lets the run breathe. Larger *and* more numerous steps, a far more
forgiving stopping rule, on a model whose gradients do not vanish — every choice is the tree's robustness
lesson applied to a sequence model.

One design choice I interrogate rather than inherit is the loss. What I care about is a *ranking* — IC and
Rank IC are cross-sectional correlations and the portfolio uses only the order of the scores — so an MSE
regression loss looks like a mismatch, penalizing numerical distance even when the order is right. The
tempting move is a listwise or pairwise ranking loss. But the frozen label pipeline already settles it:
the label passes through `CSRankNorm`, which replaces each day's cross-section of returns with normalized
ranks before the model sees them. So the target the MSE fits is *already* the cross-sectional rank, and
squared-error to a rank target is in effect a smooth ranking objective — a model that gets every stock's
rank right drives the MSE to its floor. That is why the benchmark can use plain MSE and still produce good
Rank IC. A bespoke ranking loss would add optimization fragility (pairwise losses are sensitive to
sampling and scale) for a gain the label normalization has largely delivered — precisely the high-wire
complexity this attempt is trying to get *away* from. So I keep the masked MSE and let `CSRankNorm` do the
ranking work.

One processor note, the inverse of the lgbm edit and not to get backwards. The LSTM, like the transformer,
is a gradient-descent model that needs standardized inputs, so this *keeps* the default processor block —
`RobustZScoreNorm` + `Fillna` on the features — and does *not* strip it the way the tree did. A NaN cannot
pass through the gate matrix multiplies at all, and un-standardized ratios of wildly different scales would
push the sigmoids straight into saturation on the first step, the very regime the carousel is meant to
avoid. So the `Fillna` and clip-and-standardize that were harmful to the tree are load-bearing here. (The
full module is in the answer.)

So the change from the LightGBM run is: keep the optimization robustness that made the tree beat the
transformer, but restore the temporal inductive bias the tree discarded — interpret the same 360 numbers
as a 60×6 sequence and run a gated memory cell over it so a day at the start of the window can both
influence and receive gradient from the end, trained at a healthy learning rate with generous patience and
the neural feature normalization the tree did not need.

The single number I will judge this on is the csi300 information ratio against the tree's 0.280: the
temporal credit assignment should sharpen the ranking and the forgiving training should let it actually
converge, so clearing 0.280 decisively is the evidence that bringing back the time axis — done robustly —
was right, and failing to clear it would mean the tree's robust tabular view was already extracting most of
the signal. csi100 is the sharper, more uncertain test: its IR should improve over the tree's −0.334 if
the temporal awareness genuinely orders the ranking better, but I am honestly unsure it turns positive,
because csi100's weakness looks structural — a small universe where holding 50 of 100 names and churning 5
a day punishes any ranking — rather than a signal-quality problem the LSTM can fix. On csi300_recent I
expect at least the tree's positive IR of 0.266.
