The LightGBM rung did what I expected on the most important axis and left a precise gap on another. On
csi300 the IR turned **positive** — 0.280, up from the transformer's −0.44 — and the IC climbed to 0.0399
from 0.0117, with Rank IC 0.0492. That confirms the robustness diagnosis: a learner with no optimization
fragility formed real signal where the brittle transformer formed noise. On csi300_recent the same story,
IR 0.266, return positive. But csi100 came back exactly as I worried it might: a perfectly respectable IC
of 0.0363, yet a *negative* portfolio return (−0.017) and IR (−0.334).

Let me read across the two rungs' numbers carefully, because the pattern is what tells me where to push.
The IC jumps are real and uniform: csi300 `0.0117 → 0.0399` is a factor of `3.4`, csi100 `0.0204 →
0.0363` is `1.8`, csi300_recent `0.0014 → 0.0253`, from essentially nothing to a genuine signal. So the
tree formed signal everywhere. The IR, though, splits by universe. On csi300 and csi300_recent the tree's
IR is positive and similar (0.280, 0.266) at similar IC (0.0399, 0.0253) — and I can back out the active
volatility as return-over-IR: `0.0203/0.280 ≈ 0.072` and `0.0197/0.266 ≈ 0.074`, both running about 7%.
On csi100 the IC (0.0363) actually *sits between* those two, higher than csi300_recent's, yet the IR is
−0.334 with a negative return; the implied vol `0.0172/0.334 ≈ 0.052` is if anything lower. So it is not
that csi100 is noisier or riskier — it is that the *same quality of ranking* that earns a positive return
on csi300 loses money on csi100. That decoupling of IC from IR, which I first flagged on the transformer
(where csi100 had the higher IC and the worse IR), has now survived a completely different model class.
That is strong evidence it is a property of the *universe and the backtest*, not of any one model: holding
fifty names out of a hundred is half the universe, so the TopkDropout churn and the constraint bite far
harder than holding fifty out of three hundred, and a decent ranking cannot overcome it. I should keep
that in mind so I do not over-promise on csi100 at this rung.

And there is a second, quieter signal in the numbers: the tree's IC lands in the high-0.03s on csi300
(0.0399) and the mid-0.02s on csi300_recent (0.0253), which is decent but not the ceiling — the LightGBM
run is deterministic (all three seeds identical, so the feedback's three rows are literally the same
numbers), so there is no variance to blame;
this is simply the ranking quality a per-row tabular view of the window can extract. The tree threw away
the temporal structure of the 60×6 window, and on csi300 especially, where portfolio return *does* track
IC, pushing the IC higher should push the IR higher too. That lost temporal structure is leaving ranking
quality on the table that a model *aware* of the time axis could recover.

So the directed move is to go back to a sequence model — but emphatically *not* the brittle transformer.
The transformer's failure was never its inductive bias; it was that its training was a high-wire act
(warmup-less Adam, init sensitivity, patience-5). Let me be explicit about the fork, because "go back to a
sequence model" has more than one door. One door is to reopen the transformer and finally give it the
warmup schedule it wanted, now that I have diagnosed the failure as optimization: I could write a manual
ramp inside `CustomModel`. But I already argued against that at the previous rung and the lgbm result only
strengthens the case — what actually won was a learner with *no* delicate optimization at all, and a
hand-tuned warmup on a 94%-FFN, dropout-off, init-sensitive model is still the high-wire, just with a net
I have to sew myself. The better door is a sequence model whose *own* dynamics are forgiving, so I get the
temporal inductive bias with the optimization robustness the tree showed me matters most on this data. A
recurrent net trained at a healthy learning rate with generous early-stop patience is exactly that
compromise: it reads the window as a sequence, but it trains the way the tree was robust — no warmup
needed, no delicate schedule. The question is which recurrence, and here I have to be careful, because the
obvious one fails for a reason I can write down.

Take a plain recurrent net first, so I understand why it is not enough. Its state is `h_t = f(h_{t-1},
x_t)`, and I want a feature from day 1 of the window to be able to influence the read-out at day 60.
Follow a single error signal backward through the sixty steps. The error that lands on a unit at the end
and reaches a unit `q` steps earlier is, telescoped, a *product* of `q` factors, each of the form
`f'(net)·w`. Let me actually put numbers on that product for a 60-step window. With the logistic squashing
the derivative peaks at `0.25`, so each factor is at most `0.25·|w|`; take a well-behaved `|w| = 1` and
the backward factor is `0.25`, so the gradient reaching day 1 from day 60 is scaled by `0.25⁶⁰ = 4⁻⁶⁰ =
2⁻¹²⁰ ≈ 10⁻³⁶`. That is not "attenuated," it is annihilated — the day-1 gradient is thirty-six orders of
magnitude below the day-60 gradient, indistinguishable from zero in any finite-precision optimizer. Push
`|w|` up to `4` to make the peak factor `1` and two things go wrong: the product is only marginally
non-vanishing at the exact knife-edge, and any input that is not tiny drives the logistic toward
saturation where `f'` collapses back toward zero faster than `w` grew. A bigger learning rate does not
help either — it scales long-range and short-range credit identically, so the *ratio* is unchanged and the
recent days still dominate every update. The vanishing is structural — the lag sits in the exponent, and
no ordinary knob touches it. So a plain RNN over a 60-step window would learn mostly from the last few
days and be nearly as blind to day 1 as the tree was, just for a different reason. That is not the fix I
want.

The cure has to make the product over the lag *not* shrink — ideally exactly 1 for any `q`. In the
simplest setting, a single unit with a self-connection of weight `w_jj`, the per-step backward factor is
`f'(net)·w_jj`, and demanding it equal 1 forces `f'(net) = 1/w_jj`, a constant — the squashing must be
*linear*. Take the cleanest case, the identity with `w_jj = 1`: now the activation simply *persists*
unchanged step to step, and the backpropagated error riding through it is multiplied by exactly 1 each
step, so `1⁶⁰ = 1` — the gradient survives the full window undamped, the exact opposite of the `10⁻³⁶` a
logistic recurrence gave. A linear self-loop of weight one — a constant error carousel — is a channel down
which gradient survives an arbitrarily long lag. That is the seed.

But a bare carousel cannot be wired to the rest of the net without two conflicts appearing. A single
incoming weight has to do two opposed jobs: at the moment the relevant day's information arrives it must
let it *in* (write to memory), but on all the other days it must *not* let irrelevant inputs overwrite
what is stored (protect). One weight receiving contradictory update signals — the input weight conflict.
The mirror image is on the output side: one outgoing weight must release the stored content when it is
needed and protect the downstream units from it when it is not — the output weight conflict. A static
weight cannot resolve either, because it is one number and cannot be context-sensitive. But another
*unit* can be context-sensitive, and a *multiplicative* gate can do what an additive bias cannot: a
control in `[0,1]` multiplying the input can zero an irrelevant signal *completely* (perfect protection)
or pass it through entirely. So wrap the carousel in multiplicative gates that are themselves learned
sigmoid units reading the rest of the net: an input gate deciding when to write, an output gate deciding
when to read. And a third issue — on a stream that is never reset the linear state accumulates without
bound and the output squashing saturates — is fixed by a forget gate multiplying the carried state, which
recovers the exact carousel when it is open (weight 1) and lets the cell reset itself when it is closed.

That gives the gated memory cell, which in compact form is `i_t = σ(W_ix x_t + W_ih h_{t-1} + b_i)`,
`f_t = σ(...)`, `g_t = tanh(...)`, `c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t` (the carousel, now gated),
`o_t = σ(...)`, `h_t = o_t ⊙ tanh(c_t)`. The backward pass carries the state error as
`ε_s^t = o_t ⊙ tanh'(c_t) ⊙ ε_h^t + f_{t+1} ⊙ ε_s^{t+1}` — when the forget gate is near 1, the state
error flows back across the lag at unit gain, the constant carousel as a statement about gradients. So
over the 60-day window, a feature from day 1 can influence the day-60 read-out *and* its gradient can
flow back to day 1 undamped, which is precisely the temporal credit the tree could not assign and the
plain RNN would have lost to the vanishing product. I briefly consider the lighter gated recurrence that
merges the cell and hidden states and drops the separate output gate — fewer parameters, and often a wash
on benchmarks — but the whole derivation I just walked leans on an *explicit* cell state whose forget gate
gives me the unit-gain carousel as a clean, separable object; the LSTM realizes that argument most
directly, and it is the benchmark-supported choice on this data, so I take the LSTM.

Now ground it in this task's edit surface, because there is a same-named subtlety I must respect. This
rung uses qlib's **non-TS** LSTM, wired to the ordinary `DatasetH` (not `TSDatasetH`). That means each
training sample is *still* the flat 360-dim Alpha360 row — the same rows the tree saw — and the time axis
is recovered *inside* the model by `x.reshape(N, 6, 60).permute(0, 2, 1) -> [N, 60, 6]`, not by the
dataset handing over genuine overlapping time-series windows. So the LSTM here is reconstructing the
60-step sequence from the same flat feature vector the tree treated as tabular; the difference is purely
that the LSTM *interprets* those 360 numbers as a 60×6 sequence and runs the gated recurrence over it,
while the tree saw 360 unordered columns. Let me trace the shapes to be sure the read-out is what I think.
The batch enters as `[N, 360]`; `reshape(N, 6, 60)` reads the flat vector feature-major into
`[stock, feature, day]`, `permute(0, 2, 1)` gives `[N, 60, 6]` so each timestep row is the six ratios on
that day; the `batch_first` LSTM returns `out` of shape `[N, 60, 64]`, I slice `out[:, -1, :]` to take the
day-60 hidden state `[N, 64]` — the cell's summary of the whole window conditioned on the most recent day
— and `Linear(64, 1)` maps it to `[N, 1]`, squeezed to `[N]` scores. The `-1` is the last *day*, not the
last feature, because time is the middle axis after the permute; the trace closes.

Two LSTM layers (`num_layers = 2`) of `hidden_size = 64` are stacked, and it is worth counting the
parameters, because the contrast with the transformer is instructive. Layer one has four gates, each with
a `64×6` input matrix, a `64×64` recurrent matrix, and two bias vectors of 64:
`4·(64·6 + 64·64 + 64 + 64) = 4·4608 = 18432`. Layer two takes the 64-dim hidden as its input, so its
input matrix is `64×64`: `4·(64·64 + 64·64 + 128) = 4·8320 = 33280`. With the `Linear(64, 1) = 65`
read-out, the whole model is about `51.8k` parameters — roughly a *tenth* of the transformer's `~563k`.
So the model that finally has the right temporal bias *and* the forgiving training is also by far the
smallest, which is entirely consistent with the through-line: on a low-signal regression the danger is
fitting noise, and a compact recurrent net is a regularizer, the same logic that drove the narrow
attention width and the heavily-penalized trees. The width 64 and depth 2 are modest by design, not by
accident.

It is worth being clear about what the second stacked layer actually buys, since it is where a third of
the parameters go. The first LSTM consumes the raw 60×6 sequence and emits a 60×64 sequence of hidden
states — a re-encoding of each day in terms of the window's running memory. The second LSTM runs its own
gated recurrence *over that sequence of hidden states*, so it can form higher-order temporal features:
patterns in how the first layer's memory itself evolves, rather than in the raw ratios. Two layers is the
point where that compositional gain is real but the capacity is still tightly bounded; a deeper stack
would multiply the parameter count and the fit-noise risk on a signal this faint, the same reason I
stopped the attention stack at two layers. And a note on compute, because unlike the transformer the LSTM
pays for its temporal bias in *serial* time: each layer must step through all sixty days in order, doing
about `4·(64·6 + 64·64) ≈ 1.8·10⁴` multiply-adds per step in layer one and `4·64·64·2 ≈ 3.3·10⁴` in layer
two, and those steps cannot be parallelized across time the way attention's single matrix multiply can. At
`T = 60` that serial chain is short enough to be a non-issue, but it is the structural price of the
recurrence, and it is exactly why the any-to-any attention was attractive in the first place — a trade I
am now reversing because, on this data, the forgiving *training* the recurrence allows mattered far more
than the parallelism the attention offered.

The training loop is where this rung deliberately differs from the transformer rung, and it is the whole
point. Adam at `lr = 1e-3` — *ten times* the transformer's `1e-4` — because the LSTM is far less
initialization-sensitive than the attention stack (its gates start near the identity carousel rather than
in the softmax's saturation-prone regime) and can take a healthy step from the start; no warmup schedule
is needed. Early-stop patience is `20`, four times the transformer's 5, so the run is not cut off on a
noisy validation dip the way I suspect the transformer was. Batch size 800, MSE loss masked over the
finite labels, gradient value-clipped at 3.0 — the clip is the one nod to the *exploding* side of the
error-flow analysis, capping a parameter update without touching the cell equations. Let me check that
these settings actually buy more optimization steps, not just a bigger rate: on the same `~5·10⁵` training
stock-days, batch 800 gives about `620` steps per epoch versus the transformer's `~250` at batch 2048, and
patience 20 over up to 200 epochs lets the run breathe. So this rung takes both larger *and* more numerous
steps, under a far more forgiving stopping rule, on a model whose gradients do not vanish — every one of
those choices is the robustness lesson from the tree applied to a sequence model: train it the forgiving
way, not the high-wire way.

One design choice I should interrogate rather than inherit is the loss. The thing I ultimately care about
is a *ranking* — IC and Rank IC are cross-sectional correlations, and the portfolio only ever uses the
order of the scores, not their magnitudes — so an MSE regression loss looks like a mismatch: it penalizes
being numerically far from a target return even when the *order* is right. The tempting move is a listwise
or pairwise ranking loss that optimizes the ordering directly. But I decide against it, and the reason is
already sitting in the frozen label pipeline. The label passes through `CSRankNorm`, which replaces each
day's cross-section of returns with their normalized ranks before the model ever sees them. So the target
the MSE is fitting is *already* the cross-sectional rank, and squared-error to a rank target is, in
effect, a smooth ranking objective — a model that gets every stock's rank right drives the MSE to its
floor. That is why the benchmark can use plain MSE and still produce a good Rank IC: the ranking is
smuggled into the target, not the loss. Swapping in a bespoke ranking loss would add optimization
fragility (pairwise losses are notoriously sensitive to sampling and scale) for a gain the label
normalization has already largely delivered — precisely the kind of high-wire complexity this rung is
trying to get *away* from. So I keep the masked MSE, matching the transformer rung's loss, and let
`CSRankNorm` do the ranking work.

One processor note, because it is the inverse of the lgbm rung's edit and I should not get it backwards.
The LSTM, like the transformer, is a gradient-descent model that needs standardized inputs to train
stably, so this rung *keeps* the default workflow processor block — `RobustZScoreNorm` + `Fillna` on the
features — exactly as the transformer rung did, and does *not* strip it the way the tree rung did. The
neural preprocessing that I removed for the trees is the preprocessing the LSTM depends on: a NaN cannot
pass through the gate matrix multiplies at all, and un-standardized ratios of wildly different scales would
push the sigmoids straight into saturation on the first step — the very regime the carousel is meant to
avoid. So the `Fillna` and the clip-and-standardize that were harmful to the tree are load-bearing here.
(The full scaffold module is in the answer.)

So the delta from the LightGBM rung is: keep the optimization robustness that made the tree beat the
transformer, but restore the temporal inductive bias the tree discarded — interpret the same 360 numbers
as a 60×6 sequence and run a gated memory cell over it so a day at the start of the window can both
influence and receive gradient from the end, trained at a healthy learning rate with generous patience
and the neural feature normalization the tree did not need.

What do I expect against the measured numbers, and what is falsifiable? The cleanest claim is on csi300:
the LSTM should beat both prior rungs on IC and IR together — IC a clear step above the tree's 0.0399
and IR well above the tree's 0.280, because the temporal credit assignment should sharpen the
ranking and the forgiving training should let it actually converge, unlike the transformer. If the
csi300 IR does not clear the tree's 0.280, the temporal bias did not pay and the tree's robust tabular
view was already extracting most of the signal. The sharper, more uncertain test is csi100: that is where
the tree had a fine IC but a *negative* IR, the universe where — by the IC-to-IR decoupling I traced
across both prior rungs — ranking quality fails to survive the backtest. If the LSTM's temporal awareness
genuinely produces a better-ordered ranking, csi100's IR should improve over the tree's −0.334 — but I am
honestly unsure it will turn positive, because csi100's problem looks structural (a small universe where
holding 50 of 100 names and churning 5 a day is punishing regardless of signal quality) rather than a
signal-quality problem the LSTM can fix; if csi100 stays negative even as its IC improves, that confirms
the csi100 weakness is a portfolio-construction artifact, not a model failure. On csi300_recent I expect
the LSTM to at least match the tree's positive IR of 0.266. The single number I will judge this rung on is
the csi300 information ratio against the tree's 0.280: clearing it decisively is the evidence that
bringing back the time axis — done robustly — was the right top of the ladder.
