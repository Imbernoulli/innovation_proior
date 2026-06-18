The LightGBM rung did what I expected on the most important axis and left a precise gap on another. On
csi300 the IR turned **positive** — 0.280, up from the transformer's −0.44 — and the IC climbed to 0.0399
from 0.0117, with Rank IC 0.0492. That confirms the robustness diagnosis: a learner with no optimization
fragility formed real signal where the brittle transformer formed noise. On csi300_recent the same story,
IR 0.266, return positive. But csi100 came back exactly as I worried it might: a perfectly respectable IC
of 0.0363, yet a *negative* portfolio return (−0.017) and IR (−0.334). So the tree ranks csi100 stocks
about as well as it ranks csi300 stocks by IC, but that ranking quality does not survive the TopkDropout
backtest on the smaller universe. And there is a second, quieter signal in the numbers: the tree's IC
sits in the high-0.03s on the two csi300 universes, which is decent but not the ceiling — the LightGBM run
is deterministic (all three seeds identical), so there is no variance to blame; this is simply the
ranking quality a per-row tabular view of the window can extract. The tree threw away the temporal
structure of the 60×6 window, and on csi100 especially that lost structure is leaving ranking quality on
the table that a model *aware* of the time axis could recover.

So the directed move is to go back to a sequence model — but emphatically *not* the brittle transformer.
The transformer's failure was never its inductive bias; it was that its training was a high-wire act
(warmup-less Adam, init sensitivity, patience-5). I want the temporal inductive bias *with* the
optimization robustness the tree showed me matters most on this data. A recurrent net trained at a
healthy learning rate with generous early-stop patience is exactly that compromise: it reads the window
as a sequence, but it trains the way the tree was robust — no warmup needed, no delicate schedule. The
question is which recurrence, and here I have to be careful, because the obvious one fails for a reason I
can write down.

Take a plain recurrent net first, so I understand why it is not enough. Its state is `h_t = f(h_{t-1},
x_t)`, and I want a feature from day 1 of the window to be able to influence the read-out at day 60.
Follow a single error signal backward through the sixty steps. The error that lands on a unit at the end
and reaches a unit `q` steps earlier is, telescoped, a *product* of `q` factors, each of the form
`f'(net)·w`. With the logistic squashing the derivative peaks at 0.25, so each factor is below 1 whenever
`|w| < 4` — and any reasonable weight is — so the product over `q` steps decays geometrically. The
gradient from sixty days ago is exponentially attenuated by the time it reaches the read-out, and bigger
weights only push the unit toward saturation where `f'` collapses faster than `w` grows, so they make it
worse, not better. A bigger learning rate scales long-range and short-range credit identically, so the
*ratio* is unchanged and the recent days still dominate every update. The vanishing is structural — the
lag sits in the exponent, and no ordinary knob touches it. So a plain RNN over a 60-step window would
learn mostly from the last few days and be nearly as blind to day 1 as the tree was, just for a different
reason. That is not the fix I want.

The cure has to make the product over the lag *not* shrink — ideally exactly 1 for any `q`. In the
simplest setting, a single unit with a self-connection of weight `w_jj`, the per-step backward factor is
`f'(net)·w_jj`, and demanding it equal 1 forces `f'(net) = 1/w_jj`, a constant — the squashing must be
*linear*. Take the cleanest case, the identity with `w_jj = 1`: now the activation simply *persists*
unchanged step to step, and the backpropagated error riding through it is multiplied by exactly 1 each
step. A linear self-loop of weight one — a constant error carousel — is a channel down which gradient
survives an arbitrarily long lag. That is the seed.

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
plain RNN would have lost to the vanishing product. This is the right recurrence.

Now ground it in this task's edit surface, because there is a same-named subtlety I must respect. This
rung uses qlib's **non-TS** LSTM, wired to the ordinary `DatasetH` (not `TSDatasetH`). That means each
training sample is *still* the flat 360-dim Alpha360 row — the same rows the tree saw — and the time axis
is recovered *inside* the model by `x.reshape(N, 6, 60).permute(0, 2, 1) -> [N, 60, 6]`, not by the
dataset handing over genuine overlapping time-series windows. So the LSTM here is reconstructing the
60-step sequence from the same flat feature vector the tree treated as tabular; the difference is purely
that the LSTM *interprets* those 360 numbers as a 60×6 sequence and runs the gated recurrence over it,
while the tree saw 360 unordered columns. I run two LSTM layers (`num_layers=2`) of `hidden_size=64`
stacked, take the final step's hidden state — the cell's summary of the whole window conditioned on the
most recent day — and read it through a `Linear(64, 1)` to one score. The width 64 and depth 2 are again
modest by design: on a low-signal regression the danger is fitting noise, and a compact recurrent net is
a regularizer, the same logic that drove the narrow transformer and the heavily-penalized trees.

The training loop is where this rung deliberately differs from the transformer rung, and it is the whole
point. Adam at `lr = 1e-3` — *ten times* the transformer's `1e-4` — because the LSTM is far less
initialization-sensitive than the attention stack and can take a healthy step from the start; no warmup
schedule is needed. Early-stop patience is `20`, four times the transformer's 5, so the run is not cut
off on a noisy validation dip the way I suspect the transformer was. Batch size 800, MSE loss masked over
the finite labels, gradient value-clipped at 3.0 — the clip is the one nod to the *exploding* side of the
error-flow analysis, capping a parameter update without touching the cell equations. Up to 200 epochs,
restoring the best-validation parameters. Every one of these choices is the robustness lesson from the
tree applied to a sequence model: train it the forgiving way, not the high-wire way.

One processor note, because it is the inverse of the lgbm rung's edit and I should not get it backwards.
The LSTM, like the transformer, is a gradient-descent model that needs standardized inputs to train
stably, so this rung *keeps* the default workflow processor block — `RobustZScoreNorm` + `Fillna` on the
features — exactly as the transformer rung did, and does *not* strip it the way the tree rung did. The
neural preprocessing that I removed for the trees is the preprocessing the LSTM depends on. (The full
scaffold module is in the answer.)

So the delta from the LightGBM rung is: keep the optimization robustness that made the tree beat the
transformer, but restore the temporal inductive bias the tree discarded — interpret the same 360 numbers
as a 60×6 sequence and run a gated memory cell over it so a day at the start of the window can both
influence and receive gradient from the end, trained at a healthy learning rate with generous patience
and the neural feature normalization the tree did not need.

What do I expect against the measured numbers, and what is falsifiable? The cleanest claim is on csi300:
the LSTM should beat both prior rungs on IC and IR together — IC above the tree's 0.0399 (toward the
mid-0.04s) and IR well above the tree's 0.280, because the temporal credit assignment should sharpen the
ranking and the forgiving training should let it actually converge, unlike the transformer. If the
csi300 IR does not clear the tree's 0.280, the temporal bias did not pay and the tree's robust tabular
view was already extracting most of the signal. The sharper, more uncertain test is csi100: that is where
the tree had a fine IC but a *negative* IR, the universe where ranking quality failed to survive the
backtest. If the LSTM's temporal awareness genuinely produces a better-ordered ranking, csi100's IR
should improve over the tree's −0.334 — but I am honestly unsure it will turn positive, because csi100's
problem may be structural (a small universe where holding 50 of 100 names and churning 5 a day is
punishing regardless of signal quality) rather than a signal-quality problem the LSTM can fix; if csi100
stays negative even as its IC improves, that confirms the csi100 weakness is a portfolio-construction
artifact, not a model failure. On csi300_recent I expect the LSTM to at least match the tree's positive
IR. The single number I will judge this rung on is the csi300 information ratio against the tree's 0.280:
clearing it decisively is the evidence that bringing back the time axis — done robustly — was the right
top of the ladder.
