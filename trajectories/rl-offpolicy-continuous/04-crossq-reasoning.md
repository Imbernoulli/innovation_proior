TD3 won the leaderboard, and it won it the way I hoped — by being strong on all three environments at
once rather than trading one region for another. HalfCheetah recovered to 11055 (from SAC's 10360),
Reacher snapped back to −3.83 (from SAC's −4.76, essentially level with DDPG's −3.94), and Ant came in
highest of all, 4642 (from SAC's 3577). So the deterministic actor reclaimed the entropy tax exactly
where I predicted, and the twin-min carried over the overestimation control that fixed Ant. But the
residual I flagged at the close of rung 3 is sitting right there in the Ant column: the per-seed Ant
returns are {4763, 3407, 5757} — a spread of well over two thousand, the widest of any environment on
the board. That is not a strong-but-stable result; it is a strong-on-average result with one seed
(3407) lagging the best (5757) by 40%. And I named the suspect already: every rung so far, DDPG, SAC,
TD3, learns its critic through *stale target networks*. The smoothed twin-min target TD3 reads is
computed off slow copies `Q_{θ'}`, lagged by `tau=0.005`, and `tau` is a knob balancing a deliberately
delayed signal against jitter. A lagged bootstrap is a slower, noisier learning signal, and on the
high-dimensional task where the critic has the most to learn and the most room to be wrong, that
staleness is the most likely source of the seed that fell behind. So the move is to attack the target
network itself.

Let me re-derive *why* the target network is there, because the folklore answer — "the bootstrap would
chase itself" — is not specific enough to attack. The critic minimizes `(Q_θ(s,a) − [r + γ Q_θ(s',a')])²`.
Standard practice already detaches the next-state value, so there is no degenerate "shrink the target"
gradient path; the question is what diverges when that detached value is computed from the *live*
critic instead of a frozen copy. Two things. One, the target value moves every step because θ moves —
the regression objective is non-stationary, the DQN story. Two, and I think this is the one actually
doing the damage, the input distributions at `(s,a)` and `(s',a')` are *not the same*. The actions `a`
at the current state come from the replay buffer — a mixture of stale policies. The actions `a'` at the
next state come from the *current* policy. Those are different action distributions over different
states, and a single critic is being asked to be accurate on both and to relate them through the
Bellman equation. A deep net's behavior on one input cloud is unconstrained by its behavior on
another; if the critic drifts to different scales on the `(s,a)` cloud versus the `(s',a')` cloud — and
nothing prevents that — the bootstrap relates two numbers on different scales and the recursion
amplifies the mismatch. The target network masks this by freezing one side, so the mismatch cannot run
away; but it does not *address* it, it just slows everything enough to contain it, at the cost of the
staleness that is hurting Ant.

That reframing is the lever. If the real disease is a *distribution mismatch between the current and
next state-action inputs to one critic*, then the cure is not "freeze one side" — it is "make the two
sides share a normalized distribution so the critic sees them as one population." And the standard tool
whose entire job is to normalize a layer's inputs to a controlled distribution is Batch Normalization.
So: put normalization inside the critic and remove the very mismatch the target network was
compensating for. If it works, the target network is redundant and I delete it — which kills the
staleness *and* the `tau` knob at once.

But BatchNorm has a bad reputation in value learning, and I should reproduce the failure before I trust
a fix. Insert vanilla BatchNorm the obvious way: forward `(s,a)` through the critic for the prediction,
*separately* forward `(s',a')` (no-grad) for the bootstrap value, form the target, regress. In the
first pass BatchNorm normalizes by the `(s,a)` batch's mean/variance; in the second pass by the
`(s',a')` batch's mean/variance. Those statistics differ — exactly because the two batches have
different distributions, the mismatch I am trying to fix — so the critic is literally a *different
function* in the two passes: same weights, different normalization, different effective transform. The
Bellman equation now relates the outputs of two non-identical functions, which is incoherent, and the
separately-updated running statistics lurch. That is precisely why naive BatchNorm destabilizes
critics, and naming the cause names the fix.

Do not let the normalization see the two batches as two populations. *Concatenate* them: stack `(s,a)`
on top of `(s',a')` into one batch of size `2N`, do a *single* forward pass through the normalized
critic, split the output into the current-state predictions and the next-state values. Now the
normalization computes its moments from the *union* of both sub-batches — one shared distribution — so
every input, current or next, is normalized identically and the critic is one consistent function
across both. The prediction and the bootstrap target now live on the same normalized scale by
construction. This is almost free: one concatenation, one forward pass (cheaper than two separate
passes), one split. The next-state half is detached before forming the target — it is a bootstrap
value, no gradient flows into it — but it shares the forward pass and therefore the normalization with
the current-state half. That shared normalization is what the target network was crudely approximating,
and now I get it exactly, from the *live* critic, with no lag. So: normalized critic, current and next
forwarded jointly, and no target network at all.

Two roles the target network played — stationarity and cross-batch consistency — and the joint-batch
normalization takes over both. It directly kills the cross-batch mismatch (both sides share moments),
and it substantially tames the non-stationarity, because normalization fixes the scale and center of
each layer's activations every step, so the representation the later layers see is far more stationary
than raw activations even as θ moves. The target network has nothing left to do; delete it. The
bootstrap value is the live critic's next-state output, detached, normalized jointly with the
prediction.

One refinement, because plain BatchNorm is not quite enough under RL's non-stationarity. BatchNorm
normalizes by *batch* statistics in training but by *running* statistics at inference, and that gap is
benign with large i.i.d. batches and harmful here: the policy is changing, the replay distribution
drifts, the running statistics chase a moving target, and a minibatch's stats can swing far from them.
Batch Renormalization (Ioffe 2017) is the patch: keep normalizing by batch statistics but add a
clipped affine correction `(r, d)` that ties the batch normalization back to the running statistics,
with `r` clamped to `[1/r_max, r_max]`, `d` to `[−d_max, d_max]`, and both detached (constants, not
differentiated). That keeps training-time and running-time normalization consistent under drifting
data — the robustness the critic needs. So the critic's normalization layer is BatchRenorm, with a
small momentum on the running stats and the corrections stop-gradiented.

Now everything else stays SAC, deliberately, because the contribution is a normalization change and I
do not want to confound it. The actor is the stochastic tanh-Gaussian, reparameterized, with the
squash log-prob correction — the same actor rung 2 used, returning `(action, log_prob, mean)` so the
loop's `eval_actor` unwraps it. Twin critics with a `min` next-state target, minus the entropy term,
and automatic temperature tuning to `H_target = −dim(A)`. The maximum-entropy objective is back, but
note the difference from rung 2's concern: SAC's entropy tax came from a *deliberately* stochastic
*deterministic-task* policy at evaluation, and here evaluation still reads the policy mean
(`mean_action`), so the entropy serves exploration during training while evaluation stays the sharp
mean — and the real lever this rung pulls is not the actor at all, it is the normalized,
target-free critic. The entropy temperature self-tunes per environment, which is what I want across
HalfCheetah, Reacher, and Ant.

This task's harness shapes two things I must be honest about, because the baseline named after this
method here is not the paper's configuration. First, the network *dimensions are fixed* — a
parameter-count check enforces 256-wide layers — so I cannot widen the critic to the paper's 2048
units. The paper's accuracy comes partly from that widening, which the normalization makes trainable;
here I keep 256-wide critics and rely only on the normalization-plus-no-target mechanism, so I should
expect the *mechanism's* benefit (un-stale bootstrap, consistent cross-batch scale) without the
extra-capacity benefit. Second, with the target network gone there is no `tau` and the critic learns
from a fresh signal, so I update the actor *every* step (policy delay 1) rather than every two — the
critic no longer needs to settle behind a lagged target before each policy move, because there is no
lag. I also keep the BatchRenorm eps at the scaffold's numerical setting rather than the paper's, and
toggle the critics to eval mode for the actor's Q-evaluation (the actor reads the critic on a single
current-state batch, which should use running statistics, not recompute batch stats on a one-sided
batch). So the scaffold delta from TD3 is: bring back SAC's stochastic actor and entropy tuning, put
BatchRenorm into the `QNetwork`, forward current and next state-action jointly through each twin critic,
delete both target critics and the target actor and all soft-updates, and update the actor every step.
The full scaffold module is in the answer.

The bar this has to clear is TD3's measured board, and the claims are falsifiable seed by seed. The
headline test is Ant: if the staleness diagnosis is right, removing the target network should
*tighten* the Ant seed spread that TD3 left at {4763, 3407, 5757} — the lagging 3407 seed should come
up, because an un-stale bootstrap learns the high-dimensional critic faster and more consistently — and
the Ant mean should land at or above TD3's 4642. If Ant's spread stays as wide, the target network was
not the cause and the variance is coming from somewhere else. HalfCheetah and Reacher are near
saturation for the good methods (TD3 at 11055 and −3.83), so the test there is *holding*, not jumping:
the normalized critic and the entropy-explored-but-mean-evaluated actor should match TD3 on both, and a
*drop* on Reacher would mean the residual evaluation behavior is not as sharp as TD3's deterministic
policy. There is a real risk specific to this constrained version: stripped of the paper's critic
widening, the normalization-only CrossQ may not separate from TD3 by much — the honest expectation is
parity-to-better with a tighter Ant, not a blowout. What this rung adds to the ladder is not a bigger
number guaranteed; it is the demonstration that the target network — the one fixture every prior rung
inherited unquestioned — is removable, and that the seed-to-seed instability still visible in the
strongest baseline's Ant column is exactly what a target-free, jointly-normalized critic is built to
cure. The leaderboard has no CrossQ row to confirm it; what the trajectory ends on is the construction
whose joint-batch normalization makes the stale bootstrap unnecessary.
