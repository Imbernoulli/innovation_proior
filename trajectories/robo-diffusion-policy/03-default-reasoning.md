IDQL did exactly what I predicted and the numbers draw the boundary of the inference-time-selection idea
sharply. It beat the diffusion-BC floor on every environment — hopper `0.49 -> 0.62`, walker2d
`0.66 -> 0.83`, halfcheetah `0.42 -> 0.48` — and the two seeds it ran are tight (hopper `0.630/0.617`,
walker2d `0.826/0.828`), so the gain is real and reproducible, not a lucky seed. Let me turn the arrows
into arithmetic, because the *shape* of the gains is the whole diagnosis. Hopper lifted `+0.134` (a `+27%`
relative jump), walker2d `+0.171` (`+26%`), halfcheetah `+0.053` (`+13%`). So the two branchy
environments got roughly double the *relative* lift of halfcheetah, and that confirms the core hypothesis
with a mechanism attached: the same diffusion-BC actor, plus a value function and an inference-time
advantage reranking, lifts the score in proportion to how much exploitable spread `mu` has between good
and mediocre actions. Halfcheetah barely moved because its `medium` buffer is a competent-but-low-ceiling
runner and even perfect selection over 50 samples from `mu` cannot conjure actions better than the best
`mu` would ever draw — its `+13%` is the signature of a nearly-exhausted ceiling. Walker2d, where `mu` has
a coherent gait with exploitable spread, jumped most in absolute terms. And hopper, at `0.62`, is still a
long way from the expert `1.0` — a `0.38` gap remaining after IDQL closed only `0.13` of the original
`0.51`. That last point is the diagnosis I have to act on: IDQL's actor is *pure BC*. It only ever samples
from `mu`, and the critic merely picks the best of those samples. The ceiling of "best of 50 samples from
`mu`" is bounded by what `mu` puts mass on. If the genuinely good action lives in a region `mu` samples
rarely or never, no amount of reranking will surface it — the candidate set never contains it, and the
`0.38` hopper gap is where that ceiling bites.

Before I reach for the fix, I should ask whether the cheaper move — just make the selection stronger —
could close that gap, because if it could I would not need to touch actor training. Two knobs are
available: raise `weight_temperature` toward the argmax limit, or raise `num_candidates` past `50`. Walk
the second concretely. IDQL already draws `50` candidates and takes essentially the best; drawing `N`
candidates from `mu` and keeping the max gives the max of `N` i.i.d. draws, whose expected value grows
only like the tail of `mu`'s advantage distribution — logarithmically slow once `N` is already `50`, and
*hard-capped* by the support of `mu`. If the good hopper action has probability near zero under `mu`, then
`P(at least one of N candidates is good) = 1 - (1 - p)^N`, and for `p ~ 0` this stays near zero even at
`N = 500`. So more candidates buys a vanishing return against a support wall; the halfcheetah `+13%`
already shows selection running into that wall. Raising `weight_temperature` is worse — it only sharpens
the pick among the *same* candidate set, so it cannot add an action the set never contained, and pushed
too far it just amplifies whichever candidate got the luckiest critic read. Both knobs are variations on
"select better from `mu`," and the hopper stall is precisely the statement that selecting from `mu` is not
enough. So the obvious next move is to stop merely *selecting* good actions and start *training the actor
to prefer them*: push the gradient of `Q` into the actor's own parameters so that the distribution it
samples from drifts toward high-`Q` regions. Then the candidate set itself improves — at inference the
actor is already sampling near the good actions, and selection only sharpens. This is the structural
difference from IDQL — value enters *during actor training*, not only at inference — and it is exactly the
lever that can break past the "best-of-`mu`" ceiling that capped IDQL on hopper.

Let me derive what that actor objective should be, because the danger is the same OOD wall as before, just
relocated. I want the actor to maximize `E_{a ~ pi}[Q(s,a)]` — sample an action from the diffusion actor
at state `s` and ascend `Q` on it. But if I *only* maximize `Q`, the actor will happily march off the data
support to wherever the critic is erroneously optimistic, and I am back to exploiting phantom optima. The
fix is a behavior-regularized objective: keep the diffusion-BC term as an anchor that pins the actor near
`mu`, and add the `Q`-maximization term as the pull toward better actions. So the actor loss is
`L_actor = L_BC + eta * L_Q`, where `L_BC` is the noise-prediction denoising loss (the same objective that
the floor and IDQL both used to clone `mu`, and that I know from the floor's `3%` seed spread trains
stably) and `L_Q = -E[Q(s, a_new)]` with `a_new` *freshly sampled from the actor's own reverse chain*, so
the gradient flows through the diffusion sampler back into the actor's weights. The coefficient `eta`
trades cloning against improvement, and I can read its two limits: at `eta = 0` the actor is pure BC and I
have literally recovered the floor's training (and IDQL's actor), so DQL contains the previous rungs'
actor as its `eta -> 0` boundary; as `eta` grows the actor chases `Q` harder but risks the OOD blow-up. So
`eta` is the single dial that slides from "clone `mu`" to "maximize `Q`," and the whole bet of this rung
is that a modest `eta > 0` lands in the band where the actor places mass on actions *better* than any
single `mu` mode while the BC anchor keeps it from leaving the support entirely — the band that selection
alone could not reach because selection never moves `mu` itself.

Two things need care, and both are visible in the harness's exact form. First, sampling the new action
*through* the diffusion chain with gradients enabled (`requires_grad=True`) is what makes `Q`-maximization
meaningful — the actor's parameters move so that the *generated* action has higher `Q`, not so that some
external action does. The default fill samples `new_act` from the actor with `use_ema=False` (I want
gradients on the live weights, not the EMA copy) and `requires_grad=True`, then evaluates `Q` on it inside
a `FreezeModules([critic])` block so the `Q`-gradient updates only the actor, not the critic. Second — and
this is the subtle, harness-specific part I must get exactly right — the `Q`-maximization term needs a
scale that does not let one of the twin heads dominate or blow up. The default uses a *randomized,
normalized double-Q* trick: with probability one half it forms `q_loss = - q1.mean() /
q2.abs().mean().detach()`, otherwise `- q2.mean() / q1.abs().mean().detach()`. The numerator is the head
being maximized; the denominator is the *detached* absolute scale of the *other* head. Put arithmetic on
why the normalization is not optional. On these MuJoCo tasks the per-step reward is order `1` and returns
accumulate over ~1000 steps at `discount = 0.99`, so `Q` values sit at a scale of order `10^1` to `10^2`.
If I used the textbook `-Q.mean()`, the `Q`-gradient would enter the actor at a magnitude ~`10` to `100`x
the BC gradient (which is an MSE on box-`[-1,1]` actions, order `1`), so `eta` would have to be retuned
per environment to undo whatever the return scale happened to be, and a single `eta` could not possibly
work across hopper, walker2d, and halfcheetah at once. Dividing by the detached other-head magnitude
`q.abs().mean()` rescales the `Q`-term to roughly *unit* magnitude, so `eta` multiplies a normalized
quantity and has a stable, dataset-independent meaning — one `eta` for all three environments. Randomizing
which head is numerator versus denominator each step symmetrizes the two critics and prevents the actor
from over-fitting to one head's idiosyncratic over-estimates, the same twin-head hygiene that IDQL got
from its `q_min`. This is not the textbook `-Q.mean()`; it is the literal mechanism this harness exposes,
and it is what makes the BC+Q balance robust across the three environments without per-env retuning of
`eta`. I derive against *this*, not the generic version.

The critic side, by contrast, is simpler than IDQL's — and that simplification is itself the point. DQL
does not need the expectile/value-net machinery, because its actor is no longer pure BC: the actor is
being pushed toward high-`Q` actions, so the natural critic target uses the actor's *own* next action
rather than an in-sample expectile. The critic is a twin-Q (`DQLCritic`), and the TD target is
`r + gamma (1-done) * min(Q1_target(s', a'), Q2_target(s', a'))` with `a'` sampled from the diffusion
actor at `s'`. This *does* query an action the actor proposes at `s'`, which is precisely the OOD query
IDQL refused — but here it is tolerable for two linked reasons: the actor is BC-anchored so `a'` stays
near the support, and the twin-min `torch.min(*critic_target(next_obs, next_act))` is the standard
clipped-double-Q underestimation pressure that fights the residual overestimation. The two mechanisms are
complementary — the BC anchor bounds *how far* `a'` can drift from data, and the min bounds *how
optimistic* the value of that drift can read — so the aggressive bootstrap that would diverge on its own is
fenced in from both sides. That is why DQL can afford the query IDQL structurally banned: IDQL had no
actor anchor at training time, so it *had* to refuse the query; DQL earns it back by regularizing. So DQL
trades IDQL's strict in-sample safety for a more aggressive but still-regularized bootstrap, and that
aggression is the source of its higher ceiling: the critic can value actions slightly better than `mu`'s,
and the actor is trained to produce them. The critic loss is the plain `F.mse_loss(current_q1, target_q) +
F.mse_loss(current_q2, target_q)`, Polyak target update at `0.995`, Adam at `3e-4`, cosine scheduler — all
the usual choices, no expectile, no separate `V`.

It is worth being explicit about what "the gradient flows through the sampler" means mechanically,
because that is the one part of this rung that has no analogue in the two below and it is where a subtle
bug would hide. When I sample `new_act` with `requires_grad=True`, the reverse chain is not detached: each
of the 5 denoising steps calls the actor network `eps_theta(a^k, obs, k)`, and every one of those calls
keeps its computation graph, so the final `a^0 = new_act` is a differentiable function of the actor's
weights *through all five steps*. Then `q1_new_action, q2_new_action = critic(obs, new_act)` inside
`FreezeModules([critic])` makes `q_loss` a function of `new_act` and hence of the actor weights, and
`actor_loss.backward()` propagates `d q_loss / d(weights)` back through the whole chain. Two consequences
follow that I have to respect. First, this is why the sample must use `use_ema=False`: the EMA copy is a
detached running average with no live graph, so sampling through it would give a `q_loss` with zero
gradient to the trainable weights — the `Q`-term would silently do nothing, and the only way I would catch
it is that DQL would collapse to the floor. Second, this is where the ~10x training cost comes from: five
graph-retaining forward passes per actor step for `new_act`, plus five more (detached, `requires_grad
=False`) for the critic's `next_act`. The `FreezeModules` guard is not cosmetic either — without it the
`Q`-gradient would also flow into the critic and the actor could "improve" its `Q` by corrupting the
critic downward, a degenerate optimum; freezing the critic forces the actor to actually move the action.

I should also justify *why* the critic target uses an actor-sampled `a'` rather than reusing IDQL's
in-sample expectile machinery, since I proved that machinery works one rung down. The reason is that the
two designs answer different actors. IDQL's `V` was the expectile of `Q` over `mu`, i.e. the value of a
*fixed* reweighting of the behavior policy — appropriate because IDQL's actor never moved off `mu`. DQL's
actor *does* move: it is trained toward high-`Q` regions, so the value I need to bootstrap is the value of
*that* actor at `s'`, which is exactly `Q(s', a')` with `a' ~ pi(.|s')` — sampling the current actor at
the next state. Reusing an in-sample expectile `V(s')` here would bootstrap the value of `mu`, not the
value of the actor I am actually training, and would leave the critic systematically pessimistic about the
improved actions the actor is learning to take, choking off the very improvement that is the point of the
rung. So the actor-sampled bootstrap is not a simplification of convenience; it is the target that matches
a moving actor, and the twin-min plus BC anchor are what make sampling that (mildly OOD) `a'` safe. This
is the clean division of labor across the ladder: IDQL keeps the actor fixed and puts all the cleverness
in an in-sample critic; DQL lets the actor move and pays for it with a coupled, actor-aware critic.

Now the literal edit, which is the *default fill* — the unmodified template — so the "edit" here is
"change nothing" (`OPS = []`). But I want to state precisely how it differs from the two rungs below so
the ladder is legible. Versus diffusion BC: DQL adds the entire `DQLCritic` (twin-Q + target +
optimizer), adds the critic TD-update reading `next_obs/rew/tml`, and replaces the actor's pure BC step
with `actor.loss(act, obs) + eta * q_loss` where `q_loss` flows through a gradient-enabled actor sample.
Versus IDQL: DQL swaps the actor backbone back to `DQLMlp` and the critic from `IDQLQNet + IDQLVNet`
(expectile `V`, in-sample SARSA) to a single `DQLCritic` twin-Q with an actor-sampled bootstrap target; it
*adds* the `eta * q_loss` actor term that IDQL deliberately omits; and at inference it reranks
`num_candidates` actions by a softmax over the *bare* `Q` (`critic_target.q_min`) rather than over the
*advantage* `Q - V`. That last difference is consistent with the training difference: IDQL needed the
advantage because its value baseline `V` was the expectile and subtracting it centered each state's
candidate scores; DQL has no `V`, so it reranks on `Q` directly, and because the actor is already trained
toward high `Q`, both the candidate set and the ranking pull the same direction — the state-offset problem
that made me subtract `V` in IDQL matters less here because the actor has already concentrated the
candidates. The inference block repeats each observation `num_candidates` times, samples that many
actions, computes `q = critic_target.q_min(obs, act).view(-1, num_candidates, 1)`, forms
`w = softmax(q * weight_temperature, dim=1)`, and resamples one with `multinomial`. Note that since DQL's
actor was *trained* to prefer high-`Q` actions, the reranking matters less than it does for IDQL (where it
was the only value mechanism) — the heavy lifting has already happened in the actor's weights, and the
inference reranking is a final sharpening on an already-good candidate set rather than the sole source of
improvement.

I also want to settle the training schedule, since it differs from IDQL's every-other-step critic. DQL
updates the critic and the actor on *every* gradient step (not every other), because both are now doing
coupled work — the critic has to keep up with an actor that is actively moving toward high-`Q` regions, and
a stale critic would let the actor exploit lag, ascending a `Q` the critic has not yet corrected. IDQL
could afford to update its critic every other step precisely because its critic chased a *fixed* target
(the value of a `mu` that never moved); DQL's critic chases a moving actor, so it cannot fall behind. The
EMA on the actor still kicks in only after 1000 steps, and the critic target is Polyak-updated every
`ema_update_interval` steps. This tighter coupling is more expensive than IDQL, and I can count the cost:
every DQL training step samples *two* full 5-step diffusion chains (one for the critic's `next_act`, one
for the actor's `new_act`), where the floor sampled *none* in training and IDQL sampled none in training
either (IDQL only sampled at inference; its training was the floor's single-forward-pass BC step plus cheap
MLP critic updates). So per gradient step DQL does on the order of `2 * 5 = 10` network forward passes
through the actor where the floor and IDQL did `1`. That ~10x per-step training cost is the reason DQL is
the slowest rung to train, and I should expect its walltime to dwarf both others — roughly an
order-of-magnitude over the floor's `~2300s`, and several times IDQL's own inflated cost.

One arithmetic point about the aggregate is worth pinning, because it tells me where the geometric mean
will actually be won. The final score is the geometric mean over the three environments, and a geometric
mean is dominated by its *smallest* factor: computing IDQL's, `(0.6235 * 0.8268 * 0.4756)^(1/3) = 0.626`,
and the floor's, `(0.49 * 0.6558 * 0.4224)^(1/3) = 0.514`, I can see halfcheetah's `~0.48` is the factor
dragging the product down, and a fixed *relative* improvement on the low halfcheetah factor moves the
geometric mean more than the same relative improvement on the already-high walker2d factor. That would
argue for pouring effort into halfcheetah — except my selection-ceiling analysis says halfcheetah is
precisely where the `medium` buffer offers the least headroom (IDQL's `+13%` there against `+27%` on
hopper). So the aggregate lift from DQL has to come from the environments where actor improvement is
actually *possible*, hopper above all, and it has to be large enough there to move the geometric mean even
though the halfcheetah factor barely budges. Concretely, if DQL takes hopper from `0.62` toward the
expert `1.0` while halfcheetah only inches up near its `0.48` ceiling, the geometric mean still rises
because the hopper factor moves a lot in absolute terms; I am counting on the branchy environments to carry the aggregate past IDQL's
`0.626`, not on prying open halfcheetah's near-exhausted ceiling.

My falsifiable expectations against the rungs below. DQL must beat IDQL on hopper and (especially) where
the best action lives outside `mu`'s frequent samples, because training the actor toward high `Q` breaks
the "best-of-`mu`" ceiling that capped IDQL at `0.62` on hopper — I expect hopper to approach the
expert-level `~1.0` normalized score, since hopper is the task where `medium`-buffer stitching toward a
clean hop pays off most and where IDQL left the largest gap (`0.38`) to close. On halfcheetah I expect a
smaller gain (IDQL was already near the `medium` ceiling at `0.48`, and my own selection-ceiling argument
above says the dataset simply does not contain much better); DQL's training-time `Q`-maximization can push
it modestly higher, but the dataset ceiling is low for everyone and I expect only a small gain there.
Walker2d, already strong for IDQL at `0.83`, should hold or improve slightly. The geometric mean across the three
must come out clearly above IDQL's — that is the claim that training-time `Q`-maximization beats
inference-time selection on these datasets, and it is exactly the claim IDQL's hopper stall set up. The
cost is walltime: DQL trains two diffusion chains per step plus a coupled critic, so I expect it to be
several times slower than IDQL and far slower than the floor; if it were not slower, something would be
wrong with the gradient-enabled sampling — a flat `training_time` would mean the reverse chains were not
actually being run with gradients (the `use_ema=False`, `requires_grad=True` sampling silently detached),
and the `Q`-term would be doing nothing — which would show up as DQL collapsing back onto IDQL's numbers. If DQL failed to beat IDQL,
that would falsify the whole premise that moving value from inference into actor training helps here — but
the hopper ceiling IDQL hit is precisely the evidence that it should, because that ceiling is exactly the
symptom that no amount of selection from `mu` can cure. The full scaffold module — the twin-Q critic, the
coupled BC+Q actor training with the randomized normalized double-Q `q_loss`, and the `Q`-reranking
inference block — is in the answer.
