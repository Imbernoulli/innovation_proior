IDQL did what I predicted and the numbers draw the boundary of inference-time selection sharply. It beat
the diffusion-BC floor on every environment — hopper `0.49 -> 0.62`, walker2d `0.66 -> 0.83`, halfcheetah
`0.42 -> 0.48` — with tight seeds (hopper `0.630/0.617`, walker2d `0.826/0.828`), so the gain is real and
reproducible. In relative terms hopper lifted `+27%`, walker2d `+26%`, halfcheetah only `+13%`: the two
branchy environments got roughly double the relative lift, confirming the core hypothesis with a mechanism
— the same BC actor plus value-based reranking lifts the score in proportion to how much exploitable
spread `mu` has between good and mediocre actions. Halfcheetah barely moved because its `medium` buffer is
a competent-but-low-ceiling runner and even perfect selection over 50 samples from `mu` cannot conjure
actions better than the best `mu` would draw. But hopper, at `0.62`, is still `0.38` short of expert,
having closed only `0.13` of the original `0.51` gap. That is the diagnosis I must act on: IDQL's actor is
*pure BC*, sampling only from `mu`, and the critic merely picks the best of those samples. The ceiling of
"best of 50 samples from `mu`" is bounded by what `mu` puts mass on; if the genuinely good hopper action
lives where `mu` samples rarely or never, no reranking surfaces it — the candidate set never contains it —
and the `0.38` gap is where that ceiling bites.

Before reaching for the fix I should ask whether merely stronger selection could close the gap, because if
it could I would not need to touch actor training. Two knobs: raise `weight_temperature` toward the argmax
limit, or raise `num_candidates` past `50`. Drawing `N` candidates from `mu` and keeping the max gives the
max of `N` i.i.d. draws, whose expected value grows only like the tail of `mu`'s advantage distribution —
logarithmically slow once `N = 50`, and *hard-capped* by the support of `mu`: if the good action has
probability near zero under `mu`, `P(any of N good) = 1 - (1-p)^N` stays near zero even at `N = 500`.
Raising `weight_temperature` is worse — it only sharpens the pick among the *same* candidate set, so it
cannot add an action the set never contained, and pushed too far just amplifies whichever candidate got
the luckiest critic read. Both knobs are "select better from `mu`," and the hopper stall is precisely the
statement that selecting from `mu` is not enough. So the move is to stop merely *selecting* good actions
and start *training the actor to prefer them*: push the gradient of `Q` into the actor's own parameters so
the distribution it samples drifts toward high-`Q` regions. Then the candidate set itself improves, and
selection only sharpens it. That is the structural difference from IDQL — value enters *during actor
training*, not only at inference — and it is the lever that can break the best-of-`mu` ceiling.

The danger is the same OOD wall, relocated. I want the actor to maximize `E_{a~pi}[Q(s,a)]` — sample from
the diffusion actor and ascend `Q` — but pure `Q`-maximization marches off the support to wherever the
critic is erroneously optimistic. So I keep a behavior-regularized objective: `L_actor = L_BC + eta * L_Q`,
where `L_BC` is the denoising loss anchoring the actor near `mu` (the same objective whose `3%` seed spread
I know trains stably) and `L_Q = -E[Q(s, a_new)]` with `a_new` *freshly sampled through the actor's own
reverse chain*, so the gradient flows through the sampler back into the actor's weights. At `eta = 0` this
is literally the floor's (and IDQL's) pure-BC actor, so DQL contains the previous rungs as its `eta -> 0`
boundary; as `eta` grows the actor chases `Q` harder but risks the OOD blow-up. The bet of this rung is
that a modest `eta` lands in the band where the actor places mass on actions *better* than any single `mu`
mode while the BC anchor keeps it inside the support — the band selection alone could not reach because
selection never moves `mu`.

Two things need care, both visible in the harness's exact form. First, sampling `new_act` through the
chain with `requires_grad=True` and `use_ema=False` is what makes `Q`-maximization meaningful: the reverse
chain is not detached, so each of the 5 denoising calls to `eps_theta(a^k, obs, k)` keeps its graph and
`new_act` is a differentiable function of the actor's weights *through all five steps*; the `Q`-gradient
then moves the weights so the *generated* action has higher `Q`. This is why the sample must not use the
EMA copy — the EMA is a detached running average with no live graph, so sampling through it would give a
`q_loss` with zero gradient to the trainable weights, silently doing nothing, detectable only as DQL
collapsing to the floor. The `Q` evaluation sits inside `FreezeModules([critic])` so the gradient updates
only the actor, not the critic — without it the actor could "improve" its `Q` by corrupting the critic
downward, a degenerate optimum.

Second, the `Q`-term needs a scale that does not let one twin head dominate. On these tasks per-step reward
is order `1` and returns accumulate over ~1000 steps at `discount = 0.99`, so `Q` sits at order
`10^1`–`10^2`. Textbook `-Q.mean()` would enter the actor at `~10`–`100x` the BC gradient (an MSE on
box-`[-1,1]` actions, order `1`), so `eta` would need per-environment retuning to undo the return scale —
no single `eta` could work across hopper, walker2d, and halfcheetah at once. The default instead uses a
randomized normalized double-Q: with probability one half `q_loss = - q1.mean() / q2.abs().mean().detach()`,
else the symmetric form. The detached other-head magnitude rescales the `Q`-term to roughly unit
magnitude, so `eta` multiplies a normalized quantity with a stable, dataset-independent meaning;
randomizing which head is numerator symmetrizes the critics and stops the actor over-fitting one head's
idiosyncratic over-estimates. I derive against this literal mechanism, not the generic `-Q.mean()`.

The critic is simpler than IDQL's, and that simplification is the point. DQL's actor is no longer pure BC —
it is pushed toward high-`Q` actions — so the natural TD target uses the actor's *own* next action: a
twin-Q `DQLCritic` with target `r + gamma(1-done) min(Q1_t(s',a'), Q2_t(s',a'))`, `a'` sampled from the
actor at `s'`. This *does* query an action the actor proposes at `s'` — the OOD query IDQL refused — but it
is tolerable here because the actor is BC-anchored so `a'` stays near support, and the clipped twin-min is
the standard underestimation pressure fighting residual overestimation. The two fences are complementary:
the anchor bounds *how far* `a'` drifts from data, the min bounds *how optimistic* its value can read.
IDQL had no actor anchor at training time, so it *had* to refuse the query; DQL earns it back by
regularizing, and that aggression is the source of its higher ceiling — the critic can value actions
slightly better than `mu`'s, and the actor is trained to produce them. Reusing IDQL's in-sample expectile
`V(s')` here would bootstrap the value of `mu`, not of the moving actor I am training, leaving the critic
systematically pessimistic about exactly the improved actions that are the point of the rung. So the
actor-sampled bootstrap is the target that matches a moving actor, not a convenience. The critic loss is
plain double MSE, Polyak `0.995`, Adam `3e-4`, cosine scheduler — no expectile, no separate `V`.

The literal edit is the *default fill* — the unmodified template, so `OPS = []`. Versus diffusion BC it
adds the whole `DQLCritic` (twin-Q + target + optimizer), the critic TD update reading `next_obs/rew/tml`,
and replaces the pure-BC step with `actor.loss(act, obs) + eta * q_loss` through a gradient-enabled actor
sample. Versus IDQL it swaps the backbone back to `DQLMlp`, the critic from `IDQLQNet + IDQLVNet` to a
single `DQLCritic` with an actor-sampled bootstrap, *adds* the `eta * q_loss` IDQL omits, and at inference
reranks by softmax over the *bare* `Q` (`critic_target.q_min`) rather than the advantage `Q - V` —
consistent because IDQL needed `V` to center each state's candidates, whereas DQL's actor has already
concentrated the candidates toward high `Q`, so reranking is a final sharpening on an already-good
candidate set rather than the sole value mechanism.

The schedule also differs: DQL updates critic and actor on *every* step (not every other), because both
are now coupled — the critic has to keep up with an actor actively moving toward high-`Q` regions, and a
stale critic would let the actor ascend a `Q` it has not yet corrected. IDQL could update every other step
precisely because its critic chased a *fixed* target (the value of a `mu` that never moved). The cost:
every DQL step samples *two* full 5-step chains (critic's `next_act`, actor's `new_act`) where the floor
and IDQL sampled none in training — on the order of `2 * 5 = 10` actor forward passes per step against
their `1`. So DQL is the slowest rung by far, and I expect its walltime several times IDQL's and roughly
an order of magnitude over the floor's `~2300s`.

One aggregate point tells me where the geometric mean is won. It is dominated by its smallest factor:
IDQL's is `(0.6235 * 0.8268 * 0.4756)^(1/3) = 0.626`, the floor's `(0.49 * 0.6558 * 0.4224)^(1/3) = 0.514`,
and halfcheetah's `~0.48` is the drag. A fixed *relative* improvement on the low halfcheetah factor moves
the mean most — except halfcheetah is exactly where the `medium` buffer offers the least headroom (IDQL's
`+13%` there against `+27%` on hopper). So DQL's aggregate lift has to come from where actor improvement is
actually possible, hopper above all, and be large enough there to move the mean even as the halfcheetah
factor barely budges.

Against the fills below: DQL must beat IDQL on hopper and wherever the best action lives outside `mu`'s
frequent samples, because training the actor toward high `Q` breaks the best-of-`mu` ceiling that capped
IDQL at `0.62` — largest gain on hopper, where `medium`-buffer stitching toward a clean hop pays off most
and IDQL left the biggest gap. On halfcheetah I expect only a small gain (near its low ceiling already),
and walker2d to hold or improve slightly. The geometric mean must come out clearly above IDQL's `0.626` —
the claim that training-time Q-maximization beats inference-time selection on these datasets. The full
module is in the answer.
