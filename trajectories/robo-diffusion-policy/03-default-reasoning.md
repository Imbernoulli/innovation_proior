IDQL did exactly what I predicted and the numbers draw the boundary of the inference-time-selection idea
sharply. It beat the diffusion-BC floor on every environment — hopper `0.49 -> 0.62`, walker2d
`0.66 -> 0.83`, halfcheetah `0.42 -> 0.48` — and the two seeds it ran are tight (hopper `0.630/0.617`,
walker2d `0.826/0.828`), so the gain is real and reproducible, not a lucky seed. That confirms the core
hypothesis: the same diffusion-BC actor, plus a value function and an inference-time advantage
reranking, lifts the score wherever `Q`/`V` carry useful information. But look at *where* it lifted and
where it stalled. Walker2d, where `mu` has a coherent gait with exploitable spread, jumped most (`+0.17`).
Halfcheetah barely moved (`0.42 -> 0.48`), because its `medium` buffer is a competent-but-low-ceiling
runner and even perfect selection over 50 samples from `mu` cannot conjure actions better than the best
`mu` would ever draw. And hopper, at `0.62`, is still a long way from solved. That last point is the
diagnosis I have to act on: IDQL's actor is *pure BC*. It only ever samples from `mu`, and the critic
merely picks the best of those samples. The ceiling of "best of 50 samples from `mu`" is bounded by what
`mu` puts mass on. If the genuinely good action lives in a region `mu` samples rarely or never, no amount
of reranking will surface it — the candidate set never contains it.

So the obvious next move is to stop merely *selecting* good actions and start *training the actor to
prefer them*. Instead of leaving all value information at inference, push the gradient of `Q` into the
actor's own parameters so that the distribution it samples from drifts toward high-`Q` regions. Then the
candidate set itself improves: at inference the actor is already sampling near the good actions, and
selection (if any) only sharpens. This is the structural difference from IDQL — value enters *during
actor training*, not only at inference — and it is exactly the lever that can break past the
"best-of-`mu`" ceiling that capped IDQL on hopper.

Let me derive what that actor objective should be, because the danger is the same OOD wall as before,
just relocated. I want the actor to maximize `E_{a ~ pi}[Q(s,a)]` — sample an action from the diffusion
actor at state `s` and ascend `Q` on it. But if I *only* maximize `Q`, the actor will happily march off
the data support to wherever the critic is erroneously optimistic, and I am back to exploiting phantom
optima. The fix is a behavior-regularized objective: keep the diffusion-BC term as an anchor that pins
the actor near `mu`, and add the `Q`-maximization term as the pull toward better actions. So the actor
loss is `L_actor = L_BC + eta * L_Q`, where `L_BC` is the noise-prediction denoising loss (the same
objective that the floor and IDQL both used to clone `mu`) and `L_Q = -E[Q(s, a_new)]` with `a_new`
*freshly sampled from the actor's own reverse chain*, so the gradient flows through the diffusion
sampler back into the actor's weights. The coefficient `eta` trades cloning against improvement: small
`eta` stays close to `mu` (safe, like IDQL's pure BC in the limit `eta -> 0`); large `eta` chases `Q`
harder but risks the OOD blow-up. This regularized policy class is what diffusion buys here — an
expressive actor that can place mass on a *better* action than any single `mu` mode, while the BC anchor
keeps it from leaving the support entirely.

Two things need care, and both are visible in the harness's exact form. First, sampling the new action
*through* the diffusion chain with gradients enabled (`requires_grad=True`) is what makes
`Q`-maximization meaningful — the actor's parameters move so that the *generated* action has higher `Q`,
not so that some external action does. The default fill samples `new_act` from the actor with
`use_ema=False` (I want gradients on the live weights, not the EMA copy) and `requires_grad=True`, then
evaluates `Q` on it inside a `FreezeModules([critic])` block so the `Q`-gradient updates only the actor,
not the critic. Second — and this is the subtle, harness-specific part I must get exactly right — the
`Q`-maximization term needs a scale that does not let one of the twin heads dominate or blow up. The
default uses a *randomized, normalized double-Q* trick: with probability one half it forms
`q_loss = - q1.mean() / q2.abs().mean().detach()`, otherwise `- q2.mean() / q1.abs().mean().detach()`.
The numerator is the head being maximized; the denominator is the *detached* absolute scale of the
*other* head. Dividing by the other head's magnitude normalizes the `Q`-gradient to roughly unit scale
so that `eta` has a stable, dataset-independent meaning, and randomizing which head is numerator vs
denominator each step symmetrizes the two critics and prevents the actor from over-fitting to one head's
idiosyncratic over-estimates. This is not the textbook `-Q.mean()`; it is the literal mechanism this
harness exposes, and it is what makes the BC+Q balance robust across the three environments without
per-env retuning of `eta`. I derive against *this*, not the generic version.

The critic side, by contrast, is simpler than IDQL's — and that simplification is itself the point. DQL
does not need the expectile/value-net machinery, because its actor is no longer pure BC: the actor is
being pushed toward high-`Q` actions, so the natural critic target uses the actor's *own* next action
rather than an in-sample expectile. The critic is a twin-Q (`DQLCritic`), and the TD target is
`r + gamma (1-done) * min(Q1_target(s', a'), Q2_target(s', a'))` with `a'` sampled from the diffusion
actor at `s'`. This *does* query an action the actor proposes at `s'`, which is precisely the OOD query
IDQL refused — but here it is tolerable for two linked reasons: the actor is BC-anchored so `a'` stays
near the support, and the twin-min `torch.min(*critic_target(next_obs, next_act))` is the standard
clipped-double-Q underestimation pressure that fights the residual overestimation. So DQL trades IDQL's
strict in-sample safety for a more aggressive but still-regularized bootstrap, and that aggression is
the source of its higher ceiling: the critic can value actions slightly better than `mu`'s, and the
actor is trained to produce them. The critic loss is the plain
`F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)`, Polyak target update at `0.995`,
Adam at `3e-4`, cosine scheduler — all the usual choices, no expectile, no separate `V`.

Now the literal edit, which is the *default fill* — the unmodified template — so the "edit" here is
"change nothing." But I want to state precisely how it differs from the two rungs below so the ladder is
legible. Versus diffusion BC: DQL adds the entire `DQLCritic` (twin-Q + target + optimizer), adds the
critic TD-update reading `next_obs/rew/tml`, and replaces the actor's pure BC step with
`actor.loss(act, obs) + eta * q_loss` where `q_loss` flows through a gradient-enabled actor sample.
Versus IDQL: DQL swaps the actor backbone back to `DQLMlp` and the critic from
`IDQLQNet + IDQLVNet` (expectile `V`, in-sample SARSA) to a single `DQLCritic` twin-Q with an
actor-sampled bootstrap target; it *adds* the `eta * q_loss` actor term that IDQL deliberately omits;
and at inference it reranks `num_candidates` actions by a softmax over the *bare* `Q`
(`critic_target.q_min`) rather than over the *advantage* `Q - V`. That last difference is consistent
with the training difference: IDQL needed the advantage because its value baseline `V` was the
expectile; DQL has no `V`, so it reranks on `Q` directly, and because the actor is already trained toward
high `Q`, both the candidate set and the ranking pull the same direction. The inference block repeats
each observation `num_candidates` times, samples that many actions, computes
`q = critic_target.q_min(obs, act).view(-1, num_candidates, 1)`, forms
`w = softmax(q * weight_temperature, dim=1)`, and resamples one with `multinomial`. Note that since DQL's
actor was *trained* to prefer high-`Q` actions, the reranking matters less than it does for IDQL (where
it was the only value mechanism) — the heavy lifting has already happened in the actor's weights.

I also want to settle the training schedule, since it differs from IDQL's every-other-step critic. DQL
updates the critic and the actor on *every* gradient step (not every other), because both are now doing
coupled work — the critic has to keep up with an actor that is actively moving toward high-`Q` regions,
and a stale critic would let the actor exploit lag. The EMA on the actor still kicks in only after 1000
steps, and the critic target is Polyak-updated every `ema_update_interval` steps. This tighter coupling
is more expensive than IDQL — every step samples *two* full diffusion chains (one for the critic's
`next_act`, one for the actor's `new_act`), where the floor sampled none in training and IDQL sampled
none in training either (it only sampled at inference). That cost is the reason DQL is the slowest rung
to train, and I should expect its walltime to dwarf both others.

My falsifiable expectations against the rungs below. DQL must beat IDQL on hopper and (especially)
where the best action lives outside `mu`'s frequent samples, because training the actor toward high `Q`
breaks the "best-of-`mu`" ceiling that capped IDQL at `0.62` on hopper — I expect hopper to approach the
expert-level `~1.0` normalized score, since hopper is the task where `medium`-buffer stitching toward a
clean hop pays off most. On halfcheetah I expect a smaller gain (IDQL was already near the `medium`
ceiling at `0.48`); DQL's training-time `Q`-maximization can push it modestly higher (toward `~0.51`)
but the dataset ceiling is low for everyone. Walker2d, already strong for IDQL at `0.83`, should hold or
improve slightly toward `~0.90`. The geometric mean across the three must come out clearly above IDQL's
— that is the claim that training-time `Q`-maximization beats inference-time selection on these
datasets, and it is exactly the claim IDQL's hopper stall set up. The cost is walltime: DQL trains two
diffusion chains per step plus a coupled critic, so I expect it to be roughly three times slower than
IDQL and far slower than the floor; if it were not slower, something would be wrong with the
gradient-enabled sampling. If DQL failed to beat IDQL, that would falsify the whole premise that moving
value from inference into actor training helps here — but the hopper ceiling IDQL hit is precisely the
evidence that it should. The full scaffold module — the twin-Q critic, the coupled BC+Q actor training
with the randomized normalized double-Q `q_loss`, and the `Q`-reranking inference block — is in the
answer.
</content>
