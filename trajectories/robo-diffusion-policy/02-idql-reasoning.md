The diffusion-BC floor came in exactly where the theory said it would, and reading the numbers tells me
precisely what to fix. Pure cloning landed at `0.49` on hopper, `0.66` on walker2d, `0.42` on halfcheetah
(means over seeds 42/123/456), and the seeds are tight — hopper `0.482 / 0.496 / 0.492`, walker2d
`0.667 / 0.643 / 0.657`. Let me actually quantify "tight" rather than assert it: the hopper spread is
`0.496 - 0.482 = 0.014`, about `3%` of the mean, and halfcheetah is tighter still at
`0.4247 - 0.4200 = 0.005`. So this is not a variance story, it is a *ceiling* story. The actor is doing
its one job well: the seed-to-seed spread is small, which means the diffusion model is faithfully and
reproducibly reproducing the behavior policy; there is no mode-averaging collapse pulling one seed into
the floor, exactly the outcome I said would falsify the floor if it did *not* appear. The relative shape
across environments is the second tell. Measure each against the expert-level `1.0`: hopper is
`1.0 - 0.49 = 0.51` short, walker2d `0.34` short, halfcheetah `0.58` short. And the *cloning quality*
ordering — walker2d `0.66` best, hopper `0.49`, halfcheetah `0.42` worst, a `0.66/0.42 = 1.55x` spread —
matches what I predicted from buffer structure: walker2d's `medium` buffer is a fairly coherent gait so it
clones best, halfcheetah's is a competent-but-unremarkable runner whose ceiling is low to begin with, and
hopper sits in between. Nothing here is broken. The actor simply has no idea which of the actions it can
sample are *good*. It was given no reward signal at all, so its ceiling is `mu` by construction, and
`0.49` on hopper is `mu`'s ceiling — the `0.51` gap to expert is precisely the reward information the
floor threw away. The obvious next move is to let reward in — to push the actor above `mu` toward the
better-than-average actions the buffer demonstrably contains.

So I need a value function. And the instant I reach for one, I hit the wall that defines offline RL. The
buffer is fixed; the moment I bootstrap — fit `Q(s,a)` toward `r + gamma * Q(s', a')` — I have to choose
which `a'` to plug into the next-state value. If I take `a'` from the policy I am improving, that policy
by construction wants to deviate from `mu`, so `a'` slides off the data support, the function approximator
returns a value for an action it has never seen, and because nothing pins those values down they come out
too high far more often than too low. The backup carries that phantom optimism backward, the policy chases
it, and the whole thing diverges. The real enemy is querying the value of out-of-distribution actions. I
have to add reward *without* ever writing down `max_{a'} Q(s',a')`, because that max is exactly the
operator that reaches outside the data.

Before I derive the in-sample fix, I should walk the routes I am *not* taking, because more than one
family of offline-RL method could bolt value onto this floor and I want the elimination to be by
argument. The first is conservative Q-learning: keep bootstrapping but push down the Q-values of OOD
actions with a penalty, then extract a policy. But CQL's extracted policy in the standard recipe is a
Gaussian, and I have just spent the floor establishing that a Gaussian smears across the modes of a
multimodal `mu` and emits the stalling in-between action — I would be throwing away the expressive actor I
proved I need, and the coarse dataset-wide penalty is a blunt instrument that also depresses good
in-support actions. The second is advantage-weighted regression: fit the policy by
`E[exp(beta A) log pi]`, a maximum-likelihood fit to `pi ∝ mu exp(beta A)`. This is the tempting one
because it keeps the actor near data by construction, so let me walk it a step. To use it with my
expressive actor I would bake the weights into the diffusion BC loss, weighting each example's
noise-prediction MSE by `exp(beta A(s,a))`. The trouble is specific and known: a high-capacity generative
model trained by importance-weighted maximum likelihood raises the likelihood of *all* its training points
regardless of their weights — it has the capacity to fit the whole buffer — so the `exp(beta A)` skew that
was supposed to concentrate mass on high-advantage actions washes out, and I get back something close to
unweighted BC. So the weighting has to enter somewhere the model cannot absorb it away. The third route is
to push the `Q`-gradient directly into the actor's parameters — the `eta * q_loss` term the default fill
in front of me uses — but that reintroduces exactly the OOD bootstrap I am trying to avoid at this rung,
and it couples critic and actor training; I want to see first how far a *decoupled* value function gets,
so I defer that route. That leaves the in-sample-value-plus-inference-extraction route, and its two pieces
answer the two failures above directly: an in-sample critic that never forms `max_{a'} Q`, and an
inference-time reweighting that the model cannot wash out because it never touches training.

The cleanest way to refuse the OOD query is to never form it. Instead of maximizing over `a'`, estimate
"the value of the best in-support action" from dataset actions alone, using expectile regression. The
`tau`-expectile of a random variable is `argmin_m E[ |tau - 1(x<m)| (x-m)^2 ]`: an asymmetric squared
loss that for `tau > 0.5` punishes `x < m` less than `x > m`, so the minimizer is pulled above the mean,
and as `tau -> 1` it climbs toward the supremum. Put a number on the asymmetry I will actually run: at
`tau = 0.7` the loss weights over-predictions (`x < m`) by `0.3` and under-predictions (`x > m`) by `0.7`,
a `0.7/0.3 = 2.33x` heavier pull upward, so `V` settles well above the mean of `Q` over the dataset
actions but short of the max — a soft, tunable "best in-support" estimate. Apply that conditionally to the
distribution of `Q(s,a)` as `a` ranges over `mu(.|s)`, and put it on a *separate* value net `V(s)`: a high
expectile of that distribution is the value of the best action the data supports, recovered purely from
in-sample actions, with the max done implicitly by the loss asymmetry. I never name the maximizing action,
so I never query an OOD one. The split onto a separate `V` matters: if I expectile-regressed
`r + gamma Q(s',a')` directly, the asymmetry would also reward lucky *transitions* — a single `s'` that
happened to be good — and confuse "this action is reliably good" with "this sample got lucky." Letting `V`
take the action-expectile and then having `Q` do an ordinary SARSA TD backup against `V(s')` isolates the
max over actions from the dynamics noise:

  `L_V(psi) = E_{(s,a)~D}[ |tau - 1(Q_targ(s,a) - V_psi(s) < 0)| (Q_targ(s,a) - V_psi(s))^2 ]`,
  `L_Q(theta) = E_{(s,a,s')~D}[ (r + gamma (1-done) V_psi(s') - Q_theta(s,a))^2 ]`.

This interpolates from SARSA at `tau = 0.5` to support-constrained Q-learning as `tau -> 1`, stable and
multi-step, never touching an OOD action. Let me check the `tau = 0.5` endpoint by hand: there the weight
`|0.5 - 1(x<m)|` is `0.5` on both sides, so the loss is symmetric squared error and its minimizer is the
mean, `V(s) = E_{a~mu}[Q(s,a)]` — the value of the behavior policy itself. That is the sanity floor: at
`tau = 0.5` the critic evaluates `mu`, and cranking `tau` up is what turns "value of `mu`" into "value of
the best in-support action." The value side is solved.

But now the half that actually drives the score is the extraction, and this is where I have to be careful,
because I have a critic with no policy anywhere in sight. I trained `V` to be the `tau`-expectile of `Q`
over `mu`. At `tau = 0.5` that is the mean, so `V` is the value of `mu` itself; at `tau -> 1` it is the
max, so the policy is greedy. For the `tau` in between that I will actually use, which policy does my
critic evaluate? I do not want to guess, so let me derive it. Generalize the value loss to an arbitrary
convex `f` with `f'(0) = 0` (expectile is the special case `f = |tau-1(u<0)| u^2`), defining
`V*(s) = argmin_V E_{a~mu}[ f(Q(s,a) - V(s)) ]`. At the optimum the derivative in `V` vanishes:
`E_{a~mu}[ f'(Q - V*) ] = 0`. Convexity with `f'(0)=0` means `f'` has the sign of its argument, so write
`f'(x) = |f'(x)| * x/|x|` and substitute: `E_{a~mu}[ |f'(Q-V*)|/|Q-V*| * (Q - V*) ] = 0`. The factor
`|f'(Q-V*)|/|Q-V*|` is a nonnegative scalar weight; fold it into the sampling distribution as
`pi_imp(a|s) ∝ mu(a|s) * |f'(Q-V*)|/|Q-V*|` and the condition collapses to
`E_{a~pi_imp}[ Q(s,a) - V*(s) ] = 0`, i.e. `V*(s) = E_{a~pi_imp}[Q(s,a)]`. So `V*` is the *value of the
implicit actor* `pi_imp` — a reweighting of the behavior policy. The in-sample critic is secretly an
actor-critic method, and the actor it evaluates is `mu` skewed by the critic's weights. For the expectile
loss the weight is the strikingly simple two-valued `|tau - 1(Q < V*)|` — `tau` above the value, `1-tau`
below — so at `tau = 0.7` the implicit actor upweights the good half of `mu` by `0.7` and the bad half by
`0.3`, broadening the good half and shrinking the bad half.

That derivation hands me the fix to the diffusion-BC ceiling almost directly, and it also tells me what
*not* to do — it is the same wash-out warning I hit walking AWR above, now derived from the other side.
The implicit actor is `mu` reweighted, and on a multimodal `medium` buffer the reweighting of a multimodal
`mu` is itself multimodal — which a unimodal Gaussian extraction cannot represent; it would smear across
the modes and put mass in the low-density valley where the OOD, over-valued actions live, throwing away
the careful in-sample value learning at the last step. So I keep the expressive diffusion actor I already
have. The naive instinct is to retrain it with the advantage weights baked into the loss, but that is the
known dead end from the design-space walk: importance-weighted maximum likelihood on a high-capacity model
raises the likelihood of all training points regardless of their weights, so the skew washes out. The
escape is sitting in the form of `pi_imp = mu * w / Z`: I do not have to bake the weighting into training.
I can train the actor to represent `mu` alone — pure diffusion BC, *the exact thing I already validated at
the floor* — and apply the critic weights at *inference* by importance resampling. At a state, draw `N`
candidates from the behavior model, score each with the critic, and resample one with probability
proportional to the advantage weight. The behavior model does the one easy job it already does well, and
the critic does the skewing where there is no training pathology to wash anything out.

This is why IDQL is the natural rung above diffusion BC and not a different animal: it *reuses the floor's
actor verbatim* and adds exactly the missing piece — a value function and an inference-time reweighting —
without touching the actor's training. The decoupling (the critic never sees the actor during training;
the actor never sees `Q`/`V` during training) is the whole source of the method's stability, and it is
precisely what lets me keep the floor's clean, reproducible BC — the `3%`-spread reproducibility I
measured — and just bolt value on top. Concretely the inference rule I adopt is the practical
exponential-style one: draw `N` candidates, compute advantage `A = Q - V` for each, form
`w = softmax(A * weight_temperature)` over the candidates, and resample one. This is the finite-sample
realization of the exponential/KL implicit actor (the linex loss `f(u) = exp(alpha u) - alpha u` whose
value is the log-partition of `pi ∝ mu exp(alpha A)`), and it skews toward high-advantage candidates while
staying inside the support the behavior model sampled from. Two limit checks pin what
`weight_temperature` does: at temperature `0`, `softmax` is uniform, so I resample one of `N` candidates
at random — which is just sampling from `mu`, i.e. I recover the floor's behavior and gain nothing, the
correct degenerate; as temperature `-> inf`, `softmax` becomes an argmax over advantage, so I always take
the single highest-`A` candidate, greedy selection over the `N`-sample support. The knob slides between
"draw from `mu`" and "take the best of `N` draws from `mu`," which is exactly the band an inference-time
selector can occupy.

A tiny worked example makes the reranking mechanics concrete and shows why subtracting `V` is not
cosmetic. Suppose at some state the `50` sampled candidates have advantages that cluster near zero except
a handful — say three candidates at `A = +1.0` and the rest near `A = 0` — and take `weight_temperature`
of order `3`. Then the three good candidates carry unnormalized weight `exp(3*1.0) = exp(3) ~ 20` each
while the rest carry `exp(0) = 1`; with roughly `47` near-zero candidates the good triplet holds
`3*20 / (3*20 + 47) ~ 56%` of the resampling mass, so more than half the time I execute one of the three
advantage-positive actions rather than a random draw from `mu`. That is the whole mechanism: the floor
would have drawn uniformly and executed a good candidate only `3/50 = 6%` of the time, and the critic
lifts that to `~56%` without the actor ever changing. And it is why I score on `A = Q - V` rather than `Q`:
if two states have very different `Q` scales, a bare-`Q` softmax pooled the right way could let a
high-`Q` *state* dominate, but here the softmax is per-state over `dim=1`, and subtracting the state's own
`V` centers each state's candidate advantages near zero so `weight_temperature` has the same meaning in
every state — a dataset-independent knob, which is what lets one setting work across all three
environments. I take the twin-`Q` `min` (via `iql_q_target`, the frozen target copy) precisely so the
advantage does not reward a candidate that merely got an optimistic read from one of the two heads.

Now I derive the literal edit against the harness, because a same-named method can differ from its generic
form and I want the exact fill. The substrate's default builds a single `DQLCritic` (a twin-Q) and trains
the actor with `bc_loss + eta * q_loss`. IDQL needs a *different* critic shape: twin-Q *plus* a value net,
with the expectile rule. The harness exposes exactly these — `IDQLQNet` (twin-Q, with `.both(obs,act)` and
a `q_min`) and `IDQLVNet` — and a matching actor backbone `IDQLMlp`, so I do not hand-roll the residual
score net or the value heads; I import them. I swap the actor backbone from `DQLMlp` to `IDQLMlp` (which
the wrapper takes the same way), construct `iql_q`, a frozen `iql_q_target = deepcopy(iql_q)`, and
`iql_v`, with their own Adam optimizers and cosine schedulers. The training loop now reads
`obs, next_obs, act, rew, tml`. I update the critic on *every other* step (`n_gradient_step % 2 == 0`) —
the behavior model is the harder fit and deserves at least as many updates as the in-sample critic, which
converges quickly because it is a plain regression with no moving policy underneath it — doing first the
expectile `V` step (`v_loss = (|tau - 1((q-v)<0)| * (q-v)^2).mean()` with `q = iql_q_target(obs,act)`),
then the `Q` step against the bootstrapped `td_target = rew + discount*(1-tml)*iql_v(next_obs)` over both
twin heads, then a Polyak update of the `Q` target at `0.995`. And on *every* step I take the weight-free
diffusion BC step `actor.update(act, obs)["loss"]` — identical to the floor's actor update, the single
forward pass per step, no reverse chain. Critically, IDQL drops the default's `eta * q_loss` term
entirely: there is no Q-maximization gradient flowing into the actor; the actor is pure BC, and all value
information enters only at inference. That is the structural difference from the rung above, and I expect
it to be exactly what costs IDQL the top spot.

The inference edit is the reranking, and here the diff from the floor is that the reranking comes *back*
(the floor deleted it). I load the actor checkpoint plus the critic checkpoint
(`iql_q`, `iql_q_target`, `iql_v`), repeat each observation `num_candidates` times, sample that many
actions, compute `adv = (iql_q_target(obs,act) - iql_v(obs))` reshaped to `(-1, num_candidates, 1)`,
`w = softmax(adv * weight_temperature, dim=1)`, normalize, and resample one candidate per env with
`torch.multinomial`. The shape bookkeeping is the concrete cost: with `num_envs = 50` and
`num_candidates = 50`, the repeated observation is `(50*50, obs_dim) = (2500, obs_dim)`, the sampled
action batch is `(2500, act_dim)`, and the softmax runs over `dim=1` of the `(50, 50, 1)` advantage tensor
so each environment's `50` candidates compete only among themselves. That is a `50x` larger sampling batch
per env-step than the floor carried — the fan-out the floor explicitly deleted, now paid back. With
`num_candidates = 50` fixed by the protocol so reranking methods see equal compute, this is a fair,
identical fan-out to what the default DQL uses; only the score is the advantage `Q - V` rather than a bare
`Q`, because here `V` is a genuine expectile baseline and subtracting it removes the state-dependent
offset so the softmax compares actions *within* a state rather than being dominated by which states happen
to have large `Q`.

Where does this leave walltime, so my `training_time` prediction is grounded? The floor trained at roughly
one forward pass per step and reranked nothing, landing around `2300s` per environment. IDQL adds a twin-Q
and a value net updated every other step (each a cheap MLP forward/backward, but non-trivial in
aggregate over a million steps) and, more expensively, reranks `50` candidates through the full 5-step
chain at every one of the ~1000 environment steps across `50` envs and `3` episodes. So I expect IDQL to
land clearly above the floor's `~2300s` — plausibly around double — but nowhere near the cost of a method
that samples full reverse chains *during training*, because IDQL's actor training is still the floor's
single-forward-pass BC step. It should sit between the floor and the default in both score and time.

So my falsifiable expectations against the floor: IDQL must beat diffusion BC on every environment,
because it has the same actor plus a value-based selection that can only help if `Q`/`V` are informative —
anything else would mean the critic learned nothing useful, or `weight_temperature` collapsed to the
uniform limit I identified. I expect the gain to be largest where the behavior policy has the most
exploitable spread between good and mediocre actions (hopper, where lifting `0.49` well above `0.6` should
be reachable) and present but smaller on halfcheetah (whose `medium` ceiling is low at `0.42`, so even
perfect selection over `50` samples cannot lift it far). I expect it to cost more walltime than the floor
— it trains a twin-Q and a value net and reranks 50 candidates per step where the floor reranked none — so
it should land between diffusion BC and the default in both score and time. And I expect it to *not* catch
the default: because IDQL's actor is pure BC and only the *selection* chases value, its ceiling is "the
best of 50 samples from `mu`," whereas a method that pushes the actor's own distribution toward high-`Q`
actions during training can place mass on actions better than anything `mu` would sample. If IDQL matched
or beat that method, it would falsify the claim that training-time Q-maximization buys anything over
inference-time selection on these datasets; I expect it will sit clearly above the floor and clearly
below it. One more internal consistency check I can watch: `normalized_score` and `episode_reward` are
just affine images of each other per environment, so a genuine value gain must move both in the same
direction on every environment — if `normalized_score` rose while `episode_reward` did not, or the seed
spread widened past the floor's `3%`, I would suspect the reranking was picking on noise rather than
signal rather than the critic having found real advantage structure. The full scaffold module — the IDQL critic construction, the decoupled expectile/BC train loop,
and the advantage-reranking inference block — is in the answer.
