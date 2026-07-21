The diffusion-BC floor came in where the theory said it would, and the numbers tell me what to fix. Pure
cloning landed at `0.49` on hopper, `0.66` on walker2d, `0.42` on halfcheetah (means over seeds
42/123/456), and the seeds are tight — the hopper spread is `0.496 - 0.482 = 0.014`, about `3%` of the
mean, halfcheetah tighter still at `0.005`. So this is a *ceiling* story, not a variance story: the actor
does its one job well, reproducing the behavior policy faithfully and reproducibly, with no mode-averaging
collapse pulling a seed into the floor — exactly the outcome I said would falsify the floor if it did
*not* appear. The cross-environment shape is the second tell: cloning quality orders walker2d `0.66` >
hopper `0.49` > halfcheetah `0.42`, a `0.66/0.42 = 1.55x` spread that matches buffer structure —
walker2d's coherent gait clones best, halfcheetah's competent-but-low-ceiling runner worst, hopper
between. Nothing is broken. The actor simply has no idea which of the actions it can sample are *good*; it
was given no reward, so its ceiling is `mu`, and the `0.51` gap to expert on hopper is precisely the
reward information the floor threw away. The next move is to let reward in — to push the actor above `mu`
toward the better-than-average actions the buffer demonstrably contains.

So I need a value function. And the instant I reach for one I hit the wall that defines offline RL. The
buffer is fixed; the moment I bootstrap `Q(s,a)` toward `r + gamma Q(s',a')` I must choose which `a'` to
plug into the next-state value. If I take `a'` from the policy I am improving, that policy by construction
wants to deviate from `mu`, so `a'` slides off the data support, the function approximator returns a value
for an action it has never seen, and because nothing pins those values down they come out too high far
more often than too low. The backup carries that phantom optimism backward, the policy chases it, and the
whole thing diverges. The enemy is querying the value of out-of-distribution actions, so I must add reward
*without* ever writing `max_{a'} Q(s',a')`, the operator that reaches outside the data.

Several offline-RL families could bolt value onto this floor; I take them in turn. Conservative Q-learning
keeps bootstrapping but pushes down OOD Q-values with a penalty; but its extracted
policy is a Gaussian, and I just spent the floor showing a Gaussian smears the modes of a multimodal `mu`
and emits the stalling in-between action — I would be throwing away the expressive actor I proved I need,
and the coarse dataset-wide penalty also depresses good in-support actions. Advantage-weighted regression
fits `pi ∝ mu exp(beta A)` by weighted maximum likelihood, keeping the actor near data — tempting, but to
use it with my expressive actor I would weight each example's noise-prediction MSE by `exp(beta A)`, and a
high-capacity generative model trained by importance-weighted maximum likelihood raises the likelihood of
*all* its training points regardless of weight, so the `exp(beta A)` skew washes out and I get back
unweighted BC. The weighting has to enter somewhere the model cannot absorb it. The third route — push the
`Q`-gradient directly into the actor (the default fill's `eta * q_loss`) — reintroduces exactly the OOD
bootstrap I am avoiding, and couples critic and actor; I want to see first how far a *decoupled* value
function gets, so I defer it. That leaves in-sample value plus inference-time extraction, whose two pieces
answer the two failures directly.

The cleanest way to refuse the OOD query is to never form it. Instead of maximizing over `a'`, estimate
the value of the best in-support action from dataset actions alone by expectile regression. The
`tau`-expectile is `argmin_m E[|tau - 1(x<m)|(x-m)^2]`, an asymmetric squared loss that for `tau > 0.5`
punishes `x < m` less than `x > m`, pulling the minimizer above the mean and, as `tau -> 1`, toward the
supremum — at `tau = 0.7` the upward pull is `0.7/0.3 = 2.33x`. Apply it to the distribution of `Q(s,a)`
as `a` ranges over `mu(.|s)`, on a *separate* value net `V(s)`: a high expectile is the value of the best
in-support action, recovered purely from in-sample actions, with the max done implicitly by the loss
asymmetry — I never name the maximizing action, so I never query an OOD one. The split onto a separate `V`
matters: expectile-regressing `r + gamma Q(s',a')` directly would also reward lucky *transitions* — a
single `s'` that happened to be good — confusing "this action is reliably good" with "this sample got
lucky." Letting `V` take the action-expectile and having `Q` do an ordinary SARSA backup against `V(s')`
isolates the max over actions from dynamics noise:

  `L_V(psi) = E_{(s,a)~D}[ |tau - 1(Q_targ(s,a) - V_psi(s) < 0)| (Q_targ(s,a) - V_psi(s))^2 ]`,
  `L_Q(theta) = E_{(s,a,s')~D}[ (r + gamma (1-done) V_psi(s') - Q_theta(s,a))^2 ]`.

This interpolates from SARSA at `tau = 0.5` — there the weight `|0.5 - 1(x<m)|` is `0.5` on both sides, so
the loss is symmetric and its minimizer is `V = E_{a~mu}[Q]`, the value of `mu` itself — to
support-constrained Q-learning as `tau -> 1`, stable and multi-step, never touching an OOD action.

But the half that drives the score is the extraction, and here I have a critic with no policy anywhere in
sight: which policy does an intermediate `tau` evaluate? Generalize the value loss to a convex `f` with
`f'(0) = 0` (expectile is `f = |tau-1(u<0)| u^2`), so `V*(s) = argmin_V E_{a~mu}[f(Q(s,a) - V(s))]`. At
the optimum `E_{a~mu}[f'(Q - V*)] = 0`; convexity with `f'(0)=0` makes `f'` share the sign of its
argument, so writing `f'(x) = |f'(x)| x/|x|` and folding the nonnegative scalar `|f'(Q-V*)|/|Q-V*|` into
the sampling distribution as `pi_imp(a|s) ∝ mu(a|s) |f'(Q-V*)|/|Q-V*|` collapses the condition to
`V*(s) = E_{a~pi_imp}[Q(s,a)]`. So `V*` is the value of an *implicit actor* — a reweighting of `mu`. For
the expectile loss the weight is the two-valued `|tau - 1(Q < V*)|`, so at `tau = 0.7` the implicit actor
upweights the good half of `mu` by `0.7` and the bad half by `0.3`.

That reweighting is multimodal whenever `mu` is, so a unimodal Gaussian extraction cannot represent it —
it would smear across the modes and put mass in the low-density valley where the OOD, over-valued actions
live, throwing away the value learning at the last step. So I keep the expressive diffusion actor. The
naive instinct is to retrain it with the advantage weights baked in — but that is the AWR wash-out again.
The escape is sitting in `pi_imp = mu * w / Z`: I do not have to bake the weighting into training. Train
the actor to represent `mu` alone — pure diffusion BC, *the exact thing I validated at the floor* — and
apply the critic weights at *inference* by importance resampling: draw `N` candidates from the behavior
model, score each with the critic, resample one with probability proportional to the advantage weight. The
behavior model does the one easy job it already does well; the critic does the skewing where there is no
training pathology to wash anything out.

So this reuses the floor's actor verbatim and adds exactly the missing piece — a value function and an
inference-time reweighting — without touching the actor's training. The decoupling (critic never sees the
actor during training, actor never sees `Q`/`V`) is the whole source of stability and is what preserves the
floor's `3%`-spread reproducibility. Concretely the inference rule is the practical exponential form: draw
`N` candidates,
compute `A = Q - V`, form `w = softmax(A * weight_temperature)` over the candidates, resample one. This
realizes the exponential/KL implicit actor (the linex `f(u) = exp(alpha u) - alpha u`, whose value is the
log-partition of `pi ∝ mu exp(alpha A)`) and slides, as `weight_temperature` goes from `0` (uniform → draw
from `mu`, gaining nothing) to large (argmax → best of `N` draws from `mu`), across exactly the band an
inference-time selector can occupy.

I score on `A = Q - V` rather than bare `Q` because the softmax runs per-state over `dim=1`, and
subtracting the state's own `V` centers each state's candidate advantages near zero so `weight_temperature`
means the same thing in every state — a dataset-independent knob that lets one setting work across all
three environments; without it a high-`Q` state could dominate the wrong pooling. I take the twin-`Q`
`min` (the frozen `iql_q_target`) so the advantage does not reward a candidate that merely got an
optimistic read from one head. The mechanism is real leverage: if only a handful of the `50` candidates
are advantage-positive, uniform sampling would execute one only `~6%` of the time, while a
`weight_temperature` of a few lifts that to a majority of the resampling mass — all without the actor ever
changing.

Now the literal edit. The default builds a single `DQLCritic` twin-Q and trains the actor with
`bc_loss + eta * q_loss`; IDQL needs a different critic shape — twin-Q *plus* a value net with the
expectile rule — and the harness exposes exactly `IDQLQNet` (twin-Q, with `.both` and a `q_min`),
`IDQLVNet`, and the matching backbone `IDQLMlp`, so I import rather than hand-roll. I swap
`DQLMlp -> IDQLMlp`, construct `iql_q`, a frozen `iql_q_target = deepcopy(iql_q)`, and `iql_v`, each with
Adam and cosine schedulers. I update the critic on *every other* step — the behavior model is the harder
fit and deserves at least as many updates as the in-sample critic, which converges fast because it is a
plain regression with no moving policy underneath — doing the expectile `V` step then the `Q` SARSA step
against `td_target = rew + discount*(1-tml)*iql_v(next_obs)` over both heads, then a Polyak `Q`-target
update at `0.995`. On *every* step the actor takes the weight-free `actor.update(act, obs)["loss"]` —
identical to the floor's, one forward pass, no reverse chain. IDQL drops the default's `eta * q_loss`
entirely: no Q-maximization gradient reaches the actor; all value enters at inference. That is the
structural difference from the Q-maximization route, and I expect it to be exactly what costs IDQL the top spot.

The inference edit brings the reranking back (the floor deleted it). I load the actor plus
`iql_q`/`iql_q_target`/`iql_v`, repeat each observation `num_candidates` times, sample that many actions,
compute `adv = iql_q_target(obs,act) - iql_v(obs)` reshaped to `(-1, num_candidates, 1)`,
`w = softmax(adv * weight_temperature, dim=1)`, and resample one per env with `torch.multinomial`. With
`num_envs = 50` and `num_candidates = 50` the sampled batch is `(2500, act_dim)` per env-step — the 50x
fan-out the floor deleted, now paid back, and identical to what the default DQL uses; only the score is the
advantage `Q - V` rather than a bare `Q`, because here `V` is a genuine expectile baseline and subtracting
it removes the state-dependent offset.

On walltime: the floor trained at one forward pass per step and reranked nothing, landing around `2300s`.
IDQL adds a twin-Q and value net updated every other step (cheap MLPs, non-trivial in aggregate over a
million steps) and reranks `50` candidates through the full 5-step chain at every env-step across `50`
envs and `3` episodes — so I expect it clearly above the floor, plausibly around double, but nowhere near
a method that samples full reverse chains *during training*, since IDQL's actor training is still the
floor's single-pass BC step.

Against the floor: IDQL must beat diffusion BC on every environment, since it has the same actor plus a
value-based selection that can only help if `Q`/`V` are informative. I expect the gain largest where `mu`
has the most exploitable spread between good and mediocre actions — hopper, lifting the `0.49` floor — and
smaller on halfcheetah, whose low `0.42` ceiling caps what selection over `50` samples can do. And it
should *not* catch the default: IDQL's actor is pure BC and only the *selection* chases value, so its
ceiling is "best of 50 samples from `mu`," whereas training-time Q-maximization can place mass on actions
better than anything `mu` samples. The full module is in the answer.
