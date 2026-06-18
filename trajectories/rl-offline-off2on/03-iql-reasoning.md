The AWAC numbers confirmed exactly the boundary I drew and then drew a new one. The good news first:
hammer-expert recovered to a mean of 126.8 (126.2 / 126.8 / 127.4, almost no seed spread) — removing the
deterministic-maximizer OOD query did stop the collapse I saw under SPOT (2.5). So the diagnosis was
right: on clean expert data, an actor that only reweights logged actions retains competence where a
`Q`-maximizing deterministic actor rode an over-valued OOD action off the manifold. Pen-cloned held at
63.7 (104.9 / 39.0 / 47.2), a touch above SPOT's 56.4 but with the same high seed variance — seed 42 is
strong, the other two are mediocre. The bad news is the one I explicitly worried about: hammer-cloned
*collapsed* to 0.336 — 0.439 / 0.323 / 0.247, dead flat, near zero on every seed, well below even SPOT's
one-lucky-seed 19.7. That is the falsifiable signal I named at the close of the last rung firing exactly
as predicted: the advantage-weighted update is too conservative on a heavily-noisy mixture. Hammer-cloned
is mostly noise with rare good demonstrations buried in it; AWAC reweights only logged actions by
`exp(adv/0.1)`, and with `λ = 0.1` that is a *sharp* temperature, so the weight concentrates on a tiny
handful of transitions — but if the critic cannot reliably tell the rare good actions from the noise, the
weights land on the wrong actions or smear uselessly, and the policy never finds the manipulation
behavior at all. The improvement signal is only ever "of the actions I logged, which scored well," and
when the logged good actions are sparse and the critic is noisy, that signal is too weak to bootstrap a
competent policy.

So the binding constraint has moved. SPOT's problem was the actor's OOD query; AWAC fixed that but left
the *critic* doing only behavior-policy evaluation — its `Q` is `Q^π` bootstrapped at policy actions, and
its improvement is one reweighting step away from `π_β`. What hammer-cloned needs is a critic that does
genuine multi-step dynamic programming — that can *stitch* the rare good fragments scattered across noisy
trajectories into a value function for the good behavior — while still never querying an unseen action.
AWAC could not stitch because its only improvement lever was reweighting `π_β`'s own actions. The next
rung has to put the improvement *into the value function itself*, safely.

Name precisely what the safe-but-insufficient piece is. AWAC's critic, and the SARSA objective behind
it, fits `Q(s,a)` to the *mean* of the TD targets over dataset actions — it learns `Q^{π_β}`, the value
of the average behavior action. The mean is why it only evaluates `π_β`. What I want in the backup is not
the mean over `a'` but the value of the *best in-support action*: a max restricted to actions the
behavior policy could actually produce at `s'`. That restriction is the whole game — it improves (so
iterating it does real dynamic programming and can stitch) but never reaches an OOD action (so it stays
safe). The obstacle is that I cannot *compute* that restricted max directly: to take a max over
in-support `a'` I would have to enumerate or sample actions and query `Q` at each — and the moment I
sample an action and query `Q`, I am back to the OOD evaluation that broke SPOT.

Reframe. Fix a state `s`. As `a'` ranges over `π_β(·|s)`, the quantity `Q(s, a')` is a *random variable*,
its randomness coming from the action. SARSA's MSE gives me the *mean* of that random variable. The
*maximum over the support* is what I want. So I need a statistic that sits high in this action-induced
random variable's distribution — ideally at the top of its support — estimable from samples (the dataset
actions at `s`) without evaluating `Q` anywhere except at those in-data actions. Mean regression gives
the mean; what gives the upper tail is **expectile regression**. The `τ`-expectile minimizes the
asymmetric squared loss `L_2^τ(u) = |τ − 1(u<0)|·u²`: for a positive residual (a sample above the
estimate) the weight is `τ`, for negative `1 − τ`; at `τ = 0.5` both are ½ and the minimizer is the mean,
and for `τ > 0.5` the upper samples dominate so the estimate climbs toward the top of the distribution.
As `τ → 1` it approaches the supremum of the support. The argument is short: every expectile lies within
the support and shares its supremum, the expectile is monotone non-decreasing in `τ`, and a bounded
monotone function has a limit at `τ → 1` which the asymmetry pushes to `x*`. So the upper expectile, in
the limit, *is* the in-support max — expressed as a regression I can do with SGD on in-sample data only.

Before splitting into a value network, try the directest thing and watch it break, because the failure
dictates the architecture. The direct move is to swap the SARSA MSE for the expectile loss on the same TD
residual `r + γ Q(s',a') − Q(s,a)`. That takes an upper expectile of `r + γ Q(s',a')` — but that target
carries *two* sources of randomness: the action `a' ~ π_β(·|s')` (which I *want* to be optimistic over —
the best `a'` is the improvement signal) and the stochastic transition `s' ~ p(·|s,a)` (which I
emphatically do *not*). An upper expectile rewards high targets indiscriminately, so it would reward a
target that is high merely because the environment got lucky with the next state — conflating "there
exists a better action here" with "I rolled well." Optimism over the dynamics, compounded over a horizon,
is exactly the overoptimism that produces a runaway value. So I must take the expectile over actions
*only* and average honestly over transitions.

Separate the two with a value network. `V_ψ(s)` does purely the action-expectile with the transition
held fixed: `L_V(ψ) = E_{(s,a)~D}[ L_2^τ( Q_θ̂(s,a) − V_ψ(s) ) ]`. Here both `s` and `a` come from `D`,
so for a given `s` the only randomness in the regression target is the action, and `V_ψ(s)` becomes the
`τ`-expectile of `Q` over dataset actions — optimistic over actions, with no dynamics in it. Then back
this up into `Q` with an *ordinary* MSE that averages the transition honestly:
`L_Q(θ) = E_{(s,a,s')~D}[ ( r + γ V_ψ(s') − Q_θ(s,a) )² ]`. The MSE is correct precisely because
`V_ψ(s')` has already done the optimistic action selection; what remains is to average `γ V_ψ(s')` over
`s'`, and a mean is the right way to average dynamics. The division of labor is the whole point: `V`
takes the upper expectile over actions, `Q` takes the mean over transitions, and both losses touch only
dataset `(s,a,s')` — no policy appears in value training, no OOD action is ever queried. This is
multi-step dynamic programming: `V_τ` is provably monotone in `τ`, bounded by the in-support optimum, and
converges to it as `τ → 1`, spanning SARSA (`τ = 0.5`, what AWAC's critic effectively did) to in-support
Q-learning (`τ → 1`). That spectrum is exactly the lever hammer-cloned was missing — at higher `τ` the
critic propagates the value of the rare good downstream states *backward across transitions from
different noisy trajectories*, stitching the fragments AWAC could not.

Now extract a policy, obeying the same commandment — never query `Q` at an unseen action. I cannot
argmax `Q` (searches OOD) or do DDPG-style ascent (evaluates `Q` at the policy's possibly-OOD actions).
I reweight the dataset's own actions: advantage-weighted regression
`L_π(φ) = E_{(s,a)~D}[ exp(β(Q_θ̂(s,a) − V_ψ(s)))·log π_φ(a|s) ]`, weight clipped to ≤ 100. This is the
same family of update AWAC used — and that is fine, because the OOD-safe improvement was never the
problem; the difference that fixes hammer-cloned is *what advantage feeds it*. AWAC's advantage came from
a `Q^π` that only evaluated `π_β`; IQL's comes from a `Q` that has done in-support dynamic programming,
so the weights now reward actions that are good *for the stitched optimal-in-support policy*, not just
good relative to the behavior average. Same extraction, far stronger signal.

In this task's harness the IQL fill is specific. The actor is a 2×256 `GaussianPolicy` with a Tanh output
activation and a state-independent `log_std` (`nn.Parameter`), a plain `Normal` (not `TanhTransform`),
with **dropout 0.1** in the actor MLP and a **CosineAnnealingLR** schedule over the 1M offline steps —
both stabilizers that matter on the noisy `cloned` data. The critic is `TwinQ` (two 2×256 MLPs, squeezed,
with a `.both()` for the twin outputs), and `ValueFunction` is a 2×256 squeezed MLP. The hyperparameters
match the CORL Adroit config: `iql_tau = 0.8` (the expectile — high enough to do real improvement on the
manipulation tasks), `beta = 3.0` (advantage temperature, softer than AWAC's `1/0.1 = 10` effective
sharpness, which on noisy data spreads the weights over more transitions rather than betting on a few),
`exp_adv_max = 100`. The update order per step is V (expectile) → Q (MSE onto `r + γ V(s')`) → Polyak the
target critic → policy (advantage-weighted), and the actor LR cosine-steps. The decisive transition fact:
`on_online_start` is a *no-op* — IQL needs no special handling at the handoff, for the same reason AWAC
did not (the value training is policy-free and the same update runs offline and online as the buffer
grows). So IQL keeps AWAC's transition-robustness while replacing the critic's behavior-evaluation with
in-support dynamic programming.

The falsifiable expectations against the AWAC numbers. My central claim is that the expectile critic's
stitching fixes hammer-cloned: I expect it to climb from AWAC's dead-flat 0.336 into a genuinely positive
range (tens of points, on *every* seed, not one lucky one), because the value function can now propagate
the rare good fragments instead of waiting for the reweighting to stumble onto them. On hammer-expert I
expect to *retain* the recovered competence AWAC found (~127) — the expectile critic should be at least as
safe as AWAC's on clean data, so a regression there would falsify the claim that IQL strictly dominates.
On pen-cloned I expect to *exceed* AWAC's 63.7, and ideally tighten the seed spread, because the stronger
advantage signal plus dropout-and-cosine stabilization should lift the two mediocre seeds. If instead
hammer-cloned stays near zero, the conclusion would be that the binding problem on that dataset is not the
critic's improvement capacity but something else — reward sparsity or the offline/online replay balance —
and I would need a calibration or replay-mixing mechanism rather than a better in-support backup. But the
prediction I am betting on is that the in-support max is exactly the missing lever, and that IQL clears
AWAC on hammer-cloned and pen-cloned while holding hammer-expert. The full scaffold module is in the
answer.
