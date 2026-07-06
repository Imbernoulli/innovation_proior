The AWAC numbers confirmed exactly the boundary I drew and then drew a new one. The good news first:
hammer-expert recovered to a mean of 126.8, and — the property I said I needed — it was *tight*:
126.2 / 126.8 / 127.4, a range of barely one point across seeds. That is the reproducible success on
clean data that SPOT's flat 2.5 collapse could not manage, so the diagnosis was right. Removing the
deterministic-maximizer OOD query did stop the collapse: on clean expert data an actor that only
reweights logged actions retains competence where a `Q`-maximizing deterministic actor rode an
over-valued OOD action off the manifold. Pen-cloned held at 63.7 (104.9 / 39.0 / 47.2), a touch above
SPOT's 56.4, and I note the spread did *not* tighten the way I hoped — seed 42 is strong at 104.9, the
other two are mediocre in the 40s, so the run-to-run range is still about 66 points, even wider than
SPOT's. The implicit constraint stabilized the *worst* failure but not the general variance on noisy
data. The bad news is the one I explicitly worried about: hammer-cloned *collapsed* to 0.336 —
0.439 / 0.323 / 0.247, dead flat, near zero on every seed. Compare it honestly across the ladder: SPOT
managed a one-lucky-seed 19.7 mean there, so AWAC is *worse* on hammer-cloned than SPOT was, not just
weak. That is the falsifiable signal I named at the close of the last rung firing exactly as predicted:
the advantage-weighted update is too conservative on a heavily-noisy mixture. Recall the arithmetic —
`awac_lambda = 0.1` saturates the weight at an advantage of about 0.46 and batch normalization hands the
top slice ~80% of the gradient, so the update is effectively top-k cloning of the critic's best-ranked
logged actions. Hammer-cloned is mostly noise with rare good demonstrations buried in it; if the critic
cannot reliably tell the rare good actions from the noise, that top-k pull lands on the wrong actions or
smears uselessly, and the dead-flat 0.24–0.44 across all three seeds is the signature of an update whose
entire improvement signal — "of the actions I logged, which scored well" — is too weak to bootstrap a
competent policy when the logged good actions are sparse and the critic is noisy.

So the binding constraint has moved, and I can state precisely where. SPOT's problem was the actor's OOD
query; AWAC fixed that but left the *critic* doing only behavior-policy evaluation — its `Q` is `Q^π`
bootstrapped at policy actions and its improvement is one reweighting step away from `π_β`. Trace why
that caps hammer-cloned: to solve that task the value function has to recognize that a rare good action
early in some noisy trajectory leads, several steps later, to the sparse completion reward, and it has to
do so by *stitching* that credit across transitions that came from different, mostly-bad trajectories.
AWAC's critic cannot stitch, because its only improvement lever is reweighting `π_β`'s own logged actions
— it never propagates "there exists a better action at `s'`" backward; it only propagates the value of
the average behavior action. What hammer-cloned needs is a critic that does genuine multi-step dynamic
programming — that can stitch the rare good fragments scattered across noisy trajectories into a value
function for the good behavior — while still never querying an unseen action. The next rung has to put the
improvement *into the value function itself*, safely.

Name precisely what the safe-but-insufficient piece is. AWAC's critic, and the SARSA objective behind
it, fits `Q(s,a)` to the *mean* of the TD targets over dataset actions — it learns `Q^{π_β}`, the value
of the average behavior action. The mean is why it only evaluates `π_β`. What I want in the backup is not
the mean over `a'` but the value of the *best in-support action*: a max restricted to actions the
behavior policy could actually produce at `s'`. That restriction is the whole game — it improves (so
iterating it does real dynamic programming and can stitch) but never reaches an OOD action (so it stays
safe). The obstacle is that I cannot *compute* that restricted max directly: to take a max over
in-support `a'` I would have to enumerate or sample actions and query `Q` at each — and the moment I
sample an action and query `Q`, I am back to the OOD evaluation that broke SPOT and produced the flat 2.5.

Reframe. Fix a state `s`. As `a'` ranges over `π_β(·|s)`, the quantity `Q(s, a')` is a *random variable*,
its randomness coming from the action. SARSA's MSE gives me the *mean* of that random variable. The
*maximum over the support* is what I want. So I need a statistic that sits high in this action-induced
random variable's distribution — ideally at the top of its support — estimable from samples (the dataset
actions at `s`) without evaluating `Q` anywhere except at those in-data actions. Mean regression gives
the mean; what gives the upper tail is **expectile regression**. The `τ`-expectile minimizes the
asymmetric squared loss `L_2^τ(u) = |τ − 1(u<0)|·u²`: for a positive residual (a sample above the
estimate) the weight is `τ`, for negative `1 − τ`; at `τ = 0.5` both are ½ and the minimizer is the mean,
and for `τ > 0.5` the upper samples dominate so the estimate climbs toward the top of the distribution.
Put the number in for the value I will use, `τ = 0.8`: a sample lying above the current estimate is
weighted `0.8` and one below `0.2`, a `4:1` asymmetry, so the estimate settles where four-fifths of the
squared-error pressure comes from above-estimate samples — well up the tail, but not at the raw supremum.
As `τ → 1` the weight on below samples vanishes and the estimate approaches the support's supremum. The
argument is short: every expectile lies within the support and shares its supremum, the expectile is
monotone non-decreasing in `τ`, and a bounded monotone function has a limit at `τ → 1` which the
asymmetry pushes to `x*`. So the upper expectile, in the limit, *is* the in-support max — expressed as a
regression I can do with SGD on in-sample data only.

Before splitting into a value network, try the directest thing and watch it break, because the failure
dictates the architecture. The direct move is to swap the SARSA MSE for the expectile loss on the same TD
residual `r + γ Q(s',a') − Q(s,a)`. That takes an upper expectile of `r + γ Q(s',a')` — but that target
carries *two* sources of randomness: the action `a' ~ π_β(·|s')` (which I *want* to be optimistic over —
the best `a'` is the improvement signal) and the stochastic transition `s' ~ p(·|s,a)` (which I
emphatically do *not*). An upper expectile rewards high targets indiscriminately, so it would reward a
target that is high merely because the environment got lucky with the next state — conflating "there
exists a better action here" with "I rolled well." And this is not a small effect at `γ = 0.99`: an
overestimate injected at one state is bootstrapped with a fixed-point amplification of `1/(1−γ) = 100`,
so optimism-over-dynamics compounded over a horizon is exactly the runaway value SPOT died of, now
reached through a different door. So I must take the expectile over actions *only* and average honestly
over transitions.

Separate the two with a value network. `V_ψ(s)` does purely the action-expectile with the transition
held fixed: `L_V(ψ) = E_{(s,a)~D}[ L_2^τ( Q_θ̂(s,a) − V_ψ(s) ) ]`. Here both `s` and `a` come from `D`,
so for a given `s` the only randomness in the regression target `Q_θ̂(s,a)` is the action — the transition
is not involved at all, because `V_ψ(s)` is a function of `s` only and the target reads `Q` at the logged
`(s,a)`, not at any `s'`. So `V_ψ(s)` becomes the `τ`-expectile of `Q` over dataset actions — optimistic
over actions, with no dynamics in it. Then back this up into `Q` with an *ordinary* MSE that averages the
transition honestly: `L_Q(θ) = E_{(s,a,s')~D}[ ( r + γ V_ψ(s') − Q_θ(s,a) )² ]`. The MSE is correct
precisely because `V_ψ(s')` has already done the optimistic action selection; what remains is to average
`γ V_ψ(s')` over `s'`, and a mean is the right way to average dynamics. The division of labor is the whole
point: `V` takes the upper expectile over actions, `Q` takes the mean over transitions, and both losses
touch only dataset `(s,a,s')` — no policy appears in value training, no OOD action is ever queried. This
is multi-step dynamic programming: `V_τ` is provably monotone in `τ`, bounded by the in-support optimum,
and converges to it as `τ → 1`, spanning SARSA (`τ = 0.5`, what AWAC's critic effectively did) to
in-support Q-learning (`τ → 1`). Verify the low end concretely: at `τ = 0.5` the expectile loss is a
plain symmetric MSE and `V_ψ(s)` regresses to `E_{a~π_β}[Q(s,a)]`, the behavior value — which is exactly
the `V` AWAC computed to form its advantage, so the `τ = 0.5` corner of this method literally *is* the
previous rung's critic. That spectrum is exactly the lever hammer-cloned was missing — at `τ = 0.8` the
critic propagates the value of the rare good downstream states *backward across transitions from
different noisy trajectories*, stitching the fragments AWAC could not, and I chose 0.8 rather than
pushing toward 1.0 because on noisy data an aggressive expectile starts to over-fit the upper sampling
noise of `Q` itself, and 0.8's `4:1` asymmetry is enough improvement without that fragility.

Trace the expectile gradient once to be sure it climbs the way I claim, because "optimistic over actions"
has to be a mechanical fact, not a slogan. The loss on one sample is `|τ − 1(u<0)|·u²` with
`u = Q̂(s,a) − V(s)`; its gradient with respect to `V(s)` is `−2|τ − 1(u<0)|·u`. When the logged action's
`Q̂` sits *above* the current `V` estimate (`u > 0`), the weight is `τ = 0.8` and `V` is pulled up with
strength `1.6·u`; when it sits *below* (`u < 0`), the weight is `0.2` and `V` is pulled down with
strength only `0.4·|u|`. So `V` rises until the above-pulls and below-pulls balance, which happens well
above the mean — at the point where `0.8·E[(Q̂−V)_+] = 0.2·E[(V−Q̂)_+]`, i.e. the upper pull mass is a
quarter the size but weighted four times as hard. A single below-average logged action barely restrains
`V`, while a handful of high-value logged actions drag it up. That is exactly "take the value of the best
in-support action, discount the mediocre ones" — the good demonstration in a noisy hammer-cloned batch
gets four times the vote of each noisy neighbor, which is how its value survives being averaged away. The
shape of the loss is doing the in-support argmax that I am forbidden to compute directly.

Make the stitching claim concrete with a two-trajectory sketch, because "propagate backward across
trajectories" is the load-bearing behavior and I want to see it move. Suppose hammer-cloned contains one
noisy trajectory that happens to pass through a good state `s'` — a configuration from which the hammer is
well-positioned — but then, being noisy, fumbles the next action and earns nothing; and a second,
separate trajectory that arrives at that same neighborhood of `s'` and *does* complete the task, earning
the late reward. AWAC's mean-backup at `s'` averages the two continuations and gets something mediocre,
so credit for the good continuation is diluted by the fumble. IQL's `V(s')` instead takes the `τ = 0.8`
expectile over the actions logged near `s'`, which is dominated by the *successful* continuation's action
— so `V(s')` reflects "there is a good action available here," the completion value. Then the MSE `Q`
backup carries that `V(s')` one step upstream into the *first* trajectory's `Q(s, a)`, where `a` was the
good action that reached `s'`. Now the first trajectory's good early action inherits the second
trajectory's late success even though that first trajectory never completed the task. Iterate the backup
and the credit walks further upstream, one honest-mean transition at a time, each hop pulling from
whichever trajectory happened to have the best in-support continuation. That is stitching, and it is
strictly impossible for a mean-backup critic that only ever evaluates the average logged continuation.

Now extract a policy, obeying the same commandment — never query `Q` at an unseen action. I cannot
argmax `Q` (searches OOD) or do DDPG-style ascent (evaluates `Q` at the policy's possibly-OOD actions).
I reweight the dataset's own actions: advantage-weighted regression
`L_π(φ) = E_{(s,a)~D}[ exp(β(Q_θ̂(s,a) − V_ψ(s)))·log π_φ(a|s) ]`, weight clipped to ≤ 100. This is the
same family of update AWAC used — and that is fine, because the OOD-safe improvement was never the
problem; the difference that fixes hammer-cloned is *what advantage feeds it*. AWAC's advantage came from
a `Q^π` that only evaluated `π_β`; IQL's comes from a `Q` that has done in-support dynamic programming, so
the weights now reward actions that are good *for the stitched optimal-in-support policy*, not just good
relative to the behavior average. Same extraction, far stronger signal. And I deliberately soften the
temperature relative to AWAC: `β = 3.0` here against AWAC's `1/0.1 = 10` effective sharpness. That is not
a contradiction of my earlier "sharp is good on clean data" reasoning — it is a consequence of the
critic changing. AWAC needed a sharp temperature to squeeze a usable signal out of a weak behavior-value
advantage; now that the advantage comes from a genuine in-support max, the ranking is trustworthy over a
*wider* band of actions, so a softer `β` spreads the maximum-likelihood pull over more of the good
transitions rather than betting everything on the single top one — which on noisy hammer-cloned is
exactly the robustness I want, since it stops the update from collapsing onto one possibly-mis-ranked
transition the way AWAC's top-k pull did.

In this task's harness the IQL fill is specific. The actor is a 2×256 `GaussianPolicy` with a Tanh output
activation and a state-independent `log_std` (`nn.Parameter`), a plain `Normal` (not `TanhTransform`),
with **dropout 0.1** in the actor MLP and a **CosineAnnealingLR** schedule over the 1M offline steps —
both stabilizers that matter on the noisy `cloned` data, where the wide seed spread I saw under AWAC
(pen 39 to 105) is precisely the run-to-run fragility dropout and a decaying learning rate are meant to
damp. The critic is `TwinQ` (two 2×256 MLPs, squeezed, with a `.both()` for the twin outputs), and
`ValueFunction` is a 2×256 squeezed MLP. The hyperparameters match the CORL Adroit config: `iql_tau =
0.8` (the expectile — high enough to do real improvement on the manipulation tasks, low enough to avoid
over-fitting the upper `Q`-noise), `beta = 3.0` (advantage temperature, softer than AWAC's effective
10), `exp_adv_max = 100`. The `Q` backup itself is shape-plain and I check it: `rewards` and `dones` are squeezed to `(B,)`, the
target is `r + (1 − done)·γ·V(s')` with `V(s')` detached, and the twin `Q` outputs are each MSE-regressed
onto that same scalar target — the `(1 − done)` gate zeroing the bootstrap on terminal transitions so a
completion state's value is just its reward, not reward plus a spurious continuation. The update order per
step is V (expectile) → Q (MSE onto `r + γ V(s')`) → Polyak
the target critic → policy (advantage-weighted), and the actor LR cosine-steps. The order matters: `V` is
fit against the *target* `Q` and then `Q` is fit against the *current* `V(s')`, so computing `V` first
means the `Q` backup already sees the freshly-updated expectile, and Polyaking the target critic only
after both keeps the expectile regression reading a slow, stable `Q̂`. The decisive transition fact:
`on_online_start` is a *no-op* — IQL needs no special handling at the handoff, for the same reason AWAC
did not (the value training is policy-free and the same update runs offline and online as the buffer
grows). So IQL keeps AWAC's transition-robustness while replacing the critic's behavior-evaluation with
in-support dynamic programming.

Two harness choices deserve their own justification because they are where I spend the fixed capacity and
the noise budget. The networks are 2×256, not the 3×256 the deterministic and AWAC rungs used, and that
is a deliberate down-sizing: IQL now carries *three* networks — actor, `TwinQ`, and a separate
`ValueFunction` — where AWAC carried two, and the parameter cap is fixed at ~1.2× the largest baseline.
A 3×256 critic is about `150k` parameters and a 2×256 critic about `Linear(72,256)+Linear(256,256)+
Linear(256,1) ≈ 18.7k+65.8k+0.3k ≈ 85k`; dropping the middle hidden layer nearly halves each network, and
with two Q-nets plus a V-net plus the actor that saving is what keeps the three-network stack under the
same cap the two-network methods lived within. So the extra value network is bought not with more total
capacity but by making each network shallower — the contribution is the *architecture of the objective*
(expectile V feeding an MSE Q), not width. Second, the dropout 0.1 and the cosine actor-LR schedule are
aimed squarely at the seed variance I could not shake under AWAC. The policy extraction is a weighted
behavior-cloning regression, and weighted BC on noisy data is exactly the setting where a network overfits
the particular high-weight transitions of a particular seed's minibatch order — dropout regularizes that,
and annealing the actor LR to zero over the 1M offline steps means the final policy is fit with a tiny
step size that averages over the last stretch of updates rather than chasing whichever transitions came
last. Neither touches the value learning; both are there to convert AWAC's 39-to-105 pen spread into
something reproducible.

The falsifiable expectations against the AWAC numbers. My central claim is that the expectile critic's
stitching fixes hammer-cloned: I expect it to climb from AWAC's dead-flat 0.336 into a genuinely positive
range (tens of points, on *every* seed, not one lucky one), because the value function can now propagate
the rare good fragments instead of waiting for the reweighting to stumble onto them. On hammer-expert I
expect to *retain* the recovered competence AWAC found (~127) — the expectile critic should be at least as
safe as AWAC's on clean data, so a regression there would falsify the claim that IQL strictly dominates.
On pen-cloned I expect to *exceed* AWAC's 63.7, and ideally tighten that stubborn 39-to-105 seed spread,
because the stronger advantage signal plus dropout-and-cosine stabilization should lift the two mediocre
seeds toward the strong one. If instead hammer-cloned stays near zero, the conclusion would be that the
binding problem on that dataset is not the critic's improvement capacity at all — it would point somewhere
outside the backup entirely, and I would have to look past a better in-support value rather than sharpen
it further. But the prediction I am betting on is that the in-support max is exactly the missing lever,
and that IQL clears AWAC on hammer-cloned and pen-cloned while holding hammer-expert. The full scaffold
module is in the answer.
