Let me start from the thing that actually annoys me in practice. I have a tabular episodic MDP, `S` states, `A` actions, horizon `H`, and I have a perfectly good provably-efficient algorithm — UCBVI, or EULER, whatever — that, given a reward `r`, finds an `ε`-optimal policy in `Õ(SAH²/ε²)` episodes. Fine. But then someone changes the reward. In real applications the reward is never fixed: you shape it, you sweep a Lagrange multiplier in a constrained formulation, you nudge it to coax out some behavior, you try ten variants. And every single time I change `r`, my algorithm has to go back and interact with the environment from scratch, because the bonus that drove its exploration was computed from the very value function I was maximizing. Exploration and reward are welded together. Ten rewards, ten full exploration bills.

So I want to break that weld. I want to interact with the environment **once**, knowing nothing about the reward — collect some dataset `D` of trajectories — and then, after the fact, when an arbitrary `r` shows up, just *compute* a near-optimal policy from `D` with no more interaction. And not one `r`: any number of them, even adversarially chosen, all served by the same `D`.

There's already a body of work that consumes data like that — batch RL. Munos and Szepesvári, Antos, Chen and Jiang, Agarwal and company. Give them a logged dataset and, *provided the dataset has good coverage*, they return a near-optimal policy by fitted value iteration or by policy gradient on the empirical model. The catch is hiding in "provided." They assume a logging distribution with bounded concentrability — the ratio of any policy's occupancy to the logging distribution is bounded — and they assume it's handed to you. Nobody tells you how to *produce* such a logging policy. That's exactly my missing half. If I can manufacture a covering dataset with no reward, I bolt a batch-RL planner on the back and I'm done.

So the real question is: what does "good coverage" have to *guarantee*, and how do I get it cheaply without a reward to guide me?

Let me think about what could possibly go wrong in the planning phase, because that tells me what coverage must protect against. After exploration I'll have counts, and I'll form an empirical transition model `P̂`. When `r` arrives, I plan on `(P̂, r)` and get some `π̂`. The danger is that `P̂` is wrong somewhere that matters for `r`. How does the wrongness of `P̂` translate into a value error? This is the simulation lemma — the value-difference identity. For two MDPs sharing the policy `π` and the reward `r`, differing only in transitions,

  `V̂^π_1(s_1) − V^π_1(s_1) = E_π[ Σ_h (P̂_h − P_h) V̂^π_{h+1}(s_h, a_h) ]`,

where the expectation is over trajectories of `π` in the true MDP. (That's the telescoping identity: at each step the only discrepancy between the two MDPs is one application of `P̂ − P` to the next-step value, and you accumulate those, weighted by where `π` actually goes.) So the evaluation error of any policy is the transition error `(P̂ − P)V̂` *averaged against the occupancy of that policy*.

Now here's the adversary I have to beat. The reward `r` is revealed after I've fixed `D`, and it can be chosen by someone who has seen my coverage. Suppose there's a state-action `(s,a)` that some policy `π` reaches with decent probability, but which my dataset visited only a handful of times. Then `P̂_h(·|s,a)` is garbage, `(P̂ − P)V̂` there can be as large as `H`, and the adversary just writes down a reward that puts all its mass downstream of `(s,a)`. Now `π` (or `π̂`) routes through `(s,a)`, the occupancy weight there is non-trivial, and the value error blows up. I lose.

The contrapositive is the design spec. To be safe against *every* reward and *every* policy simultaneously, I need: every `(s,a)` that is reachable with non-negligible probability must be visited by my data often enough that `P̂` there is accurate. Coverage isn't a vague "spread out"; it's "for every policy `π`, the occupancy `P^π_h(s,a)` is dominated by my data distribution `μ_h(s,a)`, uniformly in `(s,a)`." That's the concentrability condition, and now I see *why* batch RL needs it and *why* it's the right target: it's exactly what makes the simulation-lemma error uniformly small over all `r`.

But wait — "every reachable `(s,a)`" can't be literally every state. This isn't a bandit where I can pull any arm. In an MDP, some states are genuinely hard to reach: in a chain with a `10^{-6}` branch, the best policy reaches a certain state with probability `10^{-6}`; another state might be unreachable outright. I cannot visit those "enough" — there aren't enough episodes in the universe. So either my whole program is hopeless, or those states don't actually matter.

Stare at that. A state `s` whose maximum reaching probability over *all* policies, `max_π P^π_h(s)`, is tiny — call it below some threshold `δ` — can contribute at most `H·δ` to any value function, because no policy spends more than `δ` probability there and the rewards are bounded by `1`, so the remaining `H` steps contribute at most `H` weighted by `δ`. Sum over all such states: at most `H·S·δ`. So if I choose `δ` small enough, the *entire* contribution of all hard-to-reach states to any value is below my error budget, and I can simply **ignore them**. That's the cut I needed. Define `s` at step `h` to be **`δ`-significant** if `max_π P^π_h(s) ≥ δ`. I only have to cover the significant states; the insignificant ones I write off, and the writeoff is provably cheap.

Now the coverage condition has teeth. I want a data distribution `μ` such that

  for every `δ`-significant `(s,h)`:  `max_{π,a} P^π_h(s,a) / μ_h(s,a) ≤ (some poly factor)`.

That is, every significant state-action is visited by `μ` with probability *proportional to its maximum possible reaching probability*. Not uniformly — proportionally. A state reachable with probability `1` gets a lot of data; a state whose best reaching probability is only `δ` gets data proportional to `δ`. That's the most I could ask for and the least I need.

How do I *produce* such a `μ` with no reward? Here's the move. For a single target `(s,h)`, the policy that visits it best is the solution to a perfectly ordinary RL problem — I just have to invent the reward. Set

  `r'_{h'}(s', a') = 1[ s' = s and h' = h ]`,

a reward of `1` exactly for landing on `s` at step `h`, zero everywhere else. Then for any policy, `V^{π}_1(s_1)` under `r'` equals the probability of reaching `(s,h)` under `π`, so `V*_1(s_1) = max_π P^π_h(s)`, and the optimal policy for `r'` is precisely the policy that maximizes the reaching probability. So I can take *any* episodic regret-minimizing RL algorithm, feed it this fabricated indicator reward through the reward-free protocol, and it will learn to reach `(s,h)` as well as anything can. The reward is a *bonus I design to point at a target*, not a reward I care about — and crucially it never touches the real, later reward. I do this for every `(s,h)` pair, collect the resulting policies, mix them, and I have my `μ`.

Now, which RL algorithm, and how many episodes per target? This is where the significance threshold bites, and it's where a naive choice fails. If I use a worst-case-regret algorithm whose suboptimality scales with the full range `H` of the value, then to learn a target that is only reachable with probability `δ ≈ ε/(SH²)` — small — I'd pay a cost that doesn't know the target is faint, and the bill explodes. I need an algorithm whose cost *scales with how reachable the target actually is*. That's EULER. Its regret is problem-dependent: the leading term scales with the optimal value `V*_1(s_1)` rather than the worst-case range. For my indicator reward, `V*_1(s_1) = max_π P^π_h(s)`, which is exactly the reaching probability — so EULER's bill for a faint target is automatically small. That's not a convenience; it's the whole reason the final complexity comes out sharp.

Let me make EULER's guarantee concrete in my setting. EULER is stated for stationary MDPs; mine is non-stationary, so I augment the state to `(s,h)` and the effective state count becomes `SH`. And my reward is the indicator, so the total reward along any trajectory is at most `1` and `V^π_1 ≤ 1` always. Look at the variance proxy EULER carries — the normalized sum `(1/(N_0H))Σ_k E_{π_k}[(Σ_h r(s_h,a_h) − V^{π_k}_1(s_1))²]`. Expand: `(a − b)² ≤ 2(a² + b²)`; with `a = Σ r ≤ 1` and `b = V^{π_k}_1 ≤ 1`, each square is bounded by the variable itself. Also `E_{π_k}Σ_h r = V^{π_k}_1`, so this normalized average is at most `(4/(N_0H))Σ_k V^{π_k}_1 ≤ 4V*_1(s_1)/H`. In the convention of EULER's theorem this lets me replace the generic `𝒢²` parameter by `4V*_1(s_1)`. Substituting that into EULER's regret (with `S → SH` for the augmentation, and `T = N_0 H` total steps),

  `Σ_{k=1}^{N_0} [ V*_1(s_1) − V^{π_k}_1(s_1) ] ≤ Õ( √( V*_1(s_1) · SAH·N_0 ) + S²AH⁴ )`,

i.e. dividing by `N_0`, the *average* suboptimality of the policy set `Φ^{(s,h)}` it returns is

  `max_π P^π_h(s) − (1/N_0) Σ_{π∈Φ} P^π_h(s) ≤ c·√( SAH·ι₀·max_π P^π_h(s) / N_0 ) + S²AH⁴ι₀³/N_0`.

I want the average reaching probability of my policy set to be at least half of the maximum. So I need both error terms below `(1/2)·max_π P^π_h(s)`. The first, `√(SAHι₀·M/N_0) ≤ c₁ M` with `M = max_π P^π_h(s)`, asks `N_0 ≳ SAHι₀/M`. The second, `S²AH⁴ι₀³/N_0 ≤ c₁ M`, asks `N_0 ≳ S²AH⁴ι₀³/M`. The second dominates. And for a `δ`-significant target, `M ≥ δ`, so

  `N_0 = O( S²AH⁴ι₀³ / δ )`

episodes suffice to make `(1/N_0)Σ_{π∈Φ} P^π_h(s) ≥ (1/2) max_π P^π_h(s)`, i.e. `max_π P^π_h(s) / ( (1/N_0)Σ_{π∈Φ} P^π_h(s) ) ≤ 2`. Good — the average reaching policy is within a factor `2` of the best, for the *state*.

But coverage has to be in terms of state-*actions*, and the optimal reaching policy might deterministically pick one action at `(s,h)`. So at the target cell itself I override: set `π_h(·|s) = Uniform(A)` for every `π ∈ Φ^{(s,h)}`. The reaching probability of `(s,h)` is fixed before step `h` and doesn't care what I do at `h`, so this override costs nothing on the reaching side, but it spreads the action mass uniformly, turning the factor `2` on the state into a factor `2A` on the state-action:

  `max_{π,a} P^π_h(s,a) / ( (1/N_0)Σ_{π∈Φ} P^π_h(s,a) ) ≤ 2A`.

Now assemble. Let `Ψ` be the union of `Φ^{(s,h)}` over all `SH` significant cells, and let my data distribution be: sample a policy uniformly from `Ψ`, run it. Then `μ_h(s,a) = (1/(N_0 SH)) Σ_{π∈Ψ} P^π_h(s,a)`. For a target `(s,h)`, only its own `Φ^{(s,h)}` block is guaranteed to cover it, and that block is a `1/(SH)` fraction of `Ψ`, so the factor `2A` becomes `2A·SH = 2SAH`:

  for every `δ`-significant `(s,h)`:  `max_{π,a} P^π_h(s,a) / μ_h(s,a) ≤ 2SAH`.

That's my coverage theorem. The factor decomposes cleanly: `2` from EULER halving the gap, `A` from the uniform action override, `SH` from diluting across the `SH` target blocks. The dataset `D` is then just `N` i.i.d. trajectories sampled from `μ`.

Now the planning side. `r` arrives, I count, form `P̂`, and call any solver that returns an `ε`-suboptimal policy for the *known* model `(P̂, r)` — exact value iteration gives optimization error `0`; an approximate planner like NPG gives some small error. I need: `π̂` is near-optimal not just on `(P̂, r)` but on the *true* `(P, r)`. The bridge is a uniform evaluation guarantee — that for *every* policy and *every* reward, the empirical value and true value are within `ε`:

  `| E_{s_1}[ V̂^π_1(s_1; r) − V^π_1(s_1; r) ] | ≤ ε`,  for all `π`, all `r`, simultaneously.

If I have that, the suboptimality decomposition is immediate. Let `π*` and `π̂*` be the optimal policies for `(P, r)` and `(P̂, r)`. Then

  `V^{π*}_1 − V^{π̂}_1 = [V^{π*}_1 − V̂^{π*}_1] + [V̂^{π*}_1 − V̂^{π̂*}_1] + [V̂^{π̂*}_1 − V̂^{π̂}_1] + [V̂^{π̂}_1 − V^{π̂}_1]`.

The first and last are evaluation errors, each `≤ ε`. The middle term `V̂^{π*}_1 − V̂^{π̂*}_1 ≤ 0` because `π̂*` is optimal on `P̂`. The third is the optimizer's own suboptimality on `(P̂, r)`, `≤ ε` by assumption. Total: `3ε`. And because the evaluation guarantee is uniform over `r` and `π`, this holds for *all* rewards at once — that's the whole point, the adversary gets nothing.

So everything reduces to proving the uniform evaluation bound. Start from the simulation lemma with shared reward:

  `| E_{s_1}[ V̂^π_1 − V^π_1 ] | ≤ E_π Σ_h | (P̂_h − P_h) V̂^π_{h+1}(s_h, a_h) |`.

Split each step `h` by significance — let `S^δ_h` be the significant states at step `h`:

  `E_π | (P̂_h − P_h)V̂^π_{h+1} | = Σ_{a, s∈S^δ_h} |(P̂−P)V̂|·P^π_h(s,a)  +  Σ_{a, s∉S^δ_h} |(P̂−P)V̂|·P^π_h(s,a)`.

The insignificant tail is the easy half: `|(P̂−P)V̂| ≤ H` since `V̂ ∈ [0,H]`, and `Σ_{s∉S^δ_h} P^π_h(s) ≤ Σ_{s∉S^δ_h} δ ≤ Sδ` because each insignificant state has occupancy at most `δ` (its max over all policies is below `δ`). So the tail is `≤ HSδ` per step, `≤ H²Sδ` over the horizon. Choosing `δ = ε/(2SH²)` makes this `≤ ε/2`. (That's also why I set the significance threshold there — it's pinned by the planning-error budget.)

The significant half needs the coverage. By Cauchy–Schwarz against the occupancy `P^π_h(s,a)` (a sub-probability, total `≤ 1`),

  `Σ_{a,s∈S^δ_h} |(P̂−P)V̂|·P^π_h(s,a) ≤ ( Σ_{a,s∈S^δ_h} |(P̂−P)V̂|² · P^π_h(s,a) )^{1/2}`.

Write `P^π_h(s,a) = P^π_h(s)·π_h(a|s)`. Now a subtle but important point: `V̂^π_{h+1}` depends on `π` only at steps `> h`, not on `π_h`. So I can replace `π_h` by the *worst* action choice without changing `V̂`, and the maximizing action distribution at step `h` is deterministic — some map `ν: S → A`:

  `Σ |(P̂−P)V̂|² P^π_h(s)π_h(a|s) ≤ max_ν Σ_{a,s∈S^δ_h} |(P̂−P)V̂|² P^π_h(s) 1[a = ν(s)]`.

Here's where I cash in coverage. For a significant `(s,h)` and any action `a`, I can build a policy `π'` agreeing with `π` before step `h` and deterministically playing `a` at `(s,h)`, so `P^π_h(s) = P^{π'}_h(s,a) ≤ 2SAH·μ_h(s,a)`. Substituting and dropping the restriction to `S^δ_h` (only adds nonnegative terms),

  `≤ 2SAH Σ_{s,a} |(P̂−P)V̂|² μ_h(s,a) 1[a=ν(s)] = 2SAH · E_{μ_h}[ |(P̂−P)V̂|² 1[a=ν(s)] ]`.

So everything now rests on bounding `E_{μ_h}[ |(P̂_h − P_h)G(s,a)|² 1[a=ν(s)] ]` uniformly over the value function `G ∈ [0,H]^S` and the deterministic action map `ν`. This is the concentration heart of the argument.

Let me get the cheap version first: for fixed `G`, `(P̂−P)G(s,a)` is the deviation of an average of `N_h(s,a)` bounded terms, so it's `O(H/√N_h(s,a))`, and `E_μ |(P̂−P)G|²` is roughly `H²·(number of cells)/N ≈ H²·SA/N`. But that has an `A` I don't want, and it's not adaptive to `ν`. I can do better with a self-bounding trick. Define, for the `i`-th sample `(s_i, a_i, s'_i) ∼ μ_h × P_h`,

  `X_i = (P̂_h G(s_i,a_i) − G(s'_i))² − (P_h G(s_i,a_i) − G(s'_i))²`,

and `Y_i = X_i · 1[a_i = ν(s_i)]`. Three properties make this work. First, the expectation: with `b = P̂_h G − G(s')`, `d = P_h G − G(s')`, write `b² − d² = (b−d)² + 2d(b−d)`; taking `E_{s'}` kills the cross term because `E_{s'}d = P_hG − E_{s'}G(s') = 0`, leaving `E[Y] = E_{μ_h}[ |(P̂−P)G|² 1[a=ν(s)] ]` — exactly the quantity I want to bound. Second, empirical-risk-minimization: `P̂_h G(s,a)` is by definition the empirical mean of `G(s')` over samples at `(s,a)`, i.e. the minimizer of `Σ_i (g − G(s'_i))²`, so `Σ_i Y_i ≤ 0`. Third, self-bounding: `b² − d² = (b+d)(b−d)`, and `|b+d| ≤ 2H`, so `Var(Y) ≤ E(Y²) ≤ 4H² · E_{μ_h}[|(P̂−P)G|² 1[a=ν(s)]] = 4H² E[Y]`. The variance is controlled by the mean.

Now Bernstein on `(1/N)Σ Y_i`, but `P̂`, `G`, and `ν` are all chosen after seeing data, so I cover. For `ν` there are `A^S` deterministic maps. For the values `P̂G(s,a)1[a=ν(s)]` and `PG(s,a)1[a=ν(s)]`, fixing `ν` they are nonzero on only `S` cells, each in `[0,H]`, so an `ε`-net has `(H/ε)^{2S}` points; the rounding costs at most `12Hε`. (This is the `A`-improvement: covering deterministic-policy-restricted values needs `A^S·(H/ε)^{2S}` balls, not the `(1/ε)^{SA}` you'd need to cover all Q-values blindly.) Bernstein, using `Σ Y_i ≤ 0` so `E[Y] ≤ E[Y] − (1/N)ΣY_i`, gives with probability `1 − p/H`

  `E[Y] ≤ √( 2 Var(Y)·[2S log(HA/ε) + log(H/p)] / N ) + H²[2S log(HA/ε)+log(H/p)]/(3N) + 12Hε`.

Plug `Var(Y) ≤ 4H² E[Y]` and pick `ε = HS/(36N)`. The first term is `√( 8H² E[Y]·(2S log(36AN/S)+log(H/p))/N )`. This is a quadratic in `√(E[Y])`: `E[Y] ≤ √(c'·E[Y]/N) + c''/N` solves to `E[Y] ≤ O(c'/N + c''/N)`, i.e.

  `E_{μ_h}[ |(P̂−P)G|² 1[a=ν(s)] ] ≤ O( (H²S/N) log(AHN/p) )`.

Union bound over `h` to make it hold for all steps with probability `1−p`. Notice the bound is `H²S/N`, not `H²SA/N` — the self-bounding plus the deterministic-policy covering bought me the factor `A`.

Now reassemble the significant half. Each step contributes `(2SAH · O(H²S/N · log))^{1/2} = O(√(H³S²A/N · log))`, and summing the square-rooted per-step bound over `H` steps gives `O(√(H⁵S²A/N · log))`. (The horizon enters as `H` from the sum and `H³` inside, giving `H⁵` under the root after squaring out — let me keep it honest: per step `√(2SAH·H²S/N·log) = √(2H³S²A/N·log)`; times `H` steps is `H·√(H³S²A/N) = √(H⁵S²A/N)`.) Combine with the insignificant tail:

  `| E_{s_1}[ V̂^π_1 − V^π_1 ] | ≤ O( √(H⁵S²A/N · log(AHN/p)) ) + H²Sδ`.

With `δ = ε/(2SH²)` the tail is `ε/2`, and choosing `N ≥ c·H⁵S²Aι/ε²` for `ι = log(SAH/(pε))` makes the first term `≤ ε/2`. So `N = O(H⁵S²Aι/ε²)` samples in the planning dataset give the uniform evaluation bound, hence `3ε`-optimality for every reward. Rescaling `ε`, the dataset needs `Õ(H⁵S²A/ε²)` trajectories.

Total exploration budget. Sampling `D` costs `N = O(H⁵S²Aι/ε²)`. Building `Ψ` costs `N_0 = O(S²AH⁴ι³/δ)` per cell with `δ = ε/(2SH²)`, i.e. `N_0 = O(S³AH⁶ι³/ε)`, times `SH` cells: `O(S⁴AH⁷ι³/ε)`. So

  `K ≤ c·[ H⁵S²Aι/ε² + S⁴AH⁷ι³/ε ]`,

dominated by the first term: `Õ(H⁵S²A/ε²)`. The reward-free protocol decouples cleanly — phase one builds `Ψ` and samples `D` with no reward; phase two counts and plans for any `r`.

Now I should ask: is the `S²` real, or an artifact of my analysis? Single-reward RL is `Θ̃(SAH²/ε²)` — only one `S`. Could reward-free be the same? I don't think so, and I want a matching lower bound to be sure where the extra `S` comes from. Consider the simplest hard instance: a single start state `0` that transitions, in one step, to one of `2n` absorbing terminal states according to an unknown distribution `q(·|0,a)` that I take near-uniform, `|q(s,a) − 1/(2n)| ≤ ε/(2n)`. Rewards live on the terminals, action-independent, `r_ν(s) = ν(s)`. To be reward-free-correct here, for any `ν` the learner must return a policy whose action at `0` maximizes `⟨q(·,a), ν⟩` — and ranging `ν` over the cube forces the learner to know `q(·,a)` for *every* action `a` in total-variation. Estimating a `2n`-dimensional near-uniform distribution to TV accuracy `ε` for each of `A` actions costs `Ω(nA/ε²)`. The interesting thing: here the factor `n ≈ S` comes from the transition being *to* `Θ(S)` states, not *from* `Θ(S)` states as in the usual single-reward lower bounds.

To make the reduction rigorous I can't go directly from "near-optimal policy" to "estimate `q` in TV"; instead I build a packing. Take `±1` vectors `v` on `[2n]` with `1^⊤v = 0`, and a family `{v_{a,j}}` that is `γ`-uncorrelated: `|⟨v_{a,j}, v_{a',j'}⟩| < 2nγ` for distinct pairs. A probabilistic-method/Chernoff argument shows such packings of size `e^{Ω(nγ²)}` exist — drawing `v` uniformly, `⟨v_{a,j},v_{a',j'}⟩ = 2n − 4Z` with `Z ∼` (roughly) `Binom(n,1/2)`, and `Pr[|Z/n − 1/2| ≥ γ/2] ≤ e^{log(4n) − nγ²}`; union over `A²M²` pairs needs `2 log M ≤ nγ² − log(4n) − 2 log A`. Define transitions `q_{a,j} = q_0 + (ε/2n)v_{a,j}` with `q_0` uniform. Now Fano: an estimator that recovers the index `J_a` for each action with probability `1−p` is lower-bounded through the per-instance KL. By Wald's identity the KL between two instances differing only in action `a` is `E[N_K(a_1=a)]·KL(Mult(q_{a,j}), Mult(q_0))`, and `KL(Mult(q_{a,j}),Mult(q_0)) = Σ_s (1+εv)/(2n) log(1+εv) ≤ ε²` using `log(1+x) ≤ x`, `Σ v = 0`, `v² = 1`. Fano gives `(1−p)log M − log 2 ≤ ε²·E[N_K(a_1=a)]` per action; summing over `a`,

  `E[K] ≥ A·((1−p)log M − log 2)/ε² ≳ nA/ε²`

for `M = e^{Ω(n)}`. The last gap is "estimation reduces to exploration": reward vectors `ν_{a_1,a_2,j_1,j_2} = (1/2)1 + (1/3)v_{a_1,j_1} + (1/6)v_{a_2,j_2}` act as separating hyperplanes. Since `<q_{a_1,j_1} − q_{a_2,j_2}, ν> = (ε/(12n))<v_{a_1,j_1} − v_{a_2,j_2}, 2v_{a'_1,j'_1} + v_{a'_2,j'_2}>`, the self-inner-product gives a positive margin `> ε/12` when the first index is the true `j_1 = J_a`, while choosing the second vector to match the competing action gives a negative margin `< −ε/12` when `j_1 ≠ J_a`. The sign of which action the returned near-optimal policy plays decodes `J_a`. Hence an `(ε/24,p)`-correct reward-free explorer yields the estimator, and the `Ω(nA/ε²)` bound transfers.

That's `Ω(SA/ε²)` — still only one `S`. The second `S` comes from embedding. Put `n = Ω(S)` independent copies of the single-state instance as the second-to-last layer of a depth-`(1+log₂ n)` binary tree: state `(x, log₂ n)` for `x ∈ [n]`, each with its own unknown `q_x`, all feeding the same `2n` absorbing leaves; action `1` always goes left, actions `> 1` go right, so the learner reaches any chosen `(x, log₂ n)` deterministically. Reward `r_{x,ν}` puts `1` on `(x, log₂ n)` and `ν` on the absorbing leaves. The reward `1` forces any `ε`-optimal policy to actually visit `(x, log₂ n)` with probability `≥ 1/2` (if it visited less, the `1`-reward shortfall alone exceeds `ε` — that's the `V^π = Pr[visit] + (H−ℓ−1)·⟨occupancy, ν⟩` accounting, with the near-uniform leaf transitions pinning the second term to within `1/8`). Conditional on visiting the copy, the persistent leaf reward makes `ε`-optimality in the embedded MDP imply `4ε/H`-optimality for the corresponding single-state problem. Varying `x` forces `n` separate copies; Fubini turns "episodes visiting `(x,·)`" into the per-copy budget, so each copy costs `Ω(nA/(ε/H)²) = Ω(nAH²/ε²)` visits and

  `E[K] = Ω(S²AH²/ε²)`.

So the `S²` is real and intrinsic to *reward-free* — the price of good coverage, exactly one factor `S` above single-reward RL, and my upper bound matches up to `H` powers and logs.

One loose end: I claimed the planner can be any approximate MDP solver. Value iteration on `(P̂, r)` solves the Bellman optimality equation exactly, so its optimization error is `0` — the cleanest instantiation. But people use policy gradient in practice, so let me also pin down natural policy gradient as an instance and bound its optimization error. NPG is multiplicative-weights on the policy: `π^{(t+1)}_h(a|s) ∝ π^{(t)}_h(a|s)·exp(η(Q^{(t)}_h(s,a) − V^{(t)}_h(s)))`, starting uniform, with `Q^{(t)}` evaluated by Bellman backups. Through the performance-difference lemma — `V^{π}_1 − V^{π'}_1 = Σ_h E_π[ Σ_a π_h(a|s)(Q^{π'}_h(s,a) − V^{π'}_h(s)) ]` — the exp-weights update telescopes into KL terms. Monotonicity first: `V^{(t+1)}_1 − V^{(t)}_1 = (1/η)Σ_h E_{π^{(t+1)}}[ KL(π^{(t+1)}_h ‖ π^{(t)}_h) + log Z^{(t)}_h ] ≥ (1/η)Σ_h E[log Z^{(t)}_h] ≥ 0`, the last step by Jensen since `log Z ≥ η Σ_a π^{(t)}(a)(Q^{(t)} − V^{(t)}) = 0`. For the rate, compare to `π*`: `V*_1 − V^{(t)}_1 = (1/η)Σ_h E_{π*}[ KL(π*_h‖π^{(t)}_h) − KL(π*_h‖π^{(t+1)}_h) + log Z^{(t)}_h ]`. Average over `t = 0..T−1`; the KL differences telescope, leaving `KL(π*‖π^{(0)}) ≤ H log A` (uniform init) plus the `log Z` sum. To bound `log Z^{(t)}_h`: for `x ≤ 1`, `exp(x) ≤ 1 + x + x²`, so if `η ≤ 1/H` then `η(Q^{(t)} − V^{(t)}) ≤ 1` and `log Z^{(t)}_h ≤ log(1 + η²Σ_a π(a)(Q−V)²) ≤ η²Σ_a π(a)(Q−V)² ≤ η²H²` (the linear term vanishes since `Σ_a π(a)(Q−V) = 0`). Therefore

  `E[V*_1 − V^{(T)}_1] ≤ (H log A)/(ηT) + ηH²`.

Optimize: `η = √(log A/(HT))`, `T = 4H³ log A/ε²` makes this `≤ ε`. So NPG is an `ε`-suboptimal approximate solver, plugs into the planning slot, and the `3ε` decomposition closes.

Let me also sanity-check why I didn't just run R-max with no reward, since that's the obvious baseline. Take R-max, set the reward to `0` on the known set and `1` on unknown pairs (call it ZeroRMax), so the agent is purely driven to grow the known set `K`. Optimism plus the escape-probability pigeonhole (the agent can leave `K` at most `O(mSA)` times, so few episodes have large escape probability) gives a coverage guarantee, and chasing the value-difference lemma through the truth `M`, the truncated `M_K`, and the empirical `M̂_K` yields suboptimality `H³ε_escape + Õ(H⁴√(S/m))`. Forcing that to `ε` needs `m = Ω(SH⁸/ε²)` and `ε_escape = O(ε/H³)`, hence `N = Ω(H^{11}S²A/(ε³p)·log²)`. That's `ε^{−3}` and *polynomial* in `1/p` — far worse than my `ε^{−2}` and `log(1/p)`. The lesson is exactly the one I started with: reward-dependent exploration, even with the reward gamed, is the wrong objective; the indicator-per-target construction with a value-dependent explorer is what makes coverage cheap. (The max-entropy route — maximizing `(1/S)Σ_s log d_π(s)` by Frank–Wolfe — has the same flavor of being suboptimal: chasing its objective to the error `O(1/S)` needed for a comparable coverage ratio drives sample complexity to `S⁵`, and it never closes the loop to a planning guarantee.)

So the causal chain: I refuse to re-explore per reward → coverage against an adversarial reward forces "every significantly-reachable `(s,a)` visited proportional to its max reachability" via the simulation lemma → states with tiny max-reachability contribute `≤ HSδ` and can be ignored, which defines `δ`-significance → to cover each significant `(s,h)` I fabricate an indicator reward pointing at it and run a *value-dependent* explorer (EULER), whose cost scales with the target's reachability, so faint targets are cheap → mixing the per-target policies gives `μ` with `max P^π/μ ≤ 2SAH` on significant cells → a self-bounded Bernstein concentration over deterministic-policy-restricted values turns that coverage into a uniform-over-all-rewards evaluation bound `≤ ε`, hence `3ε`-optimal planning for every reward → `Õ(H⁵S²A/ε²)` exploration episodes, matched by an `Ω(S²AH²/ε²)` lower bound whose extra `S` is the price of decoupling.

```python
import numpy as np

# ---- Phase 1: explore once, no reward -------------------------------------
def reward_free_explore(env, N0, N, rng):
    S, A, H = env.S, env.A, env.H
    Psi = []
    for h in range(H):
        for s in range(S):                       # one block per (s, h) cell
            r_ind = [np.zeros((S, A)) for _ in range(H)]
            r_ind[h][s, :] = 1.0                 # indicator reward: reach (s,h)
            Phi = regret_minimizing_explorer(env, r_ind, N0, rng)  # EULER stand-in
            for pi in Phi:
                pi[h][s, :] = 1.0 / A            # Uniform(A) at target -> cover actions
            Psi.extend(Phi)
    # mu = uniform mixture over Psi  =>  max_pi P^pi/mu <= 2SAH on significant cells
    D = [env.rollout(Psi[rng.integers(len(Psi))], rng) for _ in range(N)]
    return D

# ---- Phase 2: plan for ANY revealed reward, no more interaction ------------
def reward_free_plan(D, S, A, H, reward, solver="VI", eta=None, T=None):
    Nsas = [np.zeros((S, A, S)) for _ in range(H)]
    Nsa  = [np.zeros((S, A))     for _ in range(H)]
    for traj in D:                               # empirical transition counts
        for h, s, a, s_next in traj:
            Nsas[h][s, a, s_next] += 1
            Nsa[h][s, a] += 1
    Phat = []
    for h in range(H):
        cnt = Nsa[h][:, :, None]
        Phat.append(np.where(cnt > 0, Nsas[h] / np.maximum(cnt, 1), 1.0 / S))
    if solver == "VI":                           # exact optimizer: opt-error = 0
        pi, _ = value_iteration(Phat, reward, H)
        return pi
    return natural_policy_gradient(Phat, reward, H, eta, T)  # eps-suboptimal

# ---- approximate MDP solvers ----------------------------------------------
def value_iteration(P, r, H):
    S, A = P[0].shape[0], P[0].shape[1]
    V_next = np.zeros(S); pi = [np.zeros((S, A)) for _ in range(H)]
    for h in reversed(range(H)):
        Q = r[h] + P[h] @ V_next                 # Bellman optimality backup
        g = np.argmax(Q, axis=1)
        pi[h][np.arange(S), g] = 1.0
        V_next = Q.max(axis=1)
    return pi, V_next

def natural_policy_gradient(P, r, H, eta, T):    # pi <- pi * exp(eta (Q - V)), normalize
    S, A = P[0].shape[0], P[0].shape[1]
    pi = [np.full((S, A), 1.0 / A) for _ in range(H)]   # uniform init
    for _ in range(T):
        V_next = np.zeros(S); Q = [None] * H
        for h in reversed(range(H)):             # evaluate Q^{pi^{(t)}}
            Q[h] = r[h] + P[h] @ V_next
            V_next = (pi[h] * Q[h]).sum(axis=1)
        for h in range(H):                       # exp-weights update
            A_h = Q[h] - (pi[h] * Q[h]).sum(axis=1, keepdims=True)
            logits = np.log(pi[h] + 1e-300) + eta * A_h
            logits -= logits.max(axis=1, keepdims=True)
            w = np.exp(logits); pi[h] = w / w.sum(axis=1, keepdims=True)
    return pi                                    # eta=sqrt(logA/HT), T=4H^3 logA/eps^2
```
