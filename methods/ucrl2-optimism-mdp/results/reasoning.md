Let me start from what actually hurts. I have an unknown finite MDP — states `s`, actions `a`, a reward in `[0,1]` with unknown mean `r̄(s,a)`, and an unknown transition law `p(·|s,a)` over the `S` states. No discount; I care about the long-run average reward, and the thing I want to keep small is the regret I pile up *while learning*:

```
Δ(T) = T·ρ* − Σ_{t=1..T} r_t ,
```

where `ρ*` is the optimal average reward. Not the quality of some policy I hand in at the end — the rewards I actually miss, step by step, including every step I waste stranded somewhere unrewarding because I made a bad exploratory move. So exploration is not free; it is charged at full price.

The tools I have for this are PAC. E3 partitions states into "known" and "unknown," carries two model MDPs — an exploitation one and an exploration one with an absorbing bonus on the unknown frontier — and at each known state it explicitly *decides*: is there a near-optimal policy in my known-MDP? then exploit; otherwise run the exploration policy to go fetch a new sample. R-max is cleaner: initialize every unknown `(s,a)` optimistically to `R_max` with a self-loop, solve the fictitious model, follow it, and you're guaranteed to either be near-optimal or hit an unknown `(s,a)` and learn — explore-or-exploit made implicit. Both are correct. But both give me an `ε`-optimal average reward after time polynomial in `1/ε³`, and a `1/ε³` sample-complexity bound, when I convert it into regret over a horizon, gives me `T^{2/3}`. And both need the `ε`-return mixing time of an optimal policy as an *input* — a number I don't know, and if I guess it too small the bound blows up to exponential in the true one. Two things bother me here: the exponent `2/3` instead of `1/2`, and the dependence on a mixing time I have to be told.

So `√T`. Where do I actually know how to get `√T` regret with no oracle parameters? The one-state case. The multi-armed bandit. UCB1: for each arm keep the empirical mean `x̄_i` and pull the arm maximizing

```
x̄_i + sqrt( 2 ln n / T_i ) .
```

That second term is exactly the Chernoff–Hoeffding half-width — `x̄_i` is within `sqrt(2 ln n / T_i)` of the true `μ_i` with overwhelming probability. And the magic is that I never wrote down an exploration schedule. The confidence interval *is* the exploration rule: an arm pulled few times has a fat radius, so a high index, so it gets tried; once it's been pulled enough the radius drops below the gap and it's abandoned. The accounting falls out — each suboptimal arm `j` gets pulled at most `8 ln n / Δ_j²` times, and summing the widths gives `O(√(Kn))` worst case. The whole exploration/exploitation tension is dissolved by one move: *act optimistically on your confidence interval.*

That's the thing I want to lift. Not "add an exploration bonus sometimes," not "flag states known/unknown" — the bandit doesn't need any of that. It needs: a confidence region around the unknown parameters, and the decision to behave as if the truth is the most favorable point in that region. Whatever I build for the MDP, I want *that* to be the engine, and I want the explore/exploit branch and the mixing-time input both to vanish the way they vanish for UCB1.

So: keep a set `M` of MDPs that are statistically plausible given the data so far. Pick, out of that set, the MDP whose *optimal average reward is the largest* — the most optimistic plausible world — and follow its optimal policy. An under-visited `(s,a)` will have wide confidence, so the optimistic world can assign it a flatteringly high reward and a flatteringly convenient transition, which makes a good policy in that world want to go *try* it. Exploration, again, becomes a side effect of optimism. That's the bet.

Two things the bandit never had to deal with come up the moment I try this, and I should face them before I write any algorithm.

The first: in a bandit the unknown is a vector of scalar means, and a confidence interval is a scalar interval. In an MDP the unknown also includes a whole *distribution* `p(·|s,a)` over `S` next-states. How tightly does an empirical distribution concentrate? For a scalar mean from `n` samples, Hoeffding gives width `~ sqrt(1/n)`. For a distribution over `m = S` outcomes, the relevant statement is the `L1` deviation bound — the empirical and true distributions differ in `L1` by `ε` with probability at most `(2^m − 2) exp(−n ε²/2)`. Set that tail to a target and solve: `ε ~ sqrt( S / n )`. So learning a transition vector concentrates a factor of `√S` *worse* than learning a scalar. That `√S` is not an artifact I can optimize away; it is the cost of an `S`-outcome distribution, and it will show up in the bound. Fine — I note it and move on.

The second is nastier, and it's why MDPs aren't bandits. In a bandit a wrong pull costs me one round. In an MDP a wrong action can *transport* me — into a corner of the state space from which it takes many steps to crawl back to where the reward is. I need a number that measures "how bad can a single misstep be," and it should depend only on the transition structure, not on rewards and not on any policy I'm trying to learn. The natural one: for any pair `s ≠ s'`, the expected time to get from `s` to `s'` under the policy that's *best at doing exactly that*, and then take the worst pair:

```
D = max_{s≠s'} min_π E[ T(s'|M,π,s) ] .
```

The diameter. It's finite exactly when the MDP is communicating — when from anywhere you can in principle reach anywhere. It's purely about transitions. And notice it can't be tiny: to address `S` distinct states using `A` choices per step you need at least about `log_A S` steps, so `D ≥ log_A S − 3` always. I expect `D` to multiply the exploration term — the price of a misstep — and I expect the final bound to look like `D · (something with √S, √A, √T)`. Let me see if the math agrees.

Before the bound, the algorithm has to actually *plan* in the optimistic MDP, and here is the first real wall. In a bandit "optimistic planning" is trivial — `argmax` over indices. In an MDP, optimism isn't just about rewards; I also get to choose, within the confidence ball, the *transition law*. A good policy in the optimistic world will want transitions that funnel probability toward high-value states. So planning has to optimize jointly over the policy *and* the plausible transition vectors. Plain value iteration doesn't do that. I need to extend it.

Collapse the entire plausible set into a single MDP with an enriched action set. For each real action `a` at state `s`, and each admissible transition vector `p̃(·|s,a)` inside the `L1` confidence ball, and each admissible mean reward `r̃(s,a)` inside the reward interval — make that a *separate* extended action. Call this `M̃⁺`; its action set is continuous. A policy on `M̃⁺` is exactly a choice of `(a, p̃, r̃)` per state, which is exactly "a plausible MDP plus a policy on it." So maximizing average reward over plausible MDPs and their policies *is* solving `M̃⁺`. Now run undiscounted value iteration on `M̃⁺`:

```
u_0(s) = 0,
u_{i+1}(s) = max_a { r̃(s,a) + max_{ p(·) ∈ P(s,a) } Σ_{s'} p(s') u_i(s') } ,
```

with `r̃(s,a) = r̂(s,a) + (reward radius)` — push the reward to the top of its interval — and `P(s,a)` the set of transition vectors within the `L1` ball. The outer `max_a` is the usual one. The inner `max_p` is new: maximize a *linear* function `Σ p(s') u_i(s')` over the `L1`-ball of distributions. That's a linear program, so the maximum sits at a vertex, and I don't have to call an LP solver — I can read it off. To push expected next-value up, I want as much probability mass as possible on the next-state with the largest `u_i`, taken from the states with the smallest `u_i`. Concretely: sort states by `u_i` descending; start from the empirical `p̂(·|s,a)`; add the whole allowed `L1` slack to the top state — bump `p(s'_1) = min{1, p̂(s'_1|s,a) + d(s,a)/2}` — and remove that same mass from the bottom states, draining the lowest-value ones to zero first until the vector is a probability distribution again. The `L1` budget is `d(s,a)`, and shifting `d/2` up and `d/2` down moves exactly `d` of `L1` mass. Sorting once per iteration, this inner step is `O(S)` per `(s,a)`, so `O(S²A)` per iteration of value iteration. Good — the optimistic planning is essentially free on top of ordinary VI.

But does this even converge? Undiscounted value iteration on an MDP can *fail* to converge if there's an optimal policy with a periodic transition matrix — the iterates oscillate. I have to make sure extended value iteration never latches onto such a policy. Stare at the inner maximization: in every iteration there's a single state `s'_1` with the maximal `u_i`, and because `d(s,a) > 0` the inner step puts positive probability on `s'_1` for every chosen row — including the row that starts at `s'_1` itself. A transition matrix with a positive self-loop at this common target state is aperiodic. So the policy extended value iteration selects always has an aperiodic transition matrix, which is exactly the condition Puterman's convergence proof needs at its delicate step. I also need the extended MDP to have a state-independent average reward; the `L1` ball alone does not promise that, but a plausible set containing some communicating MDP does. On the episodes that matter for the main regret accounting, the true MDP is in the set, so that condition is satisfied. Then `u_{i+1} − u_i → ρ̃·1`, and the recentred iterate `u_i − min_s u_i(s)` converges to the bias vector. So I can stop when the increments have nearly equalized:

```
max_s (u_{i+1}(s) − u_i(s)) − min_s (u_{i+1}(s) − u_i(s)) < ε ,
```

and the greedy policy is then `ε`-optimal. I'll set `ε = 1/√t_k` at episode `k` — accurate enough that the planning error contributes only `O(√T)` total, cheap enough to compute. The optimistic average reward `ρ̃_k` I land on satisfies `ρ̃_k ≥ ρ* − 1/√t_k`, because the true MDP is (with high probability) inside the plausible set, so the most-optimistic plausible reward can only exceed the true optimum, up to the `1/√t_k` stopping slack.

Here's a fact about these `u_i` on a good episode. The *span* of `u_i` — `max_s u_i(s) − min_s u_i(s)` — is at most `D`. Why: `u_i(s)` is the best total `i`-step reward starting at `s` in `M̃⁺`, and when the true MDP is plausible, `M̃⁺` contains the true MDP's actions, so its diameter is `≤ D`. Suppose `u_i(s'') − u_i(s') > D` for two states. Then I could get a *better* `i`-step value starting from `s'`: first run the policy that drives `s' → s''` fastest, which takes `≤ D` steps on average, then run the optimal `i`-step policy for `s''`. I forfeit at most `D` of the `i` rewards by the detour, so this gives `u_i(s') ≥ u_i(s'') − D`, contradicting the supposition. Hence `span(u_i) ≤ D`. After I recenter — define `w_k(s) = u_i(s) − (max u_i + min u_i)/2` — I get `‖w_k‖_∞ ≤ D/2`. This recentered value vector is what's going to get multiplied by transition errors in the regret, so the diameter is going to ride in on it. There it is: the planning-error price is the span of the value function, and the span is bounded by the diameter.

Now the regret. First strip off the reward noise. The actual rewards `r_t` are bounded `[0,1]` and, given the visit counts, independent, so Hoeffding lets me replace `Σ r_t` by `Σ_{s,a} N(s,a) r̄(s,a)` at a cost of `sqrt((5/8) T log(8T/δ))`. So if I write the per-episode regret as `Δ_k = Σ_{s,a} v_k(s,a) (ρ* − r̄(s,a))`, where `v_k(s,a)` is how often I take `(s,a)` in episode `k`, then with high probability

```
Δ(T) ≤ Σ_k Δ_k + sqrt( (5/8) T log(8T/δ) ) .
```

Wait — episodes. I've been saying "episode `k`" without deciding what an episode *is*. Why split into episodes at all, and when do I recompute? Recomputing the optimistic policy every single step is the bandit habit, and for a bandit it's fine because the `argmax` is cheap. But extended value iteration is `O(S²A)` per iteration — I can't afford it per step. And I don't need to: my estimates only change meaningfully when I've gathered substantially more data about something. So I'll recompute only at the start of an *episode*, and end an episode the moment the visit count of the action I'm currently taking has *doubled* relative to what it was when the episode began. The doubling rule. It does three jobs at once, which is why it's the right rule and not, say, fixed-length episodes. First, it caps the number of episodes: each episode doubles some `N(s,a)`, and a count can double only `~log` times, so the total number of episodes is `m ≤ SA log_2(8T/(SA))` — logarithmic in `T`. That keeps both the total planning cost and certain additive `D·m` terms logarithmic. Second, it makes the within-episode counts satisfy `N_k ≤ N(s,a) ≤ 2N_k`, which is exactly what lets a sum of confidence widths telescope. Third, it's the cheapest replanning schedule that still keeps my policy roughly current. So: episodes are defined by doubling, and I have to be careful that the high-probability statements hold uniformly over the (data-dependent, random) episode boundaries.

Two kinds of episodes. The bad kind: the true MDP `M` falls *outside* my plausible set `M_k`. The good kind: `M ∈ M_k`. Deal with the bad kind first, because it should be rare. How rare? Fix a `(s,a)` and a count `n`. The transition confidence fails when `‖p̂ − p‖_1` exceeds my radius; with the `L1` tail `(2^S − 2) exp(−n ε²/2)` and `m = S` outcomes, choosing

```
ε = sqrt( 14 S log(2 A t / δ) / n )
```

drives the fixed-`n`, fixed-`(s,a)` transition failure probability to `≤ δ/(20 t^7 SA)`. The reward confidence: Hoeffding `2 exp(−2 n ε²)`, and choosing

```
ε_r = sqrt( 7 log(2 S A t / δ) / (2 n) )
```

drives the fixed-`n`, fixed-`(s,a)` reward failure to `≤ δ/(60 t^7 SA)`. Union over `n = 1..t−1` turns the denominators into `t^6`; union over all `SA` pairs gives `P(M ∉ M(t)) < δ/(15 t^6)`. So those are the radii, fixed by the requirement that the truth almost never escapes. Now the regret from bad episodes: by the doubling stopping rule, within any episode `Σ_{s,a} v_k(s,a) ≤ t_k − 1 ≤ t`, so the regret of a bad episode is at most its length, and

```
Σ_k Δ_k 1{M ∉ M_k} ≤ Σ_{t} t · 1{M ∉ M(t)} ≤ √T + Σ_{t > T^{1/4}} t · 1{M ∉ M(t)} ,
```

and `Σ_{t} t · δ/(15 t^6)` converges and is below `δ/(12 T^{5/4})` in probability, so the whole bad-episode contribution is `≤ √T` with high probability. The `t^6` in the failure rate is doing real work here: it's fast enough that even multiplying by the episode length `t` and summing over `t` leaves a convergent, negligible tail. Bad episodes cost `O(√T)`. Set them aside.

Now the good episodes, `M ∈ M_k` — this is where the real bound is. Optimism gives `ρ̃_k ≥ ρ* − 1/√t_k`, so

```
Δ_k = Σ v_k(s,a)(ρ* − r̄(s,a)) ≤ Σ v_k(s,a)(ρ̃_k − r̄(s,a)) + Σ v_k(s,a)/√t_k .
```

I want to turn `ρ̃_k − r̄(s,a)` into something I can charge to confidence widths. Use the value iteration relation. At the iteration where I stop, `u_{i+1}(s) − u_i(s) ≈ ρ̃_k` uniformly (within `1/√t_k`), and expanding `u_{i+1}(s) = r̃_k(s,π̃_k(s)) + Σ_{s'} p̃_k(s'|s,π̃_k(s)) u_i(s')`, I get, for the action `π̃_k` actually plays,

```
| ρ̃_k − ( r̃_k(s,π̃_k(s)) + Σ_{s'} p̃_k(s'|s) u_i(s') − u_i(s) ) | ≤ 1/√t_k .
```

Write this in vectors: `r_k` the optimistic reward vector of `π̃_k`, `P̃_k` the optimistic transition matrix, `v_k` the row vector of visit counts. Then summing `v_k(s,a)(ρ̃_k − r̄(s,a))` and substituting, the gain term becomes `v_k (P̃_k − I) u_i` plus reward-confidence slack plus the `1/√t_k` stopping slack. Since the rows of `P̃_k` sum to 1, I can replace `u_i` by any constant shift of it — replace it by the recentered `w_k`, which has `‖w_k‖_∞ ≤ D/2`. And `r̃_k(s,a) − r̄(s,a)` is bounded by twice the reward radius (3), because both `r̃_k` and `r̄` are within one radius of `r̂_k`. So:

```
Δ_k ≤ v_k (P̃_k − I) w_k
        + 2 Σ_{s,a} v_k(s,a) sqrt( 7 log(2SAt_k/δ) / (2 max{1,N_k(s,a)}) )
        + 2 Σ_{s,a} v_k(s,a)/√t_k .
```

The last two terms are tame — reward width and planning slack, both `O(√(SAT))`-ish when summed. The first term, `v_k(P̃_k − I)w_k`, is the whole game, and the diameter is sitting inside it as `‖w_k‖_∞ ≤ D/2`.

`P̃_k` is the *optimistic* transition matrix; I don't get to see it act — what actually moves me is the *true* matrix `P_k` of `π̃_k` in `M`. So split:

```
v_k (P̃_k − I) w_k = v_k (P̃_k − P_k) w_k + v_k (P_k − I) w_k .
```

The first piece is the optimism-vs-reality gap in transitions. Bound it by Hölder: row by row,

```
v_k (P̃_k − P_k) w_k ≤ Σ_s v_k(s,π̃_k(s)) · ‖ p̃_k(·|s) − p_k(·|s) ‖_1 · ‖w_k‖_∞ .
```

Both `p̃_k` and the true `p_k` are in the plausible set, so their `L1` difference is at most twice the radius (4), `≤ 2 sqrt(14 S log(2AT/δ)/N_k)`, and `‖w_k‖_∞ ≤ D/2`. The factor of 2 and the `1/2` cancel:

```
v_k (P̃_k − P_k) w_k ≤ D sqrt(14 S log(2AT/δ)) · Σ_{s,a} v_k(s,a)/sqrt(max{1,N_k(s,a)}) .
```

This is the dominating term. Look at what it's made of: `D` (the diameter, from the span of `w_k`), `√S` (from the `L1` radius of an `S`-outcome distribution — exactly the extra `√S` I flagged at the start), a `√log`, and the sum `Σ v_k/√N_k`. The second piece, `v_k(P_k − I)w_k`, is the one I expect to be *small*: `v_k` is roughly the stationary distribution `μ_k` of `P_k` (`μ_k P_k = μ_k`), so `v_k(P_k − I) ≈ 0`. Make it rigorous with a martingale: along the trajectory set `X_t = (p(·|s_t,a_t) − e_{s_{t+1}}) w_k`; these are martingale differences (`E[e_{s_{t+1}} | past] = p(·|s_t,a_t)`), bounded by `|X_t| ≤ (‖p‖_1 + ‖e‖_1)·‖w_k‖_∞ ≤ D`, and telescoping the unit vectors `e_{s_t}` across an episode gives `v_k(P_k − I)w_k ≤ Σ_t X_t + D` (the `+D` from the boundary terms `w_k(s_{t_k}) − w_k(s_{t_{k+1}})`, each `≤ D/2`). Azuma over the whole horizon, plus summing the `+D` over the `m` episodes:

```
Σ_k v_k(P_k − I)w_k ≤ D sqrt( (5/2) T log(8T/δ) ) + D·m .
```

And `m ≤ SA log_2(8T/(SA))`, so `D·m` is `Õ(DSA)` — lower order. The diameter rides in twice now: through the dominant `D·√S·…` transition-error term, and through this `D·√T` martingale term — both because `‖w_k‖_∞ ≤ D/2`.

Everything now reduces to that recurring sum `Σ_k Σ_{s,a} v_k(s,a)/sqrt(max{1,N_k(s,a)})`. Here the doubling rule pays off for the second time. Fix `(s,a)`. The visit count before episode `k` is `N_k`, and within an episode it at most doubles, so `v_k ≤ N_k` and the increments behave like a sequence `z_k = v_k` with running total `Z_{k-1} = N_k ≤ Z_k`. I claim `Σ_k z_k / sqrt(Z_{k-1}) ≤ (√2 + 1) sqrt(Z_n)`. Prove it by induction: in the inductive step, `√(Z_{k-1}) + z_k/√(Z_{k-1})`, and since `z_k ≤ Z_{k-1}` (the doubling guarantee `v_k ≤ N_k`), bounding `(√2+1)√(Z_{k-1}) + z_k/√(Z_{k-1}) ≤ (√2+1)√(Z_{k-1}+z_k) = (√2+1)√(Z_k)` goes through after squaring — the `z_k ≤ Z_{k-1}` is exactly what makes the cross term close. So per `(s,a)`, `Σ_k v_k/√N_k ≤ (√2+1)√(N(s,a))`, and summing over the `SA` pairs with Jensen (`Σ_{s,a} √N(s,a) ≤ √(SA · Σ N) = √(SAT)`):

```
Σ_k Σ_{s,a} v_k(s,a)/sqrt(max{1,N_k(s,a)}) ≤ (√2 + 1) √(SAT) .
```

There's the `√(SAT)`. That's the whole reason the bound is `√(AT)` and not `T`: each confidence width shrinks like `1/√N`, and the visit counts, summed against those shrinking widths, total `√(SAT)` instead of `T`. Without the doubling rule the within-episode counts could drift far from `N_k` and this telescoping would break.

Assemble. The dominant term is `D · √(14 S log(2AT/δ)) · (√2+1)√(SAT)` — that's `D · S · √(AT) · √log`, since `√S · √(SAT) = S √(AT)`. The reward-width term contributes `√(SAT)·√log` (no `D`, no extra `√S`), the planning slack `√(SAT)`, the martingale term `D√T√log`, the bad episodes `√T`, the Hoeffding reward strip `√(T log)`, the `D·m` and per-episode logs all lower order. Collecting constants and using `T ≥ 34A log(T/δ)` to fold the lower-order pieces in, with probability `≥ 1 − δ/(4T^{5/4})`,

```
Δ(T) ≤ 34 · D S sqrt( A T log( T/δ ) ) ,
```

and a union over the possible values of `T` (the `Σ_T δ/(4T^{5/4}) < δ`) makes it hold for all `T > 1` with probability `≥ 1 − δ`. So the regret is `Õ(D S √(A T))`. Reading the exponents back: the `√(AT)` is concentration (`1/√N` widths summed to `√(SAT)`); one factor of `S` rather than `√S` is `√S` from the `L1` distribution radius times the `√S` already inside `√(SAT)`; the `D` is the diameter, twice over, because a planning error is worth up to the span of the value function and the span is bounded by `D`. Every part of the shape is accounted for.

Is `D S √(AT)` even close to right, or did optimism cost me? Build a lower bound to see what's forced. Take a two-state gadget: a reset state `s∘` with reward `0` and a rewarding state `sp` with reward `1`. From `s∘`, each of about `A` actions moves to `sp` with probability `δ` and otherwise stays at `s∘`, except one special action in one special copy, which moves to `sp` with probability `δ + ε`. From `sp`, every action stays in `sp` with probability `1 − δ` and returns to `s∘` with probability `δ`. Choosing `δ = Θ(1/D)` makes each successful trip to `sp` last about `D` steps in expectation, so the diameter is `≈ D`, I make about `T/D` reset-state decisions, and each better decision is amplified by a length-`D` reward run. Tile `S/2` copies of this gadget and wire them into an `A`-ary tree on the `s∘`-states using a few extra zero-reward actions, keeping the diameter `≤ D`. If I identify all the `s∘`-states I only make the learner stronger; the hard core is then a bandit with `k A ≈ SA` arms. Finding the one special `(copy, action)` is exactly the bandit hard instance of Auer et al. — by the KL/Pinsker argument between the all-`δ` world and the one-special-arm world, distinguishing the special arm forces `Ω(√(kA · #decisions))` regret on the arm scale, which at reward-scale `D` and `T/D` decisions is

```
Ω( D · √( SA · T/D ) ) = Ω( √( D S A T ) ) .
```

Tuning `ε = Θ(√(D kA / T))` makes it tight. So no algorithm beats `Ω(√(DSAT))`, and my upper bound `Õ(DS√(AT))` matches it up to a `√(DS)` factor and the log — optimism didn't cost me the exponent in `T`, and the diameter is genuinely necessary, not an artifact of my proof.

Two more things fall out almost for free. A logarithmic-in-`T` gap-dependent bound: call an episode `ε`-bad if its average regret exceeds `ε`; rerunning the regret accounting above restricted to bad episodes (with Freedman's/Bernstein's inequality replacing Azuma to get the variance-aware constant) bounds the number of steps in `ε`-bad episodes by `342 D²S²A log(T/δ) / ε²`, since the regret in those steps is both `≥ ε L_ε` and `≤ Õ(DS√(L_ε A))`; setting `ε` to the average-reward gap `g` between best and second-best policy turns this into an `O(D²S²A log(T)/g)` expected-regret bound — the MDP analogue of UCB1's `Σ (1/Δ_j) log n`. And a sample-complexity corollary: solving `34 DS√(A log(T/δ)/T) < ε` for `T` gives per-step regret below `ε` after `Õ(D²S²A/ε²)` steps. The same machinery, read three ways.

And one robustness note: if the MDP is allowed to change `ℓ` times (diameter always `≤ D`), I don't need a new idea — just restart the whole thing with confidence `δ/ℓ²` at steps `t_i = ⌈i³/ℓ²⌉`. Up to time `T` this gives `n ≈ ℓ^{2/3}T^{1/3}` restarts; the regret burned in the `ℓ` periods where a change lands is `≤ 7 ℓ^{1/3} T^{2/3}`, and each clean segment contributes `Õ(√(T_i))`, which sums by Jensen to `Õ(√n√T · D S √A) = Õ(ℓ^{1/3} T^{2/3} D S √A)`. The growing restart gaps are the same amortization trick one layer up.

Let me write the algorithm exactly as it has to run.

```python
import numpy as np

# UCRL2: optimism over a confidence set of MDPs; replan per episode (doubling).

def ucrl2(mdp, T, delta):
    S, A = mdp.S, mdp.A
    # Accumulated statistics across all episodes.
    Nsa  = np.zeros((S, A))            # total visits to (s,a) before this episode
    Rsa  = np.zeros((S, A))            # accumulated reward in (s,a)
    Psas = np.zeros((S, A, S))         # transition counts (s,a)->s'
    s = mdp.reset()
    t = 1
    while t <= T:
        tk = t
        vk = np.zeros((S, A))          # in-episode counts (reset each episode)

        # --- Step 2-3: empirical model from data before episode k ----------
        rhat = Rsa  / np.maximum(1, Nsa)
        phat = Psas / np.maximum(1, Nsa)[:, :, None]

        # --- Step 4: confidence radii (Hoeffding rewards, Weissman L1 trans) -
        log_r = np.log(2 * S * A * tk / delta)
        log_p = np.log(2 * A * tk / delta)
        dr = np.sqrt(7 * log_r / (2 * np.maximum(1, Nsa)))           # eq (3)
        dp = np.sqrt(14 * S * log_p / np.maximum(1, Nsa))            # eq (4)

        # --- Step 5: extended value iteration -> optimistic policy ----------
        policy = extended_value_iteration(rhat, dr, phat, dp, S, A, 1/np.sqrt(tk))

        # --- Step 6: follow policy until some (s,a) count doubles -----------
        # episode ends when v_k(s,a) reaches N_k(s,a) for the current (s,a)
        while t <= T and vk[s, policy[s]] < max(1, Nsa[s, policy[s]]):
            a = policy[s]
            r, s2 = mdp.step(s, a)
            vk[s, a]   += 1
            Rsa[s, a]  += r
            Psas[s, a, s2] += 1
            s = s2
            t += 1
        Nsa += vk                      # roll in-episode counts into the totals


def extended_value_iteration(rhat, dr, phat, dp, S, A, eps):
    # Optimistic rewards: push each mean to the top of its interval (cap at 1).
    r_opt = np.minimum(1.0, rhat + dr)
    u = np.zeros(S)
    while True:
        order = np.argsort(-u)                  # states by value, descending
        q = np.empty((S, A))
        for s in range(S):
            for a in range(A):
                # inner max_p <p, u> over the L1 ball of radius dp[s,a].
                p = max_l1_transition(phat[s, a], dp[s, a], order)
                q[s, a] = r_opt[s, a] + p @ u
        u_next = q.max(axis=1)
        # stop when the value-iteration increments have (nearly) equalized:
        # span(u_next - u) < eps  <=>  greedy policy is eps-optimal
        d = u_next - u
        if d.max() - d.min() < eps:
            return q.argmax(axis=1)
        u = u_next


def max_l1_transition(p_hat, radius, order):
    # Put up to radius/2 mass on the top-value state, then remove the excess
    # from the lowest-value states until the row is a probability vector again.
    p = p_hat.copy()
    top = order[0]
    p[top] = min(1.0, p[top] + radius / 2.0)
    total = p.sum()
    j = len(order) - 1
    while total > 1.0 and j > 0:
        low = order[j]
        cut = min(p[low], total - 1.0)
        p[low] -= cut
        total -= cut
        j -= 1
    return p
```

So the causal chain, start to finish: I want `√T` regret during learning with no mixing-time oracle; UCB1 already does this for one state by acting optimistically on a confidence interval, which dissolves explore-vs-exploit; lifting that means keeping a confidence *set of MDPs* and following the optimal policy of the most optimistic one; an MDP forces two new costs — learning `S`-outcome transition distributions concentrates a factor `√S` worse, and a misstep can strand me for up to the diameter `D` steps; planning optimistically over transitions as well as rewards needs extended value iteration, whose iterate has span `≤ D` (that's where `D` enters); I replan only on count-doubling, which keeps episodes logarithmic and makes the confidence widths telescope into `√(SAT)`; decomposing the regret, the dominant term is `D·√S·√(SAT) = DS√(AT)` from the transition confidence times the span, and a matching `Ω(√(DSAT))` lower bound — a bandit hidden inside the MDP — shows the shape is right.
