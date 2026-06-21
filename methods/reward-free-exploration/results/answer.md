# Reward-Free Exploration

## Method

Input: an unknown episodic tabular MDP with `S` states, `A` actions, horizon `H`, accuracy `epsilon`, failure probability `p`.

Exploration phase, with no downstream reward:

1. Set `delta = epsilon/(2 S H^2)`.
2. For every target state-time pair `(s,h)`, create the synthetic reachability reward
   `r_{h'}(s',a') = 1[s'=s and h'=h]`.
3. Run EULER for `N_0` episodes on this synthetic reward, obtaining a policy multiset `Phi^(s,h)`.
4. For every `pi in Phi^(s,h)`, replace `pi_h(.|s)` by `Uniform(A)`.
5. Let `Psi` be the union of all target policy multisets.
6. Collect `N` trajectories by sampling `pi ~ Uniform(Psi)` and rolling out `pi`; call the resulting dataset `D`.

Planning phase, for any revealed reward `r`:

1. Form the empirical transition model `Phat_h(s'|s,a)` from `D`.
2. Run any planner that returns an `epsilon`-suboptimal policy for the known MDP `(Phat,r)`, such as exact value iteration or the finite-horizon NPG update
   `pi_{h}^{t+1}(a|s) proportional to pi_h^t(a|s) exp(eta Q_h^{pi^t}(s,a))`.
3. Return the planner's policy.

## Coverage Invariant

For every `delta`-significant state-time pair,

`max_pi P_h^pi(s) >= delta`,

the exploration distribution `mu` induced by sampling uniformly from `Psi` satisfies

`max_{pi,a} P_h^pi(s,a) / mu_h(s,a) <= 2 S A H`.

This invariant is the core object. It says the data distribution visits every meaningfully reachable state-action pair in proportion to the best possible reaching probability, while ignoring only states whose maximum total value contribution is below the chosen error budget.

## Sample Complexity

Choose

`N_0 = O(S^2 A H^4 iota_0^3 / delta)`,

where `iota_0 = log(S A H/(p delta))`, and

`N = O(H^5 S^2 A iota / epsilon^2)`,

where `iota = log(S A H/(p epsilon))`.

Then the total number of exploration episodes is

`K <= c [ H^5 S^2 A iota / epsilon^2 + S^4 A H^7 iota^3 / epsilon ]`,

which is `Otilde(H^5 S^2 A / epsilon^2)` in the leading term.

## Guarantee

With probability at least `1-p`, the same dataset `D` supports arbitrarily many adaptively chosen downstream reward functions. For every such reward, if the empirical planner has optimization error at most `epsilon`, the returned policy is `3 epsilon`-suboptimal in the true MDP:

`E_{s_1}[V_1^*(s_1;r) - V_1^{pihat}(s_1;r)] <= 3 epsilon`.

The proof uses the value-difference lemma, splits insignificant states from covered states, and bounds the covered part by a self-bounded Bernstein inequality:

`E_mu |(Phat_h - P_h)G(s,a)|^2 1[a=nu(s)] <= O(H^2 S log(A H N / p)/N)`.

This uniform evaluation event holds over all policies and rewards, which is why one exploration phase can support many later reward functions.

## Lower Bound

Any algorithm with this reward-agnostic first phase needs

`Omega(S^2 A H^2 / epsilon^2)`

episodes in expectation when `A >= 2`, `S >= C log_2 A`, `H >= C log_2 S`, `epsilon <= min{1/4, H/48}`, and the success probability is fixed at `1/2`. The lower bound first forces transition-distribution identification at a single state using packed near-uniform leaf distributions and reward vectors as separating hyperplanes, then embeds `Omega(S)` such copies in a binary tree. The extra factor of `S` over fixed-reward tabular RL is therefore the price of coverage for all possible downstream rewards, not an artifact of the upper-bound analysis.
