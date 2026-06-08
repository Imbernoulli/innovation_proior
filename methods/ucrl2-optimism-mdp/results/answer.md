# UCRL2 — optimism in the face of uncertainty for whole MDPs

## Problem

Online undiscounted (average-reward) reinforcement learning in an unknown finite communicating MDP `M = (S, A, p, r̄)`. The learner wanders continuously (no resets), and we bound the **total regret during learning** against the optimal average reward `ρ*`:

```
Δ(M, A, s, T) = T·ρ*(M) − Σ_{t=1..T} r_t .
```

Only `S = |S|` and `A = |A|` are known. The MDP's difficulty is measured by its **diameter**

```
D(M) = max_{s ≠ s'} min_π E[ T(s' | M, π, s) ] ,
```

the worst-case expected time to travel between two states under the best policy aimed at the target — a quantity that depends only on the transition structure.

## Key idea

Lift "optimism in the face of uncertainty" from bandits (UCB1) to whole MDPs. Maintain a **confidence set `M_k` of plausible MDPs** around the empirical reward/transition estimates, and at the start of each episode act optimally in the **most optimistic** MDP in the set. Optimism inflates the value of under-visited state-action pairs, so a good policy in the optimistic MDP automatically seeks them out — exploration becomes a side effect, with no explicit explore/exploit branch and no mixing-time input. Two MDP-specific costs shape the bound: learning an `S`-outcome transition distribution concentrates a factor `√S` worse than a scalar mean, and a planning error can strand the learner for up to `D` steps. Planning over optimistic *transitions* (not just rewards) requires **extended value iteration**; replanning only when a visit count **doubles** keeps episodes logarithmic and makes the confidence widths telescope.

## Algorithm (UCRL2)

Proceed in episodes `k = 1, 2, …`. At the start of episode `k` (time `t_k`), with counts `N_k(s,a)`, empirical `r̂_k`, `p̂_k`:

1. **Confidence set `M_k`** — all MDPs with
   ```
   |r̃(s,a) − r̂_k(s,a)| ≤ sqrt( 7 log(2 S A t_k / δ) / (2 max{1, N_k(s,a)}) )            (Hoeffding)
   ‖p̃(·|s,a) − p̂_k(·|s,a)‖_1 ≤ sqrt( 14 S log(2 A t_k / δ) / max{1, N_k(s,a)} )         (Weissman L1)
   ```
   These guarantee `P(M ∉ M_k) < δ/(15 t_k^6)`.

2. **Extended value iteration** — collapse `M_k` into one MDP `M̃⁺` with extended actions `(a, p̃)`, `p̃` ranging over the `L1` ball, and run
   ```
   u_0(s) = 0,
   u_{i+1}(s) = max_a { r̂_k(s,a) + d_r(s,a) + max_{p ∈ ball} Σ_{s'} p(s') u_i(s') } ,
   ```
   the inner max solved by sorting states by `u_i`, adding up to `d_p/2` mass to the top-value state, and draining the excess from the lowest-value states (`O(S²A)` per iteration). Since `d_p > 0`, the selected transition matrix gives every state positive probability of moving to the top-value state, including a self-loop at that state, so the selected policy is aperiodic; when the plausible set contains the communicating true MDP, VI converges. Stop when `span(u_{i+1} − u_i) < 1/√t_k`, yielding a `1/√t_k`-optimal greedy policy `π̃_k` of the optimistic MDP `M̃_k`, with `ρ̃_k ≥ ρ* − 1/√t_k`.

3. **Execute** `π̃_k` until, for the current `(s,a)`, the in-episode count `v_k(s,a)` reaches `max{1, N_k(s,a)}` — i.e. the count doubles. Then start episode `k+1`.

```python
import numpy as np

def ucrl2(mdp, T, delta):
    S, A = mdp.S, mdp.A
    Nsa  = np.zeros((S, A)); Rsa = np.zeros((S, A)); Psas = np.zeros((S, A, S))
    s = mdp.reset(); t = 1
    while t <= T:
        tk = t; vk = np.zeros((S, A))
        rhat = Rsa  / np.maximum(1, Nsa)
        phat = Psas / np.maximum(1, Nsa)[:, :, None]
        dr = np.sqrt(7 * np.log(2*S*A*tk/delta) / (2*np.maximum(1, Nsa)))     # reward radius
        dp = np.sqrt(14 * S * np.log(2*A*tk/delta) / np.maximum(1, Nsa))      # L1 transition radius
        policy = extended_value_iteration(rhat, dr, phat, dp, S, A, 1/np.sqrt(tk))
        while t <= T and vk[s, policy[s]] < max(1, Nsa[s, policy[s]]):
            a = policy[s]; r, s2 = mdp.step(s, a)
            vk[s, a] += 1; Rsa[s, a] += r; Psas[s, a, s2] += 1
            s = s2; t += 1
        Nsa += vk

def extended_value_iteration(rhat, dr, phat, dp, S, A, eps):
    r_opt = np.minimum(1.0, rhat + dr)
    u = np.zeros(S)
    while True:
        order = np.argsort(-u)
        q = np.empty((S, A))
        for s in range(S):
            for a in range(A):
                p = max_l1_transition(phat[s, a], dp[s, a], order)
                q[s, a] = r_opt[s, a] + p @ u
        u_next = q.max(axis=1); d = u_next - u
        if d.max() - d.min() < eps:
            return q.argmax(axis=1)
        u = u_next

def max_l1_transition(p_hat, radius, order):
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

## Regret theorem and proof sketch

**Theorem (upper bound).** With probability `≥ 1 − δ`, for any initial state and any `T > 1`,
```
Δ(M, UCRL2, s, T) ≤ 34 · D S sqrt( A T log(T/δ) )  =  Õ( D S √(A T) ) .
```

**Proof sketch.**
- *Reward noise.* Hoeffding replaces `Σ r_t` by `Σ N(s,a) r̄(s,a)` at cost `sqrt((5/8) T log(8T/δ))`; write `Δ_k = Σ v_k(s,a)(ρ* − r̄(s,a))`.
- *Failing confidence.* By doubling, `Σ_{s,a} v_k ≤ t_k`, and `P(M ∉ M(t)) < δ/(15 t^6)`, so episodes with `M ∉ M_k` cost `≤ √T` w.h.p.
- *Span ≤ diameter.* On good episodes, `M̃⁺` contains the true MDP's actions, so its diameter is `≤ D`; a detour `s' → s''` in `≤ D` expected steps gives `span(u_i) ≤ D`, so the centred `‖w_k‖_∞ ≤ D/2`.
- *Good episodes.* Optimism + the VI relation give `Δ_k ≤ v_k(P̃_k − I)w_k + 2 Σ v_k·(reward radius) + 2 Σ v_k/√t_k`. Split `P̃_k − I = (P̃_k − P_k) + (P_k − I)`:
  - `v_k(P̃_k − P_k)w_k ≤ D √(14 S log(2AT/δ)) · Σ v_k/√N_k` (Hölder, `L1` radius × span) — **dominant**;
  - `v_k(P_k − I)w_k`: martingale `X_t = (p(·|s_t,a_t) − e_{s_{t+1}})w_k`, `|X_t| ≤ D`, Azuma ⇒ `Σ_k v_k(P_k−I)w_k ≤ D√((5/2)T log(8T/δ)) + D·m`.
- *Counting.* Number of episodes `m ≤ SA log_2(8T/(SA))` (each episode doubles some `N(s,a)`). Doubling gives `v_k ≤ N_k`, so `Σ_k v_k/√N_k ≤ (√2+1)√N(s,a)` (induction), and Jensen `Σ_{s,a}√N(s,a) ≤ √(SAT)`. Hence `Σ_k Σ_{s,a} v_k/√N_k ≤ (√2+1)√(SAT)`.
- *Assemble.* Dominant term `D·√(S log)·√(SAT) = DS√(AT log)`; all others (`√(SAT)`, `D√T`, `√T`, `D·m`) are lower order. Collecting constants and union over `T` gives the theorem.

**Theorem (lower bound).** For any algorithm and any `S, A ≥ 10`, `D ≥ 20 log_A S`, `T ≥ DSA`, there is an MDP with `S` states, `A` actions, diameter `D` on which expected regret is `≥ 0.015 √(DSAT) = Ω(√(DSAT))`. (A bandit with `≈ SA` arms is embedded in the MDP via `S/2` two-state gadgets whose rewarding state returns with probability `δ = Θ(1/D)`; KL/Pinsker on the special arm.)

So UCRL2 is near-optimal: upper `Õ(DS√(AT))` vs lower `Ω(√(DSAT))`, matching in `A` and `T` up to `√(DS)·√log`.

## Corollaries

- **Sample complexity.** Per-step regret `< ε` after `T = Õ(D²S²A / ε²)` steps.
- **Logarithmic (gap-dependent) regret.** With `g` the average-reward gap between best and second-best policy, expected regret is `O(D²S²A log(T) / g)`.
- **Changing MDPs.** Restarting UCRL2 with confidence `δ/ℓ²` at steps `t_i = ⌈i³/ℓ²⌉` yields `Õ(ℓ^{1/3} T^{2/3} D S √A)` regret against the per-segment optima when the MDP changes `ℓ` times.
