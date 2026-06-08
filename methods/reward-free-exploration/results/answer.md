# Reward-Free Exploration

## The problem

In a tabular episodic MDP (`S` states, `A` actions, horizon `H`, rewards in
`[0,1]`), explore **once with no reward**, collecting a dataset `D`, so that
afterward, for **any** reward function `r` revealed later — and for arbitrarily
many, even adversarially/adaptively chosen — planning on `D` alone (no further
interaction) returns an `ε`-optimal policy. This decouples *where to go* from
*what to optimize*, which matters when rewards are engineered by trial and error.

## Key idea

1. **Coverage is the target, dictated by an adversary.** By the simulation
   (value-difference) lemma, the evaluation error of any policy under any reward
   is `E_π Σ_h (P̂_h − P_h)V̂^π_{h+1}`. To bound this for every reward and policy
   at once, the data distribution `μ` must satisfy, for every reachable `(s,a)`,
   `P^π_h(s,a)/μ_h(s,a) ≤ poly` — every significantly-reachable state-action is
   visited proportional to its maximum reaching probability.
2. **Significant vs insignificant states.** A state is `δ`-significant if
   `max_π P^π_h(s) ≥ δ`. Insignificant states contribute at most `HSδ` per step,
   hence `H²Sδ` over the simulation-lemma sum. Set `δ = ε/(2SH²)`.
3. **Reward-as-bonus.** To learn to reach `(s,h)`, feed the explorer the
   indicator reward `r'_{h'}(s',a') = 1[s'=s, h'=h]`. Then `V*_1 = max_π P^π_h(s)`
   and the optimal policy maximizes reaching probability. Use a value-dependent
   explorer (EULER) whose cost scales with `V*_1` — so faint targets are cheap.
4. **Mixture = coverage.** Mix the per-target policy sets (uniform action at the
   target cell) into `Ψ`; `μ` = run a uniform draw from `Ψ`.

## The protocol

**Exploration phase (no reward).** For each `(s,h)`: run EULER on the indicator
reward for `N_0` episodes to get a policy set `Φ^{(s,h)}` whose average reaching
probability is `≥ ½ max_π P^π_h(s)`; set the action at `(s,h)` to `Uniform(A)`.
Let `Ψ = ⋃_{(s,h)} Φ^{(s,h)}`. Sample `N` trajectories, each under a uniformly
drawn `π ∈ Ψ`, to form `D`.

**Planning phase (reward `r` given).** Count to form the empirical transition
`P̂_h(s'|s,a) = N_h(s,a,s')/N_h(s,a)`; call any approximate MDP solver on
`(P̂, r)` — value iteration (exact) or NPG.

## Guarantees

**Coverage (exploration).** With `N_0 = O(S²AH⁴ι₀³/δ)`, w.p. `1−p`, `D` is i.i.d.
from a `μ` with, for every `δ`-significant `(s,h)`,
`max_{π,a} P^π_h(s,a)/μ_h(s,a) ≤ 2SAH`.

**Uniform evaluation (planning).** With `δ = ε/(2SH²)` and `N ≥ c H⁵S²Aι/ε²`,
w.p. `1−p`, for **every** reward `r` and policy `π` simultaneously,
`|E_{s_1}[V̂^π_1(s_1;r) − V^π_1(s_1;r)]| ≤ ε`.
Proof: split the simulation-lemma error into insignificant states (`≤ H²Sδ ≤ ε/2`)
and significant states; Cauchy–Schwarz, reduce to a deterministic action map `ν`,
import coverage `P^π_h(s) ≤ 2SAH·μ_h(s,a)`, then a self-bounded Bernstein bound
`E_{μ_h}|(P̂−P)G|²1[a=ν(s)] ≤ O(H²S/N·log(AHN/p))` (ERM `ΣY_i≤0`,
`Var(Y)≤4H²E[Y]`, covering `A^S·(H/ε)^{2S}` values — the `A`-improved net),
summed over `h` to `O(√(H⁵S²A/N·log))`.

**Planning suboptimality.** Via the decomposition
`V^{π*}_1 − V^{π̂}_1 ≤ EvalErr1 + (≤0) + OptErr + EvalErr2 ≤ ε + 0 + ε + ε = 3ε`,
holding for all rewards simultaneously.

**Main theorem.** There is a reward-free exploration algorithm that, w.p. `1−p`,
outputs `ε`-optimal policies for arbitrarily many adaptively chosen rewards, using

  `K ≤ c·[ H⁵S²Aι/ε² + S⁴AH⁷ι³/ε ]`,  `ι = log(SAH/(pε))`,

i.e. `Õ(H⁵S²A/ε²)` episodes.

**Lower bound.** Any algorithm with this guarantee needs `Ω(S²AH²/ε²)` episodes
(for `A≥2`, `S ≥ C log₂ A`, `H ≥ C log₂ S`, `ε ≤ min{1/4, H/48}`), even with
randomized/non-Markov policies. Construction: a single state transitioning to
`2n` near-uniform terminals forces TV-learning of `q(·,a)` for every action
(`Ω(nA/ε²)` via an uncorrelated `±1` packing + Fano + Wald); embed `n = Ω(S)`
copies in a binary tree so the learner is forced to learn `n` of them
and the persistent terminal rewards reduce embedded `ε`-optimality to single-copy
accuracy `4ε/H`, giving `Ω(n²AH²/ε²) = Ω(S²AH²/ε²)`. The extra factor `S` over
single-reward RL's `Θ̃(SAH²/ε²)` is the price of good coverage.

## NPG as the approximate solver

`π^{(t+1)}_h(a|s) ∝ π^{(t)}_h(a|s)·exp(η(Q^{(t)}_h(s,a) − V^{(t)}_h(s)))`,
uniform init. Via the performance-difference lemma and `exp(x) ≤ 1+x+x²`
(`η ≤ 1/H`):
`E[V*_1 − V^{(T)}_1] ≤ (H log A)/(ηT) + ηH²`. With `η = √(log A/(HT))`,
`T = 4H³ log A/ε²`, this is `≤ ε`.

## Implementation

```python
import numpy as np

class EpisodicMDP:
    def __init__(self, P, H, p1):
        self.P, self.H, self.p1 = P, H, p1
        self.S, self.A = P[0].shape[0], P[0].shape[1]
    def rollout(self, policy, rng):
        s = rng.choice(self.S, p=self.p1); traj = []
        for h in range(self.H):
            a = rng.choice(self.A, p=policy[h][s])
            s_next = rng.choice(self.S, p=self.P[h][s, a])
            traj.append((h, s, a, s_next)); s = s_next
        return traj

def value_iteration(P, r, H):                    # exact planner: opt-error 0
    S, A = P[0].shape[0], P[0].shape[1]
    V_next = np.zeros(S); pi = [np.zeros((S, A)) for _ in range(H)]
    for h in reversed(range(H)):
        Q = r[h] + P[h] @ V_next; g = np.argmax(Q, axis=1)
        pi[h][np.arange(S), g] = 1.0; V_next = Q.max(axis=1)
    return pi, V_next

def natural_policy_gradient(P, r, H, eta, T):
    S, A = P[0].shape[0], P[0].shape[1]
    pi = [np.full((S, A), 1.0 / A) for _ in range(H)]
    for _ in range(T):
        V_next = np.zeros(S); Q = [None] * H
        for h in reversed(range(H)):
            Q[h] = r[h] + P[h] @ V_next; V_next = (pi[h] * Q[h]).sum(axis=1)
        for h in range(H):
            A_h = Q[h] - (pi[h] * Q[h]).sum(axis=1, keepdims=True)
            lo = np.log(pi[h] + 1e-300) + eta * A_h
            lo -= lo.max(axis=1, keepdims=True); w = np.exp(lo)
            pi[h] = w / w.sum(axis=1, keepdims=True)
    return pi

def reward_free_explore(env, regret_minimizing_explorer, N0, N, rng):
    S, A, H = env.S, env.A, env.H; Psi = []
    for h in range(H):
        for s in range(S):
            r_ind = [np.zeros((S, A)) for _ in range(H)]; r_ind[h][s, :] = 1.0
            Phi = regret_minimizing_explorer(env, r_ind, N0, rng)
            for pi in Phi:
                pi[h][s, :] = 1.0 / A            # Uniform(A) at target cell
            Psi.extend(Phi)
    return [env.rollout(Psi[rng.integers(len(Psi))], rng) for _ in range(N)]

def reward_free_plan(D, S, A, H, reward, solver="VI", eta=None, T=None):
    Nsas = [np.zeros((S, A, S)) for _ in range(H)]
    Nsa  = [np.zeros((S, A))     for _ in range(H)]
    for traj in D:
        for h, s, a, s_next in traj:
            Nsas[h][s, a, s_next] += 1; Nsa[h][s, a] += 1
    Phat = []
    for h in range(H):
        cnt = Nsa[h][:, :, None]
        Phat.append(np.where(cnt > 0, Nsas[h] / np.maximum(cnt, 1), 1.0 / S))
    if solver == "VI":
        return value_iteration(Phat, reward, H)[0]
    return natural_policy_gradient(Phat, reward, H, eta, T)
```
