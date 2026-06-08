# Average-reward MDPs: the gain/bias characterization and Blackwell optimality

## Problem

Optimize a stationary policy for a Markov decision process that runs forever (a continuing,
recurrent task), where the discounted criterion `Σ γ^t r_t` is awkward: the discount `γ<1` is an
artifact that imposes an arbitrary horizon, can rank long-run-worse policies above better ones,
and forces `V_γ → ∞` as `γ → 1`. The honest objective is the **long-run average reward (gain)**

```
g = lim_{N→∞} (1/N) · E[ Σ_{t=0}^{N-1} r_t ].
```

## Key idea

For a unichain policy the gain `g` is **state-independent** — a single scalar — so it cannot rank
states. The optimum is therefore characterized not by one value function but by a **pair**:

- the **gain** `g` (constant on each recurrent class), the per-step "level", and
- the **bias / relative value** `h`, the finite offset after subtracting the common `N g`
  growth from an `N`-step value. In aperiodic unichain cases this is the ordinary centered-return
  limit; in general it is pinned by the Poisson equation or the Abel/Laurent limit. `h` plays the
  role of the value function and is determined only up to an additive constant (pin
  `h(reference)=0`).

The two are exactly the leading coefficients of the **Laurent expansion** of the discounted value
about its pole at `γ=1`:

```
V_γ(s) = g/(1−γ) + h(s) + e_γ(s),     e_γ(s) → 0 as γ → 1,
```

where the `1/(1−γ)` pole coefficient is the gain and the `O(1)` constant coefficient is the bias.

## The optimality equations

**Policy evaluation (average-reward Poisson equation).** For a fixed policy `π`,

```
g + h(s) = r(s, π(s)) + Σ_{s'} p(s'|s,π(s)) h(s'),       g·1 + (I − P_π) h = r_π,
```

solvable because `g = d · r_π` (with `d P_π = d`) puts `r_π − g·1` in the range of the singular
`I − P_π`; pin the `span{1}` null direction with `h(reference)=0`.

**Average-reward Bellman optimality equation.** A scalar `g*` and vector `h*` satisfy

```
g* + h*(s) = max_a [ r(s,a) + Σ_{s'} p(s'|s,a) h*(s') ],
```

and the greedy policy with respect to `h*` attains the optimal gain `g* = max_π g_π`. For a
unichain or communicating MDP such a `(g*, h*)` and a stationary deterministic optimal policy
exist (Howard 1960; Blackwell 1962; cf. Puterman ch. 8).

## Optimality ladder: gain → bias → Blackwell

With `V_γ = g/(1−γ) + h + O(1−γ)`, call `π*` **`n`-discount-optimal** if
`lim_{γ→1}(1−γ)^{-n}(V_γ^{π*}(s) − V_γ^π(s)) ≥ 0` for all `s, π` (Veinott 1969). Then:

- `n = −1` ⟺ **gain-optimal**: multiplying by `(1−γ)` isolates `g^{π*} − g^π ≥ 0`.
- `n = 0` ⟺ **bias-optimal**: among gain-equal policies the pole cancels and the limit is
  `h^{π*} − h^π ≥ 0`. (Tie-break gain by bias — e.g. among goal-reaching policies, prefer the
  fastest, which is the one with larger `h`.)
- `n = ∞` ⟺ **Blackwell-optimal**: a single stationary policy is discount-optimal for every
  `γ ∈ (γ*, 1)`; it maximizes the Laurent coefficients lexicographically (gain, then bias, then
  the rest). Such a policy exists for finite MDPs (Blackwell 1962). The sets nest:
  all ⊃ gain-optimal ⊃ bias-optimal ⊃ … ⊃ Blackwell-optimal.

## Algorithms

**Policy iteration (Howard 1960).** Alternate the Poisson solve for `(g, h)` with greedy
improvement `π'(s) = argmax_a [ r(s,a) + Σ p(s'|s,a) h(s') ]`; terminates in finitely many steps
at a gain-optimal policy under the standard finite/unichain assumptions. Bias and Blackwell
tie-breaking use the Laurent ordering among gain-equal policies.

**Relative value iteration (White 1963).** The raw backup `V ← T(V)` drifts up by `g*` per step
(`T(h*) = h* + g*·1`), so compute `U_k = T(V_k)` and subtract a reference:
`V_{k+1}(s) = U_k(s) − U_k(ref)`. The drift vector `U_k − V_k` approaches `g*·1`, and the
normalized iterates converge to `h`. Stop on the **span seminorm**
`sp(U_k−V_k) = max−min < ε`, which ignores the common drift. Requires aperiodicity (use the
transform `P̃ = (1−τ)I + τP`, `R̃ = τR`, which scales `g̃ = τg`, leaves `h` and the optimal policy
unchanged); the original gain is the measured drift divided by `τ`. The naïve asynchronous version
can diverge (Tsitsiklis), since the relative-value map is neither monotone nor a max-norm
contraction.

## Implementation

```python
import numpy as np

class MDP:
    """Finite MDP. P[a]: (n,n) row-stochastic; R[a]: length-n reward."""
    def __init__(self, P, R):
        self.P, self.R = P, R
        self.n, self.m = P[0].shape[0], len(P)

def evaluate_policy(mdp, pi, ref=0):
    """Gain g and bias h from the average-reward Poisson equation, h(ref)=0."""
    n = mdp.n
    Ppi = np.stack([mdp.P[pi[s]][s] for s in range(n)])
    rpi = np.array([mdp.R[pi[s]][s] for s in range(n)])
    A = np.zeros((n + 1, n + 1)); b = np.zeros(n + 1)
    A[:n, :n] = np.eye(n) - Ppi    # (I - Ppi) h
    A[:n, n]  = 1.0                # + g
    b[:n]     = rpi
    A[n, ref] = 1.0                # h(ref) = 0
    x = np.linalg.solve(A, b)
    return x[n], x[:n]             # g, h

def policy_iteration(mdp):
    pi = np.zeros(mdp.n, dtype=int)
    while True:
        g, h = evaluate_policy(mdp, pi)
        Q = np.stack([mdp.R[a] + mdp.P[a] @ h for a in range(mdp.m)], axis=0)
        pi_new = Q.argmax(axis=0)
        if np.array_equal(pi_new, pi):
            return g, h, pi        # g + h = max_a [ r + P h ]
        pi = pi_new

def relative_value_iteration(mdp, ref=0, eps=1e-9, tau=0.5):
    P = [(1 - tau) * np.eye(mdp.n) + tau * mdp.P[a] for a in range(mdp.m)]
    R = [tau * mdp.R[a] for a in range(mdp.m)]
    V = np.zeros(mdp.n)
    while True:
        Q = np.stack([R[a] + P[a] @ V for a in range(mdp.m)], axis=0)
        raw = Q.max(axis=0)
        drift = raw - V                                    # tends to tau*g times 1
        V_next = raw - raw[ref]                            # kill the gain drift
        if drift.max() - drift.min() < eps:                # span seminorm stop
            g = 0.5 * (drift.max() + drift.min()) / tau    # undo g~ = tau*g
            return g, V_next, Q.argmax(axis=0)             # gain, bias, greedy policy
        V = V_next
```
