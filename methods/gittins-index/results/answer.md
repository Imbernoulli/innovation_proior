# The Gittins index (dynamic allocation index) and the index theorem

## Problem

A family of `n` independent Markov reward processes ("arms" / "bandit processes" /
"projects"). At each discrete step you advance **exactly one**; it pays a random reward
and its own state moves Markovianly, while the other `n−1` are frozen (no reward, no
state change). Reward is discounted by `β ∈ (0,1)`. Maximize total expected discounted
reward over an infinite horizon. The exact dynamic program exists (Blackwell) but its
state is the **product** of the per-arm state spaces — `k^n`, or infinite-dimensional for
Bayesian arms — so direct backward induction is intractable.

## Key idea

The problem **decouples**. Attach to each arm a single scalar — the **dynamic allocation
index** (Gittins index) — computed from that arm *alone*, ignoring all others. The
optimal policy is simply: **at every step, advance an arm of currently maximal index.**
The `k^n` joint problem collapses to `n` independent one-dimensional index computations.

The index of an arm in state `x` has three equivalent readings:

1. **Best discounted reward-rate** achievable over a stopping time you choose:

       ν(x) = sup_{τ ≥ 1}  E[ Σ_{t=0}^{τ-1} β^t R(x(t)) | x(0)=x ]
                          / E[ Σ_{t=0}^{τ-1} β^t | x(0)=x ].

   The denominator is *discounted* time `Σβ^t`, not step count — the only choice for
   which a constant-reward arm has reward-rate equal to its constant. The supremum is
   attained by the stopping rule "continue until the index first falls below `ν(x)`."

2. **Fair charge**: the largest constant per-play charge `γ` at which optimal play of
   the arm still at least breaks even,

       γ(x) = sup{ γ : sup_{τ ≥ 1} E[ Σ_{t=0}^{τ-1} β^t ( R(x(t)) − γ ) | x(0)=x ] ≥ 0 }.

   Equal to reading 1 by `Σβ^t(R−γ) = Σβ^t R − γΣβ^t`, so `γ(x)=ν(x)`. On a
   lump-sum scale, `G(x) = γ(x)/(1−β)` (a perpetuity of `γ` per period is worth
   `γ/(1−β)`; pure rescaling, same ordering, same policy).

3. **Calibration** against a *standard* arm that pays a constant `λ` forever (its own
   index is `λ`). `ν(x)` is the `λ` at which you are indifferent between starting on the
   arm and starting on the constant arm. Because the standard arm has one frozen state,
   the pair `{D, λ}` has the same state space as `D` alone, giving a **one-dimensional**
   Bellman equation:

       R({D,λ}, x) = max[ λ/(1−β),  R(x) + β·E_x R({D,λ}, y) ],

   and `ν(x)` is the largest `λ` for which the play-on branch is still at least as good
   as retiring to the standard arm; in the regular finite-state case this is the tie point
   of the two branches.

## Index theorem (with proof)

**Theorem.** For a family of independent Markov reward processes (one advanced per step,
the rest frozen), any policy that advances a process of maximal current index at every
step is optimal.

**Proof (Weber's prevailing-charge argument).**

*Single arm.* Let the fair charge after `t` plays be `g_t = γ(x(t))`, and let the
prevailing charge be the running minimum `\bar g_t = min_{s≤t} g_s`. This sequence is
(i) **nonincreasing** in the number of plays, (ii) random, and (iii) **policy-independent**
because it depends only on the arm's own trajectory. If the arm is played up to any
stopping time `σ`,

    E[Σ_{t=0}^{σ-1} β^t R(x(t))] ≤ E[Σ_{t=0}^{σ-1} β^t \bar g_t],

with equality when the arm is stopped only at times where fair charge equals prevailing
charge. Thus charges upper-bound reward for any play of the arm, and optimal play makes
the bound tight.

*Many arms.* Reset every arm's charge this way.

- **Upper bound (any policy):** summing the single-arm inequalities over arms gives, for
  *every* policy, expected total discounted reward ≤ expected total discounted charges
  paid.
- **Maximizing charges:** the charges paid are a discounted sum `Σ_t β^t·(charge at t)`,
  an interleaving of `n` nonincreasing streams. Discounting weights early steps most, so
  the largest possible charge total is the nonincreasing rearrangement of all stream
  values. The greatest-index policy produces that rearrangement: it keeps an arm active
  while its fair charge exceeds its prevailing charge, and a switch occurs only when the
  played arm's fair charge has fallen to the prevailing charge, so the next paid charge
  cannot exceed the previous one.
- **Attainment:** the greatest-index policy never leaves an arm idle while its fair charge
  strictly exceeds its prevailing charge. It maximizes the charges (the universal upper
  bound on reward) *and* meets that bound. Hence it is optimal. ∎

The decoupling holds because each arm's charge stream is policy-independent: the charges
of arm `j` are unaffected by what you do with arm `i`, so the family genuinely splits into
single-arm problems.

## Special cases (consistency checks)

- **Deteriorating arm** (`ν(x(1)) ≤ ν(x(0))` a.s.): the best window is one play, so
  `ν(x) = R(x,1)` and **one-step lookahead is optimal**. For single-machine scheduling with
  non-increasing completion hazard `p(t)`, the index is `p_i(t)·V_i` — the classical
  priority/`cμ`-type rule, derived not assumed.
- **Improving arm:** never stop early; closed form by summing the reward-rate to the first
  state change.
- **Bernoulli/Beta bandit:** with posterior-exponent state `(a,b)`, immediate mean
  `(a+1)/(a+b+2)`, and updates `(a+1,b)` or `(a,b+1)`, `ν(a,b)` exceeds the immediate
  mean because learning has option value; compute it by horizon truncation + backward
  calibration over the arm's `(a,b)` state.
- **Normal-mean bandit:** posterior `N(ξ, m^{-1})` over the unknown mean. Adding a constant
  to rewards shifts every weighted average, and multiplying rewards by a constant scales
  every weighted average by the discount, giving the shift-scale law
  `ν(ξ,m,β) = ξ + β·ν(0,m,1)`. This reduces tabulation to a single one-variable function
  `ν(0,m,1)` — iterations over functions of **one** real variable versus `2n` for the joint
  Bellman equation.

## Implementation

```python
def gittins_index_by_calibration(states, transition, reward, beta,
                                 lo, hi, tol=1e-9):
    """Index of every state of ONE Markov reward process, by calibrating against a
    constant-charge yardstick. The pair's Bellman max is solved over the process's
    OWN states (1-D, no product). The index of state x is the largest lambda for
    which it is still optimal to keep playing rather than retire to lambda/(1-beta)."""
    def plays_on(x0, lam):
        V = {s: reward(s) / (1 - beta) for s in states}
        retire = lam / (1 - beta)
        while True:
            newV, delta = {}, 0.0
            for s in states:
                cont = reward(s) + beta * sum(p * V[s2]
                                              for s2, p in transition(s).items())
                newV[s] = max(retire, cont)
                delta = max(delta, abs(newV[s] - V[s]))
            V = newV
            if delta < tol:
                break
        cont0 = reward(x0) + beta * sum(p * V[s2]
                                        for s2, p in transition(x0).items())
        return cont0 >= retire

    index = {}
    for x0 in states:
        a, b = lo, hi
        while b - a > tol:                 # largest lambda with plays_on => index(x0)
            m = 0.5 * (a + b)
            if plays_on(x0, m):
                a = m
            else:
                b = m
        index[x0] = a
    return index


def greatest_index_policy(processes, states, indices):
    """Advance the process whose current state has the largest index."""
    return max(range(len(processes)), key=lambda j: indices[j][states[j]])
```
