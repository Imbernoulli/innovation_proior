# Nash equilibrium and its existence theorem

## The problem it solves

Von Neumann's minimax theorem gives a complete solution theory only for **two-person
zero-sum** games. It has nothing to say about games with three or more players, or two
players whose payoffs are not strictly opposed, when the players act **independently** — no
communication, no binding coalitions, no side-payments. The need is a solution concept for
*arbitrary* finite non-cooperative games, plus a proof that a solution always exists.

## The key idea

Stop summarizing a game by a single "value" — that only makes sense under antagonism.
Replace it with a **stability** condition that needs no antagonism: a profile of (mixed)
strategies is an **equilibrium point** if **no player can increase their own payoff by
unilaterally changing their strategy**, holding everyone else fixed. Equivalently, every
player is simultaneously best-responding to the others. This reduces to the von Neumann
saddle point exactly in the two-person zero-sum case, so it is a genuine generalization.

Existence follows by turning best-response into a **continuous self-map of the (compact,
convex) product of strategy simplices** whose **fixed points are exactly the equilibria**,
and applying **Brouwer's fixed point theorem** (or, set-valued, **Kakutani's**).

## Setup and definition

A finite game has players `i = 1..n`; player `i` has finitely many pure strategies
`pi_{i,alpha}`. A **mixed strategy** `s_i` is a probability vector on player `i`'s pure
strategies — a point of player `i`'s simplex. A **profile** `s = (s_1,...,s_n)` lives in the
product of the simplices, a compact convex polytope (a "cell"). The payoff `p_i(s)` is the
expected payoff to player `i`; it is **multilinear** — linear in each `s_i` separately. Write
`(s; r_i)` for `s` with player `i`'s strategy replaced by `r_i`, and
`p_{i,alpha}(s) = p_i(s; pi_{i,alpha})` for the payoff `i` gets by switching to pure strategy
`alpha`.

**Definition (equilibrium point).** `s` is an equilibrium point iff for every `i`:

    p_i(s) = max_{r_i} p_i(s; r_i).

Because `p_i` is linear in `r_i` and `r_i` ranges over a simplex, the maximum is attained at
a pure strategy, so equivalently `p_i(s) = max_alpha p_{i,alpha}(s)` for every `i`; and since
`p_i(s)` is the average of the `p_{i,alpha}(s)` over the used strategies, this is equivalent
to: **every pure strategy used with positive probability is a best response.** In a
two-person zero-sum game the set of equilibria equals the set of pairs of von Neumann optimal
("good") strategies.

## Existence theorem and proof (Brouwer)

**Theorem. Every finite game has an equilibrium point in mixed strategies.**

*Proof.* Define the **gain** of deviating to pure strategy `alpha` from the current mix:

    phi_{i,alpha}(s) = max( 0,  p_{i,alpha}(s) - p_i(s) )   >= 0.

It is the amount player `i` would gain by switching to pure `alpha`; it is `0` for all
`alpha` exactly when every pure-deviation payoff is no larger than the current payoff.
Since the current payoff is the average of those pure-deviation payoffs under `s_i`, this
is exactly the condition that `i` is best-responding. Define the map `T : s |-> s'` by

    s'_i = ( s_i + sum_alpha phi_{i,alpha}(s) * pi_{i,alpha} ) / ( 1 + sum_alpha phi_{i,alpha}(s) ).

The numerator's total mass is `1 + sum_alpha phi_{i,alpha}(s)`, equal to the denominator, so
`s'_i` is again a probability vector (`T` maps the cell into itself); the denominator is
`>= 1 > 0`, and every operation is continuous, so `T` is continuous.

*Fixed points of `T` are exactly the equilibria.*
- **(=>)** At any `s`, a least-profitable *used* pure strategy `pi_{i,alpha}` satisfies
  `p_{i,alpha}(s) <= p_i(s)` (it is at most the average), so `phi_{i,alpha}(s) = 0`; its
  weight in `s'_i` is `c_{i,alpha} / (1 + sum_beta phi_{i,beta}(s))`. If `s` is fixed and
  `c_{i,alpha} > 0`, the denominator must be `1`, i.e. `sum_beta phi_{i,beta}(s) = 0`, so
  *all* gains vanish. Then every pure-deviation payoff is `<= p_i(s)`, while `p_i(s)` is
  their average under `s_i`; hence `p_i(s)` equals the maximum pure-deviation payoff and
  player `i` is best-responding. True for all `i`, so `s` is an equilibrium.
- **(<=)** If `s` is an equilibrium, every `phi_{i,alpha}(s) = 0`, so `s'_i = s_i / 1 = s_i`:
  `s` is fixed.

The profile space is compact and convex, and `T` is a continuous self-map, so **Brouwer's
fixed point theorem** gives a fixed point of `T`, which is therefore an equilibrium. ∎

**Kakutani route (equivalent, simpler hypotheses).** Map each profile `s` to the set of
profiles in which every player best-responds to `s` (the **best-response correspondence**).
Its values are non-empty (a best response exists by compactness/continuity), convex (the
argmax of a linear payoff over a simplex is a face), and it has closed graph: if
`s^k -> s`, `t^k -> t`, and `t^k` best-responds to `s^k`, then for every player `i` and
pure strategy `alpha`, the inequality saying `t_i^k` pays at least as well against
`s^k_{-i}` as `alpha` does passes to the limit by continuity. Thus `t` best-responds to
`s`. By **Kakutani's theorem** the correspondence has a fixed point — a profile in which
everyone best-responds to it — i.e. an equilibrium.

## Corollaries

- **Symmetric equilibrium.** `T` is intrinsic (depends only on payoffs), so it commutes with
  every automorphism of the game. The symmetric profiles form a non-empty (the barycenter is
  symmetric) compact convex subcell that `T` maps into itself; Brouwer there gives a
  profile fixed by every automorphism, hence a **symmetric equilibrium** in this sense.
- **Polyhedral structure.** In a *solvable* game (equilibrium set interchangeable), each
  player's set of equilibrium strategies is the simplex intersected with finitely many linear
  inequalities `p_i(t; ·) - p_{i,alpha}(t; ·) >= 0` — a **polyhedral convex set** (convex hull
  of finitely many vertices).
- **Dominance.** No equilibrium has a strictly dominated mixed strategy as a component; by
  multilinearity dominance is checkable against opponents' pure profiles only, so dominance
  elimination plus contradiction analysis can help locate equilibria in concrete games.

## A worked check (best-response / gain map)

```python
import numpy as np

def expected_payoffs(payoff_tensors, profile):
    """payoff_tensors[i]: player i's payoff over the grid of pure profiles.
       profile: list of mixed strategies (probability vectors). Returns one payoff per player."""
    n = len(payoff_tensors)
    out = np.empty(n)
    for i in range(n):
        t = payoff_tensors[i]
        for s in profile:                  # contract each player's axis with its mixture
            t = np.tensordot(t, s, axes=([0], [0]))
        out[i] = t
    return out

def deviation_payoffs(payoff_tensors, profile, i):
    """p_{i,alpha}(s): player i's payoff for each pure strategy alpha, others fixed at `profile`."""
    t = payoff_tensors[i]
    for k, s in enumerate(profile):
        if k == i:
            continue                       # leave player i's own axis uncontracted
        t = np.tensordot(t, s, axes=([1 if k > i else 0], [0]))
    return t                               # vector over player i's pure strategies

def gains(payoff_tensors, profile):
    """phi_{i,alpha}(s) = max(0, p_{i,alpha}(s) - p_i(s)) for every player i."""
    p = expected_payoffs(payoff_tensors, profile)
    return [np.maximum(0.0, deviation_payoffs(payoff_tensors, profile, i) - p[i])
            for i in range(len(payoff_tensors))]

def T(payoff_tensors, profile):
    """One application of the continuous gain map: s_i -> (s_i + bonus_i)/(1 + sum gains_i)."""
    phi = gains(payoff_tensors, profile)
    new = []
    for i, s in enumerate(profile):
        new.append((s + phi[i]) / (1.0 + phi[i].sum()))   # bonus mass phi_{i,alpha} on pure alpha
    return new

def is_equilibrium(payoff_tensors, profile, tol=1e-9):
    """s is an equilibrium iff all gains vanish (no profitable unilateral deviation)."""
    return all(g.max() <= tol for g in gains(payoff_tensors, profile))
```

Iterating `T` from any starting profile is *not* a guaranteed solver (Brouwer is
non-constructive), but `is_equilibrium` checks the fixed-point/equilibrium condition directly,
and `T` makes the existence proof concrete: a fixed point of `T` is exactly an equilibrium.
