## Research question

Place `26` circles inside the unit square `[0,1]²`, pairwise non-overlapping, and make the
**sum of their radii** `Σ rᵢ` as large as possible. The single thing being designed is a
**constructor**: a program that emits one concrete packing — centers `(xᵢ, yᵢ)` and radii
`rᵢ ≥ 0` — and it is scored by `Σ rᵢ` alone. Nothing about the harness is learned; the
constructor's output is a fixed list of `26` circles, the evaluator checks they are legal and
adds up the radii, and that sum is the whole result.

Unlike the classical *equal-circle* packing problem (where every circle has the same radius and
one maximizes that common radius), here the radii are **free and unequal**. That freedom is the
whole character of the problem: a good packing mixes a few large circles with many small ones
that fill the gaps, and the optimum is a genuinely irregular arrangement with no symmetry to lean
on. The count `26` is chosen deliberately — it sits in the range where the best known value is
the product of dedicated optimization, not a closed form, which is exactly why it is used as a
discovery target.

## How the score is defined

The score is simply

```
score = Σ_{i=1}^{26} r_i        (higher is better)
```

subject to the hard feasibility constraints, with the convention (following the AlphaEvolve /
OpenEvolve / ShinkaEvolve / AutoEvolver harnesses that made this a standard benchmark) that a
packing is accepted when it violates no constraint by more than a small absolute tolerance
`atol`. The constraints are:

- **Inside the square:** for every circle, `rᵢ ≤ xᵢ ≤ 1 − rᵢ` and `rᵢ ≤ yᵢ ≤ 1 − rᵢ`.
- **Pairwise disjoint:** for every `i ≠ j`, `√((xᵢ−xⱼ)² + (yᵢ−yⱼ)²) ≥ rᵢ + rⱼ`.
- **Non-negative radii:** `rᵢ ≥ 0`.

This is a nonconvex quadratically-constrained problem (the pairwise distance constraints are
nonconvex), so there is no convexity to exploit globally; the landscape is riddled with local
optima, and the strong methods are all local-refinement-plus-restart schemes. A useful structural
fact the constructor can exploit: **for fixed centers, the optimal radii are the solution of a
linear program** — maximize `Σ rᵢ` subject to `rᵢ + rⱼ ≤ dᵢⱼ` and `rᵢ ≤ wallᵢ` (the distance to
the nearest wall), all linear in the `rᵢ`. So the genuinely hard, nonconvex part is *where to put
the centers*; once they are fixed the radii are free in closed/LP form.

The headline numbers to keep in view — the published frontier for `n = 26`:

| Reference point | Σ rᵢ | source |
|---|---|---|
| Friedman (2012), prior human best | ~2.634 | E. Friedman, Packing Center tables |
| 5×5 grid + interstitial (this scaffold baseline) | 2.5414 | structured equal-circle start |
| **AlphaEvolve** (Novikov et al. 2025) | **2.63586276** | arXiv:2506.13131 |
| **ShinkaEvolve** (Lange et al. 2025) | **2.635983283** | arXiv:2509.19349 |
| **ThetaEvolve** | **2.63598308** | reported |
| **AutoEvolver / Claude Code** (record) | **2.635988438568** | github.com/tengxiaoliu/autoevolver |

The record (`2.635988438568`, AutoEvolver) and the ShinkaEvolve value (`2.635983283`) sit within
`~5×10⁻⁶` of one another — the frontier here is a band that successive evolutionary / agentic
systems have been shaving by parts in the sixth decimal place. So the ladder is not chasing a
distant target; it is climbing from a trivial structured baseline (`~2.54`) up into this frontier
band (`~2.636`), and the honest measure of a rung is how many of those decimal places it buys.

## Prior art before the first rung

- **Equal-circle packing (Graham, Lubachevsky, Specht, et al.).** Decades of work on packing `n`
  *equal* circles in a square, tabulated by Specht (packomania.com). *Gap here:* the radii are
  forced equal, so the optimum is a single common radius; the unequal-radius sum-of-radii problem
  is a different, less-studied objective where mixing sizes wins.
- **Friedman's packing-center tables (2012).** Hand- and computer-found configurations for many
  small-`n` packing objectives, including sum-of-radii variants. *Gap:* the `n = 26` value
  (`~2.634`) was the standing human best and was improved only recently by automated search.
- **Multi-start SLSQP / nonlinear programming.** The standard strong recipe for nonconvex QCQPs:
  many random initial center layouts, each refined by Sequential Least-Squares Quadratic
  Programming (`scipy.optimize.minimize(method='SLSQP')`) on the full constrained problem,
  keeping the best feasible result. *Gap:* a single SLSQP run lands in whatever basin its
  initialization sits in; reaching the frontier needs either many restarts or a smarter search
  over basins.
- **Program-evolution / agentic search (AlphaEvolve, OpenEvolve, ShinkaEvolve, ThetaEvolve,
  AutoEvolver, 2025).** LLM-driven evolutionary systems that mutate the *constructor program*
  itself, discovering hybrid pipelines: structured initialization (golden-angle spirals, corner
  seeding) + joint SLSQP refinement of centers *and* radii together + iterated perturbation
  chains / simulated annealing to escape local optima. AutoEvolver's agent specifically reported
  that the key jump came from **jointly optimizing centers AND radii with SLSQP** rather than
  splitting into LP-for-radii and a separate center optimizer, then polishing with **iterated
  perturbation chains**, reaching the record `2.635988438568` after ~16.6 hours of autonomous
  compute. *Gap:* these systems spend large search budgets; the question for a single constructor
  is how close a principled hybrid pipeline gets within a bounded run.

## The fixed substrate

The harness is a thin, deterministic evaluator. It calls the constructor once, receives `26`
centers and `26` radii, checks the shape, that no radius is negative, that every circle lies
inside the unit square, and that no pair overlaps — all within an absolute tolerance `atol`
(`1e-6` in the AutoEvolver/OpenEvolve harness; ShinkaEvolve uses `1e-7`). If feasible it returns
`Σ rᵢ`; otherwise the packing is rejected. The tolerance and the count `n = 26` are frozen.

## The editable interface

Exactly one function is editable: `construct_packing()`, returning `(centers, radii)` — a
`(26, 2)` array of centers and a length-`26` array of radii. Every rung on the ladder is a
different body for it. The feasibility checker and the scorer are fixed.

```python
import numpy as np

N = 26

def feasible(centers, radii, atol=1e-7):
    """True iff all circles are inside [0,1]^2 and pairwise disjoint within atol."""
    centers = np.asarray(centers, float); radii = np.asarray(radii, float)
    if np.any(radii < -atol):
        return False
    if np.any(radii - np.minimum(centers[:,0], centers[:,1]) > atol):     return False
    if np.any(radii - np.minimum(1-centers[:,0], 1-centers[:,1]) > atol): return False
    for i in range(N):
        for j in range(i+1, N):
            if (radii[i]+radii[j]) - np.hypot(*(centers[i]-centers[j])) > atol:
                return False
    return True

def score(centers, radii):
    assert feasible(centers, radii)
    return float(np.sum(radii))

# ---- EDITABLE: the constructor. Default = a structured 5x5 grid + interstitial circle. ----
def construct_packing():
    k = 5; r = 1.0/(2*k)
    centers = [[(i+0.5)/k, (j+0.5)/k] for i in range(k) for j in range(k)]
    centers.append([2.0/k, 2.0/k])                 # 26th in an interstitial gap
    radii = [r]*25 + [(np.sqrt(2)-1)*r]
    return np.array(centers), np.array(radii)
```

Every valid output must satisfy: shapes `(26, 2)` and `(26,)`; all circles inside the square and
pairwise disjoint within `atol`. There are no other constraints — the constructor is free to
return any feasible packing, structured or searched.

## Evaluation settings

A single deterministic instance: `n = 26`, scored by `Σ rᵢ`. Because a constructor may search
internally with randomness, the run is fixed to a stated seed so the reported number is
reproducible, and the harness reports the sum of the *returned* packing after verifying its
feasibility. The frontier values — AlphaEvolve `2.63586276`, ShinkaEvolve `2.635983283`,
AutoEvolver record `2.635988438568` — are the fixed yardsticks every rung is read against. There
is no partial credit beyond the radius sum, no held-out set, and no way to game the metric except
by actually producing a feasible packing with a larger `Σ rᵢ`.
