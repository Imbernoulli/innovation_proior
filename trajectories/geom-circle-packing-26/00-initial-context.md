## Research question

Place 26 circles inside the unit square `[0,1]²`, pairwise non-overlapping, and maximize the sum of their radii, `Σ rᵢ`. The only artifact produced is a **constructor**: a program that emits 26 centers `(xᵢ, yᵢ)` and radii `rᵢ ≥ 0`. The evaluator checks feasibility and returns `Σ rᵢ`; that sum is the entire result.

This is the **unequal-radius**, sum-of-radii variant of circle packing in a square. Unlike equal-circle packing, where a single common radius is optimized, the radii here are free and can differ, so a good packing mixes a few large circles with many small gap-fillers. For `n = 26` the best known values are not closed forms; they come from intensive optimization.

A packing is feasible when, within tolerance `atol`:

- every circle stays inside the walls: `rᵢ ≤ xᵢ ≤ 1 − rᵢ` and `rᵢ ≤ yᵢ ≤ 1 − rᵢ`;
- no pair overlaps: `√((xᵢ−xⱼ)² + (yᵢ−yⱼ)²) ≥ rᵢ + rⱼ`;
- all radii are non-negative.

## Prior art / Background / Baselines

- **Equal-circle packing.** Core idea: optimize a single common radius for `n` circles in a square, tabulated extensively for many `n`. Gap: forcing equal radii ignores the advantage of mixing sizes, so the best equal-circle configurations are far from optimal for the sum-of-radii objective.

- **Friedman's packing-center tables.** Core idea: collect hand- and computer-found configurations for small-`n` sum-of-radii problems. Gap: the standing value for `n = 26` is about `2.634`, and these tables do not provide a practical way to push it higher.

- **5×5 grid + interstitial circle (scaffold baseline).** Core idea: place 25 equal circles on a regular grid and insert a 26th circle in the central gap. Gap: the rigid grid leaves boundary and interstitial space unused, giving a sum near `2.5414`, well below the known frontier.

- **Multi-start SLSQP / nonlinear programming.** Core idea: generate many random initial center layouts and refine each with a constrained gradient-based local solver. Gap: each refinement is trapped in the basin of its initialization; approaching the frontier requires an impractical number of restarts.

- **Program-evolution / agentic search.** Core idea: treat the constructor itself as the object of search, using evolutionary or agent-driven mutation to build pipelines of initialization, local refinement, and restart/perturbation strategies. Gap: the published systems reach the frontier only after large autonomous search budgets, so a single bounded constructor still does not match them.

Current published frontier for `n = 26`:

| Source | Σ rᵢ |
|---|---|
| Friedman tables | ~2.634 |
| 5×5 grid + interstitial | ~2.5414 |
| AlphaEvolve | 2.63586276 |
| ShinkaEvolve | 2.635983283 |
| AutoEvolver / Claude Code | 2.635988438568 |

## Fixed substrate / Code framework

The harness is a deterministic evaluator. It calls `construct_packing()` once, receives 26 centers and 26 radii, checks that all radii are non-negative, every circle lies inside `[0,1]²`, and no pair overlaps (all within `atol = 1e-7`), and returns `Σ rᵢ` if feasible. The tolerance and `n = 26` are fixed.

## Editable interface

Only `construct_packing()` may be edited. It must return `(centers, radii)` where `centers` has shape `(26, 2)` and `radii` has shape `(26,)`.

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

Any valid output is allowed—structured, random, or searched—as long as it is feasible.

## Evaluation settings

One deterministic instance: `n = 26`, score = `Σ rᵢ`. If the constructor uses randomness, the run is fixed to a stated seed so the reported value is reproducible. The published frontier values above are the yardsticks against which results are compared. There is no partial credit; the only way to improve the score is to return a feasible packing with a larger radius sum.
