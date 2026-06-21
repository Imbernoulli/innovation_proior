## Research question

When does a self-map have a stable point `x` with `f(x) = x` if the space has no metric and no useful topology? A contraction argument would ask for distances that shrink. A topological argument would ask for continuity and compactness. The ordered setting asks for something different: a partially ordered space where arbitrary collections have least upper bounds and greatest lower bounds, and a self-map that respects the order.

The question is what can be said about fixed points of a monotone self-map on a complete lattice, using only the order structure.

## Background

A complete lattice is a partial order in which every subset has a join and a meet. This includes the empty subset, so there is a bottom element `bot = join emptyset` and a top element `top = meet emptyset`. Power sets under inclusion, closed intervals or boxes under componentwise order, and finite grids under componentwise order are standard examples.

An order-preserving map is a self-map satisfying `x <= y => f(x) <= f(y)`. This is weaker than continuity and unrelated to metric contraction. It allows jumps and discontinuities, and it does not imply uniqueness. The object to control is not a distance between iterates, but the way `f` moves elements relative to the order.

In concrete computer-science and numerical settings, monotone operators are often also continuous in the order-theoretic sense of preserving suprema of nonempty chains. Then the least fixed point can sometimes be reached as the supremum of the finite approximation chain `bot, f(bot), f(f(bot)), ...`.

Game-theoretic equilibrium gives a separate reason to care about this level of generality. In supermodular games and games with strategic complementarities, strategy spaces are complete lattices and extremal best-response selections are monotone maps. A pure equilibrium can then be viewed as a fixed point of a monotone self-map, and the order can distinguish lowest and highest equilibria.

## Baselines

- **Metric contraction fixed points.** A contraction proof turns repeated application into a Cauchy sequence and usually gives uniqueness. A complete lattice need not be a metric space, and monotone maps can naturally have many fixed points.

- **Topological fixed points.** Brouwer-style equilibrium existence uses compactness, convexity, and continuity. This fits many economic fixed-point proofs, where the space is a convex compact subset of Euclidean space.

- **Finite-height iteration.** On a finite grid, starting from the bottom and repeatedly applying a monotone map yields an increasing sequence; finite height forces termination. This is a useful algorithmic baseline for finite structures.

- **Chain-continuous approximation.** If `f` preserves suprema of nonempty chains, the ascending chain from bottom has a clear limiting candidate. This is valuable in program semantics, where finite approximations often build a meaning by increasing definedness.

## Evaluation settings

The proof is judged in the abstract order-theoretic setting: arbitrary complete lattice `L`, arbitrary monotone `f : L -> L`, and no assumptions of finiteness, metric completeness, contraction, compactness, convexity, or continuity.

The natural examples for checking any statement about this setting are power sets ordered by inclusion, intervals and boxes ordered coordinatewise, finite grids, and strategy-profile lattices in supermodular games.

## Code framework

For this theorem, the final artifact is a proof rather than code. The available proof primitives are the definitions of partial order, monotone self-map, complete lattice, interval sublattice, fixed point, join, and meet. A neutral proof scaffold has one open slot:

```text
Input:
  complete lattice L
  monotone map f : L -> L

Definitions available:
  join(S) for every S subset of L
  meet(S) for every S subset of L
  Fix(f) = {x in L | f(x) = x}

Open proof obligation:
  fill in an order-only construction that proves Fix(f) is nonempty,
  identifies its extremal elements, and closes arbitrary joins and meets inside Fix(f).
```
