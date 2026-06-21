# Context

## Repeated Decisions With Late Costs

A decision maker faces the same feasible region over and over, but the cost rule
for the current round is unknown until after the decision is committed. The
feasible region is fixed and known in advance, such as resource constraints for a
farm, production constraints in a factory, a routing polytope, or a simplex of
mixed strategies. On round `t`, the decision maker chooses a feasible point
`x^t`; only then is the round's convex cost function `c^t` revealed.

There is no useful promise that the next cost resembles the previous one. The
sequence may be adversarial, may be chosen by a strategic opponent, and may have
no distribution. The problem is therefore not ordinary stochastic prediction and
not ordinary offline minimization of one fixed objective. The learner must make a
new feasible commitment before seeing the loss that will evaluate it.

## What Counts As Success

The goal cannot be to choose the per-round minimizer of `c^t`, because `c^t` is
not known at the time of the commitment. A fair comparator has to respect the
same repeated-decision nature while still expressing hindsight. The standard
choice is the single best fixed feasible point after the entire sequence of
costs is known.

For horizon `T`, the learner's cumulative cost is compared with
`min_x sum_{t=1}^T c^t(x)` over feasible `x`. The excess is regret. A successful
method should make average regret vanish, so its cumulative cost differs from
the best fixed hindsight decision by a sublinear amount. This is the same
conceptual target as Hannan consistency in repeated games and the best-expert
comparison in prediction with expert advice.

## Available Geometry

Convexity is the only structure robust enough for this adversarial setting. A
convex feasible set contains every segment between feasible points. A convex
differentiable cost lies above each of its tangent planes:
`f(y) >= f(x) + grad f(x) dot (y - x)`. This first-order lower bound is a way to
compare a curved loss at two points using local information at the point that was
actually played.

A closed convex feasible set also supports Euclidean projection: every outside
point has a nearest feasible point. The key fact is not just that projection
restores feasibility. It is that projection cannot increase distance to any
feasible comparator. If the feasible set has bounded diameter and gradients are
bounded, then squared distances and gradient norms can become finite terms in a
regret proof.

## Earlier Pieces That Do Not Yet Compose

Blackwell's approachability theorem and Hannan's consistency results show that
adversarial repeated interaction can be controlled through average performance
and hindsight comparison. These results establish the right game-theoretic
ambition but do not by themselves give a simple general-purpose convex
optimization update.

The experts and multiplicative-weights line gives a concrete adversarial
algorithm for a finite action set. A learner plays a distribution over experts,
then sees a cost vector, and competes with the best fixed expert. The decision
set is a simplex and the loss is linear in the played distribution. The proof is
based on an entropic or weight potential and is specialized to finite experts.

Online prediction and regression supply gradient-style updates with relative
loss bounds, but those analyses are tied to particular losses or divergences.
Infinitesimal gradient ascent in games supplies another nearby object: in a
two-player, two-action game, players move their mixed strategies by gradient
ascent and average payoffs converge in the special `2 x 2` geometry. Its proof
depends on low-dimensional game dynamics rather than a general convex-set
argument.

## Constraints On A Satisfactory Resolution

A satisfactory resolution must preserve the adversarial nature of the problem:
no distribution over costs, no stationarity, and no relation between successive
rounds. It must work for any closed bounded convex feasible set for which
projection can be computed, and for any convex differentiable round cost with
bounded gradients.

It should also explain the older examples rather than merely coexist with them.
On a simplex with linear costs it should look like an experts problem. On a
mixed-strategy simplex with utility revealed through an opponent's action it
should give a no-regret repeated-game strategy beyond the `2 x 2` case. Its
analysis should reveal why the regret comparator, convex tangent inequality,
projection geometry, and step-size schedule fit together.
