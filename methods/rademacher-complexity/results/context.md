# Context

## Uniform Gaps After Choosing From Data

A learner does not evaluate one fixed function. It sees a sample, searches a class, and returns the function that looks best on that sample. For each fixed function, an empirical average can be close to its true expectation, but the selected function is chosen after all empirical accidents are visible.

The quantity that matters is therefore a uniform gap over the whole class:

$$
\sup_{f\in\mathcal F}\left(Pf-P_n f\right).
$$

If this gap is large, empirical risk minimization can choose a function whose training performance is an artifact of the sample. A useful learning guarantee must control this supremum, not merely a pointwise deviation.

## Fixed Penalties And Model Selection

Classical learning bounds already have the right outer shape:

$$
\text{true risk}\le \text{empirical risk}+\text{complexity penalty}.
$$

The difficulty is the penalty. VC dimension and related worst-case combinatorial measures are distribution-free. They charge for the richest configuration the class could realize anywhere, even when the observed sample has a much simpler geometry.

This is especially damaging for model selection. Choosing among model classes requires penalties that track the actual problem instance closely enough to rank competing classes. A fixed penalty can be theoretically valid and still too coarse to guide the choice.

## Existing Capacity Controls

The older route is to count or cover what a class can do. For binary classes, the growth function and VC dimension control the number of dichotomies on finite samples. For real-valued classes, covering numbers, metric entropy, fat-shattering dimension, and margin-based arguments give finer capacity controls.

These quantities explain uniform convergence and can imply meaningful rates. They also have drawbacks: many are worst-case over samples, many are hard to compute, and several enter through upper bounds rather than through a capacity measured directly on the realized data.

Empirical-process theory supplies sharper proof tools. A ghost sample can replace a true expectation, and concentration inequalities can turn expected suprema into high-probability statements. But before the target move, the symmetric empirical process is usually a proof device rather than the final sample-dependent penalty.

## Margins And Composite Predictors

The relevant predictors are often real-valued. Boosting, neural networks, and kernel machines output scores, then classify by the sign of the score. A useful bound should not discard the margin by applying a hard step function too early.

The standard workaround is a Lipschitz surrogate that upper-bounds the classification error. This creates a new problem: the capacity of the composed loss class must be related back to the capacity of the underlying score class.

Composite classes add another demand. Voting methods use convex hulls, neural networks use sums and Lipschitz nonlinearities, decision trees use Boolean combinations, and kernel methods use norm balls in feature space. A practical theory needs algebraic rules that reduce such classes to simpler parts.

## Open Design Pressure

Only the training sample is visible, but the guarantee concerns a fresh draw from the distribution. A useful penalty therefore has to live in that narrow space: observable from data, distribution-free enough to certify a uniform gap, and still sensitive enough to distinguish an easy sample from an adversarial one.

The same pressure appears in composite classes. The penalty should interact predictably with monotonicity, scaling, convex hulls, sums, Lipschitz composition, translations, loss transformations, and kernel norm bounds. Without those algebraic rules, it remains a proof existence result rather than a tool for model selection.
