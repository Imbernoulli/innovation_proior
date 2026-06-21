## Statistical Setting

A statistical hypothesis fixes, or partly fixes, a probability law for the data. A test is a rule that divides the possible samples into cases where the hypothesis is rejected and cases where it is not. In a continuous sample space this rule is a region; in a discrete sample space it is a set of sample points. Either way, the same rule can be judged under more than one possible probability law.

The older question asks whether the observed sample looks unusual under the hypothesis being tested. That can produce useful procedures, but it does not by itself say which unusual samples should matter most when several different rejection regions have the same probability under the tested hypothesis.

## Error Tradeoff

There are two ways for the rule to go wrong. It can reject the hypothesis when that hypothesis is true, or it can fail to reject it when another specified hypothesis is true. The first error can be controlled by choosing a rejection region with a small probability under the tested law. The second error depends on how much probability the same region has under the alternative law.

The hard part begins after the first error has been fixed. Many regions can have exactly the same probability under the tested law. They spend the same amount of error allowance, yet they can have very different probabilities under the alternative.

## Existing Criteria

Common tests often begin by choosing a statistic, such as a sample mean, a sample variance, a distance from a fitted center, or a small-sample standardized mean. The statistic then determines a tail or extreme region. These tests can be excellent in special models, but the statistic by itself does not give a general comparison principle across all regions with the same first-error probability.

The weakness is not that these statistics are arbitrary in practice. The weakness is that their optimality is not visible from the statistic alone. A statistic can be natural, easy to tabulate, and still fail to explain why its rejection region is the best way to spend a fixed first-error allowance.

## Decision Geometry

In sample-space language, every possible rejection rule is a region. The tested law assigns a cost to that region: the chance of rejecting when the tested hypothesis is true. The alternative law assigns a reward: the chance of rejecting when that alternative is true.

A useful theory must compare regions that have the same cost. It cannot merely say that a region is extreme, rare, or visually far from the center. It must explain which pieces of sample space are worth including when the total first-error probability is fixed in advance.

## Required Resolution

The desired result should turn test construction into a constrained optimization problem. It should begin with a fixed allowable first-error probability, then choose among all regions with that allowance by asking which one gives the largest rejection probability under a specified alternative.

It must also handle boundary cases. In continuous models a boundary can usually be chosen so the allowance is used exactly. In discrete or atomic models the boundary may have positive probability, so the rule may need randomization on that boundary to hit the desired size.
