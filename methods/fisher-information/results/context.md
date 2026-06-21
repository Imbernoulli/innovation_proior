## Research Question

A sample may contain many observations, but an estimation problem usually asks for only a few population parameters. The question is how to say, before choosing a final estimator, how much parameter-relevant content the sample has and how much of it is lost when the sample is replaced by a statistic.

The desired object has to answer a local question. If the parameter is moved by a tiny amount, does the probability law move sharply enough for the data to distinguish the two nearby possibilities, or is the movement nearly invisible?

## Statistical Setting

Assume a parametric family `p_theta(x)` and independent observations drawn from one member of that family. The likelihood of a parameter value is proportional to the probability that this value would have produced the observed sample. It ranks parameter values from the data, but it is not itself a probability distribution over the parameter.

A statistic is useful only insofar as it preserves the parameter-relevant part of the sample. A sufficient statistic preserves all of it. An inefficient statistic discards some of it, even if it remains consistent in large samples.

## Existing Tools

The method of moments supplies equations by matching sample moments to population moments. It is often easy to compute, but consistency alone does not say whether the resulting statistic uses the sample well.

Maximum likelihood supplies an invariant way to select the parameter value best supported by the observed sample. Around that optimum, second derivatives of the log-likelihood already hint at a local accuracy scale, because a sharply peaked likelihood separates nearby parameter values better than a flat one.

Older probable-error calculations can approximate the sampling variability of fitted constants. They need a principle telling which statistic has the smallest attainable error and how to compare other statistics against it.

## Failure Modes

A raw likelihood value has no absolute scale, so it cannot by itself be the amount of information. A first derivative at one observed sample can point uphill or downhill, but in a regular model its average is zero at the true parameter, so the sign does not give a persistent model property.

Estimator variance is procedure-specific. It describes the result after a statistic has been chosen, while the underlying problem asks for a property of the statistical model and the experiment itself.

Global distances between full distributions are too coarse for local estimation. The estimation problem needs the infinitesimal separation between neighboring model points.

## Evaluation Setup

A satisfactory local measure should be nonnegative, should add across independent observations, should transform correctly when the parameter is re-labeled, and should expose directions in which the model is weakly identified.

It should also explain lower bounds on variance. If an estimator is required to track a parameter locally, the model should determine how small its sampling variance can be, and equality should correspond to an estimator aligned with the informative local direction.
