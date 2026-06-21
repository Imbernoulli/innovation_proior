## Research Question

A sample may contain many observations, but an estimation problem usually asks for only a few population parameters. The question is how to say, before choosing a final estimator, how much parameter-relevant content the sample has and how much of it is lost when the sample is replaced by a statistic.

This is a local question. If the parameter is moved by a tiny amount, the probability law moves with it; one wants to characterize how sharply it moves for nearby parameter values.

## Statistical Setting

Assume a parametric family `p_theta(x)` and independent observations drawn from one member of that family. The likelihood of a parameter value is proportional to the probability that this value would have produced the observed sample. It ranks parameter values from the data, but it is not itself a probability distribution over the parameter.

A statistic preserves the parameter-relevant part of the sample to varying degrees. A sufficient statistic preserves all of it. An inefficient statistic preserves less, while remaining consistent in large samples.

## Existing Tools

The method of moments supplies equations by matching sample moments to population moments. It is often easy to compute and yields consistent estimators.

Maximum likelihood selects the parameter value best supported by the observed sample, in an invariant way. Around that optimum, second derivatives of the log-likelihood describe a local accuracy scale: a sharply peaked likelihood separates nearby parameter values better than a flat one.

Probable-error calculations approximate the sampling variability of fitted constants.

## Properties Under Study

Several local features of a regular model are available to work with. The log-likelihood is differentiable in the parameter; its first derivative at the true parameter has mean zero in a regular model, while its second derivative measures the curvature of the likelihood. Estimator variance describes the result after a particular statistic has been chosen. Global distances between full distributions describe how far apart two model points are, whereas a local estimation question concerns the infinitesimal separation between neighboring model points.

## Evaluation Setup

One wants a measure attached to the model and the experiment rather than to any one estimator: a local quantity defined from the parametric family `p_theta(x)` itself. Of interest is how such a measure behaves under combining independent observations and under re-labeling the parameter, and how it relates to the sampling variance achievable by an estimator that tracks the parameter locally.
