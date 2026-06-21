## Estimation Setting

A regular parametric model gives a family of probability laws indexed by an unknown parameter. A sample is observed, and a statistic is chosen as an estimate of the parameter or of a function of it. The statistic is itself random, so its accuracy is judged through its sampling distribution under each possible parameter value.

The central difficulty is that there are many plausible statistics for the same target. Some are easy to compute, some follow likelihood principles, some are unbiased, and some have small variance in special cases. A theory of estimation needs a way to say when a proposed statistic is intrinsically limited by the model rather than merely by poor construction.

## Existing Ideas

Least-squares theory had already posed a minimum-variance problem under linear unbiasedness. Fisher had introduced likelihood, consistency, efficiency, sufficiency, intrinsic accuracy, and loss of information, mainly in large-sample language. Aitken and Silverstone had pushed a finite-sample program based on unbiased estimating functions and minimum sampling variance, and had compared it with maximum likelihood.

These ideas give important pieces: likelihood tells how the model moves with the parameter, efficiency asks how much information an estimate retains, and sufficiency asks whether a statistic preserves all relevant sample information.

## Unsettled Gap

Variance alone is not enough, because a statistic can have very small variance while failing to track the parameter. Unbiasedness alone is not enough either, because it only fixes the average value, not the spread. Maximum likelihood gives a powerful construction, but a construction is not the same as a universal limit applying to every regular unbiased statistic.

The missing object is a local obstruction: a statement that any statistic whose mean responds to parameter motion at a specified rate must carry a corresponding amount of sampling variability.

## Regularity Pressures

The model must be smooth enough that small parameter changes can be represented by differentiating the density. The support should not move with the parameter, and differentiation must pass through the integral defining expectations. These assumptions are substantive because boundary terms or discontinuities can create information not represented by an ordinary local derivative.

Under the smooth setup, the derivative of the log density is the natural local direction in which the distribution changes. The question is how a statistic's local mean response can be connected to that direction.

## Desired Artifact

The needed result should not depend on choosing a sample mean, a likelihood estimate, or a sufficient statistic in advance. It should start from a regular model and a statistic with a specified mean response, then constrain attainable variance using only quantities determined by the model and that response.

It should also make clear which smoothness assumptions are doing real work, so exceptional cases do not get mistaken for failures of the ordinary smooth-model calculation.
