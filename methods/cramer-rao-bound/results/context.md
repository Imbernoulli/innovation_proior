## Estimation Setting

A regular parametric model gives a family of probability laws indexed by an unknown parameter. A sample is observed, and a statistic is chosen as an estimate of the parameter or of a function of it. The statistic is itself random, so its accuracy is judged through its sampling distribution under each possible parameter value.

Many plausible statistics are available for the same target. Some are easy to compute, some follow likelihood principles, some are unbiased, and some have small variance in special cases. Estimation theory studies how to characterize the accuracy attainable for a given regular model.

## Existing Ideas

Least-squares theory had already posed a minimum-variance problem under linear unbiasedness. Fisher had introduced likelihood, consistency, efficiency, sufficiency, intrinsic accuracy, and loss of information, mainly in large-sample language. Aitken and Silverstone had pushed a finite-sample program based on unbiased estimating functions and minimum sampling variance, and had compared it with maximum likelihood.

These ideas give several pieces: likelihood tells how the model moves with the parameter, efficiency asks how much information an estimate retains, and sufficiency asks whether a statistic preserves all relevant sample information. Variance measures the spread of a statistic, unbiasedness fixes its average value, and maximum likelihood gives a construction for an estimate.

## Regularity Pressures

The model must be smooth enough that small parameter changes can be represented by differentiating the density. The support should not move with the parameter, and differentiation must pass through the integral defining expectations. These assumptions are substantive because boundary terms or discontinuities can create information not represented by an ordinary local derivative.

Under the smooth setup, the derivative of the log density is the natural local direction in which the distribution changes. A statistic's local mean response can be described relative to that direction.

## Research Question

Starting from a regular model and a statistic with a specified mean response, characterize the attainable sampling variance using only quantities determined by the model and that response, without choosing a sample mean, a likelihood estimate, or a sufficient statistic in advance. The treatment should make clear which smoothness assumptions are doing real work, so exceptional cases are handled separately from the ordinary smooth-model calculation.
