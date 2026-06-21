# Context: finite-sample model choice under unknown risk

## Research Question

A supervised learner receives a finite sample from an unknown joint distribution and must choose a predictor whose future risk is small. The risk is an expectation over the unknown distribution, while the quantity directly available to the learner is the sample average loss.

## Existing Theory

Empirical risk minimization covers familiar estimators such as least squares and maximum likelihood. Classical concentration can control a fixed candidate function, and uniform convergence theory extends that control to an entire function class through quantities such as growth functions, VC dimension, and confidence intervals.

## Failure Modes

A rule chosen after seeing the data is not a fixed rule. If the candidate class is too rich, a learner can drive training error to zero while leaving true error large. If the class is too small, the learner may underfit no matter how stable the empirical estimate is. Parameter counts alone do not reliably measure this complexity.

## Baselines

One baseline is to pick a single hypothesis class and minimize training loss inside it. Other baselines add model-order penalties such as AIC or MDL, choose architectures by validation, constrain norms, or compare families of hyperplanes. Each baseline addresses part of the fit-versus-complexity tension but leaves open how to combine finite-sample fit with a distribution-free capacity guarantee.

## Evaluation Interface

The setup exposes a training sample, a loss, a sequence of candidate function families, and an estimate or bound for each family's capacity. A candidate solution must specify how the learner solves the empirical problem inside each family, how it assigns finite-sample confidence to that family, and how it chooses the final predictor before testing on fresh data.
