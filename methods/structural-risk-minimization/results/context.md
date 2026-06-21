# Context: finite-sample model choice under unknown risk

## Research Question

A supervised learner receives a finite sample from an unknown joint distribution and must choose a predictor whose future risk is small. The risk is an expectation over the unknown distribution, while the quantity directly available to the learner is the sample average loss.

## Existing Theory

Empirical risk minimization covers familiar estimators such as least squares and maximum likelihood. Classical concentration can control a fixed candidate function, and uniform convergence theory extends that control to an entire function class through quantities such as growth functions, VC dimension, and confidence intervals.

## Baselines

One baseline is to pick a single hypothesis class and minimize training loss inside it. Other baselines add model-order penalties such as AIC or MDL, choose architectures by validation, constrain norms, or compare families of hyperplanes.

## Evaluation Interface

The setup exposes a training sample, a loss, and a sequence of candidate function families. A candidate solution must specify how the learner solves the empirical problem inside each family and how it selects the final predictor before testing on fresh data.
