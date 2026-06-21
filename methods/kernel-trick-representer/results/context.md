## Research question

We want simple linear learning machinery to work after a very rich feature expansion, while keeping the computation finite. The useful geometry may live in many coordinates, or even infinitely many, but the data set supplies only finitely many labeled observations.

The same setting arises in regularized function estimation. A smoothness penalty is naturally defined over a function space, yet a numerical algorithm must return a finite object that can be evaluated on new inputs.

## Background

Classical pattern-recognition procedures already use separating hyperplanes, perceptron-style correction rules, and expansions in chosen basis functions. Spline theory supplies another source of structure: smooth functions can be selected by minimizing a roughness functional subject to interpolation or data-fitting constraints.

Hilbert-space language is available for both settings. Inner products measure geometry, continuous linear functionals have representers, and point evaluation can be treated as an inner product when the function space has the right reproducing structure.

## Baselines

One baseline is to explicitly construct all feature coordinates and run a linear method there.

A second baseline is to solve a variational problem directly over an abstract function space. This is mathematically clean and gives a principled way to encode smoothness assumptions.

A third baseline is memorization or nearest-prototype comparison in the original input space. That uses finite data directly.

## Evaluation settings

The method should handle binary classification rules whose lifted decision boundary is linear, smoothing and interpolation problems with bounded observation functionals, and regularized empirical-risk objectives whose data term depends only on the fitted values at the training inputs.

The important checks are structural rather than benchmark-specific: whether a proposed similarity behaves like a legitimate inner product, whether the finite numerical problem remains well conditioned in degenerate cases, and whether held-out predictions are determined by quantities available at training time.

## Code framework

A software testbed can compare an explicit feature-map implementation with a callable similarity function `k(x,z)`. Small finite samples should be enough to check basic matrix validity and behavior when the resulting matrix is singular or nearly singular.

For spline-like variants, the framework should keep any unpenalized basis terms separate from the roughness-penalized part, because those two pieces have different identifiability and conditioning requirements.
