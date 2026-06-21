## Research question

Wide neural networks are powerful enough to interpolate finite data, but their training objective in parameter space is highly non-convex. A useful theory should explain not only what random wide networks look like before training, but also what gradient descent does to the function they compute.

The central question is whether the evolution of network predictions during gradient descent can be described as a dynamical system in function space rather than in parameter space.

## Background

There is already a mature kernel language for learning from pairwise similarities. Positive definite kernels define feature spaces without exposing their coordinates, and kernel gradient or kernel ridge dynamics can be written through finite Gram matrices on the data.

There is also a known infinite-width limit for randomly initialized networks. With variance scaled against layer width, sums over many hidden units converge to Gaussian processes whose covariance is computable by a layerwise recursion.

## Baselines

One baseline is to analyze the parameter loss landscape directly. Studies characterize the geometry of critical points, symmetry groups, and optimization trajectories in weight space.

A second baseline is to freeze random features or train only the last layer. This gives a kernel method with a fixed similarity function defined by the random initialization.

A third baseline is to use a kernel inspired by a network architecture as a static surrogate. This connects network structure to kernel machines and yields practical algorithms.

## Evaluation settings

The clean setting is a fully connected feedforward network with fixed depth, growing hidden-layer widths, iid Gaussian initialization, smooth enough nonlinearity, and continuous-time gradient descent on a finite training set.

## Code framework

An implementation should expose a layerwise covariance recursion and a finite matrix solver for prediction dynamics under squared loss.

For finite networks, diagnostic code can compare empirical prediction dynamics at initialization and during training to the analytic infinite-width recursion, and measure how the induced similarity behaves as width increases.
