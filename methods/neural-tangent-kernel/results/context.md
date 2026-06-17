## Research question

Wide neural networks are powerful enough to interpolate finite data, but their training objective in parameter space is highly non-convex. A useful theory should explain not only what random wide networks look like before training, but also what gradient descent does to the function they compute.

The central tension is that the loss may be simple as a functional of predictions while complicated as a function of weights. If the evolution of predictions can be described directly, the problem may become a tractable dynamical system in function space.

## Background

There is already a mature kernel language for learning from pairwise similarities. Positive definite kernels define feature spaces without exposing their coordinates, and kernel gradient or kernel ridge dynamics can be written through finite Gram matrices on the data.

There is also a known infinite-width limit for randomly initialized networks. With variance scaled against layer width, sums over many hidden units converge to Gaussian processes whose covariance is computable by a layerwise recursion.

## Baselines

One baseline is to analyze the parameter loss landscape directly. This faces saddle points, many symmetries, and no simple convex structure.

A second baseline is to freeze random features or train only the last layer. That gives a kernel method, but it discards the question of what happens when all layers are trained.

A third baseline is to use a kernel inspired by a network architecture as a static surrogate. That can be useful, but it does not by itself justify the trajectory followed by gradient descent on the original parameters.

## Evaluation settings

The clean setting is a fully connected feedforward network with fixed depth, growing hidden-layer widths, iid Gaussian initialization, smooth enough nonlinearity, and continuous-time gradient descent on a finite training set.

The key checks are whether the prediction dynamics converge to a deterministic limit, whether the limiting operator remains positive definite on the data, whether least-squares training admits a closed-form trajectory, and whether the theory says which data directions are fitted quickly.

## Code framework

An implementation should expose a layerwise covariance recursion, a companion training-time operator, and a finite matrix solver for prediction dynamics under squared loss.

For finite networks, diagnostic code can compare empirical prediction dynamics at initialization and during training to the analytic infinite-width recursion, and measure how much the induced similarity changes as width increases.
