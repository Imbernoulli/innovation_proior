## Problem Setup

A one-hidden-layer real-valued feedforward network computes a finite ridge expansion

```text
G(x) = sum_{j=1}^N alpha_j sigma(w_j . x + theta_j),
```

where one scalar nonlinearity `sigma` is reused in every hidden unit. The output weights, directions, and thresholds are free; the activation itself is fixed.

The mathematical question is representational. Given a compact domain such as `[0,1]^n`, a continuous target `f`, and a tolerance `epsilon > 0`, can some finite choice of hidden units make the uniform error smaller than `epsilon`? A training algorithm is not part of the question.

## Earlier Representation Tools

Exact superposition results already show that continuous multivariate functions can be written using continuous functions of one variable and addition. The catch is that those representations may use specially chosen univariate functions. They do not force every component to be a shifted and scaled copy of one fixed curve.

Classical approximation theory offers nearby tools. Fourier series approximate with trigonometric systems. Stone-Weierstrass proves density for subalgebras that separate points. Tauberian theorems handle translation-invariant spans. These tools show that simple families can be complete, but none by itself answers the fixed-activation ridge span.

## Neural Network Motivation

In pattern classification and nonlinear prediction, a hidden unit turns an affine score into a scalar response, and the output layer adds many such responses. This makes a single hidden layer look like a linear span of hyperplane-oriented features.

Finite point separation was already plausible: with enough hidden units, a network can separate finite training samples in many ways. The harder question is what happens on a continuum of inputs, where the topology of `C(K)` and the uniform norm matter.

## Decision Regions

Decision functions add a second pressure. A continuous network cannot uniformly equal a discontinuous indicator of an arbitrary measurable region. The right formulation must allow approximation except on a small-measure set or away from a narrow boundary band.

This keeps the problem honest: a network output can be continuous and still support useful classification if the ambiguous region is made small enough.

## Evaluation Criteria

The main setting is `C([0,1]^n)` with the supremum norm. Related settings use `L^p` norms with respect to finite measures when approximation in measure is the correct goal.

The activation class is also part of the evaluation. Different bounded, threshold-like, oscillatory, and algebraic shapes raise the same question: which property of the activation makes the ridge span large enough?
