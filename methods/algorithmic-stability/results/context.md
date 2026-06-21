## Estimation After Selection

A training sample is used twice: first to choose a predictor and then to estimate how good that predictor is. If the predictor were fixed in advance, an empirical average would be a straightforward estimate of population loss. Here the predictor is a function of the sample itself, so the empirical loss and the true loss are computed on related quantities.

Let the sample be `S = (z_1, ..., z_m)`, drawn i.i.d. from an unknown distribution over examples. A learning rule maps `S` to a hypothesis `A(S)`, and a bounded loss `ell(A(S), z)` measures performance on an example `z`. The target quantity is the true risk `R(A,S) = E_z ell(A(S), z)`, while the observable quantity is often `R_emp(A,S) = (1/m) sum_i ell(A(S), z_i)`.

The question is how to certify that these two quantities are close for the particular learning rule being run.

## Class-Level Control

The classical response is to control a whole hypothesis class at once. Uniform-convergence arguments bound the largest possible gap between empirical and true risk over every function in a class. VC dimension, fat-shattering dimension, and related capacity measures make this program quantitative.

This view applies when a learner searches a fixed class by empirical-risk minimization. It is stated in terms of the class rather than the details inside the learning rule, and in terms of class capacity rather than a regularization parameter that changes the optimization geometry.

## Perturbing The Sample

A more local diagnostic is to ask what happens when one training example is removed or replaced. Write `S \ i` for the sample with point `i` deleted, and write `S^i` for a sample with point `i` replaced by a fresh draw.

Leave-one-out estimates already use this kind of perturbation: train without one point and test on the omitted point, giving `R_loo(A,S) = (1/m) sum_i ell(A(S \ i), z_i)`.

## Regularized Procedures

Regularized learning makes the perturbation question especially natural. A regularized objective is not just trying to fit the sample; it is also paying for movement in the hypothesis space. Removing or replacing one empirical-loss summand changes the objective by about one sample's share, and the penalty constrains how far the optimizer moves.

Kernel ridge regression, support-vector-style objectives, and Tikhonov regularization all fit this pattern. The ambient function space can be large, while the selected solution is shaped by the penalty.

## Certificate Requirements

A certificate for such a procedure works with quantities the learning rule actually exposes: the effect of a one-coordinate perturbation, the comparison between a training-point loss and a fresh-point loss, and the concentration of a quantity whose value depends on all independent sample coordinates. The question is how to turn these into a high-probability bound on `R(A,S)`.
