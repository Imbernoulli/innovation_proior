## Research Question

Modern prediction systems can be trained until every training example is fit exactly, including examples whose labels contain noise. In the classical story, an exact fit to noisy data is the canonical overfitting case, and its performance on fresh data is expected to degrade. The empirical fact is sharper than ordinary overparameterization: zero training error can coexist with useful test accuracy.

The cleanest place to ask whether this is mathematically possible is linear regression with squared loss. Let responses follow a well-specified linear model, with an optimal linear parameter vector `theta^*`, and let the number of available directions in parameter space exceed the number of samples. Then interpolation is not a special accident; there are typically many parameters `theta` with `X theta = y`.

The question is: among exact fits to noisy training data, when can a natural interpolating rule have excess prediction risk close to the optimal linear rule, and what property of the data distribution governs the difference between a harmless exact fit and a harmful one?

## Background

The classical bias-variance trade-off says that a model class must balance underfitting against overfitting. Textbook examples treat interpolating fits as unlikely to predict well, and ordinary low-dimensional least-squares variance calculations describe the variance behavior near a square design.

Recent deep-learning experiments add to this view. Standard architectures trained by stochastic gradient methods can fit random labels or partially corrupted labels, often reaching zero training error without the optimization procedure collapsing. These experiments show that the effective capacity of the models is large enough to memorize the training set, and that explicit regularization alone is not the full account of generalization.

The double-descent picture organizes the empirical risk curve. As capacity rises, test risk can first follow a classical U-shape, spike around the interpolation threshold, and then decline again in the overparameterized regime.

## Baselines

Ridge regression offers the classical controlled solution. It minimizes squared error with an explicit norm penalty, shrinking the fitted parameter to reduce variance. In the overparameterized setting, the zero-penalty limit is an exact interpolant whenever the design has full row rank.

The least-norm least-squares solution is the canonical interpolating baseline. Among all exact fits, it chooses the parameter vector of smallest Euclidean or Hilbert-space norm. Equivalently, when `p > n` and `XX^T` is invertible, it has the pseudoinverse form `X^T(XX^T)^{-1}y`.

High-dimensional asymptotic analyses of ridgeless least squares show rich behavior, including peaks near the interpolation threshold and improved risk beyond it. These analyses typically fix a proportional limit or a specific feature model.

## Mathematical Ingredients

For squared loss, excess risk can be measured as prediction error relative to the best linear rule. Once a fitted parameter is fixed, this risk depends on the covariance of the covariates, because parameter error in a high-variance direction is expensive and parameter error in a low-variance direction is cheap.

The data model can be represented through the spectral decomposition of the covariance operator. The covariate is a linear transform of independent standardized coordinates, allowing weighted sums of independent outer products to describe the random Gram matrices that appear in the estimator.

The available proof tools include pseudoinverse algebra, the Sherman-Morrison-Woodbury identity, concentration inequalities for subgaussian and subexponential random variables, epsilon-net bounds for random matrix operator norms, and covariance-estimation bounds controlled by spectral size rather than ambient dimension.

## Evaluation Setting

The estimator is evaluated by its excess prediction risk on a fresh draw from the same distribution. The noise has mean zero conditional on the covariates and nonzero conditional variance, so an exact fit must absorb genuine randomness in the training labels.

The regime of interest has more parameter directions than samples, possibly infinitely many directions in a separable Hilbert space. The finite-sample question concerns whether the natural norm-biased interpolant predicts nearly as well as the optimal linear rule.

The starting objects are a generic data sampler, a ridge or least-squares baseline, a placeholder for the exact-fit rule, and a way to compute excess risk from the covariance spectrum.
