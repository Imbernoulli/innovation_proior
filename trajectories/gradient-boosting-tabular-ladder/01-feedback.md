Measured result — AdaBoost (Freund & Schapire, 1997, "A Decision-Theoretic Generalization of On-Line
Learning and an Application to Boosting," *J. Comput. Syst. Sci.* 55(1):119–139), the multiclass form
realized as SAMME in scikit-learn `_weight_boosting.py`.

This is a foundational/canonical bookend: AdaBoost predates the fixed tabular speed/accuracy benchmark
suite (Higgs/MS-LTR/Amazon) used by the later rungs, so it has **no separately logged number on that
suite** — its contribution is the algorithm itself, not a benchmark figure on these datasets. What it
established, stated as the paper does:

- The reweighting and vote-weight rules are computed from the *realized* weighted error εₜ each round —
  βₜ = εₜ/(1−εₜ) for the multiplicative weight update, vote weight log(1/βₜ) = log((1−εₜ)/εₜ) — with
  **no advance knowledge of the weak learner's edge** required. The paper states the update "reduces
  the probability assigned to those examples on which the hypothesis makes a good prediction and
  increases the probability of the examples on which the prediction is poor."
- The training error of the weighted-vote classifier is bounded by a product of per-round factors; with
  the update convention that down-weights correct examples, the relevant factor is Zₜ/√βₜ where
  Zₜ = (1−εₜ)βₜ + εₜ. The choice βₜ = εₜ/(1−εₜ) minimizes that factor and makes the just-used learner
  have weighted error 1/2 under the next distribution. The bound decreases geometrically as long as
  every weak learner clears the random bar (εₜ < ½ binary; εₜ < (K−1)/K for K classes, via the
  +log(K−1) term in SAMME).

Limitation carried into the next rung: the scheme is built on a 0/1 misclassification indicator and an
implicit exponential loss, so it captures the *sign* of an error but not its magnitude, and does not
generalize to arbitrary differentiable losses or to regression. The exponential-loss / additive-model
view that exposes this limitation came later (Friedman, Hastie & Tibshirani, 2000, "Additive Logistic
Regression: A Statistical View of Boosting," *Ann. Statist.* 28(2):337–407) and motivates the gradient
formulation of the next rung.
