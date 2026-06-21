I begin by observing that the central problem of supervised learning is not merely to fit the training sample, but to choose a predictor whose expected loss on future draws from the unknown distribution remains small. The expected risk is the integral of the loss against the true data-generating measure, while the only quantity I can evaluate directly is the empirical average over the observed sample. A naive strategy would minimize the empirical risk alone, yet that ignores the fact that the hypothesis I select is itself a function of the sample. Once the data participates in the choice, a concentration statement that holds for one fixed function no longer applies automatically; I need uniform control over the entire class of candidate predictors.

This observation leads me to organize the candidate functions into a nested structure of hypothesis spaces, each equipped with a capacity measure such as a VC dimension or a comparable uniform-convergence guarantee. I denote the structure by S_1 subset S_2 subset ... subset S_n, with corresponding capacities h_1 < h_2 < ... < h_n. Inside each space I solve ordinary empirical risk minimization, producing a candidate alpha_k. The decisive step is not to keep the candidate with the smallest training error, because that candidate typically comes from the richest and most over-capacity class. Instead I choose the level that minimizes a finite-sample upper bound on the true risk, namely the empirical risk of the level's best candidate plus a confidence term that grows with the capacity of the level and shrinks with the sample size.

For binary classification, Vapnik's classical VC bound supplies the additive confidence term Omega_0(h, l, eta) = sqrt((h(ln(2l/h)+1) - ln eta) / l), so that with probability at least 1 - eta the true error of any predictor in a class of VC dimension h is bounded by its training error plus Omega_0. Applying this logic level by level gives the structural risk minimization principle: select k* = argmin_k [R_emp(alpha_k) + Omega(h_k, l, eta_k)], and return alpha_{k*}. The selected predictor therefore trades the approximation error of a small class against the estimation error of a large class.

When the structure contains countably many classes, I must ensure that all bounds hold simultaneously. I assign prior weights w(k) with sum_k w(k) <= 1 and set eta_k = w(k) eta for each level. The union bound then guarantees that with probability at least 1 - eta, every level satisfies its own bound, so the selected level inherits an oracle-style guarantee against the best predictor in each candidate class plus that class's confidence penalty.

The principle is not merely a penalty on parameter count. Capacity measures such as VC dimension reflect the class's ability to realize dichotomies, not the number of scalar parameters. A concrete illustration is the margin bound for hyperplane classifiers: if inputs lie in a ball of radius R and a hyperplane separates them with margin Delta, the VC dimension is bounded by min(R^2 / Delta^2, n) + 1. Maximizing the margin therefore reduces capacity. A hard-margin support vector machine in the separable case keeps empirical error zero while selecting the separator from the smallest-capacity margin class compatible with the data; soft-margin and kernel variants preserve the same tradeoff between empirical violations and capacity control.

The canonical name of the method is Structural Risk Minimization, and its essential message is that model selection should be driven by a finite-sample guarantee rather than by training loss alone. The following Python script illustrates the idea on a one-dimensional regression problem: it fits polynomials of increasing degree, computes an empirical mean-squared-error plus a capacity penalty, and selects the degree with the smallest structural risk bound.

```python
import numpy as np

np.random.seed(0)
n_samples = 80
x = np.sort(np.random.uniform(-1, 1, n_samples))
y_true = np.sin(np.pi * x)
y = y_true + np.random.normal(scale=0.2, size=n_samples)


def fit_poly(degree, x, y):
    return np.polyfit(x, y, degree)


def empirical_mse(coef, x, y):
    pred = np.polyval(coef, x)
    return np.mean((y - pred) ** 2)


def srm_score(mse, degree, n, eta=0.05):
    # Use degree + 1 as a simple capacity proxy and a small union-bound weight.
    h = degree + 1
    penalty = np.sqrt((h * (np.log(2 * n / h) + 1) - np.log(eta)) / n)
    return mse + penalty


best_degree = None
best_score = float("inf")
print("degree  train_mse  srm_score")
for d in range(0, 12):
    coef = fit_poly(d, x, y)
    mse = empirical_mse(coef, x, y)
    score = srm_score(mse, d, n_samples)
    print(f"{d:3d}    {mse:.4f}     {score:.4f}")
    if score < best_score:
        best_score = score
        best_degree = d

print(f"\nSelected degree by structural risk minimization: {best_degree}")
```
