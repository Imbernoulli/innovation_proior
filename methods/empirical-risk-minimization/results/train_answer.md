The problem is to choose a rule from a fixed class after seeing a finite sample, so that the chosen rule performs well on future data drawn from the same unknown distribution. For any single rule fixed before the data are observed, the sample average loss is a reliable estimate of its true expected loss by the law of large numbers and standard concentration. But the rule we actually use is not fixed in advance. It is selected because it looks best on the realized sample, which biases it toward unusually low sample loss. A guarantee for one preselected rule therefore does not cover the rule we care about. The gap we must control is not pointwise but uniform over the whole class: the largest deviation between true risk and empirical risk among all candidate rules. If that uniform gap is small, then the empirical minimizer inherits a bound on its excess risk relative to the best rule in the class.

A naive appeal to training loss alone fails when the class is rich enough to memorize the sample, because then a rule can have zero training error while its true error remains large. A finite-class union bound handles selection correctly but does not extend to infinite classes such as linear separators or neural networks. What saves the argument is the observation that on a finite sample of size n, an infinite class can only exhibit finitely many label patterns. The effective complexity is measured by the number of dichotomies the class can realize on n points, or more coarsely by its VC dimension. When this effective complexity grows polynomially rather than exponentially, uniform convergence holds and empirical minimization becomes a sound principle. This is empirical risk minimization.

The method is called Empirical Risk Minimization. Given a sample, a loss function, and a fixed hypothesis class, one simply chooses the hypothesis with the smallest average loss on the sample. The theoretical guarantee comes from a uniform bound on the deviation between true and empirical risk over the class. For binary classification with a class of finite VC dimension h, Vapnik's bound states that with probability at least 1 − δ, every hypothesis f satisfies R(f) ≤ R_emp(f) + c0(n, h, δ), where c0 is proportional to sqrt((h log(n/h) − log δ)/n). Because the bound holds uniformly, it applies to the empirical minimizer even though that minimizer is chosen after seeing the data. The excess risk of the chosen rule relative to the best rule in the class is then at most twice the uniform gap.

The practical prescription is therefore to fix the hypothesis class before looking at the data, minimize the empirical risk within that class, and rely on a capacity control such as VC dimension, Rademacher complexity, or a growth function to ensure that the class-wide risk gap is small. If the class is too small, the best rule inside it may still be inaccurate; if the class is too large, the uniform gap can be large and low empirical risk may mean memorization. The art is to choose a class whose capacity matches the available data.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def empirical_risk_minimizer(X_train, y_train, X_test, y_test, learning_rate=0.1, steps=1000):
    """Minimal linear ERM for binary classification with a VC-style generalization bound."""
    n, d = X_train.shape
    # Add bias term.
    X = np.concatenate([X_train, np.ones((n, 1))], axis=1)
    X_te = np.concatenate([X_test, np.ones((X_test.shape[0], 1))], axis=1)
    w = np.zeros(d + 1)

    # Minimize empirical logistic risk with gradient descent.
    for _ in range(steps):
        z = y_train * (X @ w)
        p = 1.0 / (1.0 + np.exp(np.clip(z, -500, 500)))
        grad = -(1.0 / n) * (X.T @ (y_train * p))
        w -= learning_rate * grad

    # Empirical risk (logistic loss).
    def logistic_loss(Xb, yb, w):
        z = yb * (Xb @ w)
        return np.mean(np.log(1 + np.exp(-np.clip(z, -500, 500))))

    emp_risk = logistic_loss(X, y_train, w)
    test_risk = logistic_loss(X_te, y_test, w)

    # Linear classifiers in R^d have VC dimension d + 1.
    h = d + 1
    delta = 0.05
    c0 = np.sqrt((h * (np.log(2 * n / h) + 1) - np.log(delta)) / n)
    bound = emp_risk + c0

    return {
        "weights": w,
        "empirical_risk": emp_risk,
        "test_risk": test_risk,
        "uniform_bound": bound,
        "vc_dimension": h,
    }


if __name__ == "__main__":
    X, y = make_classification(n_samples=500, n_features=10, n_informative=8,
                               n_redundant=2, n_classes=2, random_state=0)
    y = 2 * y - 1  # map to {-1, +1}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    result = empirical_risk_minimizer(X_train, y_train, X_test, y_test)
    print(f"Empirical risk: {result['empirical_risk']:.4f}")
    print(f"Test risk:      {result['test_risk']:.4f}")
    print(f"VC dimension:   {result['vc_dimension']}")
    print(f"Uniform bound:  {result['uniform_bound']:.4f}")
```
