Fitting one model family is a solved routine: write down a likelihood, choose a prior or regularizer, find the most plausible parameters, and use the curvature of the posterior for error bars. The harder problem begins when several families have all been fitted and must be ranked. A polynomial of degree ten can mimic a polynomial of degree three and then spend its extra coefficients chasing noise; a dense radial-basis expansion can reproduce a sparse one and wiggle between the data points. Because richer models can always match or improve the best attainable training fit, ranking by maximum likelihood is a ratchet that pushes toward over-parameterization. A usable comparison rule must reward fit, but it must also make a model pay for freedom that the data do not require.

The usual repairs attach a complexity cost from outside the inference. Penalized-likelihood criteria add a term proportional to the number of parameters, sometimes with a sample-size correction; description-length methods need an arbitrary choice of parameter precision; validation methods are noisy when data are scarce and do not explain how to set a regularization strength from the fitted posterior. These shortcuts are not wrong, but they ignore the actual geometry of the posterior and the width of the prior. The complexity charge should come from the model's own predictions, not from a rule invented after the fact.

The method is the Bayesian Occam factor. It ranks models by the model evidence, the normalizing constant of the parameter posterior:

P(D | H) = ∫ P(D | w, H) P(w | H) dw.

This is the probability the whole model assigns to the observed data before any particular parameter value is selected. A simple model concentrates its predictive mass on a narrow region of data space; a flexible model spreads the same total mass over many more possible data sets. Even if the flexible model can fit the observed data after tuning, it assigned that data set less probability beforehand because it had to reserve probability for all the other data sets it could explain. The evidence therefore contains a built-in razor against unused flexibility.

When the posterior is peaked at the most probable parameters w_MP, Laplace's method gives an intuitive decomposition:

P(D | H) ≈ P(D | w_MP, H) × Occam factor,

where the Occam factor is P(w_MP | H) (2π)^(k/2) |A|^(-1/2) and A is the Hessian of the negative log posterior at w_MP. For one parameter with an approximately uniform prior width σ_w, this reduces to the ratio σ_w|D / σ_w, the fraction of the prior volume that remains plausible after seeing the data. The Occam factor is not a penalty bolted on from outside; it is the width of the integral that best-fit methods ignore.

In the standard quadratic interpolation case, with Gaussian noise precision β and a Gaussian weight prior precision α, the evidence for fixed hyperparameters is the ratio of normalizing constants P(D | α, β, H) = Z_M / (Z_w Z_n). Maximizing this evidence yields the effective number of parameters γ = Σ_a λ_a / (λ_a + α), where λ_a are the eigenvalues of β Φ^T Φ. The stationary conditions become 2α E_w^MP = γ and 2β E_n^MP = N − γ, where E_w and E_n are the regularizer and data-misfit energies at w_MP. These equations set the regularization strength and the noise scale together: well-measured parameters each explain about one unit of noise, while poorly measured parameters are controlled by the prior and do not count. Whole model families are then ranked by integrating the evidence over their hyperparameter priors; when that surface is peaked, the integral is again well approximated by the height times the width in log α and log β. The result is the characteristic Occam trade-off: too-simple models lose by data misfit, and over-flexible models lose by unused parameter volume.

```python
import numpy as np
from numpy.linalg import slogdet, solve

def polynomial_basis(x, degree):
    return np.vander(x, degree + 1, increasing=True)

def fit_evidence(Phi, t, alpha_init=1.0, beta_init=1.0, max_iter=100, tol=1e-6):
    """Type-II ML estimate of alpha, beta and log evidence for a linear-Gaussian model."""
    N, k = Phi.shape
    alpha, beta = alpha_init, beta_init
    for _ in range(max_iter):
        A = alpha * np.eye(k) + beta * Phi.T @ Phi
        w = solve(A, beta * Phi.T @ t)
        Ew = 0.5 * np.dot(w, w)
        En = 0.5 * np.sum((Phi @ w - t) ** 2)
        eigenvalues = np.linalg.eigvalsh(beta * Phi.T @ Phi)
        gamma = np.sum(eigenvalues / (eigenvalues + alpha))
        alpha_new = gamma / (2.0 * Ew)
        beta_new = (N - gamma) / (2.0 * En)
        if max(abs(alpha_new - alpha), abs(beta_new - beta)) < tol:
            alpha, beta = alpha_new, beta_new
            break
        alpha, beta = alpha_new, beta_new
    A = alpha * np.eye(k) + beta * Phi.T @ Phi
    w = solve(A, beta * Phi.T @ t)
    Ew = 0.5 * np.dot(w, w)
    En = 0.5 * np.sum((Phi @ w - t) ** 2)
    sign, logdetA = slogdet(A)
    assert sign > 0
    log_ev = (
        -alpha * Ew
        - beta * En
        - 0.5 * logdetA
        + 0.5 * k * np.log(alpha)
        + 0.5 * N * np.log(beta)
        - 0.5 * N * np.log(2.0 * np.pi)
    )
    return w, alpha, beta, log_ev

# Synthetic noisy interpolation data
gen_degree = 3
x = np.linspace(-1, 1, 30)
true_w = np.array([0.0, -1.0, 0.5, 0.2])
Phi_true = polynomial_basis(x, gen_degree)
t = Phi_true @ true_w + np.random.normal(scale=0.2, size=x.shape)

# Compare polynomial degrees by Bayesian Occam factor
for deg in range(1, 9):
    Phi = polynomial_basis(x, deg)
    _, alpha, beta, log_ev = fit_evidence(Phi, t)
    print(f"degree {deg}: log_ev={log_ev:.3f}, alpha={alpha:.3f}, beta={beta:.3f}")
```
