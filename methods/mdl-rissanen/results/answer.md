# Minimum Description Length (MDL)

## Problem

Choose a model from a family indexed by an integer structure parameter (autoregression order, polynomial degree, number of Markov states) given finite data. Maximum likelihood cannot do this: because the families are nested, the maximized likelihood is non-decreasing in the number of parameters, so likelihood always prefers the largest model and overfits. A criterion is needed that trades fit against complexity in one common unit, with the complexity charge growing correctly with sample size, assuming nothing about a "true" data-generating distribution.

## Key idea

**Model selection is data compression.** The best model is the one yielding the shortest total description of the data. A decodable description has two parts — you must transmit the model before you can use it to encode the data — so the cost is

  L(model) + L(data | model),

both measured in bits. This is Occam's razor made quantitative (regularity = compressibility, after Kolmogorov/Solomonoff), with complexity penalized automatically: a richer model costs more to transmit (L(model) up) but encodes the data more cheaply (L(data|model) down); the minimum of the sum sits at a model simple enough to state cheaply yet rich enough to compress the data. It needs a restricted (parametric) description language rather than Solomonoff's universal one, which is uncomputable.

Two identities make it concrete:

1. **Code length = −log probability (Kraft / Shannon).** Ideal lengths ℓ(z)=−log₂ P(z) satisfy Kraft exactly; integer prefix lengths use ⌈−log₂ P(z)⌉ and add only bounded overhead, while any prefix code induces the subprobability q(z)=2^−ℓ(z). Thus, up to the standard O(1) coding constant, large probability ⇔ short code. A model in the family *is* a distribution P(·|θ), so the data-encoding cost is L(data|model) = −log P(D|θ) + O(1) — the negative log-likelihood up to coding constants. MDL thus recovers likelihood as the fit term and adds the model-transmission cost likelihood omits.

2. **(k/2) log n model cost (optimal precision).** The k real parameters must be quantized to be transmitted. Encoding each coordinate to grid spacing δ costs ≈ log(1/δ) code units to name; using the rounded value worsens the fit by the second-order term ≈ ½·(nI)·δ² because the negative-log-likelihood curvature at θ̂ is the observed information H ≈ nI, growing linearly in n. Minimizing f(δ)=−log δ + ½nIδ² gives f′(δ)=−1/δ+nIδ=0, hence δ* = 1/√(nI) ~ 1/√n — the resolution at which the data can still distinguish parameter values. Substituting, the naming cost is ½ log n per coordinate; the fit penalty, Fisher determinant, parameter volume, and cost of k fold into O(1). Thus L(model) ≈ (k/2) log n.

## The criterion (final form)

For a model class with k real parameters and maximum-likelihood estimate θ̂, the two-part description length of n observations is

  **L(D, k) = −log P(D | θ̂ₖ) + (k/2) log n + O(1).**

Select the order k (and the coefficients θ̂ₖ) minimizing it. Properties:

- **Automatic complexity penalty.** Each extra parameter costs +½ log n in model bits and is worth adding only if it shortens the data encoding by more than ½ log n — a stopping rule with no tuning constant.
- **Correct n-scaling.** The penalty grows like log n. AIC uses 2k on the −2 log scale, equivalently k on the −log code-length scale, so its constant charge leaves a non-vanishing over-selection probability in nested settings. Here the charge is (k/2)log n on the −log scale, equivalently k log n on the −2 log scale, because it is the cost of resolving a parameter to the data's own precision 1/√n.
- **Joint estimation.** Minimizing over k and θ together yields the integer structure and the real coefficients at once.
- **No assumed truth.** Derived by counting bits to describe *this* data in a chosen language; it never posits a true distribution (contrast Akaike's expected-KL bias, or the Bayesian prior of MML). Its leading two terms match Schwarz's BIC when both are put on the same scale: BIC's −2 log P(D|θ̂) + k log n is twice the leading code length above. The agreement is leading-order only; O(1) terms, priors, parameter volume, Fisher determinant, and cases where k is not small relative to n can separate the criteria.

## Reference algorithm

```python
import numpy as np

def description_length_bits(data, k, fit_mle, neg_log_likelihood_bits):
    """Two-part code length: -log2 P(D | theta_hat_k) + (k/2) log2 n.
    Term 1 (fit): data encoded through the fitted model = negative log-likelihood
    at the MLE, by the ideal Kraft-Shannon code-length = -log2 P identity.
    Term 2 (model cost): k parameters each transmitted to precision ~1/sqrt(n),
    the resolution at which data still distinguish parameters (likelihood
    curvature grows like n), costing (1/2) log2 n bits per parameter.
    """
    n = len(data)
    theta_hat = fit_mle(data, k)                       # reals: MLE at fixed order k
    fit_bits = neg_log_likelihood_bits(data, theta_hat, k)  # -log2 P(D | theta_hat)
    model_bits = 0.5 * k * np.log2(n)                  # (k/2) log2 n, optimal precision
    return fit_bits + model_bits                       # O(1) dropped

def select_order(data, k_max, fit_mle, neg_log_likelihood_bits):
    """Pick the order with the shortest total description.
    Maximum likelihood alone decreases fit_bits monotonically in k and prefers
    k_max; the (k/2) log2 n term opposes it, stopping at the order where the next
    parameter no longer pays for its half-log2-n of overhead.
    """
    scores = {k: description_length_bits(data, k, fit_mle, neg_log_likelihood_bits)
              for k in range(k_max + 1)}
    k_star = min(scores, key=scores.get)
    return k_star, fit_mle(data, k_star)               # integer structure AND reals
```

Worked instance (Gaussian-noise regression / AR, where −log P reduces to residual sum of squares plus constants): in a fixed log base, −log P(D|θ̂ₖ) = (n/2) log σ̂²_k + const, so the criterion becomes (n/2) log σ̂²_k + (k/2) log n + const — minimize over the order k. The residual variance σ̂²_k falls with k while (k/2) log n rises; their sum is minimized at the order that best compresses the data.
