# Context: choosing a model from data

## Research question

Given a finite stretch of observed data — a time series, a regression table, the transition counts of a sequence — and a *family* of candidate models indexed by an integer structure parameter (the order of an autoregression, the degree of a polynomial, the number of states), how do we choose which model to fit?

The hard part is not estimating the real-valued coefficients once the structure is fixed; least squares or maximum likelihood does that. The hard part is choosing the **structure**: how many parameters. Maximum likelihood is no help here, because likelihood is monotone in model size — a higher-order model contains the lower-order ones as special cases, so its best fit can only be better. Push the order up and the fitted log-likelihood keeps improving until the model interpolates the data exactly, having learned the noise. The data themselves, scored by their own likelihood, vote for the most complex model on offer, which is the one that generalizes worst.

So a usable criterion must do something maximum likelihood cannot: it must make complexity *cost* something, and it must trade that cost against fit on a common scale, with no free tuning knob that lets the answer be argued either way. A solution would have to (i) define what "fit" and "complexity" mean in one shared unit, (ii) make the complexity charge grow correctly with the amount of data, and (iii) yield, as a by-product, both the integer structure and the real coefficients — all without assuming we know the true data-generating distribution, which in practice we never do.

## Background

**Occam's razor, unquantified.** The intuition that among hypotheses compatible with the evidence one should prefer the simplest is ancient, but "simplest" is qualitative. It gives no procedure: simpler in what units, and how much worse a fit is one willing to tolerate to buy how much simplicity? Any operational model-selection rule has to turn this slogan into a number.

**Shannon source coding and the code-length / probability correspondence.** Information theory supplies the unit. For a source emitting symbols with probabilities P, the shortest achievable expected code length is the entropy H(P), and an ideal code spends −log₂ P(x) bits on symbol x. More sharply, the Kraft inequality says that lengths ℓ(z) are prefix-decodable exactly when ∑z 2^−ℓ(z) ≤ 1. Given P, the ideal lengths ℓ(z)=−log₂ P(z) satisfy Kraft equality; if integer codewords are required, ⌈−log₂ P(z)⌉ loses less than one bit per item, and block or arithmetic coding pushes that overhead into an O(1) term. Conversely, any prefix code defines a subprobability q(z)=2^−ℓ(z), so probability and code length differ only by the same constant-completion issue. Large probability ⇔ short code: the cost of recording an outcome under an assumed distribution is its negative log-probability, up to the harmless integer-code constant. Shannon's setting, though, assumes the source distribution is *given*; it offers a unit for measuring description length but not a rule for picking the distribution.

**Kolmogorov complexity and Solomonoff's inductive inference.** In the 1960s Solomonoff (1964), Kolmogorov (1965) and Chaitin (1966, 1969) independently defined the complexity of a string as the length of the shortest program, in a fixed universal language, that prints it and halts. The invariance theorem says this length is language-independent up to an additive constant for long enough strings. Solomonoff's *A Theory of Inductive Inference* proposed that the best explanation of data is the shortest program generating it — regularity is compressibility, and learning is finding the shortest description. This is exactly Occam made quantitative. Its defect for practice is fatal in two ways: the shortest program is **uncomputable** (no algorithm finds it for arbitrary data), and for the short data sets one actually has, the additive-constant slack of the invariance theorem swamps everything, so the answer depends on arbitrary syntax. The idea is compelling, but in this universal form it is unusable for the finite data sets and concrete model families a statistician actually confronts.

**Algorithmic randomness.** The work of Kolmogorov, Chaitin and Martin-Löf on algorithmic randomness — that a random string is one with no description shorter than itself — makes the view that regularity is what can be compressed concrete and precise.

**The diagnostic fact about likelihood.** It is well documented, and immediate from the nesting of model families, that the maximized log-likelihood is non-decreasing in the number of parameters: adding a coefficient never worsens the best attainable fit, and generically strictly improves it by an amount that does not vanish. Plotting fitted log-likelihood against model order gives a curve that keeps rising; there is no internal signal in the likelihood that says "stop here." How a per-parameter charge ought to depend on the sample size n, if it is to oppose this drift without being an arbitrary knob, is the quantitative crux left open.

## Baselines

**Maximum likelihood / least squares with fixed order.** Estimate θ by maximizing P(D|θ); for Gaussian noise this is least squares. Core math: θ̂ = argmax_θ P(D|θ), and −log P(D|θ̂) is the residual "fit" cost. Gap: gives no way to choose the order — applied across orders it always prefers the largest, i.e. it overfits. It solves the real-parameter problem and leaves the integer-structure problem wide open.

**Akaike's AIC (1973).** The first model-selection rule built from information theory. Akaike estimates the expected Kullback–Leibler divergence between the fitted model and the truth and shows the maximized log-likelihood is an optimistically biased estimate of out-of-sample fit, the bias being approximately k (the number of parameters). This yields: choose the model minimizing −2 log P(D|θ̂) + 2k. Core idea/math: penalty = 2k on the deviance scale, independent of sample size n. Gap: the penalty does not grow with n, so in nested settings where extra parameters are unnecessary the O(1) likelihood-ratio gains from noise still beat the constant charge with non-vanishing probability; AIC is not consistent. And the penalty is derived from an expected-KL/bias argument that presumes a true model, which the data-only criterion sought here wants to avoid assuming.

**Wallace–Boulton Minimum Message Length (1968).** Predates the present line by a decade. Choose the hypothesis minimizing the total length of a two-part message that first states a hypothesis and then states the data given it. Core math: minimize L(H) + L(D|H) with both parts in bits, where L(H) is computed from a **prior** over hypotheses, −log w(H), discretized to an optimal precision. Gap (relative to the goal of a non-Bayesian, data-only criterion): MML is explicitly Bayesian — it requires committing to a prior distribution over hypotheses and interprets the message length through that prior as a degree of belief. A criterion that wants to make no claim that the data were "generated" by any distribution, and to depend on the data alone, cannot rest on such a prior, leaving open where its hypothesis-description cost would instead come from.

**Schwarz's BIC (1978, concurrent).** From a Bayesian large-sample approximation of the marginal likelihood (Laplace's method on the posterior), Schwarz derives the rule: minimize −2 log P(D|θ̂) + k log n. Core math: the penalty grows with n, unlike AIC. Gap as background: it arrives via a Bayesian motivation (approximating the evidence ∫P(D|θ)w(θ)dθ) and only retains leading terms, dropping the O(1) dependence on priors, parameter volume, and the model's functional form; it is a by-product of an integral approximation that presumes a prior over θ, and offers no account of the rule in data-only terms that make no such Bayesian commitment.

## Evaluation settings

The natural proving grounds are the classical model-order problems where the integer structure is the whole question:

- **Autoregressive / time-series order selection.** Data: a scalar or vector time series x₁,…,x_n. Family: AR(k) (or ARMA) models for k = 0,1,2,…. Quantity to choose: the order k and the coefficients. Metric: the chosen score value as a function of k; out-of-sample one-step prediction error as a sanity check.
- **Markov-chain order / variable-order sources.** Data: a discrete sequence; sufficient statistics are the transition counts n[symbol | context]. Family: k-th order Markov models, with 2^k (or m^k) free transition probabilities. Choose the order k.
- **Polynomial-degree / subset regression.** Data: pairs (xᵢ, yᵢ) with assumed Gaussian noise Y = h(X) + Z. Family: polynomials of degree k, or subsets of candidate regressors. Choose the degree / which regressors enter. Fit measured through the residual sum of squares, which under Gaussian noise equals the negative log-likelihood up to constants.

These data types, the nested model families, and the residual-sum-of-squares / log-likelihood scoring all pre-exist; the yardstick is whether a criterion recovers a sensible order rather than the maximal one, and predicts unseen data well.

## Code framework

The available scaffold is a likelihood evaluator, maximum-likelihood / least-squares fitting at a fixed order, and a loop over candidate orders. The unresolved function is whatever extra term lets an order be scored without trusting fitted likelihood alone — and how that term should depend on the order k and the sample size n.

```python
import numpy as np

def neg_log_likelihood_bits(data, model):
    """Bits to encode data under a fully specified model (a single distribution).
    If model.log_prob is a natural log, divide by log(2) to put the fit in bits."""
    return -model.log_prob(data) / np.log(2.0)

def fit_mle(data, k):
    """Maximum-likelihood / least-squares fit of the k-parameter model to the data.
    Returns the fitted point hypothesis theta_hat (the reals), given the order k."""
    ...  # standard estimation at fixed order; this is solved

def complexity_term(theta_hat, k, n):
    """The unresolved extra term that charges for the order-k fit, in the same
    units as the fit cost. Its dependence on k and n is what has to be pinned down."""
    # TODO: what is this term, and how should it scale with k and n?
    pass

def score_candidate(data, k):
    """Score a candidate order k by a single number that trades fit against
    something that opposes overfitting."""
    theta_hat = fit_mle(data, k)
    fit_bits = neg_log_likelihood_bits(data, theta_hat_model(theta_hat))
    # TODO: combine fit_bits with the unresolved complexity term.
    pass

def select_order(data, k_max):
    """Choose the order by minimizing the score over candidate orders.
    Maximum likelihood alone can't do this (it monotonically prefers k_max);
    the score must include the missing complexity term."""
    return min(range(k_max + 1), key=lambda k: score_candidate(data, k))
```
