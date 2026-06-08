# Context: choosing a model from data

## Research question

Given a finite stretch of observed data — a time series, a regression table, the transition counts of a sequence — and a *family* of candidate models indexed by an integer structure parameter (the order of an autoregression, the degree of a polynomial, the number of states), how do we choose which model to fit?

The hard part is not estimating the real-valued coefficients once the structure is fixed; least squares or maximum likelihood does that. The hard part is choosing the **structure**: how many parameters. Maximum likelihood is no help here, because likelihood is monotone in model size — a higher-order model contains the lower-order ones as special cases, so its best fit can only be better. Push the order up and the fitted log-likelihood keeps improving until the model interpolates the data exactly, having learned the noise. The data themselves, scored by their own likelihood, vote for the most complex model on offer, which is the one that generalizes worst.

So a usable criterion must do something maximum likelihood cannot: it must make complexity *cost* something, and it must trade that cost against fit on a common scale, with no free tuning knob that lets the answer be argued either way. A solution would have to (i) define what "fit" and "complexity" mean in one shared unit, (ii) make the complexity charge grow correctly with the amount of data, and (iii) yield, as a by-product, both the integer structure and the real coefficients — all without assuming we know the true data-generating distribution, which in practice we never do.

## Background

**Occam's razor, unquantified.** The intuition that among hypotheses compatible with the evidence one should prefer the simplest is ancient, but "simplest" is qualitative. It gives no procedure: simpler in what units, and how much worse a fit is one willing to tolerate to buy how much simplicity? Any operational model-selection rule has to turn this slogan into a number.

**Shannon source coding and the code-length / probability correspondence.** Information theory supplies the unit. For a source emitting symbols with probabilities P, the shortest achievable expected code length is the entropy H(P), and an ideal code spends −log₂ P(x) bits on symbol x. More sharply, the Kraft inequality says that lengths ℓ(z) are prefix-decodable exactly when ∑z 2^−ℓ(z) ≤ 1. Given P, the ideal lengths ℓ(z)=−log₂ P(z) satisfy Kraft equality; if integer codewords are required, ⌈−log₂ P(z)⌉ loses less than one bit per item, and block or arithmetic coding pushes that overhead into an O(1) term. Conversely, any prefix code defines a subprobability q(z)=2^−ℓ(z), so probability and code length differ only by the same constant-completion issue. Large probability ⇔ short code. This is the bridge that lets "how well a model fits" (a likelihood) and "how complex a model is" (a description) be measured in the *same* currency, bits. The cost of recording an outcome under an assumed distribution is its negative log-probability, up to the harmless integer-code constant. Shannon's setting, though, assumes the source distribution is *given*; it offers the unit but not a rule for picking the distribution.

**Kolmogorov complexity and Solomonoff's inductive inference.** In the 1960s Solomonoff (1964), Kolmogorov (1965) and Chaitin (1966, 1969) independently defined the complexity of a string as the length of the shortest program, in a fixed universal language, that prints it and halts. The invariance theorem says this length is language-independent up to an additive constant for long enough strings. Solomonoff's *A Theory of Inductive Inference* proposed that the best explanation of data is the shortest program generating it — regularity is compressibility, and learning is finding the shortest description. This is exactly Occam made quantitative. Its defect for practice is fatal in two ways: the shortest program is **uncomputable** (no algorithm finds it for arbitrary data), and for the short data sets one actually has, the additive-constant slack of the invariance theorem swamps everything, so the answer depends on arbitrary syntax. The idea is right; it must be *scaled down* to a restricted, computable description method before it can be used.

**Algorithmic randomness as the trigger.** The work of Kolmogorov, Chaitin and Martin-Löf on algorithmic randomness — that a random string is one with no description shorter than itself — makes the compression view of regularity concrete and is the immediate intellectual spark for a coding-based theory of statistical modeling.

**The diagnostic fact about likelihood.** It is well documented, and immediate from the nesting of model families, that the maximized log-likelihood is non-decreasing in the number of parameters: adding a coefficient never worsens the best attainable fit, and generically strictly improves it by an amount that does not vanish. Plotting fitted log-likelihood against model order gives a curve that keeps rising; there is no internal signal in the likelihood that says "stop here." Any honest accounting must add a term that rises with model size to oppose it — and the rate at which that term should rise (per parameter, as a function of sample size n) is the quantitative crux.

## Baselines

**Maximum likelihood / least squares with fixed order.** Estimate θ by maximizing P(D|θ); for Gaussian noise this is least squares. Core math: θ̂ = argmax_θ P(D|θ), and −log P(D|θ̂) is the residual "fit" cost. Gap: gives no way to choose the order — applied across orders it always prefers the largest, i.e. it overfits. It solves the real-parameter problem and leaves the integer-structure problem wide open.

**Akaike's AIC (1973).** The first model-selection rule built from information theory. Akaike estimates the expected Kullback–Leibler divergence between the fitted model and the truth and shows the maximized log-likelihood is an optimistically biased estimate of out-of-sample fit, the bias being approximately k (the number of parameters). This yields: choose the model minimizing −2 log P(D|θ̂) + 2k; on the −log code-length scale, that is −log P(D|θ̂) + k. Core idea/math: penalty = 2k on the deviance scale, independent of sample size n. Gap: the penalty does not grow with n, so in nested settings where extra parameters are unnecessary the O(1) likelihood-ratio gains from noise still beat the constant charge with non-vanishing probability; AIC is not consistent. And the penalty is derived from an expected-KL/bias argument about a presumed true model, not from a coding account of what it costs to write the model down.

**Wallace–Boulton Minimum Message Length (1968).** Predates the present line by a decade and shares the central instinct: choose the hypothesis minimizing the total length of a two-part message that first states a hypothesis and then states the data given it. Core math: minimize L(H) + L(D|H) with both parts in bits, where L(H) is computed from a **prior** over hypotheses, −log w(H), discretized to an optimal precision. Gap (relative to the goal of a non-Bayesian, data-only criterion): MML is explicitly Bayesian — it requires committing to a prior distribution over hypotheses and interprets the message length through that prior as a degree of belief. A criterion that wants to make no claim that the data were "generated" by any distribution, and to depend on the data alone, must derive the model-description cost from coding considerations rather than from a prior.

**Schwarz's BIC (1978, concurrent).** From a Bayesian large-sample approximation of the marginal likelihood (Laplace's method on the posterior), Schwarz derives the rule: minimize −2 log P(D|θ̂) + k log n; on the −log code-length scale, that is −log P(D|θ̂) + (k/2)log n. Core math: the penalty grows with n, unlike AIC. Gap as background: it arrives via a Bayesian motivation (approximating the evidence ∫P(D|θ)w(θ)dθ) and only retains leading terms, dropping the O(1) dependence on priors, parameter volume, and the model's functional form; it is a by-product of an integral, not a coding account of the model-transmission cost. That the same (k/2)log n falls out of a different coding argument is exactly the convergence to be explained.

## Evaluation settings

The natural proving grounds are the classical model-order problems where the integer structure is the whole question:

- **Autoregressive / time-series order selection.** Data: a scalar or vector time series x₁,…,x_n. Family: AR(k) (or ARMA) models for k = 0,1,2,…. Quantity to choose: the order k and the coefficients. Metric: the chosen score value as a function of k; out-of-sample one-step prediction error as a sanity check.
- **Markov-chain order / variable-order sources.** Data: a discrete sequence; sufficient statistics are the transition counts n[symbol | context]. Family: k-th order Markov models, with 2^k (or m^k) free transition probabilities. Choose the order k.
- **Polynomial-degree / subset regression.** Data: pairs (xᵢ, yᵢ) with assumed Gaussian noise Y = h(X) + Z. Family: polynomials of degree k, or subsets of candidate regressors. Choose the degree / which regressors enter. Fit measured through the residual sum of squares, which under Gaussian noise equals the negative log-likelihood up to constants.

These data types, the nested model families, and the residual-sum-of-squares / log-likelihood scoring all pre-exist; the yardstick is whether a criterion recovers a sensible order rather than the maximal one, and predicts unseen data well.

## Code framework

The available scaffold is a likelihood evaluator, maximum-likelihood / least-squares fitting at a fixed order, and a loop over candidate orders. The unresolved function is a complexity charge for the fitted family, expressed in the same log base as the fit cost, so an order can be scored without trusting fitted likelihood alone.

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

def model_cost_bits(theta_hat, k, n):
    """Bits to name the structure k and the fitted real parameters theta_hat.
    The scaling with k and n is the unresolved part."""
    # TODO: how many bits does it cost to write down a k-parameter model,
    #       and how should that cost scale with the sample size n?
    pass

def score_candidate(data, k):
    """Score a candidate order k by a single number that trades fit against
    complexity in one unit."""
    theta_hat = fit_mle(data, k)
    fit_bits = neg_log_likelihood_bits(data, theta_hat_model(theta_hat))
    # TODO: combine fit_bits with the unresolved model cost.
    pass

def select_order(data, k_max):
    """Choose the order by minimizing the score over candidate orders.
    Maximum likelihood alone can't do this (it monotonically prefers k_max);
    the score must include the missing model-cost term."""
    return min(range(k_max + 1), key=lambda k: score_candidate(data, k))
```
