# Minimum Description Length

## Core Move

Rissanen's distinctive move is to treat a statistical model as a code or language for the data, not as a claim that the model is literally true. Model selection then asks for the complete decodable description with the smallest total length. A complete description must include both:

`L(model) + L(data | model)`.

This turns Occam's razor into an inference rule. A complex model can fit the data well, but it must be transmitted before it can be used. A simple model is cheap to transmit, but may leave the data expensive to encode. The selected model is the one that minimizes the total.

## Why Likelihood Appears

Shannon coding supplies the shared unit. For a distribution `P`, the ideal code length of an outcome `x` is `-log P(x)` up to bounded coding constants. Logs must be kept in one base throughout; base 2 reads as bits, while natural logs read as nats and give the same minimizer after rescaling. Thus, once a fitted statistical model `P(. | theta)` is known, the cost of encoding the data through it is:

`L(data | theta) = -log P(data | theta) + O(1)`.

Likelihood is therefore the data-code part of the message. Maximum likelihood remains the right fixed-order estimator because it minimizes this part, but it cannot select order because it omits the model-code part.

## Continuous Parameter Cost

The main technical obstacle is that real-valued parameters cannot be transmitted exactly. They must be quantized. If a parameter is encoded on a grid with spacing `delta`, the model-description part costs about `log(1/delta)` per coordinate. Rounding away from the maximum-likelihood estimate worsens the negative log likelihood by a quadratic term. Since likelihood curvature grows like sample size `n`, the fit loss is of order `n delta^2`.

Balancing these terms gives an optimal precision of order `1/sqrt(n)`. One real parameter therefore costs about `log sqrt(n) = (1/2) log n` to transmit, up to lower-order constants. For `k` regular parameters, the leading model cost is:

`(k/2) log n`.

## Criterion

For an order-`k` regular parametric model with maximum-likelihood estimate `theta_hat_k`, the leading two-part description length is:

`L(data, k) = -log P(data | theta_hat_k) + (k/2) log n + O(1)`.

Equivalently, on the usual deviance scale:

`-2 log P(data | theta_hat_k) + k log n + O(1)`.

Select the order `k` and fitted parameters `theta_hat_k` that minimize this expression. An extra parameter is accepted only when the improvement in data encoding exceeds its model-description cost.

## Why This Is Not Just Recombination

Shannon provides the code-length/probability identity, but assumes the source distribution is already given. Kolmogorov and Solomonoff provide the idea that regularity is compressibility, but universal shortest-program induction is not a computable finite-sample model-selection rule. Akaike supplies an information-theoretic likelihood correction, but his penalty is constant in `n` and is derived from expected predictive bias relative to a true distribution.

The new step is the restricted, decodable coding formulation for statistical model classes: use likelihood as the data-code term, make the fitted model itself pay its transmission cost, and derive the sample-size-dependent penalty from the finite precision at which real parameters can be specified. This makes the shortest-description principle an operational model-selection and inference rule.
