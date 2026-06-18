I stop asking whether a statistical model is the truth. In the problems I actually face, that question is too strong. An autoregression, a Markov chain, or a Gaussian regression model is not the mechanism of nature; it is a way to speak about the data. Once I see it that way, the model-selection problem changes shape. I am not choosing the true distribution from a list. I am choosing the language in which the observed data can be said most economically.

The immediate obstacle is maximum likelihood. At a fixed order, it is exactly the right tool: it gives me the parameter value that makes the data easiest to account for inside that order. Across orders, it breaks. A richer nested model can always imitate the poorer one and then use the extra degree of freedom to improve the fitted likelihood. If I judge only by fitted likelihood, I always select the largest model I allow. The fit term has no built-in memory that the larger language itself has become harder to specify.

So the missing principle must make the model pay for being available. If a receiver is going to decode my data with the help of a fitted model, the receiver must know which fitted model I used. I cannot use a codebook privately and charge only for the encoded data. A complete message has two parts: first I identify the model, then I encode the data with that model. The total cost is therefore

`L(model) + L(data | model)`.

This is the point at which an old qualitative principle becomes operational. Simplicity is no longer an aesthetic preference. It is the number of bits I must spend before the data part can even be decoded. A very rich model can make the data part short, but only after I have paid to describe the rich model. A very poor model is cheap to name, but it leaves a long data part. The right model is wherever the total message is shortest.

Now I need both terms in one currency. Shannon gives me that. If a distribution assigns probability `P(x)` to an outcome, an ideal prefix code spends `-log P(x)` units on that outcome, up to the usual bounded integer-code overhead. I keep the logarithm base fixed throughout; base 2 gives bits, natural logarithms give nats, and the minimizing rule is unchanged as long as both terms use the same base. Conversely, prefix code lengths satisfy the Kraft inequality, so code lengths define probability weights. Probability and code length are two faces of the same budget. This means that, once I have fixed a model `P(. | theta)`, the cost of sending the data through that model is just the negative log likelihood:

`L(data | theta) = -log P(data | theta) + O(1)`.

Likelihood is not discarded. It reappears as the data-encoding part of the message. What likelihood omitted was the first part, the cost of telling the receiver which fitted distribution to use.

If the model were one of finitely many fully specified distributions, the rest would be straightforward: give each candidate a decodable name, add that name length to its negative log likelihood, and pick the shortest total. But statistical models usually have real-valued parameters, and a real number cannot be sent exactly in a finite message. This is the conceptual obstacle. The complexity of a parametric model is not just the number of parameters; it is the number of bits needed to specify those parameters to a useful precision.

I therefore have to choose the precision, and the data themselves tell me how fine it should be. Suppose an order-`k` model has maximum-likelihood estimate `theta_hat`. I quantize each coordinate to a grid with spacing `delta` and transmit the grid point. Finer grids cost more to name. Per coordinate, the naming cost is about `log(1/delta)`. But if I use a rounded parameter instead of `theta_hat`, the data encoding gets worse. Near `theta_hat`, the first derivative of the negative log likelihood vanishes and the second derivative controls the loss:

`-log P(data | theta) ~= -log P(data | theta_hat) + 0.5 (theta - theta_hat)^T H (theta - theta_hat)`.

The Hessian `H` is observed information. It grows like the sample size `n`, because the log likelihood is a sum over observations. A rounding error of size `delta` therefore costs on the order of `n delta^2` in fit. Per coordinate, the tradeoff has the form

`log(1/delta) + constant * n * delta^2`.

If `delta` is too small, I waste model bits distinguishing parameters the data cannot distinguish. If `delta` is too large, I save a few model bits but lose too much in fitted likelihood. Minimizing the tradeoff gives `delta` of order `1/sqrt(n)`. That scale is not arbitrary. It is the statistical resolution of a regular parameter estimate: the likelihood peak narrows like `1/sqrt(n)` as observations accumulate.

At that resolution, the cost to name one real parameter is about `log sqrt(n)`, namely `(1/2) log n`, plus constants depending on curvature, parameter range, and coding convention. For `k` regular real parameters, the leading model cost is therefore

`(k/2) log n`.

The total leading description length for the order-`k` fitted model is

`-log P(data | theta_hat_k) + (k/2) log n + O(1)`.

Now the rule is visible. I fit each candidate order by maximum likelihood, because that makes the data part shortest within the order. Then I add the cost of specifying the fitted parameters to the precision the data justify. An extra parameter is worth adding only if it shortens the data encoding by more than its half-log-`n` model cost. The selected order and the fitted coefficients are the components of the shortest complete message.

This also explains why the move is not a mere splice of earlier ideas. Shannon gives me the identity between probability and code length, but only after a source distribution is given; he does not solve model choice. Kolmogorov and Solomonoff give me the deep intuition that regularity is compressibility, but the universal shortest program is not a computable finite-sample statistical criterion. Akaike gives me an information-theoretic correction to maximum likelihood, but his penalty is a constant per parameter and comes from expected predictive bias relative to a true distribution. The present move is to restrict the shortest-description idea to statistical model classes and then make the model itself part of the decodable message.

That restriction is the bridge. It keeps the compression interpretation, but avoids the uncomputable universal search. It keeps likelihood, but reinterprets it as only the second part of a code. It keeps an Occam penalty, but derives its sample-size dependence from the precision at which parameters can be transmitted without wasting bits. The model is not rewarded for being true. It is rewarded for earning back, in data compression, the cost of saying which language it uses.

On the usual twice-negative-log scale, the same leading expression becomes

`-2 log P(data | theta_hat_k) + k log n + O(1)`.

That is why the result resembles the Bayesian dimension penalty found by Laplace approximation: both see that a `k`-dimensional likelihood peak has width about `n^{-k/2}`. But the interpretation here is different. I do not integrate likelihood against a prior to express belief in a model. I count the bits needed to describe the data through a fitted statistical language, including the bits needed to identify that language at the resolution the data can support.
