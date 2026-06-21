When I have to choose the structure of a statistical model from finite data, the hardest decision is usually the integer part, not the real-valued part. In an autoregression I need an order, in a Markov chain I need a context length, and in regression I need a number of predictors or a polynomial degree. Once that integer is fixed, ordinary maximum likelihood gives me the coefficients. The problem is that maximum likelihood cannot choose the integer itself, because within a nested family every richer model can imitate a poorer one and then use its extra freedom to improve the fit. If I judge models only by their maximized likelihood, I will always end up with the largest model I allowed. I need a way to charge for the complexity of the model on the same scale as the fit it provides.

Rissanen's Minimum Description Length criterion, which I will call MDL, solves this by treating a statistical model as a code or language for describing the data rather than as a literal claim about the true mechanism. The central move is to ask for the length of the shortest complete decodable message that a receiver could use to recover the observed data. That message has two parts. First I must tell the receiver which model I have selected and what its fitted parameters are. Then I must encode the data using that model. The total description length is the sum of these two costs, and the best model is the one that makes the total smallest.

Shannon's coding theorem gives me the common currency for both parts. If a distribution assigns probability P(x) to an observation x, then an ideal prefix code spends about -log P(x) units on that observation. Probability and code length are interchangeable as long as I keep the logarithm base fixed throughout. This means that once I have fitted a model P(. | theta), the cost of encoding the data is just the negative log likelihood. Maximum likelihood is therefore the right fixed-order estimator, because it minimizes the data-encoding part of the message. What maximum likelihood omits is the cost of naming the model itself.

The real difficulty is that the parameters in most statistical models are real numbers, and I cannot transmit a real number exactly with a finite message. I have to quantize each parameter to a grid. Suppose I transmit a parameter by naming the nearest grid point with spacing delta. Naming that grid point costs about log(1/delta) per coordinate. At the same time, rounding the parameter away from the maximum-likelihood estimate worsens the data encoding. Near the maximum-likelihood estimate, the negative log likelihood behaves like a quadratic whose curvature is the observed information matrix. Because the log likelihood is a sum over observations, that curvature grows with the sample size n. A rounding error of size delta therefore increases the data cost on the order of n delta squared.

So for each parameter there is a tradeoff: a finer grid costs more to describe but loses less in fit, while a coarser grid costs less to describe but loses more in fit. The total contribution for one coordinate looks like log(1/delta) plus a constant times n delta squared. Minimizing this gives an optimal precision proportional to one over the square root of n. At that precision the cost to name one regular real parameter is about log sqrt(n), which is one half log n. For a model with k parameters the leading model-description cost is therefore k/2 log n.

Putting the two parts together, the leading description length for an order-k model with maximum-likelihood estimate theta_hat_k is -log P(data | theta_hat_k) plus k/2 log n, up to lower-order terms. If I work on the usual deviance scale, this becomes -2 log P(data | theta_hat_k) plus k log n. I fit each candidate order by maximum likelihood, compute this score, and select the order that minimizes it. An extra parameter is worth including only if it shortens the data description by more than the half-log-n bits it costs to transmit.

This is not the same as simply adding a parameter-count penalty by hand. The k/2 log n term comes from the resolution at which the data allow parameters to be distinguished. It is also not identical to Akaike's information criterion, whose penalty is a constant per parameter and is derived from expected predictive bias under a true distribution. It is not a Bayesian model comparison either, because I am not integrating likelihood against a prior over models; I am counting the bits in a decodable message. And it is not the universal shortest-program idea of Kolmogorov or Solomonoff, because universal program-size complexity is not computable in the finite-sample form needed for ordinary statistical model selection. MDL keeps the compression intuition but restricts it to statistical model classes in a way that is computable and data-driven.

I find the principle easiest to trust when I see it recover a known structure from synthetic data. The code below generates a short time series from a true second-order autoregression, fits autoregressive models of orders zero through six by least squares, and compares the MDL score to AIC and BIC. The MDL score is computed from the maximized Gaussian log likelihood plus the parameter cost. The demonstration shows that MDL typically selects the correct order, while the in-sample likelihood alone would always prefer the largest order.

```python
import numpy as np

def fit_ar(data, k):
    """Fit AR(k) by least squares and return sigma^2 and coefficient count."""
    n = len(data)
    if k == 0:
        sigma2 = np.var(data, ddof=0)
        return sigma2, 0
    y = data[k:]
    X = np.array([data[k - i - 1:n - i - 1] for i in range(k)]).T
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    sigma2 = np.mean(resid ** 2)
    return sigma2, k

def gaussian_nll(data, k, sigma2):
    """Negative log likelihood for AR(k) with Gaussian errors."""
    n = len(data)
    used = n - k
    if sigma2 <= 0:
        return np.inf
    return 0.5 * used * (np.log(2 * np.pi) + np.log(sigma2) + 1)

np.random.seed(42)
n = 200
phi = np.array([0.6, -0.4])
y = np.zeros(n)
noise = np.random.randn(n)
for t in range(2, n):
    y[t] = phi[0] * y[t - 1] + phi[1] * y[t - 2] + noise[t]

orders = range(7)
results = []
for k in orders:
    sigma2, p = fit_ar(y, k)
    nll = gaussian_nll(y, k, sigma2)
    aic = 2 * nll + 2 * p
    bic = 2 * nll + p * np.log(n - k)
    mdl = nll + 0.5 * p * np.log(n)
    results.append((k, nll, aic, bic, mdl))

print("k   NLL      AIC      BIC      MDL")
for k, nll, aic, bic, mdl in results:
    print(f"{k}   {nll:7.2f}  {aic:7.2f}  {bic:7.2f}  {mdl:7.2f}")

best_mdl = min(results, key=lambda x: x[4])
print(f"\nMDL selects order {best_mdl[0]}")
```

In practice I apply MDL by enumerating the candidate structures, fitting each one by maximum likelihood, and ranking them with the two-part description length. The selected structure and its fitted coefficients together form the shortest complete explanation of the data. The method applies wherever I face nested model classes and need a principled stopping point between underfitting and overfitting, from time-series order selection through Markov order estimation to subset regression and beyond. The canonical name I use for this criterion is Rissanen's Minimum Description Length, and its operational form is the minimization of -log P(data | theta_hat_k) plus k/2 log n.