A probabilistic model can make the joint $p(x,z)$ cheap to evaluate while making the posterior $p(z\mid x)$ effectively impossible. I can write $p(x,z)$ as a product of local factors and read off its value at any hidden configuration, but the posterior asks me to divide by the evidence $p(x)=\sum_z p(x,z)$, a sum or integral over the entire hidden space. In discrete models that sum is exponentially large; in continuous models it is a high-dimensional integral with no closed form. The same missing denominator blocks two tasks at once: it blocks the posterior because $p(z\mid x)=p(x,z)/p(x)$, and it blocks model scoring and parameter learning because that evidence is the likelihood of the observed data. Exact inference can exploit conditional independencies with a junction tree, but its cost is exponential in the largest clique, and dense networks with explaining-away effects routinely force cliques of size on the order of hundreds — so exact normalization is not a usable primitive. Monte Carlo methods sidestep the sum by building a Markov chain whose stationary distribution is the posterior, but they are stochastic, slow to converge, hard to diagnose, and they return a cloud of samples rather than a compact reusable distribution. Expectation-Maximization rewrites the likelihood objective as $E_q[\log p(x,z\mid\theta)]+H(q)$ plus a nonnegative gap, but its exact E step still presumes the posterior over hidden variables can be computed, which is precisely the assumption that fails. What is missing is a general way to choose a tractable stand-in for the posterior, to measure its error without ever touching the evidence, and to turn that into concrete update equations.

I propose variational inference built on the evidence lower bound, the ELBO, optimized in the mean-field setting by coordinate-ascent variational inference (CAVI). The move is to stop solving for the posterior itself and instead search for a tractable distribution $q(z)$ that stands in for it: I fix a family $Q$ whose expectations and entropy I can compute, and ask which member is closest to $p(z\mid x)$, solving $q^\* = \arg\min_{q\in Q}\,\mathrm{KL}(q(z)\,\|\,p(z\mid x))$. The forward direction of the KL is chosen deliberately, because its expectation is taken under $q$, the distribution I control; the reverse direction would require expectations under the unknown posterior, which is the very thing I cannot evaluate. At first this objective looks like it smuggles the impossible term back in, since expanding it gives $\mathrm{KL}(q\,\|\,p(z\mid x)) = E_q[\log q(z)] - E_q[\log p(z\mid x)]$ and the posterior log density carries $\log p(x)$ inside it. The resolution is that this impossible term does not depend on $q$. Substituting $\log p(z\mid x)=\log p(x,z)-\log p(x)$ yields $\mathrm{KL}(q\,\|\,p(z\mid x)) = E_q[\log q(z)] - E_q[\log p(x,z)] + \log p(x)$, and since the last term is constant while I vary $q$, minimizing the KL is identical to maximizing the computable quantity

$$\mathrm{ELBO}(q) = E_q[\log p(x,z)] - E_q[\log q(z)] = E_q[\log p(x,z)] + H(q).$$

The same identity, read from the other side, gives the bound for free: rearranging produces $\log p(x) = \mathrm{ELBO}(q) + \mathrm{KL}(q\,\|\,p(z\mid x))$, and because the KL is nonnegative the ELBO is a genuine lower bound on the log evidence, with slack equal exactly to the posterior-approximation error. This is the load-bearing conversion — raising the bound and approximating the posterior are one and the same operation, and the bound doubles as the diagnostic of progress. Jensen's inequality tells the identical story and makes the direction feel inevitable: writing $p(x)=E_q[p(x,z)/q(z)]$ and pushing the concave $\log$ inside gives $\log p(x) \ge E_q[\log p(x,z) - \log q(z)]$, and convex duality on log-sum-exp shows the same entropy price for introducing a distribution over configurations.

To make the optimization real I need a family that is restrictive enough to be tractable yet expressive enough to land near the posterior; a completely unrestricted $q$ would just recover the exact posterior and reproduce the original impossible problem. The mean-field choice is to factorize fully, $q(z)=\prod_{j=1}^m q_j(z_j)$. This deliberately throws away posterior dependence — so I should expect underestimated variance and missed correlations — but in exchange the expectations and entropy split into separable pieces, and I can optimize one factor while holding the rest fixed. Viewing the ELBO as a function of a single factor $q_j$, everything independent of $z_j$ folds into a constant and what remains is $E_{q_j}[E_{-j}[\log p(x,z)]] - E_{q_j}[\log q_j(z_j)]$, which has the form of a negative KL from $q_j$ to a density proportional to $\exp(E_{-j}[\log p(x,z)])$. The optimal factor is therefore not a gradient step or a guess but the normalized exponentiated expected log joint,

$$q_j^\*(z_j) \;\propto\; \exp\!\big(E_{-j}[\log p(x,z)]\big) \;\propto\; \exp\!\big(E_{-j}[\log p(z_j\mid z_{-j},x)]\big),$$

and cycling these updates monotonically raises the ELBO to a local optimum. This is exactly Gibbs sampling with the randomness removed: a Gibbs sampler draws $z_j$ from the complete conditional $p(z_j\mid z_{-j},x)$, whereas here I keep the whole factor $q_j$ and set it from the expected log of that same complete conditional under the other factors — the random local draw becomes a deterministic local distribution update. The method is cleanest when the complete conditional is in an exponential family, $p(z_j\mid z_{-j},x)=h(z_j)\exp(\eta_j(z_{-j},x)^\top z_j - a(\eta_j))$, because taking the expectation over the other factors only changes the natural parameter, leaving $q_j^\*(z_j)=h(z_j)\exp(\nu_j^\top z_j - a(\nu_j))$ with $\nu_j=E_{-j}[\eta_j(z_{-j},x)]$; the optimal factor stays in the same family and each update reduces to expected-natural-parameter bookkeeping rather than a functional optimization. This also clarifies where EM sits: for a model with parameters $\theta$, the same lower bound $\mathcal{L}(q,\theta)=E_q[\log p(x,z\mid\theta)]+H(q)$ has gap $\mathrm{KL}(q\,\|\,p(z\mid x,\theta))$, and if $q$ ranges over all distributions the best $q$ is the exact posterior so coordinate ascent in $(q,\theta)$ is precisely EM — EM is the exact-posterior special case, and restricting $q$ gives a tractable approximate E step optimizing the very same objective. As a concrete instance, for the Gaussian mixture $\mu_k\sim\mathcal{N}(0,\sigma^2)$, $c_i\sim\mathrm{Categorical}(1/K)$, $x_i\mid c_i,\mu\sim\mathcal{N}(\mu_{c_i},1)$ with $q(\mu,c)=\prod_k\mathcal{N}(\mu_k;m_k,s^2_k)\prod_i\mathrm{Categorical}(c_i;\phi_i)$, the CAVI updates are $\phi_{ik}\propto\exp(m_k x_i - (m_k^2+s^2_k)/2)$, $s^2_k=1/(1/\sigma^2+\sum_i\phi_{ik})$, and $m_k=s^2_k\sum_i\phi_{ik}x_i$. I keep the limitations in view: the bound is generally non-convex in the variational parameters so coordinate ascent only reaches a local optimum, the factorization can lock onto one mode and ignore dependence, and the chosen KL direction is mode-seeking — so multiple random restarts and monitoring the ELBO are not optional polish but part of using the method responsibly.

The same ELBO objective can be optimized directly with black-box gradient estimators rather than closed-form coordinate updates, by reparameterizing the sample from $q$ and differentiating the bound, which is what the following training step does: it samples $z$ from the encoder's $q(z\mid x)$ via `rsample`, scores the reconstruction under the decoder, subtracts the analytic KL to the prior, and ascends the bound with Adam.

```python
import torch
import torch.nn as nn


def elbo(x, encoder, decoder, prior):
    """ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))."""
    q = encoder(x)
    z = q.rsample()                      # reparameterized sample
    log_likelihood = decoder(z).log_prob(x).sum()
    kl_divergence = torch.distributions.kl_divergence(q, prior).sum()
    return log_likelihood - kl_divergence


def training_step(x, encoder, decoder, prior, optimizer):
    encoder.train()
    decoder.train()
    optimizer.zero_grad()
    loss = -elbo(x, encoder, decoder, prior).mean()
    loss.backward()
    optimizer.step()
    return float(loss)


# Example skeleton (not run)
encoder = nn.Sequential(nn.Linear(784, 256), nn.ReLU(),
                        nn.Linear(256, 2 * 10))
decoder = nn.Sequential(nn.Linear(10, 256), nn.ReLU(),
                        nn.Linear(256, 784), nn.Sigmoid())
prior = torch.distributions.Normal(0, 1)
optimizer = torch.optim.Adam(list(encoder.parameters()) +
                             list(decoder.parameters()), lr=1e-3)
```
