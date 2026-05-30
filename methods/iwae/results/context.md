# Context

## Research question

We want to learn a deep, directed latent-variable generative model `p(x) = ∫ p(x, h) dh` of high-dimensional data (binarized images of digits and handwritten characters) and to train it by approximate maximum likelihood. The model is a top-down generative network — a prior over latents followed by a cascade of stochastic layers that produce the observation — paired with a bottom-up recognition network that amortizes posterior inference. Because the marginal likelihood requires integrating out the latents, training proceeds by maximizing a tractable lower bound on `log p(x)`, with the recognition network providing the approximate posterior.

The difficulty is that the standard lower bound is only as good as the approximate posterior. When the recognition network is restricted to a simple family (a diagonal-Gaussian, amortized from the observation by a feed-forward net), and the true posterior is not in that family, the bound is loose, and — more insidiously — the training objective *reshapes the model* to make its posteriors fit the recognition network's assumptions, rather than letting the model use its full capacity. A satisfactory solution would: (a) give a strictly tighter lower bound on `log p(x)` than the standard one, ideally one that can be made arbitrarily tight at the cost of more computation; (b) relax the pressure that forces the posterior to be simple/factorial, so the model can learn richer representations that use more of its latent dimensions; (c) remain a *valid* lower bound (so we are still doing approximate maximum likelihood); and (d) keep a low-variance, reparameterized gradient so it is trainable with the same machinery as the simple bound.

## Background

**Latent-variable generative models and the inference bottleneck.** A directed latent-variable model defines `p(x) = ∫ p(x|h) p(h) dh`. The log-likelihood gradients for such models are expressed in terms of posterior statistics `p(h|x)`, which are intractable for interesting models. One classical strategy (Helmholtz machines; Dayan, Hinton, Neal & Zemel, 1995) is to train a *recognition network* alongside the generative model: a network that maps an observation to an approximate posterior over latents, much faster than generic inference like MCMC. The recognition network is the amortized inference engine.

**The evidence lower bound.** For any distribution `q(h|x)` with support covering the posterior, Jensen's inequality gives

`log p(x) = log E_{q(h|x)}[ p(x,h)/q(h|x) ] ≥ E_{q(h|x)}[ log p(x,h)/q(h|x) ] = L(x)`,

and the gap is exactly `L(x) = log p(x) − KL(q(h|x) || p(h|x))`. Maximizing `L` jointly over the generative and recognition parameters does two things: it pushes up the data log-likelihood, and it pulls the approximate posterior toward the true posterior. The KL gap means the bound is tight only when `q(h|x)` equals the true posterior; for a restricted `q` family the bound is loose by the KL between `q` and the true posterior.

**The reparameterization trick.** If one differentiates `L` with respect to the recognition parameters directly, the result is a REINFORCE-style (score-function) estimator that trains slowly and with high variance, because it does not exploit the gradient of the integrand with respect to the latents (Williams, 1992; the "backprop through a random number generator" view). For Gaussian `q`, one instead writes a sample as a deterministic, differentiable function of the parameters and a fixed-distribution auxiliary noise: `h = σ(x) ⊙ ε + μ(x)`, with `ε ∼ N(0, I)`. Since the noise distribution does not depend on the parameters, the gradient operator passes inside the expectation,

`∇_θ E_{q}[ f(h) ] = E_{ε∼N(0,I)}[ ∇_θ f(h(ε, θ)) ]`,

and the inner gradient is ordinary backpropagation. This is what makes Gaussian latent-variable models trainable with low-variance gradients.

**The cost of a restricted posterior.** With `q(h|x)` constrained to be approximately factorial and predictable from `x` by a feed-forward net, the bound `L` harshly penalizes posterior samples that fail to explain the observation. Concretely, because the objective is an expectation of `log p(x,h)/q(h|x)` under `q`, *every* sample drawn from `q` is required to be a good explanation; if the recognition net places only a small fraction of its mass in the high-posterior region, the remaining mass is penalized heavily. To avoid that penalty, training drives the generative model toward solutions whose true posteriors actually *are* approximately factorial and predictable. This is a real, observable phenomenon in these models: trained models use far fewer latent dimensions than their capacity allows — many latent dimensions become inactive (their posterior is indistinguishable from the prior across the data) — and this inactivation is driven by the *objective*, not merely by optimization plateaus. The activity of a latent dimension can be measured by `A = Cov_x( E_{q(u|x)}[u] )`: if a dimension carries information about the data, this statistic is large; for the simple bound the distribution of `A` across dimensions is bimodal, with a large cluster of essentially-dead dimensions.

**Importance sampling for marginal likelihoods.** Importance sampling estimates `p(x) = ∫ p(x,h) dh = E_{q(h|x)}[ p(x,h)/q(h|x) ]` by drawing samples from a proposal `q` and averaging the importance weights `w = p(x,h)/q(h|x)`. The average `(1/k) Σ_i w_i` is an *unbiased* estimator of `p(x)` for any `k ≥ 1`. Plain importance sampling is notorious for high (even infinite) variance when the proposal is poorly matched to the target — the weight distribution becomes heavy-tailed and a few samples dominate. Importance weighting has been used to build log-probability lower bounds for inference (e.g. Gogate, Bidyuk & Dechter, 2007, who lower-bound the probability of evidence via the Markov inequality; and inference-by-importance-sampling-from-the-prior in Tang & Salakhutdinov, 2013, and Ba, Mnih & Kavukcuoglu, 2015).

**Reweighted wake-sleep.** A recognition-network approach (Bornschein & Bengio, 2015) that combines the wake-sleep algorithm with a generative-network update that turns out to be gradient ascent on a multi-sample importance-weighted objective. It interprets this update as following a *biased* estimate of `∇ log p(x)`, and trains the recognition network with separate wake- and sleep-phase objectives rather than a single unified bound; it does not use the reparameterization trick.

## Baselines

**The standard variational autoencoder (Kingma & Welling, 2014; Rezende, Mohamed & Wierstra, 2014).** A top-down generative network `p(x,h)` (fixed `N(0,I)` prior on the top latent layer; each conditional a diagonal Gaussian, or Bernoulli at the visible layer for binary data, with mean/variance from a feed-forward net) paired with a bottom-up recognition network `q(h|x)` of the same diagonal-Gaussian form. Trained by maximizing the single-sample bound `L(x) = E_{q(h|x)}[log p(x,h)/q(h|x)]` with the reparameterization-trick gradient, Monte-Carlo–estimated as `(1/k) Σ_i ∇_θ log w(x, h(ε_i,x,θ))` with `w = p(x,h)/q(h|x)`. Strength: low-variance reparameterized gradients, competitive likelihoods, fast amortized inference. Gap: the bound is loose by `KL(q||p(h|x))`; the objective penalizes every poorly-explaining `q`-sample, which constrains the model to factorial, feed-forward-predictable posteriors and leads to many inactive latent dimensions and overly simple representations. Even drawing more samples to lower the variance of the *gradient* of `L` does not change *which* bound is being maximized — it is still the single-sample bound.

**Deep autoregressive networks (DARN; Gregor, Danihelka, Mnih, Blundell & Wierstra, 2014) and NVIL (Mnih & Gregor, 2014).** Deep generative + recognition networks trained on the same variational bound, but with score-function (REINFORCE) gradient estimators rather than reparameterization. NVIL reduces variance by training an auxiliary network to predict reward baselines. Gap: high-variance updates and the same single-sample bound; the reparameterization trick is not used.

**Wake-sleep / Helmholtz machine (Dayan et al., 1995) and reweighted wake-sleep (Bornschein & Bengio, 2015).** Recognition-network training by alternating wake and sleep phases. Classic wake-sleep trains the two networks on *different* objectives. Reweighted wake-sleep introduces a multi-sample importance-weighted generative update but keeps separate wake/sleep recognition objectives, interprets the update as a biased estimate of `∇ log p(x)`, and does not reparameterize. Gap: not a single unified objective; biased-gradient interpretation; no reparameterization.

**Richer-posterior approaches.** Instead of changing the bound, one can make the approximate posterior more expressive: normalizing flows (Rezende & Mohamed, 2015) compose invertible transformations to turn a simple `q` into a flexible one; the Hamiltonian variational approximation (Salimans, Kingma & Welling, 2015) interleaves MCMC and variational inference. Gap: these add machinery to `q` itself; they are orthogonal to the question of whether the *bound* can be tightened by using multiple samples.

## Evaluation settings

Density estimation on binarized `28×28` images: MNIST handwritten digits (LeCun et al., 1998; standard 60,000/10,000 train/test split) and Omniglot handwritten characters (Lake, Salakhutdinov & Tenenbaum, 2013; 24,345/8,070 split). Binarization follows the standard scheme of sampling binary observations with expectations equal to the real-valued training pixels (Salakhutdinov & Murray, 2008); a fixed-binarization variant (Larochelle & Murray, 2011) is also a standard protocol. Architectures: a one-stochastic-layer model (50-unit latent, with two 200-unit deterministic `tanh` layers between observation and latent) and a two-stochastic-layer model (100- and 50-unit latents, with 200- and 100-unit deterministic layers). Stochastic layers are diagonal Gaussians with an `exp` nonlinearity on predicted variances; the visible layer is Bernoulli. Optimization with Adam (Kingma & Ba, 2015), minibatches of size 20, Glorot initialization (Glorot & Bengio, 2010). The reported metric is held-out negative log-likelihood (estimated by a large-sample importance-weighted bound on the test set), together with the number of *active* latent dimensions per layer measured by the activity statistic `A_u = Cov_x(E_{q(u|x)}[u])` with an activity threshold of `10⁻²`.

## Code framework

The primitives that already exist: a deep-learning framework with autodiff and an Adam optimizer; a data pipeline that yields binarized image minibatches; and standard building blocks for amortized Gaussian inference. We assume a reusable block that maps an input vector through two `tanh` layers and emits the mean and (log-)standard-deviation of a diagonal Gaussian, plus a reparameterized sampler.

```python
import torch
import torch.nn as nn

class GaussianBlock(nn.Module):
    """Two tanh layers -> (mu, sigma) of a diagonal Gaussian; exp() keeps sigma > 0."""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.fc_mu = nn.Linear(hidden_dim, out_dim)
        self.fc_logsigma = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        h = self.body(x)
        mu = self.fc_mu(h)
        sigma = torch.exp(self.fc_logsigma(h))
        return mu, sigma

def gaussian_log_density(z, mu, sigma):
    # log N(z | mu, diag(sigma^2)), summed over the latent dimension
    return torch.sum(-0.5 * ((z - mu) / sigma) ** 2 - torch.log(sigma) - 0.5 * LOG2PI, -1)

def standard_normal_log_density(z):
    return torch.sum(-0.5 * z ** 2 - 0.5 * LOG2PI, -1)

def bernoulli_log_density(x, p):
    return torch.sum(x * torch.log(p) + (1 - x) * torch.log(1 - p), -1)


class LatentVariableModel(nn.Module):
    """Top-down generative net + bottom-up recognition net, trained on a
    lower bound of log p(x). The recognition net is a reparameterized
    diagonal Gaussian; the visible layer is Bernoulli."""
    def __init__(self, dim_latent, dim_obs):
        super().__init__()
        self.encoder = GaussianBlock(dim_obs, 200, dim_latent)
        self.decoder = nn.Sequential(
            nn.Linear(dim_latent, 200), nn.Tanh(),
            nn.Linear(200, 200), nn.Tanh(),
            nn.Linear(200, dim_obs), nn.Sigmoid())

    def encode(self, x):
        mu, sigma = self.encoder(x)
        eps = torch.randn_like(sigma)
        h = mu + sigma * eps              # reparameterized sample
        return h, mu, sigma, eps

    def decode(self, h):
        return self.decoder(h)

    def objective(self, x):
        # TODO: the training bound on log p(x) we are about to design,
        #       and its reparameterized gradient.
        pass

    def log_likelihood_estimate(self, x):
        # TODO: a held-out estimate of log p(x).
        pass


def train_step(model, x, optimizer):
    optimizer.zero_grad()
    loss = -model.objective(x)
    loss.backward()
    optimizer.step()
    return loss.item()
```
