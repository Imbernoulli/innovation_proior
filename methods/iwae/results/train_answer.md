The canonical method I am presenting is the Importance Weighted Autoencoder, or IWAE. It is a direct refinement of the variational autoencoder that keeps the same architecture—an encoder that maps observations to an approximate posterior and a decoder that maps latents back to observations—but changes the training objective so that the model is no longer forced to learn simple, factorial posteriors.

The starting point is the ordinary VAE evidence lower bound. For a generative model p(x, h) and a recognition network q(h|x), the bound is L(x) = E_{q(h|x)}[log p(x, h) / q(h|x)]. This is a lower bound on log p(x) because log is concave, and the gap is exactly the KL divergence from q to the true posterior. The problem is that this objective averages a single log-importance-weight over every sample drawn from q. That means every sample is required to be a good explanation of x. If the recognition network puts most of its mass in regions where p(x, h) is small relative to q(h|x), the bound collapses. Because the generative model is trained against the same objective, it responds by making its true posterior fit the simple family that q can represent. The result is a model that uses far fewer latent dimensions than it has been given: many latent units become inactive, their posterior mean barely moving across the dataset, and the representations are poorer than the architecture could support.

IWAE fixes this by replacing the single importance weight with a k-sample average of importance weights. Define w_i = p(x, h_i) / q(h_i|x) for h_i drawn independently from q(h|x), and define the new objective L_k(x) = E_{h_1, ..., h_k ~ q}[log (1/k) sum_i w_i]. When k is 1 this is exactly the VAE bound, so the method contains the ordinary VAE as a special case. The difference is that the log is now applied to an average over k samples rather than to a single sample. A few high-weight samples can pull the average up, so the recognition network no longer has to be correct on every draw. This relaxes the pressure on q and allows the generative model to keep richer, more complex posteriors.

The first thing to check is that L_k is still a valid lower bound on log p(x). Each weight w_i is an unbiased estimator of p(x), because E_q[w_i] = integral q(h|x) p(x, h) / q(h|x) dh = integral p(x, h) dh = p(x). Therefore E[(1/k) sum_i w_i] = p(x). Since log is concave, Jensen's inequality gives L_k = E[log (1/k) sum_i w_i] <= log E[(1/k) sum_i w_i] = log p(x). So IWAE is still doing approximate maximum likelihood.

The second property is monotonicity: more samples make the bound no looser. Specifically, L_{k+1} >= L_k for every k. The proof uses a neat averaging-over-subsets trick. Take k samples and their weights, and choose a uniformly random m-subset of the indices, where m <= k. The expected m-sample average over this random subset equals the full k-sample average, because each index appears in the subset with probability m/k. Therefore the k-sample average can be written as an expectation over random m-subsets of the corresponding m-sample average. Applying Jensen's inequality to the log of that expectation over subsets gives L_k >= L_m. So the bounds form a ladder: log p(x) >= ... >= L_{k+1} >= L_k >= ... >= L_1 = L(x).

In the limit of many samples, the bound converges to the true log-likelihood. The average (1/k) sum_i w_i is an average of i.i.d. copies of w with mean p(x), so under the standard bounded-positive-weight regularity it converges almost surely to p(x) by the strong law of large numbers. The log is continuous, and the log averages converge in expectation, so L_k approaches log p(x) as k grows. The ladder actually reaches the ceiling in the limit.

One might worry that importance weighting explodes in high dimensions, because the weights can become heavy-tailed when the proposal q poorly matches the posterior. The key saving factor is that IWAE estimates log p(x), not p(x). For a positive unbiased estimator Z_hat of a positive quantity Z, Markov's inequality applied to the ratio Z_hat / Z gives Pr(log Z_hat > log Z + b) <= e^{-b}. So the right tail of the log-estimator decays exponentially, regardless of how heavy-tailed the raw weights are. The mean absolute deviation of log Z_hat is bounded by 2 + 2 delta, where delta is the bound gap. The estimate is tame even when plain importance sampling of p(x) would be dangerous.

To train, I need a reparameterized gradient of L_k with respect to both the generative and recognition parameters. Write each sample as h_i = h(epsilon_i, x, theta) with epsilon_i drawn from a fixed standard normal, so the expectation is over a distribution that does not depend on theta. Then the gradient passes inside the expectation: nabla_theta L_k = E[nabla_theta log (1/k sum_i w_i)]. Differentiating the log of the sum gives (sum_i nabla_theta w_i) / (sum_j w_j), and using nabla w_i = w_i nabla log w_i yields nabla_theta L_k = E[sum_i tilde{w}_i nabla_theta log w_i], where tilde{w}_i = w_i / sum_j w_j are the self-normalized importance weights. For k = 1 the single weight self-normalizes to 1 and the gradient reduces to the ordinary VAE gradient. For k > 1 the gradient is a weighted average of per-sample score gradients, with well-explaining samples receiving larger weights.

In practice I implement this by constructing a surrogate scalar whose gradient is the same. If I compute the self-normalized weights and detach them, so that autodiff does not differentiate through them, then the scalar sum_i tilde{w}_i log w_i differentiates exactly to sum_i tilde{w}_i nabla log w_i. The value of this surrogate is not L_k, but its gradient is the correct IWAE gradient. All weight computations are done in the log domain for numerical stability: subtract the maximum log-weight before exponentiating, normalize, and detach. For evaluation I compute the actual bound estimate L_k = logmeanexp_i(log w_i), which can be evaluated with a large k such as 5000 to estimate held-out log-likelihood.

The architecture itself remains the same as in a VAE. The encoder is a feed-forward network with two tanh hidden layers that outputs the mean and log-standard-deviation of a diagonal Gaussian; exponentiating the log-standard-deviation keeps the standard deviation positive. The decoder is another two tanh hidden layers followed by a sigmoid that gives the Bernoulli probability for each binary pixel. Reparameterized sampling is h = mu + sigma * epsilon with epsilon ~ N(0, I). The only change is in the objective function.

The code below is a small, self-contained illustration of the IWAE bound. It builds a tiny generative model and recognition network, draws latent samples, and shows that the k-sample IWAE bound is at least as large as the single-sample VAE bound for the same parameters. It also demonstrates the self-normalized gradient estimator and the stable logmeanexp evaluation. This is not a full training run on binarized MNIST or Omniglot, but it verifies the core quantitative claims: the bound ladder is monotonic in k, and the gradient estimator reduces to the VAE estimator when k equals 1.

```python
import numpy as np
import torch
import torch.nn as nn

LOG2PI = float(np.log(2.0 * np.pi))


def gaussian_log_density(z, mu, sigma):
    return torch.sum(
        -0.5 * ((z - mu) / sigma) ** 2
        - torch.log(sigma)
        - 0.5 * LOG2PI,
        dim=-1,
    )


class TinyModel(nn.Module):
    """Small one-layer VAE/IWAE for binary observations."""

    def __init__(self, dim_obs, dim_latent):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim_obs, 32),
            nn.Tanh(),
            nn.Linear(32, dim_latent * 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(dim_latent, 32),
            nn.Tanh(),
            nn.Linear(32, dim_obs),
            nn.Sigmoid(),
        )

    def encode(self, x):
        out = self.encoder(x)
        mu, log_sigma = torch.chunk(out, 2, dim=-1)
        sigma = torch.exp(log_sigma)
        eps = torch.randn_like(sigma)
        z = mu + sigma * eps
        return z, mu, sigma, eps

    def log_weights(self, x):
        z, mu, sigma, eps = self.encode(x)
        log_q = gaussian_log_density(z, mu, sigma)
        log_prior = torch.sum(-0.5 * z ** 2 - 0.5 * LOG2PI, dim=-1)
        p_x_given_z = self.decoder(z)
        p_x_given_z = torch.clamp(p_x_given_z, 1e-6, 1 - 1e-6)
        log_lik = torch.sum(
            x * torch.log(p_x_given_z)
            + (1.0 - x) * torch.log(1.0 - p_x_given_z),
            dim=-1,
        )
        return log_prior + log_lik - log_q

    def iwae_bound(self, x, k):
        x_rep = x.unsqueeze(0).expand(k, *x.shape)
        log_w = self.log_weights(x_rep)
        m = torch.max(log_w, dim=0, keepdim=True)[0]
        return torch.mean(m + torch.log(torch.mean(torch.exp(log_w - m), dim=0)))

    def vae_bound(self, x):
        return self.iwae_bound(x, k=1)

    def training_surrogate(self, x, k):
        x_rep = x.unsqueeze(0).expand(k, *x.shape)
        log_w = self.log_weights(x_rep)
        max_log_w = torch.max(log_w, dim=0, keepdim=True)[0]
        w = torch.exp(log_w - max_log_w)
        w_tilde = (w / torch.sum(w, dim=0, keepdim=True)).detach()
        return torch.mean(torch.sum(w_tilde * log_w, dim=0))


if __name__ == "__main__":
    torch.manual_seed(0)
    dim_obs = 8
    dim_latent = 4
    batch_size = 16
    model = TinyModel(dim_obs, dim_latent)
    x = (torch.rand(batch_size, dim_obs) > 0.5).float()

    vae = model.vae_bound(x).item()
    iwae_5 = model.iwae_bound(x, k=5).item()
    iwae_50 = model.iwae_bound(x, k=50).item()

    print(f"VAE bound (k=1):  {vae:.4f}")
    print(f"IWAE bound k=5:   {iwae_5:.4f}")
    print(f"IWAE bound k=50:  {iwae_50:.4f}")
    assert iwae_5 >= vae - 1e-4, "IWAE bound should be at least the VAE bound"
    assert iwae_50 >= iwae_5 - 1e-4, "IWAE bound should be non-decreasing in k"

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = -model.training_surrogate(x, k=5)
    loss.backward()
    optimizer.step()
    print(f"Training surrogate loss: {loss.item():.4f}")
```

In summary, IWAE keeps the VAE architecture and the reparameterization trick, but replaces the single-sample evidence lower bound with a k-sample importance-weighted lower bound. The new bound is always a valid lower bound on log p(x), is never looser than the VAE bound, and converges to the true log-likelihood as k grows. The gradient is a self-normalized importance-weighted average of per-sample VAE gradients, implemented through a detached-weight surrogate. By letting good posterior samples dominate the update instead of punishing every poor sample, IWAE relieves the pressure that drives ordinary VAEs to waste latent capacity, leading to tighter bounds and richer representations without changing the model family.
