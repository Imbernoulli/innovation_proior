# Variational autoencoder objective: L2 reconstruction + KL

The method is the variational autoencoder (VAE) objective — the negative evidence lower bound (ELBO) of a deep directed latent-variable model, trained with the reparameterized gradient estimator. For one datapoint,

```text
log p_theta(x) >= L(x)
              = E_q[log p_theta(x|z)] - D_KL(q_phi(z|x) || p(z)).
```

For a diagonal Gaussian encoder `q_phi(z|x) = N(mu, sigma^2 I)` and prior `p(z) = N(0, I)`,

```text
-D_KL(q || p) = 0.5 * sum_j (1 + log sigma_j^2 - mu_j^2 - sigma_j^2)
```

is the term in the maximized lower bound. The positive KL added to the minimized loss is

```text
D_KL(q || p) = 0.5 * sum_j (mu_j^2 + sigma_j^2 - 1 - log sigma_j^2).
```

With `logvar = log sigma^2`, this is `0.5 * sum(mean^2 + exp(logvar) - 1 - logvar)`, matching the local diffusers-style `DiagonalGaussianDistribution.kl()` implementation.

The sampling path is the Gaussian change of variables:

```text
z = mu + sigma * eps,   eps ~ N(0, I).
```

Since `eps` is independent of encoder parameters, `E_q[f(z)] = E_eps[f(mu + sigma * eps)]` and gradients flow through `z` by ordinary backpropagation. The local `posterior.sample()` implements exactly this as `mean + std * eps`, with `std = exp(0.5 * logvar)`.

For a fixed-variance Gaussian pixel decoder,

```text
log p_theta(x|z) = C - (1 / (2 sigma_x^2)) * sum_d (x_d - f_theta(z)_d)^2,
```

so the reconstruction term is squared L2 error. With PyTorch's default `F.mse_loss`, the squared error is averaged over batch and pixels. For `D` pixels per image, `MSE + beta * KL` is the exact Gaussian negative ELBO up to a positive global scale when `beta = 2 * sigma_x^2 / D`; in practice `kl_weight` is the effective KL coefficient under this reduction convention.

```python
import torch.nn as nn
import torch.nn.functional as F


class VAELoss(nn.Module):
    """Negative ELBO for the fixed Gaussian-bottleneck reconstruction harness."""

    def __init__(self, device):
        super().__init__()
        self.kl_weight = 1e-6

    def forward(self, recon, target, posterior, step):
        rec_loss = F.mse_loss(recon, target)
        kl_loss = posterior.kl().mean()
        loss = rec_loss + self.kl_weight * kl_loss
        return loss, {
            "loss": loss.detach().item(),
            "rec_loss": rec_loss.detach().item(),
            "kl_loss": kl_loss.detach().item(),
        }
```

This is the l2-kl instance: one reparameterized latent sample is already used by the model to produce `recon`; the loss adds per-pixel Gaussian reconstruction error and the positive closed-form KL to the standard-normal prior.
