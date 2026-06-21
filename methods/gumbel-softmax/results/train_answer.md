We want to train neural networks that contain categorical latent variables end-to-end with gradient descent. In such a network some upstream parameters produce class probabilities, a discrete category is sampled, and downstream layers turn that choice into a loss. The quantity we actually optimize is an expectation over that sample, but backpropagation cannot reach the parameters because drawing a categorical variable is fundamentally an argmax followed by a one-hot encoding. Both operations are flat almost everywhere, so the chain rule yields zero gradients. Existing ideas fall short in opposite directions: the score-function or REINFORCE estimator is unbiased and works with black-box costs, but it multiplies the raw cost by the log-probability score and suffers from variance that grows with the dimensionality of the sample. Straight-through estimators for Bernoulli variables are cheap but biased, limited to binary choices, and suffer from a mismatch between the hard forward sample and the smooth backward proxy. Neither family gives the low-variance path derivative that makes continuous reparameterization tricks so effective.

The method I propose is the Gumbel-Softmax estimator, also known as the Concrete distribution. It begins with the Gumbel-Max trick: adding independent Gumbel noise to the log-probabilities and taking the argmax produces an exact categorical sample. The noise is independent of the parameters, so the only non-differentiable piece is the argmax itself. The key move is to relax that argmax to a softmax with a temperature parameter tau. Instead of a one-hot corner of the simplex, the sample becomes a smooth point on the simplex that concentrates near the largest perturbed logit. Because the softmax is differentiable for any positive temperature, gradients flow from the loss through the sample and back to the parameters. At very small temperature the relaxed sample approaches the exact one-hot categorical draw, while at large temperature it becomes smoother and closer to uniform. This creates a bias-variance tradeoff controlled by tau, which is typically annealed from a high value down to a small floor during training so that optimization starts stable and ends close to the true discrete distribution.

When a genuine hard discrete value is required, such as for discrete actions or quantized codes, the straight-through variant keeps a one-hot sample in the forward pass but uses the gradient of the soft relaxed sample in the backward pass. This is implemented by returning a tensor whose value is the hard one-hot vector but whose gradient is computed as if the soft sample had been used. Because the soft sample shares the same Gumbel noise as the hard sample, the forward-backward mismatch is much smaller than in earlier Bernoulli straight-through schemes.

```python
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def gumbel_softmax(logits, tau=1.0, hard=False, dim=-1):
    """Differentiable approximate categorical sample via the Gumbel-Softmax trick.

    logits[..., k]: unnormalized log-probabilities.
    Returns a sample on the probability simplex; if hard=True, returns a one-hot
    tensor whose gradient is that of the soft sample (straight-through).
    """
    # g_i ~ Gumbel(0,1), sampled as -log(-log u) = -log(e) with e ~ Exp(1).
    g = -torch.empty_like(logits).exponential_().log()
    scores = (logits + g) / tau
    y_soft = scores.softmax(dim)

    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        # Value is y_hard; gradient flows through y_soft.
        return y_hard - y_soft.detach() + y_soft
    return y_soft


class CategoricalLatentVAE(nn.Module):
    def __init__(self, x_dim, n_vars, n_classes, hidden=256):
        super().__init__()
        self.n_vars, self.n_classes = n_vars, n_classes
        self.encoder = nn.Sequential(
            nn.Linear(x_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_vars * n_classes))
        self.decoder = nn.Sequential(
            nn.Linear(n_vars * n_classes, hidden), nn.ReLU(),
            nn.Linear(hidden, x_dim))

    def forward(self, x, tau, hard=False):
        logits = self.encoder(x).view(-1, self.n_vars, self.n_classes)
        z = gumbel_softmax(logits, tau=tau, hard=hard)
        x_logits = self.decoder(z.view(x.size(0), -1))
        return x_logits, logits


def elbo(x, x_logits, logits):
    recon = F.binary_cross_entropy_with_logits(x_logits, x, reduction="none").sum(-1)
    q = F.softmax(logits, dim=-1)
    log_q = F.log_softmax(logits, dim=-1)
    k = logits.size(-1)
    kl = (q * (log_q + math.log(k))).sum(-1).sum(-1)
    return (recon + kl).mean()


def train(model, optimizer, loader, n_steps, r=1e-4, tau_min=0.5):
    step = 0
    for x in loader:
        tau = max(tau_min, math.exp(-r * step))
        optimizer.zero_grad()
        x_logits, logits = model(x, tau=tau)
        loss = elbo(x, x_logits, logits)
        loss.backward()
        optimizer.step()
        step += 1
        if step >= n_steps:
            break
```

This gives a complete, drop-in replacement for a non-differentiable categorical sample inside an ordinary autograd training loop.
