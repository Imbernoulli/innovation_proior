The variational lower bound in a VAE is tight only when the approximate posterior q(z|x) matches the true posterior p(z|x). In practice q is almost always a diagonal Gaussian, which can represent neither correlations between latent dimensions nor any non-Gaussian structure. Because true posteriors are essentially never factorized, this choice makes the bound loose by exactly the amount the posterior fails to capture dependencies. Diagonal Gaussians survive only because they satisfy three hard constraints: density evaluation and sampling must both be cheap, and both must be parallel across high-dimensional latents on a GPU. Any replacement must keep all three properties while being much more expressive.

Existing flexible posteriors fall short on at least one of those constraints. Planar and radial normalizing flows refine a Gaussian through invertible maps, but each planar step routes information through a single scalar bottleneck w^T z + b, so reshaping a high-dimensional posterior requires an impractically deep chain. Coupling-layer flows such as NICE and Real NVP are efficient in one direction but update only half the variables per layer as a function of the other half, giving weaker per-step flexibility. Fully autoregressive density estimators like MADE or PixelCNN are flexible and high-dimensional, yet their sampling recursion y_i = mu_i(y_{1:i-1}) + sigma_i(y_{1:i-1}) epsilon_i is inherently sequential, costing one step per dimension. Hamiltonian variational inference is expensive and introduces extra auxiliary variables. What is needed is a flow step that is flexible over all dimensions, parallel, and has a trivial log-determinant.

The method that achieves this is the Inverse Autoregressive Flow, or IAF. It is built from the inverse of an autoregressive transformation rather than its forward sampling direction. Given a full vector y, the inverse map recovers the noise as epsilon_i = (y_i - mu_i(y)) / sigma_i(y). Because y is available in its entirety, every mu_i and sigma_i can be computed in one parallel forward pass, and every epsilon_i can be computed in parallel. The Jacobian of y -> epsilon is lower triangular because epsilon_i does not depend on y_j for j > i, and its diagonal entries are 1 / sigma_i(y). The log-determinant is therefore just the sum over dimensions of -log sigma_i(y). This is the key insight: the forward autoregressive map is sequential and slow to sample, but its inverse is fully parallel and has a dead-simple Jacobian.

IAF uses this inverse autoregressive transformation as a normalizing-flow step inside the posterior. The encoder first outputs the parameters mu_0, sigma_0 of an initial diagonal Gaussian and a context vector h that carries information about the observation into every refinement step. A base sample is drawn by reparameterization, z_0 = mu_0 + sigma_0 * epsilon with epsilon ~ N(0, I), and then T autoregressive refinement steps are applied, each with its own masked network. The t-th step outputs mu_t and sigma_t as autoregressive functions of z_{t-1} and h, and updates z_t = mu_t + sigma_t * z_{t-1}. Because each step is autoregressive in z_{t-1}, its Jacobian is triangular with sigma_t on the diagonal. The exact log-density of the final latent is log q(z_T | x) = - sum_i (1/2 epsilon_i^2 + 1/2 log(2*pi) + sum_{t=0}^T log sigma_{t,i}).

To keep a deep chain numerically stable, the implemented step uses an LSTM-style gated blend. The autoregressive network outputs two unconstrained vectors m_t and s_t; sigma_t is obtained as sigmoid(s_t), bounded in (0, 1), and the update is rewritten as z_t = sigma_t * z_{t-1} + (1 - sigma_t) * m_t. This is algebraically equivalent to the general form with mu_t = (1 - sigma_t) * m_t, so the density formula remains unchanged, but multiplying by bounded scales prevents explosion across layers. Initializing s_t with a positive forget-gate bias makes sigma_t start near 1, so the flow begins as the identity and optimization starts from an ordinary factorized Gaussian, only learning deviations from it. Reversing the variable order between steps is a volume-preserving permutation that mixes dependencies in both directions for free.

In deep latent-variable models, IAF pairs naturally with an autoregressive prior p(z_{1:L}) = p(z_L) product_{l<L} p(z_l | z_{l+1:L}). Making the prior autoregressive makes the true posterior more flexible, and because the IAF posterior can match that structure, the KL gap shrinks without restricting the generative model. Inference is arranged by first running a deterministic bottom-up pass to collect evidence, then sampling stochastic variables top-down so each local posterior conditions on both the observation evidence and the prior context. To avoid the zero-information equilibrium q(z|x) ~ p(z) that deep stochastic stacks can collapse into early in training, a free-bits objective replaces the usual KL with max(lambda, KL) per latent group, removing the gradient that pulls unused latents toward the prior.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, mask):
        super().__init__(in_features, out_features)
        self.register_buffer("mask", mask.float())

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class AutoregressiveNN(nn.Module):
    """Masked net: output i depends only on z_{1:i-1}; h can affect every output."""
    def __init__(self, dim, hidden, context_dim):
        super().__init__()
        input_degrees = torch.arange(1, dim + 1)
        if dim == 1:
            hidden_degrees = torch.zeros(hidden, dtype=torch.long)
        else:
            hidden_degrees = torch.arange(hidden) % (dim - 1) + 1
        output_degrees = torch.cat([input_degrees, input_degrees])
        mask_in = (hidden_degrees[:, None] >= input_degrees[None, :]).float()
        mask_out = (output_degrees[:, None] > hidden_degrees[None, :]).float()
        self.input = MaskedLinear(dim, hidden, mask_in)
        self.output = MaskedLinear(hidden, 2 * dim, mask_out)
        self.context = nn.Linear(context_dim, 2 * dim)

    def forward(self, z, h):
        hidden = F.elu(self.input(z))
        m, s = torch.chunk(self.output(hidden) + self.context(h), 2, dim=1)
        return m, s


class Encoder(nn.Module):
    def __init__(self, x_dim, z_dim, context_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, 2 * z_dim + context_dim),
        )
        self.z_dim = z_dim
        self.context_dim = context_dim

    def forward(self, x):
        params = self.net(x.reshape(x.size(0), -1))
        mu0 = params[:, :self.z_dim]
        log_sigma0 = params[:, self.z_dim:2 * self.z_dim].clamp(min=-7.0, max=7.0)
        h = params[:, 2 * self.z_dim:]
        return mu0, log_sigma0, h


class RefiningStep(nn.Module):
    def __init__(self, dim, hidden, context_dim, forget_bias=1.5):
        super().__init__()
        self.net = AutoregressiveNN(dim, hidden, context_dim)
        self.forget_bias = forget_bias

    def forward(self, z, h):
        m, s = self.net(z, h)
        sigma = torch.sigmoid(s + self.forget_bias)         # starts near an identity step
        z = sigma * z + (1.0 - sigma) * m
        log_density_delta = -torch.log(sigma.clamp_min(1e-8)).sum(dim=1)
        return z, log_density_delta


class FlexiblePosterior(nn.Module):
    def __init__(self, x_dim, z_dim, context_dim, n_steps, hidden, reverse_between_steps=True):
        super().__init__()
        self.encoder = Encoder(x_dim, z_dim, context_dim, hidden)
        self.steps = nn.ModuleList(
            [RefiningStep(z_dim, hidden, context_dim) for _ in range(n_steps)]
        )
        self.reverse_between_steps = reverse_between_steps

    def sample_and_log_prob(self, x):
        mu0, log_sigma0, h = self.encoder(x)
        eps = torch.randn_like(mu0)
        z = mu0 + torch.exp(log_sigma0) * eps
        log_q = -(0.5 * eps ** 2 + 0.5 * math.log(2 * math.pi) + log_sigma0).sum(dim=1)
        for i, step in enumerate(self.steps):
            z, log_density_delta = step(z, h)
            log_q = log_q + log_density_delta
            if self.reverse_between_steps and i + 1 < len(self.steps):
                z = z.flip(dims=(1,))
        return z, log_q


def elbo(decoder, prior, x, z, log_q):
    return (decoder.log_prob(x, z) + prior.log_prob(z) - log_q).mean()


def make_optimizer(params):
    return torch.optim.Adam(params)
```
