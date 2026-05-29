# The Variational Autoencoder (VAE)

The variational autoencoder trains a deep directed latent-variable model — `z ~ p(z)`, `x ~ p_θ(x|z)` with a neural-network likelihood — even when the marginal `p(x) = ∫ p(z)p_θ(x|z) dz` and the posterior `p(z|x)` are both intractable, and does so at scale with ordinary minibatch SGD. Three ideas make it work: a **reparameterized estimator of the evidence lower bound (SGVB)** that is differentiable and low-variance; an **amortized recognition network** `q_φ(z|x)` (the encoder) that predicts per-datapoint variational parameters with one shared net and is trained jointly with the generative network `p_θ(x|z)` (the decoder); and a closed-form Gaussian KL term that turns the whole thing into a regularized autoencoder.

## The problem

For an i.i.d. dataset `{x^(i)}`, we want (1) approximate ML/MAP estimation of the generative parameters `θ`, (2) fast approximate posterior inference `z ≈ q(z|x)`, and (3) a usable handle on `p(x)`. With a nonlinear `p_θ(x|z)`: EM has no closed-form E-step (the posterior is intractable), mean-field VI loses the analytic/conjugate expectations its coordinate updates need, the score-function (REINFORCE) gradient is unbiased but too high-variance (for the ELBO, the explicit `∇_φ[-log q_φ]` term cancels because `E_q[∇_φ log q_φ]=0`, leaving a noisy integrand-times-score estimator that does not use the pathwise `∂ log p_θ(x,z)/∂z` signal), per-datapoint MCMC is too slow for large `N`, and wake-sleep optimizes two objectives that do not jointly bound `log p(x)`. The common failure point: differentiating `∇_φ E_{q_φ(z|x)}[f_φ(z)]` when the sampling measure depends on `φ`.

## Key idea: the reparameterization trick

Rewrite the latent as a deterministic, differentiable transform of a *fixed* noise variable, moving `φ` out of the measure and into the integrand:

```
z = g_φ(ε, x),    ε ~ p(ε)   (independent of φ)
```

By change of variables `q_φ(z|x) ∏_i dz_i = p(ε) ∏_i dε_i`, so `E_{q_φ(z|x)}[f(z)] = E_{p(ε)}[f(g_φ(ε, x))]`, an expectation against a `φ`-independent measure. Now `∇_φ` moves inside and flows through `g_φ` via the chain rule `∂f/∂z · ∂g_φ/∂φ` — the pathwise gradient the score-function estimator throws away — giving much lower variance at the same sample count. For a diagonal-Gaussian posterior the transform is location-scale:

```
z = μ + σ ⊙ ε,    ε ~ N(0, I).
```

## The objective (SGVB / ELBO)

The ELBO comes from the exact decomposition `log p_θ(x) = KL(q_φ(z|x) ‖ p_θ(z|x)) + L(x)`, KL ≥ 0, so

```
log p_θ(x) ≥ L(x) = - KL( q_φ(z|x) || p_θ(z) ) + E_{q_φ(z|x)}[ log p_θ(x|z) ].
```

The generic reparameterized estimator (form A, used when the KL has no closed form):

```
L̃^A(x) = (1/L) Σ_l [ log p_θ(x, z^(l)) - log q_φ(z^(l)|x) ],   z^(l) = g_φ(ε^(l), x),  ε^(l) ~ p(ε).
```

When the KL is analytic, keep it exact and sample only the reconstruction — the lower-variance form B:

```
L̃^B(x) = - KL( q_φ(z|x) || p_θ(z) ) + (1/L) Σ_l log p_θ(x | z^(l)),   z^(l) = g_φ(ε^(l), x),  ε^(l) ~ p(ε).
```

With `p(z) = N(0, I)` and `q_φ(z|x) = N(μ, σ² I)` (the encoder outputs `μ` and `log σ²`), the `(J/2)log 2π` constants in `∫q log p` and `∫q log q` cancel and the KL is closed-form (`J = dim z`):

```
- KL = (1/2) Σ_{j=1}^J ( 1 + log σ_j² - μ_j² - σ_j² ).
```

The reconstruction term `log p_θ(x|z)` is the observation log-likelihood matched to the data type: **Bernoulli** for binary data (`log p = Σ_d [ x_d log y_d + (1-x_d) log(1-y_d) ]`, i.e. negative per-pixel BCE, decoder ends in a sigmoid), or **Gaussian** for continuous data (`log N(x; μ, σ²I)`, decoder outputs `μ` and `log σ²`, mean squashed to `(0,1)` by a sigmoid).

Over the dataset, the unbiased full-objective minibatch estimator is `L(X) ≈ (N/M) Σ_{i=1}^M L(x^(i))` (the `N/M` rescales the minibatch sum's expectation back to the full-dataset sum). The PyTorch example code below optimizes the unscaled minibatch sum; for a fixed minibatch size this differs by the constant factor `M/N`, which is absorbed into the learning-rate scale. In practice `L = 1` sample per datapoint suffices when `M` is reasonably large (e.g. `M = 100`), since averaging over many distinct points in the minibatch controls the variance better than multiple samples of the same point. Maximize with SGD / Adagrad / Adam (a small weight decay on `θ` corresponds to an `N(0,I)` prior on `θ`, making it approximate MAP).

## Algorithm

```
Initialize θ, φ
repeat
    X^M ← random minibatch of M datapoints
    ε   ← random samples from p(ε)
    g   ← ∇_{θ,φ} L̃^M(X^M, ε)        # one backward pass through encoder + decoder
    θ, φ ← update with g (SGD / Adagrad / Adam)
until convergence
```

## Working code (PyTorch, MNIST, Bernoulli decoder)

```python
import torch
from torch import nn, optim
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        # encoder q_phi(z|x): 784 -> 400 -> {mu(20), logvar(20)}  (amortized: one shared net for all x)
        self.fc1  = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)   # mu
        self.fc22 = nn.Linear(400, 20)   # log sigma^2
        # decoder p_theta(x|z): 20 -> 400 -> 784 Bernoulli probabilities
        self.fc3  = nn.Linear(20, 400)
        self.fc4  = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)        # sigma; output log-variance so exp(.) is positive & stable
        eps = torch.randn_like(std)          # fixed noise, enters the graph as a constant
        return mu + eps * std                # z = mu + sigma * eps

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))   # Bernoulli probabilities

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)  # L = 1
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    # -E_q[log p(x|z)] single-sample estimate: per-pixel negative Bernoulli log-likelihood
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    # analytic KL: -(-KL) = -1/2 * sum(1 + logvar - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD                         # minimizing this maximizes the ELBO

model = VAE()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train(epoch, train_loader):
    model.train()
    for data, _ in train_loader:
        optimizer.zero_grad()
        recon, mu, logvar = model(data)
        # Unscaled minibatch sum, matching the common PyTorch example; multiply by N/M for the full-objective estimator.
        loss = loss_function(recon, data, mu, logvar)
        loss.backward()                      # grad w.r.t. theta AND phi in one pass
        optimizer.step()

# Generating new data after training: sample z ~ N(0, I), decode.
def sample(n=64):
    with torch.no_grad():
        z = torch.randn(n, 20)
        return model.decode(z).view(n, 1, 28, 28)
```

## Implementation notes

- The canonical implementation uses ReLU activations and Adam (faster to converge); the original formulation used `tanh`/`sigmoid` MLPs with Adagrad/SGD — interchangeable choices of nonlinearity and optimizer over the same objective.
- For continuous data (e.g. Frey Faces) the decoder becomes a Gaussian MLP (`μ = W₄h + b₄`, `log σ² = W₅h + b₅`, `h = tanh(W₃z + b₃)`), with the output mean squashed to `(0,1)` by a sigmoid; the reconstruction term becomes a Gaussian log-likelihood while the analytic Gaussian KL term is unchanged.
- A diagonal-covariance Gaussian `q` is a simplifying choice (cheap sampling/KL, linear in `J`), not a limitation; richer posteriors can be substituted at the cost of a harder KL.
- The framework extends to full variational Bayes over the global parameters `θ` using separate variational parameters `λ` for `q_λ(θ)` and `φ` for `q_φ(z|x)`. A second reparameterization `θ = h_λ(ζ)`, `ζ ~ p(ζ)`, gives two analytic Gaussian KLs when the priors/posteriors are Gaussian: `KL(q_λ(θ)||p(θ))` and `KL(q_φ(z|x)||p(z))`. The main algorithm uses MAP point estimation of `θ` for simplicity.
