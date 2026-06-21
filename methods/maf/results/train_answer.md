What I want is a neural density estimator for general-purpose continuous data — UCI tables, natural-image patches, digits — that is at once flexible enough to capture complex, possibly multimodal structure and tractable enough to return the *exact* density $p(x)$ of any externally provided datapoint in a single fast pass on a GPU. That last clause is the whole point: the reason to reach for a density estimator rather than a sample-only generator is to *score* external data, for likelihood-free inference, learned priors, and importance proposals. So the bar is one-pass exact $p(x)$ for any $x$, not a $D$-step sequential recursion.

Two families already give flexibility-with-tractability, and each falls short in a specific way. Autoregressive models factor the joint by the chain rule, $p(x) = \prod_i p(x_i \mid x_{1:i-1})$, and learn each one-dimensional conditional; the log-likelihood is exact and a sum of per-dimension terms. A recurrent version updates a hidden state once per variable and so costs $D$ sequential steps — GPU-unfriendly. Masking fixes that: take a fully-connected net with $D$ inputs and $D$ outputs and drop connections so output $i$ sees only inputs $1, \dots, i-1$; then output $i$ parameterizes the $i$-th conditional and all conditionals come out in one parallel pass. That is MADE. But a single masked model with single-Gaussian conditionals has a ceiling — any density whose conditional of $x_i$ given the past is multimodal is out of reach — and it is *order-sensitive*: with simple conditionals one variable ordering may fit a density perfectly while another cannot, and there is no principled way to pick among the factorially many orders. Normalizing flows are the other family: $p(x) = \pi_u(f^{-1}(x))\,|\det(\partial f^{-1}/\partial x)|$ for an invertible $f$ pushing a simple base $\pi_u$ onto the data, tractable exactly when $f$ is easy to invert and has an easy Jacobian determinant, with both properties surviving composition so a flow deepens by stacking. But Inverse Autoregressive Flow (IAF), built for variational inference, can only cheaply score its *own* samples; to score an external $x$ it must invert through a $D$-step recursion. And Real NVP's coupling layer copies a whole block of coordinates untouched, transforming the rest as a function of only the copied block — strictly less flexible than scaling and shifting *every* coordinate as a function of all previous ones.

I propose Masked Autoregressive Flow (MAF). The starting observation is that an autoregressive model with Gaussian conditionals, viewed literally as a generator, already *is* a normalizing flow — not metaphorically. With $p(x_i \mid x_{1:i-1}) = \mathcal{N}(x_i; \mu_i, (\exp\alpha_i)^2)$ where $\mu_i = f_{\mu_i}(x_{1:i-1})$ and $\alpha_i = f_{\alpha_i}(x_{1:i-1})$ are the conditional mean and log-standard-deviation, sampling draws $u_i \sim \mathcal{N}(0,1)$ and sets
$$x_i = u_i \cdot \exp(\alpha_i) + \mu_i,$$
so the vector of internal random numbers $u \sim \mathcal{N}(0, I)$ maps to data by a function $x = f(u)$. This $f$ is invertible coordinate by coordinate, because $\mu_i$ and $\alpha_i$ depend only on $x_{1:i-1}$, which I already have when scoring a given $x$:
$$u_i = (x_i - \mu_i)\cdot\exp(-\alpha_i).$$
The Jacobian of this $x \to u$ map is triangular by the autoregressive structure — $\partial u_i/\partial x_j = 0$ for $j > i$ — so its determinant is the product of the diagonal. Only the explicit $(x_i - \mu_i)$ factor touches the diagonal, since $\mu_i, \alpha_i$ are functions of earlier coordinates, giving $\partial u_i/\partial x_i = \exp(-\alpha_i)$ and
$$\log\left|\det(\partial f^{-1}/\partial x)\right| = -\sum_i \alpha_i.$$
Substituting the inverse map and this log-det into the change-of-variables formula yields the exact density of any $x$, computed in one masked pass because the entire $x$ is available up front.

What makes a single such layer too rigid is exactly diagnosable: if the $u$'s that $f^{-1}$ recovers from the data are not actually distributed like standard normals — skewed, curved, clumped on a scatter plot — that *is* the symptom of a bad single-Gaussian fit. So rather than declare them standard normal and eat the error, I model their density with another autoregressive flow of the same type, and recurse. Stack flows $M_1, \dots, M_K$: $M_2$ models the random numbers $M_1$ emits, $M_3$ models those of $M_2$, and only the final $u_K$ is declared standard normal. Because each layer is an invertible, tractable-Jacobian flow, the stack is again one, with the total log-det the sum of layer log-dets and the density of $x$ equal to the base density at the fully-transformed code plus that sum. The stack is genuinely more expressive than any one layer: each MADE has unimodal Gaussian conditionals, but a smooth invertible reshaping of a Gaussian can be multimodal in the original coordinates, so the *composition* expresses multimodal conditionals. I keep each layer's conditioners $\{f_{\mu_i}, f_{\alpha_i}\}$ a Gaussian MADE that emits all $(\mu_i, \alpha_i)$ in one masked pass, so the whole $x \to u$ direction, and hence the density, is one parallel pass per layer.

I deliberately keep single-Gaussian conditionals per layer rather than mixtures. A mixture would make each conditional individually multimodal, but then the per-coordinate map $u_i \mapsto x_i$ is no longer a clean invertible affine recursion — a mixture-CDF inverse is messy and the tidy triangular Jacobian with diagonal $\exp(\alpha_i)$ breaks. I would rather keep each layer a simple invertible affine recursion so the flow machinery stays exact and cheap, and buy flexibility by stacking, which composition hands me for free. If I want a universal-approximation guarantee I can place a mixture-of-Gaussians MADE as the *base* density under the stack and train jointly, but the workhorse layer stays single-Gaussian.

The defining design choice is which variables the conditioners read. In my layer $\mu_i, \alpha_i$ are functions of the previous *data* $x_{1:i-1}$. The single-character alternative is to read the previous *random numbers* $u_{1:i-1}$ — that is IAF, and it flips the computational trade-off entirely. Conditioning on $x$: scoring an external $x$ needs the $u$'s, and since every $\mu_i, \alpha_i$ depends on $x_{1:i-1}$ which is all given, one masked pass produces every $u_i$ — but sampling is sequential, because to generate $x_i$ I need $\mu_i, \alpha_i$, which need the $x_{1:i-1}$ I have not yet generated, so $D$ passes in order. Conditioning on $u$ mirrors this: one-pass sampling, but $D$-pass scoring of external data. My task is density estimation of arbitrary data, so I condition on $x$ and get one pass for the thing I do constantly. The duality runs deeper than cost: maximizing my likelihood, i.e. minimizing $\mathrm{KL}(\pi_x \,\|\, p_x)$, equals minimizing $\mathrm{KL}(p_u \,\|\, \pi_u)$ — exactly the variational objective an implicit IAF (with base $\pi_x$ and transform $f^{-1}$) minimizes — which I see by writing the KL out and changing variables $x \mapsto u$ so that $-\log|\det(\partial f^{-1}/\partial x)|$ becomes $+\log|\det(\partial f/\partial u)|$ and the integrand collapses to $\log p_u(u) - \log \pi_u(u)$. And Real NVP's coupling layer is just my layer with a frozen prefix: set $\mu_i = \alpha_i = 0$ for $i \le d$ and let $\mu_i, \alpha_i$ for $i > d$ depend only on $x_{1:d}$. MAF is strictly more flexible — every coordinate scaled and shifted as a function of all previous ones — at the price of the $D$-pass sampling I traced, which for a density estimator that scores constantly and samples rarely is the right trade.

Two practical pieces finish the construction. First, order: a single layer is order-sensitive and I do not know the best order, but since I am stacking I do not have to commit — I use the dataset's natural order for the first layer and *reverse* the order each successive layer, so dependencies one order handles poorly get a second chance under the opposite order deeper in the stack. Second, depth: a deep composition of unnormalized affine maps drifts in scale and is hard to train, so I renormalize activations *between* layers with a layer that is itself a legal flow. Batch normalization is elementwise affine and qualifies. Writing $x$ for the side near the data and $u$ for the side near the base, the batchnorm flow layer is $x = (u - \beta)\odot\exp(-\gamma)\odot(v+\epsilon)^{1/2} + m$ with inverse $u = (x - m)\odot(v+\epsilon)^{-1/2}\odot\exp(\gamma) + \beta$, where $m, v$ are the running mean and variance (minibatch at train, full-train at test) and $\beta, \gamma$ are learned. I exponentiate $\gamma$ rather than use a raw scale for two reasons — it forces the scale positive so the map stays invertible, and it makes the log-det fall out cleanly as
$$\log\left|\det(\partial f^{-1}/\partial x)\right| = \sum_i \left[\gamma_i - \tfrac{1}{2}\log(v_i + \epsilon)\right],$$
which I add to the running log-det sum like any other layer. For conditional estimation $p(x\mid y)$ I augment every MADE's inputs with $y$, dropping no connection from $y$, so the label becomes an extra input to every layer. The conditioners are Gaussian MADEs with degree-based masks $M[j,k] = \mathbf{1}\{\deg_{\text{next}}[j] \ge \deg_{\text{prev}}[k]\}$, trained by maximizing the exact log-likelihood with Adam (step $10^{-4}$ for the deep stack, $10^{-3}$ for a single MADE), $\ell_2$ coefficient $10^{-6}$, and early stopping after 30 epochs without validation improvement.

```python
import math
import numpy as np
from numpy.random import permutation, randint
import torch
import torch.nn as nn
from torch.nn import functional as F


class MaskedLinear(nn.Linear):
    def __init__(self, n_in, n_out, bias=True):
        super().__init__(n_in, n_out, bias)
        self.mask = None
    def set_mask(self, mask):
        self.mask = mask
    def forward(self, x):
        return F.linear(x, self.mask * self.weight, self.bias)


class MADE(nn.Module):
    """Gaussian MADE: one pass -> (mu, alpha) per coordinate, autoregressive by construction."""
    def __init__(self, n_in, hidden_dims, random_order=False, seed=None):
        super().__init__()
        np.random.seed(seed)
        self.n_in, self.n_out, self.hidden_dims = n_in, 2 * n_in, hidden_dims
        dims = [n_in, *hidden_dims, self.n_out]
        layers = []
        for i in range(len(dims) - 2):
            layers += [MaskedLinear(dims[i], dims[i + 1]), nn.ReLU()]
        layers += [MaskedLinear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)
        self._make_masks(random_order)

    def _make_masks(self, random_order):
        L, D = len(self.hidden_dims), self.n_in
        deg = {0: permutation(D) if random_order else np.arange(D)}
        for l in range(L):
            deg[l + 1] = randint(low=deg[l].min(), high=D - 1, size=self.hidden_dims[l])
        deg[L + 1] = deg[0]
        masks = []
        for i in range(len(deg) - 1):
            M = (deg[i + 1][:, None] >= deg[i][None, :]).astype(int)
            masks.append(torch.tensor(M, dtype=torch.float32))
        masks[-1] = torch.cat((masks[-1], masks[-1]), dim=0)
        it = iter(masks)
        for m in self.net.modules():
            if isinstance(m, MaskedLinear):
                m.set_mask(next(it))

    def forward(self, x):
        return self.net(x.float())


class MAFLayer(nn.Module):
    def __init__(self, dim, hidden_dims, reverse):
        super().__init__()
        self.dim, self.made, self.reverse = dim, MADE(dim, hidden_dims), reverse

    def forward(self, x):                                 # x -> u (density, one pass)
        mu, logp = torch.chunk(self.made(x), 2, dim=1)    # 0.5*logp = -alpha
        u = (x - mu) * torch.exp(0.5 * logp)
        u = u.flip(dims=(1,)) if self.reverse else u
        log_det = 0.5 * torch.sum(logp, dim=1)            # -sum_i alpha_i
        return u, log_det

    def backward(self, u):                                # u -> x (sampling, D sequential steps)
        u = u.flip(dims=(1,)) if self.reverse else u
        x = torch.zeros_like(u)
        for i in range(self.dim):
            mu, logp = torch.chunk(self.made(x), 2, dim=1)
            x[:, i] = mu[:, i] + u[:, i] * torch.exp(torch.clamp(-0.5 * logp[:, i], max=10))
        return x


class BatchNormLayer(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(dim))
        self.gamma = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        m, v = x.mean(0), x.var(0)
        u = (x - m) / torch.sqrt(v + self.eps) * torch.exp(self.gamma) + self.beta
        log_det = torch.sum(self.gamma - 0.5 * torch.log(v + self.eps))
        return u, log_det


class MAF(nn.Module):
    def __init__(self, dim, n_layers, hidden_dims, use_reverse=True):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(MAFLayer(dim, hidden_dims, reverse=use_reverse))
            self.layers.append(BatchNormLayer(dim))

    def forward(self, x):
        log_det_sum = torch.zeros(x.shape[0])
        for layer in self.layers:
            x, log_det = layer(x)
            log_det_sum = log_det_sum + log_det
        return x, log_det_sum

    def log_prob(self, x):
        u, log_det = self.forward(x)
        base = -0.5 * (u ** 2 + math.log(2 * math.pi)).sum(dim=1)
        return base + log_det


# training
maf = MAF(dim=784, n_layers=5, hidden_dims=[1024])
optimizer = torch.optim.Adam(maf.parameters(), lr=1e-4, weight_decay=1e-6)
for x in dataloader:
    loss = -maf.log_prob(x).mean()
    optimizer.zero_grad(); loss.backward(); optimizer.step()
```
