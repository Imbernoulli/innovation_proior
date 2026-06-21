We want to train a directed generative model that first draws a continuous latent $z \sim p(z)$ and then draws an observation $x \sim p_\theta(x|z)$, where the likelihood $p_\theta(x|z)$ is an arbitrarily flexible neural network with nonlinear hidden layers — flexible because only such a likelihood can bend itself around the curved manifolds real data lives on. The linear-Gaussian special case is solvable (Roweis & Ghahramani showed PCA is the maximum-likelihood solution of $p(z)=\mathcal N(0,I)$, $p(x|z)=\mathcal N(Wz,\varepsilon I)$ as $\varepsilon\to 0$), but linear is far too weak. The moment the likelihood becomes a nonlinear network, the whole machinery falls apart: the marginal $p(x)=\int p(z)\,p_\theta(x|z)\,dz$ has no closed form, so it can be neither evaluated nor differentiated, and since the posterior $p(z|x)=p_\theta(x|z)\,p(z)/p(x)$ shares that same uncomputable denominator, it is intractable too. The two intractabilities are welded together. On top of this, $N$ is large: we cannot run over the full dataset each step, we want to update on a minibatch or a single point, and any method that runs an inner optimization loop or a sampling chain per datapoint dies at this scale and also blocks inference on a fresh test point.

Every classical tool jams at the same place. The starting object is the exact decomposition that holds for any $q$, $\log p(x) = \mathrm{KL}(q(z)\,\|\,p(z|x)) + L$ with $L = \mathbb E_q[\log p(x,z) - \log q(z)]$; since $\mathrm{KL}\ge 0$, $L$ is the evidence lower bound, tight when $q=p(z|x)$. We optimize $L$ rather than $\log p(x)$ precisely because $\log p(x)$ has the integral stuck inside the log, whereas $L$ moves the log inside an expectation we can sample, and the gap $\log p(x)-L=\mathrm{KL}(q\|p(z|x))\ge 0$ means $L$ is always a guaranteed underestimate that does inference (tightening $q$) and learning (raising the model) at once. EM exploits this by setting $q=p(z|x;\theta_{\text{old}})$ in its E-step — but that requires a closed-form posterior, which a nonlinear network destroys, so EM cannot take its first step. Mean-field variational inference retreats to a tractable factorized $q=\prod_j q_j$, but its coordinate update $\log q_j^*(z_j)=\mathbb E_{-j}[\log p(x,z)]+\text{const}$ needs that expectation in closed form, which holds only under exponential-family conjugacy that the network again breaks; it also cannot represent posterior correlations. The score-function (REINFORCE) gradient drops conjugacy and estimates $\nabla_\phi L$ directly: for $f_\phi(z)=\log p_\theta(x,z)-\log q_\phi(z|x)$ the explicit $\nabla_\phi[-\log q_\phi]$ term cancels because $\mathbb E_q[\nabla_\phi\log q_\phi]=0$, leaving the unbiased estimator $\frac1L\sum_l f_\phi(z^{(l)})\,\nabla_\phi\log q_\phi(z^{(l)}|x)$ — but its variance is fatal. The cause is structural: $\phi$ lives in the measure, so the estimator is a noisy zero-mean score weighted by the integrand; when the parameter-free reward is nearly constant $\approx c$ the true gradient is near zero yet the estimator still emits $\approx c\cdot s(z)$ with variance $c^2\,\mathrm{Var}(s)$ that does not vanish, and the smooth model slope $\partial\log p_\theta(x,z)/\partial z$ is thrown away. Baselines and control variates treat the symptom, not the disease. Monte Carlo EM samples the posterior with HMC but runs a chain per point and is not online; wake-sleep already has the right *shape* — an amortized recognition network $q(z|x)$ trained alongside a generative network $p(x|z)$ — but optimizes two distinct objectives (the sleep phase minimizes a reversed KL on hallucinated data) that do not jointly bound $\log p(x)$. All roads jam on the same operation: taking $\nabla_\phi$ of an expectation whose measure depends on $\phi$.

I propose the variational autoencoder (VAE). Its breakthrough is the reparameterization trick: instead of leaving $\phi$ in the sampling measure, rewrite "draw $z$ from $q_\phi$" as drawing a fixed noise $\varepsilon\sim p(\varepsilon)$ from a distribution independent of $\phi$ and pushing it through a deterministic, differentiable transform $z=g_\phi(\varepsilon,x)$. By change of variables $q_\phi(z|x)\prod_i dz_i = p(\varepsilon)\prod_i d\varepsilon_i$, so $\mathbb E_{q_\phi(z|x)}[f(z)]=\mathbb E_{p(\varepsilon)}[f(g_\phi(\varepsilon,x))]$ — the *same* expectation, but against a $\phi$-independent measure. Now the gradient moves straight inside,
$$\nabla_\phi \mathbb E_{p(\varepsilon)}[f(g_\phi(\varepsilon,x))] = \mathbb E_{p(\varepsilon)}\!\left[\frac{\partial f}{\partial z}\frac{\partial g_\phi}{\partial \phi}\right],$$
running the chain rule all the way back through $g_\phi$. This is exactly the pathwise channel the score-function estimator could not use: when $f$ is nearly constant, $\partial f/\partial z\approx 0$ and the estimator automatically approaches the true gradient of zero, so the variance collapses with the smoothness of $f$ rather than being governed by the score's intrinsic noise. For a diagonal-Gaussian posterior this is the location-scale map $z=\mu+\sigma\odot\varepsilon$, $\varepsilon\sim\mathcal N(0,I)$, which is directly differentiable in $\mu$ and (through $\varepsilon$) in $\sigma$.

The objective is the reparameterized ELBO, which I write in the form $\log p_\theta(x)\ge L(x) = -\mathrm{KL}(q_\phi(z|x)\,\|\,p_\theta(z)) + \mathbb E_{q_\phi(z|x)}[\log p_\theta(x|z)]$. The fully generic estimator (form A, when the KL has no closed form) is $\tilde L^A(x)=\frac1L\sum_l[\log p_\theta(x,z^{(l)})-\log q_\phi(z^{(l)}|x)]$ with $z^{(l)}=g_\phi(\varepsilon^{(l)},x)$; but whenever the KL can be integrated analytically there is no reason to hand a quantity we can compute exactly to Monte Carlo — that only injects variance for nothing — so I keep it exact and sample only the reconstruction, giving the lower-variance form B,
$$\tilde L^B(x) = -\mathrm{KL}(q_\phi(z|x)\,\|\,p_\theta(z)) + \frac1L\sum_l \log p_\theta(x|z^{(l)}),\quad z^{(l)}=g_\phi(\varepsilon^{(l)},x).$$
This is a Rao-Blackwell-flavored move, and it exposes the autoencoder reading directly: the KL is a regularizer keeping the encoded posterior near the prior, and the reconstruction term scores the decode — but the regularizer's strength is fixed by the $\log p(x)$ objective itself, not a hand-tuned hyperparameter as in denoising or contractive autoencoders.

The concrete choices all earn their place. The prior is a centered isotropic $p(z)=\mathcal N(0,I)$: it is parameter-free (nothing to learn, no degenerate prior-posterior accommodation), imposes no preferred direction in latent space, and — most practically — pairs with a Gaussian posterior to give the cleanest closed-form KL. It is not too simple, because the nonlinear decoder deforms this Gaussian into an arbitrarily complex data distribution; the complexity belongs in $p(x|z)$, not the prior. The approximate posterior is a diagonal Gaussian $q_\phi(z|x)=\mathcal N(\mu^{(i)},\sigma^{(i)2}I)$ whose two output heads come from an encoder MLP run on $x^{(i)}$ — diagonal rather than full covariance because that costs only $J$ values of $\mu$ and $J$ of $\log\sigma^2$ with elementwise sampling, KL, and gradients, against the $J(J+1)/2$ parameters plus a Cholesky factorization a full covariance would demand. The encoder emits $\log\sigma^2$ rather than $\sigma$ so that $\exp$ makes the variance automatically positive without a hand-imposed constraint and is numerically stable. With both prior and posterior Gaussian over the same $J$-dimensional space, the $(J/2)\log 2\pi$ normalizers in $\int q\log p$ and $\int q\log q$ cancel exactly, leaving
$$-\mathrm{KL}(q_\phi(z|x)\,\|\,p_\theta(z)) = \tfrac12\sum_{j=1}^J\bigl(1+\log\sigma_j^2-\mu_j^2-\sigma_j^2\bigr),$$
which I derived using $\mathbb E_q[z_j^2]=\mu_j^2+\sigma_j^2$ and $\mathbb E_q[(z_j-\mu_j)^2]=\sigma_j^2$; it is zero when $\mu_j=0,\sigma_j=1$, penalizes $\mu_j$ back toward $0$, and is maximized at $\sigma_j=1$, exactly the regularize-toward-the-prior behavior. The reconstruction term must match the data type or the likelihood is simply wrong: for binary data (binarized MNIST), a Bernoulli decoder ending in a sigmoid gives $\log p(x|z)=\sum_d[x_d\log y_d+(1-x_d)\log(1-y_d)]$, which *is* per-pixel negative binary cross-entropy; for continuous data (Frey Faces), a Gaussian decoder outputting $\mu,\log\sigma^2$ (mean squashed to $(0,1)$ by a sigmoid) gives $\log\mathcal N(x;\mu,\sigma^2I)$, i.e. squared error with a learned variance.

Over the dataset the unbiased full-objective estimator draws a uniform minibatch and rescales, $L(X)\approx \frac NM\sum_{i=1}^M \tilde L(x^{(i)})$ — the $N/M$ corrects the minibatch sum's expectation back to the full-dataset sum, so each minibatch gradient is an unbiased estimate of the full gradient (common fixed-batch code optimizes the unscaled sum and absorbs the constant $M/N$ into the learning rate). For the number of samples per point, the variance scales roughly as $1/(M\cdot L)$, and under a fixed compute budget spreading the budget over more distinct $x$ beats drawing many correlated $z$ for the same $x$, so $L=1$ suffices once $M$ is reasonably large (e.g. $M=100$). The full chain is then genuinely differentiable through one low-variance channel: the encoder emits $(\mu,\log\sigma^2)$, the reparameterization $z=\mu+\sigma\odot\varepsilon$ injects fixed noise as a graph constant, the decoder emits reconstruction parameters, and a single backward pass sends the reconstruction gradient along $\partial\log p/\partial z\cdot\partial z/\partial(\mu,\sigma)$ back to the encoder and through the decoder to $\theta$, while the analytic KL differentiates directly in $(\mu,\log\sigma^2)$ — one objective training encoder and decoder together, with the encoder amortized so inference on a new point is a single forward pass. The same reparameterization reused on $\theta=h_\lambda(\zeta)$ extends this to full variational Bayes over the global parameters with two analytic Gaussian KLs, confirming it is a general operator rather than a Gaussian-only trick; the main algorithm simply does MAP on $\theta$ (a small weight decay corresponding to an $\mathcal N(0,I)$ prior) and trains by ordinary SGD/Adagrad/Adam.

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
