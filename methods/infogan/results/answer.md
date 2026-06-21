# InfoGAN — disentangling by maximizing mutual information

## The problem it solves

Learn a *disentangled*, interpretable representation — a small set of latent variables that tend to
control separate semantic factors (digit identity, rotation, width, pose, lighting, glasses) — fully
**unsupervised**, on complex image datasets, at negligible extra cost over a normal adversarial
network. Generation alone cannot deliver this: a perfect generator can carry an arbitrarily entangled
latent code, and the adversarial objective is indifferent to *how* the generator uses its input, so a
plain latent vector can entangle all factors.

## The key idea

Split the generator's input into incompressible noise `z` and a structured latent code
`c = (c_1, …, c_L)` with a factored prior `P(c) = ∏_i P(c_i)`, so `G = G(z, c)`. Left alone, the
generator will *ignore* `c` (a "trivial code": `P_G(x|c) = P_G(x)`). Force it to use `c` by adding an
information-theoretic regularizer that maximizes the **mutual information** `I(c; G(z, c))` between the
code and the generated image — high mutual information means the code is recoverable from the image,
i.e. the generator actually used it.

## The objective

    min_{G,Q}  max_D   V_InfoGAN(D, G, Q) = V(D, G) − λ L_I(G, Q),

with `V(D,G) = E_{x~p_data}[log D(x)] + E_{z~p(z),c~P(c)}[log(1 − D(G(z,c)))]` the usual adversarial
value function on the full generator input.

**Why a bound.** `I(c; G(z,c)) = H(c) − H(c | G(z,c))` requires the intractable posterior `P(c|x)`.
Introduce an auxiliary distribution `Q(c|x)` ≈ `P(c|x)` and lower-bound (Variational Information
Maximization):

    I(c; G(z,c)) = H(c) + E_{x~G(z,c)}[ D_KL(P(·|x) ‖ Q(·|x)) + E_{c'~P(c|x)}[log Q(c'|x)] ]
                 ≥ H(c) + E_{x~G(z,c)}[ E_{c'~P(c|x)}[log Q(c'|x)] ]      (drop KL ≥ 0; tight when Q = P).

**Removing the posterior sample.** An expectation identity — for `X, Y, f`:
`E_{x~X, y~Y|x}[f(x,y)] = E_{x~X, y~Y|x, x'~X|y}[f(x',y)]` (proved by inserting `∫ P(x'|y)dx' = 1`) —
lets the inner posterior expectation be replaced by sampling the code from its prior and generating:

    L_I(G, Q) = E_{c ~ P(c), x ~ G(z,c)}[ log Q(c | x) ] + H(c)   ≤   I(c; G(z, c)).

`L_I` is Monte-Carlo-able: sample `c ~ P(c)`, `z ~ noise`, form `x = G(z,c)`, evaluate `log Q(c|x)`.
`H(c)` is treated as a constant because the code prior is fixed. Maximize `L_I` w.r.t. `Q` directly
and w.r.t. `G` through the generated image. For finite discrete codes, when `Q` matches the true
posterior and the generated image determines `c`, `L_I` reaches `H(c)` and the mutual information is
maximal.

## How L_I becomes a concrete loss

`Q` is a recognition network that shares the discriminator's convolutional feature extractor and adds
a small fully-connected recognition head — so InfoGAN adds negligible compute over a plain adversarial
network. The head's parameterization follows the code type:

- **Categorical `c_i`** (e.g. 1-of-10): `Q(c_i|x)` = softmax. Then `−log Q` = cross-entropy between
  the softmax and the sampled one-hot code; `λ = 1` works.
- **Continuous `c_j`**: `Q(c_j|x)` = factored Gaussian; the head outputs mean and std (std via an
  exponential transform to stay positive); `−log Q` = Gaussian NLL (≈ MSE for fixed variance). Use a
  smaller `λ` (e.g. 0.1) because the continuous `L_I` involves differential entropy.

Built on the stable convolutional adversarial recipe (up-convolutional generator, leaky-ReLU
discriminator, batchnorm, Adam). A typical setup uses `lr = 2e-4` for `D`, `lr = 1e-3` for `G`,
`lambda = 1`, leaky-ReLU slope `0.1`, and fixed-variance Gaussian NLL for continuous MNIST codes.
No new stabilization trick is needed, and `L_I` typically converges faster than the adversarial
objective.

**Interpretation.** `(P_G(x|c), Q(c|x))` is a Helmholtz machine; maximizing `L_I` w.r.t. `Q` is the
Wake-Sleep "sleep" update, and InfoGAN *additionally* applies it to `G` (a second sleep-like update on
generated samples), explicitly forcing the generator to convey information through the code.

## Training core

```python
import math
import torch
import torch.nn.functional as F

TINY = 1e-8
latent_dim, n_classes, code_dim = 62, 10, 2
info_reg_coeff = 1.0

# G maps concat(noise, one_hot_category, continuous_code) -> image.
# D_Q shares one image feature trunk, then returns:
#   d_prob: sigmoid real/fake probability
#   q_cat_logits: logits for Q(c_discrete | x)
#   q_cont_mean, q_cont_log_std: diagonal-Gaussian parameters for Q(c_continuous | x)
G, D_Q = Generator(), DiscriminatorWithRecognitionHead()
opt_g = torch.optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.999))
opt_dq = torch.optim.Adam(D_Q.parameters(), lr=2e-4, betas=(0.5, 0.999))

def sample_codes(batch, device):
    noise = torch.randn(batch, latent_dim, device=device)
    label = torch.randint(n_classes, (batch,), device=device)
    one_hot = F.one_hot(label, n_classes).float()
    cont = torch.empty(batch, code_dim, device=device).uniform_(-1.0, 1.0)
    return noise, one_hot, cont

def log_q_categorical(one_hot, q_cat_logits):
    return (one_hot * F.log_softmax(q_cat_logits, dim=1)).sum(dim=1)

def log_q_gaussian(value, mean, log_std=None):
    # Use learned std when supplied; omitting it gives fixed-variance Gaussian NLL.
    if log_std is None:
        log_std = torch.zeros_like(mean)
    std = torch.exp(log_std)
    return (-0.5 * math.log(2.0 * math.pi) - log_std - 0.5 * ((value - mean) / std).pow(2)).sum(dim=1)

def mi_estimate(one_hot, cont, q_cat_logits, q_cont_mean, q_cont_log_std=None):
    cat_prior = math.log(n_classes)
    zero = torch.zeros_like(cont)
    cont_prior = -log_q_gaussian(cont, zero, zero).mean()
    return (
        cat_prior + log_q_categorical(one_hot, q_cat_logits).mean()
        + cont_prior + log_q_gaussian(cont, q_cont_mean, q_cont_log_std).mean()
    )

def train_step(real):
    batch, device = real.size(0), real.device
    noise, one_hot, cont = sample_codes(batch, device)
    latent = torch.cat([noise, one_hot, cont], dim=1)

    # D/Q update: canonical loss is -log D(real)-log(1-D(fake)) - lambda * L_I.
    with torch.no_grad():
        fake = G(latent)
    real_prob, _, _, _ = D_Q(real)
    fake_prob, q_cat_logits, q_cont_mean, q_cont_log_std = D_Q(fake)
    d_adv = -(torch.log(real_prob + TINY) + torch.log(1.0 - fake_prob + TINY)).mean()
    mi = mi_estimate(one_hot, cont, q_cat_logits, q_cont_mean, q_cont_log_std)
    d_loss = d_adv - info_reg_coeff * mi
    opt_dq.zero_grad()
    d_loss.backward()
    opt_dq.step()

    # G update: the reference trainer uses the non-saturating -log D(fake) generator loss
    # and subtracts the same MI estimate, but optimizes only generator parameters.
    for p in D_Q.parameters():
        p.requires_grad_(False)
    noise, one_hot, cont = sample_codes(batch, device)
    fake = G(torch.cat([noise, one_hot, cont], dim=1))
    fake_prob, q_cat_logits, q_cont_mean, q_cont_log_std = D_Q(fake)
    mi = mi_estimate(one_hot, cont, q_cat_logits, q_cont_mean, q_cont_log_std)
    g_loss = -torch.log(fake_prob + TINY).mean() - info_reg_coeff * mi
    opt_g.zero_grad()
    g_loss.backward()
    opt_g.step()
    for p in D_Q.parameters():
        p.requires_grad_(True)
    return d_loss.detach(), g_loss.detach(), mi.detach()
```

## Why it works

The mutual-information regularizer makes "ignore the code" a bad solution: the generator can only
raise the bound by producing images from which the sampled code is recoverable. The variational bound
`L_I` makes that intractable information term differentiable and cheap, and the factored code prior
pushes separate code variables to carry separate recoverable factors. The method still validates
disentanglement empirically by sweeping one code variable while holding the others fixed; it is a
training pressure, not a theorem that every coordinate must become a human factor.
