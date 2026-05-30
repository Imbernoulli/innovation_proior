# InfoGAN — disentangling by maximizing mutual information

## The problem it solves

Learn a *disentangled*, interpretable representation — one latent coordinate per semantic factor
(digit identity, rotation, width, pose, lighting, glasses) — fully **unsupervised**, on complex image
datasets, at negligible extra cost over a normal adversarial network. Generation alone cannot deliver
this: a perfect generator can carry an arbitrarily entangled latent code, and the adversarial
objective is indifferent to *how* the generator uses its input, so a plain latent vector entangles all
factors.

## The key idea

Split the generator's input into incompressible noise `z` and a structured latent code
`c = (c_1, …, c_L)` with a factored prior `P(c) = ∏_i P(c_i)`, so `G = G(z, c)`. Left alone, the
generator will *ignore* `c` (a "trivial code": `P_G(x|c) = P_G(x)`). Force it to use `c` by adding an
information-theoretic regularizer that maximizes the **mutual information** `I(c; G(z, c))` between the
code and the generated image — high mutual information means the code is recoverable from the image,
i.e. the generator actually used it.

## The objective

    min_{G,Q}  max_D   V_InfoGAN(D, G, Q) = V(D, G) − λ L_I(G, Q),

with `V(D,G) = E_{x~p_data}[log D(x)] + E_z[log(1 − D(G(z,c)))]` the usual adversarial value function.

**Why a bound.** `I(c; G(z,c)) = H(c) − H(c | G(z,c))` requires the intractable posterior `P(c|x)`.
Introduce an auxiliary distribution `Q(c|x)` ≈ `P(c|x)` and lower-bound (Variational Information
Maximization):

    I(c; G(z,c)) = H(c) + E_{x~G(z,c)}[ D_KL(P(·|x) ‖ Q(·|x)) + E_{c'~P(c|x)}[log Q(c'|x)] ]
                 ≥ H(c) + E_{x~G(z,c)}[ E_{c'~P(c|x)}[log Q(c'|x)] ]      (drop KL ≥ 0; tight when Q = P).

**Removing the posterior sample.** A change-of-variables identity — for `X, Y, f`:
`E_{x~X, y~Y|x}[f(x,y)] = E_{x~X, y~Y|x, x'~X|y}[f(x',y)]` (proved by inserting `∫ P(x'|y)dx' = 1`) —
lets the inner posterior expectation be replaced by sampling the code from its prior and generating:

    L_I(G, Q) = E_{c ~ P(c), x ~ G(z,c)}[ log Q(c | x) ] + H(c)   ≤   I(c; G(z, c)).

`L_I` is Monte-Carlo-able: sample `c ~ P(c)`, `z ~ noise`, form `x = G(z,c)`, evaluate `log Q(c|x)`.
`H(c)` is treated as a constant (fixed code prior). Maximize `L_I` w.r.t. `Q` directly and w.r.t. `G`
through the differentiable `x = G(z,c)`. At tightness `L_I = H(c)` and the mutual information is
maximized.

## How L_I becomes a concrete loss

`Q` is a recognition network that **shares the entire convolutional body of the discriminator** plus
one final fully-connected layer — so InfoGAN adds negligible compute over a plain adversarial network.
The head's parameterization follows the code type:

- **Categorical `c_i`** (e.g. 1-of-10): `Q(c_i|x)` = softmax. Then `−log Q` = cross-entropy between
  the softmax and the sampled one-hot code; `λ = 1` works.
- **Continuous `c_j`**: `Q(c_j|x)` = factored Gaussian; the head outputs mean and std (std via an
  exponential transform to stay positive); `−log Q` = Gaussian NLL (≈ MSE for fixed variance). Use a
  smaller `λ` (e.g. 0.1) because the continuous `L_I` involves differential entropy.

Built on the stable convolutional adversarial recipe (up-convolutional generator, leaky-ReLU
discriminator, batchnorm, Adam, `lr ≈ 2e-4`); no new stabilization trick is needed, and `L_I`
typically converges faster than the adversarial objective.

**Interpretation.** `(P_G(x|c), Q(c|x))` is a Helmholtz machine; maximizing `L_I` w.r.t. `Q` is the
Wake-Sleep "sleep" update, and InfoGAN *additionally* applies it to `G` (a second sleep-like update on
generated samples), explicitly forcing the generator to convey information through the code.

## Working code

```python
import torch
import torch.nn as nn
import itertools

latent_dim, n_classes, code_dim = 62, 10, 2    # 62 noise + 1 ten-way categorical + 2 continuous

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = latent_dim + n_classes + code_dim
        self.init_size = 32 // 4
        self.l1 = nn.Sequential(nn.Linear(input_dim, 128 * self.init_size ** 2))
        self.conv = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2), nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128, 0.8), nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2), nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64, 0.8), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, 1, 1), nn.Tanh(),
        )
    def forward(self, noise, code_cat, code_cont):
        x = torch.cat((noise, code_cat, code_cont), -1)
        out = self.l1(x).view(x.size(0), 128, self.init_size, self.init_size)
        return self.conv(out)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        def block(i, o, bn=True):
            b = [nn.Conv2d(i, o, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn: b.append(nn.BatchNorm2d(o, 0.8))
            return b
        self.body = nn.Sequential(*block(1, 16, bn=False), *block(16, 32),
                                  *block(32, 64), *block(64, 128))
        ds = 32 // (2 ** 4)
        self.validity = nn.Linear(128 * ds ** 2, 1)
        self.q_cat  = nn.Sequential(nn.Linear(128 * ds ** 2, n_classes), nn.Softmax(dim=1))
        self.q_cont = nn.Linear(128 * ds ** 2, code_dim)
    def forward(self, x):
        f = self.body(x).view(x.size(0), -1)
        return self.validity(f), self.q_cat(f), self.q_cont(f)

adversarial_loss = nn.BCEWithLogitsLoss()
categorical_loss = nn.CrossEntropyLoss()        # -log Q, categorical
continuous_loss  = nn.MSELoss()                 # -log Q, fixed-variance Gaussian
lambda_cat, lambda_con = 1.0, 0.1

G, D = Generator(), Discriminator()
opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_info = torch.optim.Adam(itertools.chain(G.parameters(), D.parameters()),
                            lr=2e-4, betas=(0.5, 0.999))

def sample_codes(b):
    z   = torch.randn(b, latent_dim)
    lab = torch.randint(0, n_classes, (b,))
    oh  = torch.eye(n_classes)[lab]
    con = torch.empty(b, code_dim).uniform_(-1, 1)
    return z, oh, lab, con

def train_step(real):
    b = real.size(0)
    valid = torch.ones(b, 1); fake = torch.zeros(b, 1)
    z, oh, lab, con = sample_codes(b)
    # 1) G adversarial
    opt_G.zero_grad()
    gen = G(z, oh, con); v, _, _ = D(gen)
    adversarial_loss(v, valid).backward(); opt_G.step()
    # 2) D adversarial
    opt_D.zero_grad()
    v_real, _, _ = D(real); v_fake, _, _ = D(gen.detach())
    ((adversarial_loss(v_real, valid) + adversarial_loss(v_fake, fake)) / 2).backward(); opt_D.step()
    # 3) information: maximize L_I (over G and Q jointly)
    opt_info.zero_grad()
    z, oh, lab, con = sample_codes(b)
    gen = G(z, oh, con); _, q_cat, q_cont = D(gen)
    info = lambda_cat * categorical_loss(q_cat, lab) + lambda_con * continuous_loss(q_cont, con)
    info.backward(); opt_info.step()
    return info
```

## Why it works

The mutual-information regularizer makes "ignore the code" a high-loss solution: the generator can
only avoid the penalty by producing images from which the code is recoverable, which is exactly the
condition that each code coordinate maps to a distinct, recoverable factor of variation. The
variational bound `L_I` makes that intractable information term differentiable and cheap (one shared
recognition head), and the factored code prior means the recovered factors come out independent — so
sweeping one coordinate moves one interpretable attribute, learned with no labels.
