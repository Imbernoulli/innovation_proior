What I want is a disentangled, interpretable representation of images learned with no labels at all: a small set of latent coordinates where moving one of them changes exactly one human-meaningful thing — digit identity, stroke angle, stroke width, face pose, lighting — and nothing else. This is worth wanting precisely because the downstream tasks are unknown at training time, so a representation that lays the factors out as separate, decodable axes is the one that will serve whatever task arrives later. The seductive shortcut is to assume that a good generative model hands me disentanglement for free — surely a model that can synthesize digits has "understood" their factors. But that is false, and it is the wall I have to design around. It is trivial to build a perfect generative model with a useless, arbitrarily scrambled internal code: take any generator and compose its latent input with a wild invertible mixing function, and you get an identical output distribution with a fully entangled code. Matching the data distribution does not pin down the representation at all. So generation alone buys me nothing on disentanglement, and every prior method that does recover disentangled factors pays for it with supervision of some kind. The inverse-graphics network of Kulkarni et al. (2015) clamps minibatches so only one known factor varies; the disentangling Boltzmann machine of Reed et al. (2014) clamps hidden units across pairs of examples known to differ in one factor; the bilinear, multi-view, and semi-supervised generative models all match part of the representation to a supplied label. The one prior method that disentangles fully unsupervised, the higher-order spike-and-slab RBM of Desjardins et al. (2012), handles only discrete factors and costs exponentially in their number, so it does not scale. None of them induce the structure from the data alone on complex images.

Look at what the adversarial generator actually does with its input. It takes a single unstructured noise vector $z \sim p_{\text{noise}}$, maps it to an image, and is trained only to make a discriminator unable to tell its outputs from real data. Nothing in that objective cares *how* $z$ is used; the generator will smear all the factors of variation across all the coordinates in whatever entangled way minimizes the adversarial loss. There is no pressure for any one coordinate of $z$ to mean anything. So if I want a coordinate to mean something, I have to *add* that pressure. I propose InfoGAN. Split the generator's input into two parts: keep a chunk of incompressible noise $z$, and carve out a separate structured latent code $c = (c_1, \dots, c_L)$ that I intend to carry the semantic factors, giving it a factored prior $P(c) = \prod_i P(c_i)$ so the components are independent by construction. The generator is now $G(z, c)$. But merely renaming part of the input "the code" changes nothing, because the indifference is still there: the generator can simply ignore $c$, finding a solution where $P_G(x \mid c) = P_G(x)$. From the adversarial loss's point of view that is a perfectly fine solution — the images still fool the discriminator — so $c$ becomes a *trivial code* carrying no information. What I need is a reason for the generator to not throw $c$ away: a quantity that is large only when the output genuinely depends on the code. "Ignored" means that knowing the image $x = G(z,c)$ tells you nothing about which $c$ produced it, so the thing to force up is exactly how much the image tells you about the code — the mutual information $I(c; G(z,c))$. When $c$ is ignored the output is independent of $c$ and this is zero; when the output strongly depends on $c$ it is high, and high mutual information means the code is *recoverable* from the image, which is precisely "the generator used it." So I add an information-maximizing regularizer to the adversarial game, written as one objective

$$\min_{G,Q}\ \max_D\ V_{\text{InfoGAN}}(D, G, Q) = V(D, G) - \lambda\, L_I(G, Q),$$

where $V(D,G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z,c}[\log(1 - D(G(z,c)))]$ is the ordinary adversarial value function on the full generator input and $\lambda > 0$ weights the information term. It is subtracted because the outer optimization over $G$ is a minimization and I want $G$ to *increase* the information.

The trouble is that I cannot compute $I(c; G(z,c))$ directly. Writing $I(c; G(z,c)) = H(c) - H(c \mid G(z,c))$, the first term is the entropy of my chosen code prior, which I control, but the second requires the *posterior* $P(c \mid x)$ — the distribution over which code produced a given generated image — which I cannot evaluate or sample. The standard escape when an intractable posterior blocks you is to introduce an auxiliary distribution you can evaluate and lower-bound the quantity. So introduce $Q(c \mid x)$, parameterized and meant to approximate the true posterior, and expand the conditional entropy as an expectation over the true posterior, then add and subtract $\log Q$ inside it. The piece $\mathbb{E}_{c' \sim P(c \mid x)}[\log(P/Q)]$ is the KL divergence $D_{\mathrm{KL}}(P(\cdot \mid x) \,\|\, Q(\cdot \mid x)) \geq 0$, so

$$I(c; G(z,c)) = H(c) + \mathbb{E}_{x \sim G(z,c)}\big[\, D_{\mathrm{KL}}(P(\cdot \mid x) \,\|\, Q(\cdot \mid x)) + \mathbb{E}_{c' \sim P(c \mid x)}[\log Q(c' \mid x)] \,\big] \ \geq\ H(c) + \mathbb{E}_{x \sim G(z,c)}\big[\, \mathbb{E}_{c' \sim P(c \mid x)}[\log Q(c' \mid x)] \,\big].$$

I have traded the unknown $\log P(c' \mid x)$ for the known $\log Q(c' \mid x)$ at the cost of dropping a non-negative KL — a genuine lower bound, tight exactly when $Q(\cdot \mid x) = P(\cdot \mid x)$. This is Variational Information Maximization (Barber & Agakov, 2003). But a posterior still lurks: the inner expectation requires sampling $c'$ from $P(c \mid x)$, the very object I cannot sample. The rescue is an expectation identity — for random variables $X, Y$ and any $f$,

$$\mathbb{E}_{x \sim X,\, y \sim Y\mid x}[f(x,y)] = \mathbb{E}_{x \sim X,\, y \sim Y\mid x,\, x' \sim X\mid y}[f(x', y)],$$

which one proves by writing the left side as $\int_x\int_y P(x,y) f(x,y)$, inserting the factor of one $\int_{x'} P(x' \mid y)\, dx' = 1$, then integrating out the original $x$ (since $\int_x P(x,y)\, dx = P(y)$) and renaming the remaining posterior draw $x'$. Its content: instead of "draw $x$, draw $y \mid x$, evaluate $f(x,y)$," I may "draw $x$, draw $y \mid x$, draw a *fresh* $x' \sim X \mid y$, evaluate $f(x', y)$" — same expectation. Applied with $X = c$, $Y = x$, $f = \log Q(c \mid x)$, the posterior sample disappears: instead I sample $c$ from its *prior*, push it through $G$ to form $x = G(z,c)$, and evaluate $\log Q$ on that same $c$ — exactly the forward process I can run. This leaves the variational lower bound

$$L_I(G, Q) = \mathbb{E}_{c \sim P(c),\, x \sim G(z,c)}[\log Q(c \mid x)] + H(c) \ \leq\ I(c; G(z,c)),$$

which is trivial to Monte-Carlo: sample $c \sim P(c)$, $z \sim$ noise, form $x = G(z,c)$, evaluate $\log Q(c \mid x)$. I maximize $L_I$ directly with respect to $Q$ and, through the generated image, with respect to $G$; the entropy $H(c)$ I treat as a constant since the code prior is fixed. When $Q$ reaches the true posterior the KL vanishes, $L_I$ attains its maximum $H(c)$ for discrete codes, and at that maximum the mutual information itself is maximized — so pushing $L_I$ up does the right thing.

Now make $Q$ concrete, and here is the economy that makes the whole thing nearly free. $Q(c \mid x)$ has to look at an image and produce a distribution over the code — a recognition network needing exactly the image features a discriminator already learns. So instead of a separate network, let $Q$ *share the entire convolutional body* of the discriminator $D$ and add a single extra fully-connected head that reads the shared features and outputs the parameters of $Q(c \mid x)$. The cost over a plain adversarial network is then one output layer. What the head outputs depends on the code type. For a categorical $c_i$ — say a 1-of-10 variable meant to capture digit identity — $Q(c_i \mid x)$ is a softmax over the $K$ categories, and $\log Q$ for the sampled one-hot is the log-probability of the true category, so maximizing the $L_I$ term is *minimizing the cross-entropy* between $Q$'s softmax and the code actually fed to $G$; here $\lambda = 1$ works, since the cross-entropy is naturally on the adversarial loss's scale. For a continuous $c_j$, $Q(c_j \mid x)$ is a factored Gaussian: the head outputs a mean and a standard deviation (the std via an exponential transform to stay positive), and $\log Q$ is the Gaussian log-density, so maximizing it is minimizing a Gaussian NLL — which, with fixed variance, reduces to mean-squared error between the predicted mean and the true code. Because the continuous $L_I$ term involves a *differential* entropy and can sit on a different scale, a smaller $\lambda$ (e.g. $0.1$) is used there. The information regularizer thus concretely becomes: generate an image from a sampled code, run the shared $Q$ head on it, and penalize how badly $Q$ recovers the code that produced it — cross-entropy for the categorical parts, Gaussian NLL for the continuous parts. I build on the known stable convolutional recipe — up-convolutional generator, leaky-ReLU discriminator, batchnorm, Adam, with a lower learning rate for the discriminator ($2\times10^{-4}$) than the generator ($10^{-3}$) — which keeps training stable with no new trick; in practice $L_I$ even converges faster than the adversarial objective. Seen from a distance, the pair $(P_G(x \mid c), Q(c \mid x))$ is a Helmholtz machine, and maximizing $L_I$ with respect to $Q$ is exactly the Wake-Sleep "sleep" update on dreamed samples; what is new is that I *additionally* apply $L_I$ to the generator — a second sleep-like update on generated samples — which reaches into $G$ and forbids it from ignoring the code, forcing it to convey the code rather than merely fitting the data marginal. The method is a training pressure, not a theorem: I still validate disentanglement by sweeping one code coordinate while fixing the rest and checking that a single interpretable thing changes.

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
