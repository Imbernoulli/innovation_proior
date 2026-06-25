Let me start from the thing I actually want and refuse to assume I can get it for free. I want a disentangled representation of images, learned with no labels: a handful of coordinates where moving one of them changes exactly one human-meaningful thing about the image — digit identity, stroke angle, stroke width, face pose, lighting — and nothing else. The reason this is worth wanting is that the downstream tasks are unknown at training time, so a representation that lays the factors out as separate, decodable axes is the one that will be useful for whatever task shows up later.

The seductive idea is that a good generative model will hand me disentanglement for free — surely a model that can synthesize digits has "understood" the factors. But that's false, and I should say why before I lean on it. It is trivial to build a perfect generative model with a useless, arbitrarily scrambled internal representation: take any generator and compose its latent input with some wild invertible mixing function, and you get the same output distribution with a totally entangled code. So matching the data distribution does not pin down the representation at all. Generation alone buys me nothing on disentanglement. That's the wall I have to design around.

Look at what the adversarial generator actually does with its input. It takes a single unstructured noise vector `z ~ p_noise` and maps it to an image, trained only to make a discriminator unable to tell its outputs from real data. Nothing in that objective cares *how* `z` is used. The generator will use `z` in whatever entangled way minimizes the adversarial loss, smearing all the factors of variation across all the coordinates. There is no pressure for "coordinate 3 of `z`" to mean anything. So if I want a coordinate to mean something, I have to *add* that pressure, because the existing objective is completely indifferent to it.

Here's the move. Split the generator's input into two parts: keep a chunk of unstructured, incompressible noise `z`, and carve out a separate structured latent code `c = (c_1, …, c_L)` that I *intend* to carry the semantic factors. Give `c` a factored prior `P(c) = ∏_i P(c_i)`, so the components are independent by construction — that independence is exactly what I want each one to become an independent factor of variation. The generator is now `G(z, c)`. I'd like `c_1` to end up being digit identity, `c_2` rotation, and so on.

But just renaming part of the input "the code" does nothing — the indifference problem is still there. Walk through it. The generator can simply *ignore* `c`: it can find a solution where the conditional output distribution doesn't depend on the code at all, `P_G(x | c) = P_G(x)`. From the adversarial loss's point of view that's a perfectly fine solution — the images still fool the discriminator. So `c` becomes a *trivial code*, carrying no information, and I've gained nothing. The generator pays no penalty for throwing the code away, so it throws it away.

What I need is a reason for the generator to *not* throw `c` away — a quantity that is large only when the output genuinely depends on the code and is small when the code is ignored. Stare at "ignored": ignored means that knowing the image `x = G(z,c)` tells you nothing about which `c` produced it. That is exactly the statement that `c` and `G(z,c)` are statistically independent, and there is a single scalar that vanishes precisely under independence and grows as dependence grows — the mutual information `I(c; G(z,c))`. Let me check the two endpoints aren't just slogans. If `c` is ignored, the output is independent of `c`, so `I = 0`. At the other extreme, suppose the image determines `c` outright — the map from `c` to `x` is injective for fixed `z`, so `c` is recoverable. Then `H(c | G(z,c)) = 0` and `I = H(c) − 0 = H(c)`, the full entropy of the code. So mutual information measures, in nats, how much of the code survives into the image; and "survives into the image" is just a synonym for "can be read back off the image," which is what I mean by the generator using `c`. That makes it the right thing to push up. So add an information-maximizing regularizer to the adversarial game — I want `I(c; G(z,c))` large. Written as one objective,

    min_G max_D  V_I(D, G) = V(D, G) − λ I(c; G(z, c)),

where `V(D,G)` is the ordinary adversarial value function and `λ > 0` weights the information term. (It's subtracted because the outer optimization over `G` is a minimization, and I want `G` to *increase* the information — minimizing `−λ I` is maximizing `I`.)

Now the trouble. I can't actually compute `I(c; G(z,c))`. Write it out:
`I(c; G(z,c)) = H(c) − H(c | G(z,c))`. The first term `H(c)` is fine — it's the entropy of my chosen code prior, which I control. The second term is the conditional entropy, and to evaluate it I need the *posterior* `P(c | x)` — the distribution over which code produced a given generated image. I have no handle on that posterior; I can sample `x` from `G(z,c)`, but I cannot evaluate `P(c|x)` or sample from it. So the information term, as written, is intractable. This is the same shape of obstacle that shows up whenever an objective secretly contains a posterior I can't touch.

The standard escape when an intractable posterior blocks you is to introduce an auxiliary distribution you *can* evaluate and lower-bound the quantity. So introduce `Q(c | x)`, a distribution I'll parameterize and that's meant to approximate the true posterior `P(c | x)`. Let me push the algebra and see what bound falls out. Start from the conditional-entropy form and expand the conditional entropy as an expectation over the true posterior:

    I(c; G(z,c)) = H(c) − H(c | G(z,c))
                 = H(c) + E_{x ~ G(z,c)}[ E_{c' ~ P(c|x)}[ log P(c' | x) ] ].

That's just the definition: `−H(c|x) = E_{c'~P(c|x)}[log P(c'|x)]`, and I average over `x`. Now do the trick of adding and subtracting `log Q(c'|x)` inside the inner expectation:

    E_{c'~P(c|x)}[log P(c'|x)] = E_{c'~P(c|x)}[log Q(c'|x)] + E_{c'~P(c|x)}[ log (P(c'|x) / Q(c'|x)) ].

The second piece is `E_{c'~P(c|x)}[log(P/Q)] = D_KL(P(·|x) ‖ Q(·|x))`, which is a KL divergence and therefore `≥ 0`. So

    I(c; G(z,c)) = H(c) + E_{x~G(z,c)}[ D_KL(P(·|x) ‖ Q(·|x)) + E_{c'~P(c|x)}[log Q(c'|x)] ]
                 ≥ H(c) + E_{x~G(z,c)}[ E_{c'~P(c|x)}[log Q(c'|x)] ].

I've traded the unknown `log P(c'|x)` for the *known* `log Q(c'|x)` at the cost of dropping a non-negative KL term — a genuine lower bound. And it's tight exactly when the KL is zero in expectation, i.e. when `Q(·|x) = P(·|x)`: the auxiliary distribution becomes the true posterior. I can't promise `Q` actually reaches the posterior — that depends on how expressive I make it and how well it trains — so the honest statement is just that the gap is the KL, and any progress `Q` makes toward the posterior closes it. The bound is valid regardless; its tightness is a hope I'd have to verify, not a guarantee.

But I'm not done — there's still a posterior lurking. The inner expectation `E_{c'~P(c|x)}[·]` requires me to *sample* `c'` from the true posterior `P(c|x)`, which is exactly the object I said I can't sample from. So the bound as written is still not something I can Monte-Carlo. I need to get rid of that posterior sample.

Here's the identity that rescues it. Claim: for random variables `X, Y` and any function `f(x,y)`, under mild regularity,

    E_{x~X, y~Y|x}[ f(x,y) ] = E_{x~X, y~Y|x, x'~X|y}[ f(x', y) ].

Let me actually prove it rather than wave at it. Start from the left side and write the expectations as integrals:

    E_{x~X, y~Y|x}[f(x,y)] = ∫_x P(x) ∫_y P(y|x) f(x,y) dy dx = ∫_x ∫_y P(x,y) f(x,y) dy dx.

Now insert a factor of `1` written cleverly: `∫_{x'} P(x'|y) dx' = 1`, since `P(x'|y)` is a density in `x'`. Multiply `f(x,y)` by it and rename — `f` doesn't depend on `x'`, so the integral just multiplies it:

    = ∫_x ∫_y P(x,y) f(x,y) [ ∫_{x'} P(x'|y) dx' ] dy dx
    = ∫_x ∫_y ∫_{x'} P(x,y) P(x'|y) f(x,y) dx' dy dx.

Now integrate out the first `x` instead of pretending I know which posterior sample produced `y`. Since `P(x,y) = P(y)P(x|y)`, integrating `P(x,y)` over `x` leaves `P(y)`, and I can name the remaining posterior draw `x'`:

    = ∫_y P(y) ∫_{x'} P(x'|y) f(x', y) dx' dy
    = ∫_x P(x) ∫_y P(y|x) ∫_{x'} P(x'|y) f(x', y) dx' dy dx
    = E_{x~X, y~Y|x, x'~X|y}[ f(x', y) ].

The content of the identity: instead of "draw `x`, then draw `y` given `x`, then evaluate `f(x,y)`," I can "draw `x`, draw `y` given `x`, then draw a *fresh* `x'` from `X|y` and evaluate `f(x', y)`" — same expectation. Now apply it with `X = c` (the code) and `Y = x` (the image), and `f = log Q(c|x)`. The left side has me sampling `c'` from the posterior `c | x`; the right side instead samples `c` from its *prior* `P(c)`, pushes it through `G` to get `x = G(z,c)`, and evaluates `log Q(c | x)` on that *same* `c`. The posterior sample is gone — replaced by sampling the code from its prior and generating from it, which is exactly the forward process I can run.

So define the variational lower bound

    L_I(G, Q) = E_{c ~ P(c), x ~ G(z,c)}[ log Q(c | x) ] + H(c),

and by the identity it equals `E_{x~G(z,c)}[ E_{c'~P(c|x)}[log Q(c'|x)] ] + H(c)`, which is `≤ I(c; G(z,c))`. This `L_I` is trivial to Monte-Carlo: sample `c` from the fixed prior `P(c)`, sample noise `z`, form `x = G(z,c)`, and evaluate `log Q(c | x)`. I maximize `L_I` directly with respect to `Q`, and with respect to `G` through the generated image. The entropy `H(c)` I'll just treat as a constant — I fix the code prior, and although I could optimize it too (common distributions have closed-form entropy), there's no need.

Before I build anything on `L_I`, I should make sure maximizing it actually drives the behavior I want, not just something correlated with it. Let me pin down its numerical range on the smallest case I can fully control: a single categorical code uniform over `K = 3` categories, so `H(c) = log 3 ≈ 1.0986` nats, and trace what `L_I = E_{c~P(c)}[log Q(c|x)] + H(c)` evaluates to in the two regimes I care about.

Take the "code ignored" regime first. If the image carries nothing about `c`, the best any `Q` can do is fall back on the prior, `Q(c|x) = 1/K` for every `c`. Then `E_{c~P(c)}[log Q(c|x)] = log(1/3) = −log 3`, and `L_I = −log 3 + log 3 = 0`. Good — the bound bottoms out at exactly zero, which is also the true mutual information when `c` is ignored, so the bound is tight there. Now the "code fully recovered" regime: the image determines `c`, and `Q` recovers it, so `Q(c_true|x) = 1`. Then `E[log Q] = log 1 = 0` and `L_I = 0 + log 3 = log 3 ≈ 1.0986`, hitting `H(c)` exactly. And an intermediate point to confirm it's monotone in recovery quality, not a step function: if `Q` puts probability `0.8` on the true category, `E[log Q] = log 0.8 = −0.2231`, giving `L_I = −0.2231 + 1.0986 = 0.8755`, which sits strictly between 0 and `H(c)`. So `L_I` slides from 0 (code ignored) up to `H(c)` (code recovered) as recovery improves. That's the verification I wanted: pushing `L_I` up is pushing recoverability up, with no slack in the discrete case at either endpoint, so the regularizer is aimed at the right target rather than a proxy that could be gamed.

Fold it into the game. The complete objective is

    min_{G, Q}  max_D   V(D, G) − λ L_I(G, Q),

an adversarial game with a variational mutual-information regularizer. `D` still plays its usual role; `G` and `Q` jointly try to make the code recoverable.

Now make `Q` concrete, and here's a nice economy. `Q(c|x)` has to look at an image and produce a distribution over the code — that's a recognition network, and it needs exactly the kind of image features that a discriminator already learns. So instead of a separate network, let `Q` *share the entire convolutional body* with the discriminator `D`, and add just one extra final fully-connected layer that reads the shared features and outputs the parameters of `Q(c|x)`. The discriminator's last layer already produces its real/fake scalar; `Q` gets its own little head off the same trunk. The cost over a plain adversarial network is then negligible — one extra output layer.

What does that head output? It depends on the type of each code component, and I'll work out what `log Q(c_i|x)` actually becomes in each case rather than just naming the loss. For a categorical code `c_i` — say a 1-of-10 variable meant to capture digit identity — the natural choice for `Q(c_i | x)` is a softmax over the `K` categories. The sampled `c_i` is one-hot, so `log Q(c_i|x) = ∑_k c_{i,k} log softmax_k(logits)`; every term is killed except the one where `c_{i,k}=1`, leaving `log softmax_{k*}(logits)`, the log-probability the head assigns to the true category `k*`. But `−log softmax_{k*}` is by definition the cross-entropy of the softmax against the one-hot target. So maximizing this `L_I` term is literally minimizing the standard classification cross-entropy between `Q`'s prediction and the code fed to `G` — no new machinery, the same loss a classifier head already uses. (I sanity-checked one set of logits `[2, 0.5, −1]` with true class 0: `log softmax_0 = −0.2413` and `−cross_entropy = −0.2413`, identical, so the two are the same number and not merely the same up to a constant.)

For a continuous code `c_j`, I parameterize `Q(c_j | x)` as a factored Gaussian: the head outputs a mean and a standard deviation (the std through an exponential transform of the output so it stays positive), and `log Q(c_j|x) = −½ log(2π) − log σ − ½((c_j − μ)/σ)²`. Maximizing this over the head is minimizing a Gaussian negative-log-likelihood. If I freeze the variance, the only `μ`-dependent term is `−½(c_j − μ)²/σ²`, so maximizing it is minimizing `(c_j − μ)²` — a plain mean-squared error between the predicted mean and the true continuous code, scaled by a constant `1/(2σ²)`. So the information regularizer concretely becomes: generate an image from a sampled code, run the shared `Q` head on it, and penalize how badly `Q` recovers the code that produced it — cross-entropy for the categorical parts, Gaussian NLL (≈ MSE at fixed variance) for the continuous parts.

The weight `λ`. For purely discrete codes I expect `λ = 1` to be a reasonable default: the categorical `L_I` term is a cross-entropy bounded in `[0, log K]` (for `K = 10`, at most `≈ 2.3` nats), and the per-example adversarial log-loss is `−log(sigmoid)` of order ~1 near the start of training, so the two are the same order of magnitude and neither should swamp the other at `λ = 1`. For continuous codes the `L_I` term is a *differential* entropy / Gaussian NLL, which is unbounded and whose scale depends on the variance `σ²` — `−log σ` can be large in magnitude — so it can sit on a different scale than the adversarial loss, and I'd expect to dial `λ` down (something like `0.1`) when I add continuous codes. These are scale arguments, not proofs; the actual values I'd tune by watching whether `L_I` rises without the adversarial loss diverging. And I don't need any new stabilization trick for the adversarial part — I build on the known stable convolutional recipe (up-convolutional generator, leaky-ReLU discriminator, batchnorm, Adam, with a lower learning rate for the discriminator than the generator), which is enough to keep training stable. In practice the information bound `L_I` even converges *faster* than the adversarial objective, so the extra recognition pressure is cheap on top of a normal adversarial run.

Let me note what this looks like from a distance, because it clarifies what's new. The pair `(P_G(x|c), Q(c|x))` is a generative model with a recognition model — a Helmholtz machine. In the classic Wake-Sleep training of such a machine, the "sleep" phase updates the recognition network `Q` on the generator's *own dreamed* samples: `max_Q E_{c~P(c), x~P_G(x|c)}[log Q(c|x)]`. That is exactly my `L_I` update with respect to `Q`. But Wake-Sleep's "wake" phase updates the *generator* on real data passed through `Q`. I do something different: I also update the *generator* by `L_I` — on samples from the generator, not on data — to force `G` to make the code recoverable. So both of my information updates are "sleep"-like (driven by generated samples), which is what makes the generator explicitly conditioned to *use* and *convey* the code, rather than merely fitting the data marginal. That's the whole point — the regularizer reaches into the generator and forbids it from ignoring the code.

Let me write the training core in code. Concatenate noise, a categorical code as a one-hot vector, and a continuous code as the generator input; the discriminator body is shared, with a real/fake head and a recognition head that returns logits for the categorical code and Gaussian parameters for the continuous code.

```python
import math
import torch
import torch.nn.functional as F

TINY = 1e-8
latent_dim, n_classes, code_dim = 62, 10, 2
info_reg_coeff = 1.0

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
    if log_std is None:
        log_std = torch.zeros_like(mean)  # fixed-variance Gaussian
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

    with torch.no_grad():
        fake = G(torch.cat([noise, one_hot, cont], dim=1))
    real_prob, _, _, _ = D_Q(real)
    fake_prob, q_cat_logits, q_cont_mean, q_cont_log_std = D_Q(fake)
    d_adv = -(torch.log(real_prob + TINY) + torch.log(1.0 - fake_prob + TINY)).mean()
    mi = mi_estimate(one_hot, cont, q_cat_logits, q_cont_mean, q_cont_log_std)
    d_loss = d_adv - info_reg_coeff * mi
    opt_dq.zero_grad(); d_loss.backward(); opt_dq.step()

    for p in D_Q.parameters():
        p.requires_grad_(False)
    noise, one_hot, cont = sample_codes(batch, device)
    fake = G(torch.cat([noise, one_hot, cont], dim=1))
    fake_prob, q_cat_logits, q_cont_mean, q_cont_log_std = D_Q(fake)
    mi = mi_estimate(one_hot, cont, q_cat_logits, q_cont_mean, q_cont_log_std)
    g_loss = -torch.log(fake_prob + TINY).mean() - info_reg_coeff * mi
    opt_g.zero_grad(); g_loss.backward(); opt_g.step()
    for p in D_Q.parameters():
        p.requires_grad_(True)
    return d_loss.detach(), g_loss.detach(), mi.detach()
```

Let me trace `mi_estimate` on a hand-checkable input to confirm the code computes the bound the derivation describes and not something subtly different. Use `n_classes = 3` (so `cat_prior = log 3 = 1.0986`) and skip the continuous part by feeding `cont = mean = 0`, which makes the two continuous Gaussian terms identical and cancel — `cont_prior + log_q_gaussian(cont, mean) = −E[logN] + E[logN] = 0`. Feed two examples with true classes 0 and 1. If `Q` recovers the code, the logits spike on the true class (say `[20,0,0]` and `[0,20,0]`): `log_softmax` on the true class is `≈ 0`, so `log_q_categorical ≈ 0`, and `mi_estimate ≈ 1.0986 + 0 = log 3`. If `Q` ignores the code (uniform logits `[0,0,0]`): `log_softmax` on the true class is `log(1/3) = −1.0986`, so `log_q_categorical = −1.0986`, and `mi_estimate ≈ 1.0986 − 1.0986 = 0`. Running it gives exactly `1.098612` and `0.000000` — the same two endpoints I computed by hand for the bound. So the implementation's `mi` is the variational lower bound `L_I` (the `cat_prior` is the `H(c)` term, the `log_q_categorical` mean is `E[log Q]`), and subtracting `info_reg_coeff * mi` in both the `D_Q` and `G` losses is subtracting `λ L_I`, matching `min max V − λ L_I`. The code and the math agree.

The causal chain: I wanted unsupervised disentanglement, but generation alone can't give it because a perfect generator can carry an arbitrarily scrambled code; the adversarial objective is indifferent to how the latent is used, so a plain structured code gets ignored (a trivial code); the cure is to *force* the code to be used, which means forcing the mutual information `I(c; G(z,c))` up; that information is intractable because it hides the posterior `P(c|x)`, so I lower-bound it variationally with an auxiliary `Q(c|x)` — and an expectation identity removes the last posterior sample, leaving `L_I = E_{c~P(c), x~G(z,c)}[log Q(c|x)] + H(c)`, a quantity I can Monte-Carlo and differentiate; I realize `Q` by sharing the discriminator's convolutional body plus one head, using softmax-cross-entropy for categorical codes and a Gaussian for continuous ones; and the whole thing is the ordinary adversarial game minus `λ L_I`. What I'd want to validate next is whether sweeping a single code coordinate while fixing the rest produces a single interpretable change in the generated image — one axis for digit identity, others for rotation and width — with no supervision anywhere in the loop.
