# GAN — the adversarial generative framework

## The problem it solves

Estimate a deep generative model and sample from it using only backpropagation and forward
propagation — with no intractable partition function, no Markov chain in training or sampling, and
no approximate-inference network. The generative-model landscape is organized around how you get a
tractable handle on `p(x)`: make the density explicit-and-tractable (autoregressive models — exact
but sequential, slow sampling), explicit-up-to-something (energy-based models with an intractable
normalizer `Z`; variational autoencoders with a likelihood *bound* and an inference network), or
give up the density and only *sample* (implicit models). Every explicit route pays a tax — `Z` and
MCMC mixing, an analytic unnormalized density, a variational bound plus encoder. GAN takes the
implicit branch on purpose and supplies the one missing piece an implicit model lacked: a learning
signal that isn't a likelihood. That signal is a *learned adversary*.

## The key idea

Pit two networks against each other.

- A **generator** `G(z; θ_g)` maps noise `z ~ p_z` to a sample. It implicitly defines a model
  distribution `p_g` (the pushforward of `p_z` through `G`); its density is never written down.
  Because `G` is a *differentiable deterministic function of injected noise*, gradients flow through
  it by backprop — the only stochasticity is `z` at the input.
- A **discriminator** `D(x; θ_d) ∈ (0,1)` estimates the probability that `x` is real data rather than
  a `G`-sample.

`D` is trained to classify real vs. generated correctly; `G` is trained to make `D` wrong. Two design
choices carry the whole idea. The contrast is a *learned generator* rather than fixed noise (as in
NCE) or a fixed hand-chosen statistic, so the classifier keeps looking for the current mismatch instead
of going slack after a fixed contrast is solved. And the opponent is a *classifier* rather than a
direct density-ratio estimator, so its sigmoid output `p_data/(p_data + p_g)` is the ratio squashed
into `(0,1)` — bounded and stable, where the raw ratio `p_data/p_g ∈ (0, ∞)` would have to be clipped.

## The objective

A two-player minimax game on one value function — exactly the (negative) binary cross-entropy of the
optimal real-vs-fake classifier:

    min_G max_D V(D, G) = E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 - D(G(z)))].

**Optimal discriminator (G fixed).** Rewrite the second term as an expectation under `p_g` (pushforward),
so `V = ∫ [p_data log D + p_g log(1 - D)] dx`. Pointwise this maximizes `a log y + b log(1-y)` with
`a = p_data`, `b = p_g` (concave; derivative `a/y - b/(1-y) = 0`), giving

    D*_G(x) = p_data(x) / (p_data(x) + p_g(x)),

the Bayes-optimal classifier with equal priors — literally the squashed density ratio.

**What G then minimizes.** Substitute `D*_G` and subtract `-log 4 = E_{p_data}[-log 2] + E_{p_g}[-log 2]`,
folding `log 2` into each log to form the two KLs to the mixture `m = (p_data + p_g)/2`:

    C(G) = -log 4 + KL(p_data ‖ m) + KL(p_g ‖ m)
         = -log 4 + 2·JSD(p_data ‖ p_g).

Since `JSD ≥ 0` with equality iff the distributions are equal, the **unique global optimum is
`p_g = p_data`**, at value `-log 4`, where `D ≡ 1/2`. The divergence is symmetric (because real-vs-fake
is symmetric) and finite even on disjoint supports — properties that came for free from the game, not
from a design choice. Viewing `V` as `U(p_g, D)`: it is linear hence convex in `p_g`, so `sup_D U` is
convex with that unique optimum, and the subgradient of the sup at the maximizing `D*` is a valid
descent direction. In the ideal distribution-space update where `D` reaches `D*` at each step,
sufficiently small generator steps converge `p_g → p_data`; with finite MLPs and approximate `D`,
this is a target and descent-direction argument, not a global parameter-space guarantee.

## Two practical fixes

1. **k-step schedule.** Optimizing `D` to completion each step is prohibitive and overfits, so take
   `k` discriminator steps per generator step (`k = 1` is the least expensive setting), trying to keep
   `D` tracking its moving optimum as `G` changes slowly — the persistent-chain (SML/PCD) trick, with
   `D`'s parameters as the carried-over state. This also keeps `G` from training too long against a
   stale `D`, the setup that leads to **collapse** (the "Helvetica" scenario: many `z` mapped to a few
   outputs that fool the current `D`, losing diversity).
2. **Non-saturating generator loss.** Early on `D(G(z)) ≈ 0`. With discriminator logit `a` and
   `D = σ(a)`, the minimax generator term has logit derivative
   `d/da log(1 - σ(a)) = -σ(a) = -D`, so the signal to `G` vanishes when `D` confidently rejects
   fakes. Train `G` to **maximize `log D(G(z))`** instead, equivalently minimize `-log D(G(z))`:
   `d/da[-log σ(a)] = -(1 - D) ≈ -1` when `D ≈ 0`. Same fixed point, much stronger early
   gradient. The minimax objective stays useful for the clean divergence argument; the executable
   generator loss uses the non-saturating form.

## Algorithm

```
for number of training iterations:
    for k steps:                                   # k = 1 in practice
        sample minibatch z^(1..m) ~ p_z,  x^(1..m) ~ p_data
        ascend  ∇_{θ_d}  (1/m) Σ [ log D(x^i) + log(1 - D(G(z^i))) ]
    sample minibatch z^(1..m) ~ p_z
    ascend  ∇_{θ_g}  (1/m) Σ  log D(G(z^i))        # non-saturating; minimax alternative descends log(1 - D)
```

Updates use any gradient rule; the Theano/Pylearn2 setup below uses SGD with momentum. The generator
uses rectifier + sigmoid units; the discriminator uses maxout with dropout — maxout for clean
piecewise-linear gradients, dropout to keep a powerful `D` from overfitting the moving target.
Quantitative evaluation of these implicit models uses a Gaussian Parzen-window log-likelihood
estimate on samples (bandwidth `σ` cross-validated).

## Code

The objective reuses one binary-cross-entropy loss against hard targets:
`cost(target=1, D(x)) = -log D(x)` and `cost(target=0, D(x)) = -log(1 - D(x))`.
The two players' parameter sets are disjoint, so autodiff produces the two gradients independently.
This is the Theano/Pylearn2 core: a noise-driven `Generator`, an `AdversaryPair`, the
`AdversaryCost2` loss, and the split SGD loop that takes `k` discriminator updates per generator
update.

```python
import numpy as np
from theano import tensor as T
from theano.compat import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams

from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin
from pylearn2.models import Model
from pylearn2.space import VectorSpace
from pylearn2.utils import safe_zip, sharedX


class Generator(Model):
    def __init__(self, mlp, noise="gaussian"):
        Model.__init__(self)
        self.mlp = mlp
        self.noise = noise
        self.theano_rng = MRG_RandomStreams(2014 * 5 + 27)

    def get_noise(self, size):
        if isinstance(size, int):
            size = (size, self.mlp.get_input_space().get_total_dimension())
        if self.noise == "uniform":
            return self.theano_rng.uniform(
                low=-np.sqrt(3), high=np.sqrt(3), size=size, dtype="float32"
            )
        if self.noise == "gaussian":
            return self.theano_rng.normal(size=size, dtype="float32")
        if self.noise == "spherical":
            noise = self.theano_rng.normal(size=size, dtype="float32")
            norm = T.maximum(1e-7, T.sqrt(T.sqr(noise).sum(axis=1))).dimshuffle(0, "x")
            return noise / norm
        raise NotImplementedError(self.noise)

    def sample_and_noise(self, num_samples, default_input_include_prob=1.,
                         default_input_scale=1., all_g_layers=False):
        n = self.mlp.get_input_space().get_total_dimension()
        noise = self.get_noise((num_samples, n))
        formatted_noise = VectorSpace(n).format_as(noise, self.mlp.get_input_space())
        if all_g_layers:
            rval = self.mlp.dropout_fprop(
                formatted_noise,
                default_input_include_prob=default_input_include_prob,
                default_input_scale=default_input_scale,
                return_all=all_g_layers,
            )
            other_layers, sample = rval[:-1], rval[-1]
        else:
            sample = self.mlp.dropout_fprop(
                formatted_noise,
                default_input_include_prob=default_input_include_prob,
                default_input_scale=default_input_scale,
            )
            other_layers = None
        return sample, formatted_noise, other_layers

    def sample(self, num_samples, default_input_include_prob=1., default_input_scale=1.):
        sample, _, _ = self.sample_and_noise(
            num_samples, default_input_include_prob, default_input_scale
        )
        return sample

    def get_params(self):
        return self.mlp.get_params()

    def get_input_space(self):
        return self.mlp.get_input_space()

    def get_output_space(self):
        return self.mlp.get_output_space()

    def get_lr_scalers(self):
        return self.mlp.get_lr_scalers()


class AdversaryPair(Model):
    def __init__(self, generator, discriminator):
        Model.__init__(self)
        self.generator = generator
        self.discriminator = discriminator

    def get_params(self):
        return self.generator.get_params() + self.discriminator.get_params()

    def get_input_space(self):
        return self.discriminator.get_input_space()

    def get_input_source(self):
        return self.discriminator.get_input_source()

    def get_lr_scalers(self):
        rval = self.generator.get_lr_scalers()
        rval.update(self.discriminator.get_lr_scalers())
        return rval


class AdversaryCost2(DefaultDataSpecsMixin, Cost):
    supervised = False

    def __init__(self, scale_grads=1, target_scale=.1,
                 discriminator_default_input_include_prob=1.,
                 discriminator_input_include_probs=None,
                 discriminator_default_input_scale=1.,
                 discriminator_input_scales=None,
                 generator_default_input_include_prob=1.,
                 generator_default_input_scale=1.,
                 no_drop_in_d_for_g=False):
        self.__dict__.update(locals())
        del self.self
        self.now_train_generator = sharedX(np.array(1., dtype="float32"))
        self.now_train_discriminator = sharedX(np.array(1., dtype="float32"))

    def expr(self, model, data, **kwargs):
        _, d_obj, g_obj, _ = self.get_samples_and_objectives(model, data)
        return d_obj + g_obj

    def get_samples_and_objectives(self, model, data):
        space, _ = self.get_data_specs(model)
        space.validate(data)
        g, d = model.generator, model.discriminator
        X = data
        m = data.shape[space.get_batch_axis()]
        y1 = T.alloc(1, m, 1)
        y0 = T.alloc(0, m, 1)

        S, z, other_layers = g.sample_and_noise(
            m,
            default_input_include_prob=self.generator_default_input_include_prob,
            default_input_scale=self.generator_default_input_scale,
            all_g_layers=False,
        )

        y_hat1 = d.dropout_fprop(
            X,
            self.discriminator_default_input_include_prob,
            self.discriminator_input_include_probs,
            self.discriminator_default_input_scale,
            self.discriminator_input_scales,
        )
        y_hat0 = d.dropout_fprop(
            S,
            self.discriminator_default_input_include_prob,
            self.discriminator_input_include_probs,
            self.discriminator_default_input_scale,
            self.discriminator_input_scales,
        )

        d_obj = 0.5 * (
            d.layers[-1].cost(y1, y_hat1)      # -log D(x)
            + d.layers[-1].cost(y0, y_hat0)    # -log(1 - D(G(z)))
        )

        if self.no_drop_in_d_for_g:
            y_hat0_for_g = d.dropout_fprop(S)
        else:
            y_hat0_for_g = y_hat0
        g_obj = d.layers[-1].cost(y1, y_hat0_for_g)  # -log D(G(z))
        return S, d_obj, g_obj, 0

    def get_gradients(self, model, data, **kwargs):
        S, d_obj, g_obj, _ = self.get_samples_and_objectives(model, data)
        g_params = model.generator.get_params()
        d_params = model.discriminator.get_params()
        for param in g_params:
            assert param not in d_params
        for param in d_params:
            assert param not in g_params

        d_grads = T.grad(d_obj, d_params)
        g_grads = T.grad(g_obj, g_params)

        if self.scale_grads:
            S_grad = T.grad(g_obj, S)
            scale = T.maximum(1., self.target_scale / T.sqrt(T.sqr(S_grad).sum()))
            g_grads = [g_grad * scale for g_grad in g_grads]

        rval = OrderedDict()
        rval.update(OrderedDict(safe_zip(
            d_params, [self.now_train_discriminator * dg for dg in d_grads]
        )))
        rval.update(OrderedDict(safe_zip(
            g_params, [self.now_train_generator * gg for gg in g_grads]
        )))
        return rval, OrderedDict()


def split_sgd_epoch(iterator, d_func, g_func, discriminator_steps=1):
    i = 0
    for batch in iterator:
        d_func(*batch)
        i += 1
        if i == discriminator_steps:
            g_func(*batch)
            i = 0
```

For the MNIST configuration, this core is instantiated with uniform noise of dimension `100`; a
generator MLP `100 -> 1200 -> 1200 -> 784` with rectified hidden layers and a sigmoid output; a
discriminator MLP with two maxout layers of `240` units and `5` pieces, then a sigmoid output; batch
size `100`; learning rate `.1`; momentum `.5` adjusted to `.7`; `AdversaryCost2(scale_grads=0)`;
`discriminator_steps=1`; discriminator input keep probability `.5`, first-hidden keep probability
`.8`, and corresponding dropout scales `2.` and `1.25`.

## What it buys and costs

- **Buys:** no Markov chain (a sample is one forward pass through `G`); no inference network in
  training; gradient is pure backprop through `D` into `G`; piecewise-linear units usable freely
  (no generation-time feedback loop); `G` never copies data directly (it only sees data through `D`'s
  gradient, so it can overfit only if `D` does, which is easy to control); can represent sharp /
  degenerate distributions that MCMC methods cannot.
- **Costs:** no explicit `p_g(x)` (likelihood must be estimated indirectly, e.g. Parzen window), and
  `D` must be kept synchronized with `G` to avoid the Helvetica-style collapse.
