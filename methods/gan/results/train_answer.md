Deep discriminative models train beautifully today because they combine three well-behaved ingredients: backpropagation, dropout, and piecewise-linear units such as ReLU and maxout. Generative modeling has not had the same breakthrough, and the obstacle is structural rather than a matter of scale. The standard way to learn a generative model is maximum likelihood, and maximum likelihood forces you to write down, normalize, and differentiate an explicit probability density. That requirement is exactly where the trouble begins: energy-based models carry an intractable partition function and need MCMC to approximate it; variational methods replace the likelihood with a bound and require a separate inference network; and fully-visible belief networks, while tractable, sample one coordinate at a time and offer no latent code. Implicit models escape the density entirely, but until now they lacked a clean learning signal that could drive them toward the data distribution using only forward and backward propagation.

The failure of the closest alternative, noise-contrastive estimation, points the way out. NCE trains a classifier to distinguish data from a fixed noise distribution, using the model's own unnormalized density inside the logistic loss. The problem is that the noise never gets harder: as soon as the model captures a few features, the classifier saturates and the gradient goes slack. A fixed contrast is a pushover, and beating a pushover is not the same as matching the data distribution. What is needed is a contrast that improves in lockstep with the model, so the classification task stays difficult and informative all the way to the optimum.

The method I propose is the Generative Adversarial Network, or GAN. It keeps the implicit-model branch of the design tree and supplies the missing signal with a learned adversary. There are two networks. The generator G takes a noise vector z drawn from a simple prior p_z and maps it through a differentiable feedforward network to a sample x = G(z). Because the only stochasticity is the injected noise at the input, gradients flow cleanly from any downstream scalar through G and into its parameters by the reparameterization principle. The discriminator D is a separate network that takes a data-space point x and outputs a scalar D(x) in (0, 1), interpreted as the probability that x came from the real data rather than from G. D is trained to classify correctly, while G is trained to make D misclassify generated samples as real. The opponent is thus the generator itself, and because it is being optimized, the contrast never collapses into a trivial problem.

Formally, the two networks play the minimax game V(D, G) = E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 - D(G(z)))], where D tries to maximize V and G tries to minimize it. For a fixed G, the optimal discriminator is D*_G(x) = p_data(x) / (p_data(x) + p_g(x)), the Bayes-optimal classifier between the two classes and, equivalently, the density ratio p_data / p_g squashed into the unit interval. Substituting this optimal discriminator into V reveals what G is actually minimizing: C(G) = -log 4 + 2 * JSD(p_data || p_g), where JSD is the Jensen-Shannon divergence. Since JSD is non-negative and zero only when the two distributions are equal, the unique global optimum is p_g = p_data, with D indifferent at 1/2 everywhere. The divergence is symmetric and bounded, which comes for free from the real-versus-fake formulation rather than from any hand-designed objective.

Two practical adjustments make the idea runnable. First, optimizing D to convergence at every step is too expensive and would overfit, so we take k discriminator gradient steps per generator step; k = 1 is the cheapest version that still keeps D tracking the moving optimum and prevents G from training too long against a stale discriminator. A stale D invites mode collapse: G discovers a small set of samples that fool the current classifier and stops exploring the full data distribution. Second, the raw minimax generator term log(1 - D(G(z))) has weak early gradients, because when G is poor D rejects fakes confidently and the logit-space derivative nearly vanishes. We therefore train G to maximize log D(G(z)) instead. This non-saturating loss shares the same fixed point but gives a strong gradient of roughly -1 when D(G(z)) is near zero, exactly when G most needs guidance.

I implement this in Theano and Pylearn2, where the pieces map directly onto the empty slots the framework leaves for a sampler, a learned cost, and a split optimizer. The generator is a `Generator` model wrapping a noise-driven MLP: it draws z from a Gaussian, uniform, or spherical distribution, formats it into the MLP's input space, and returns one forward pass through the MLP as the sample — no chain, no recurrence, so the piecewise-linear units are free to use throughout. An `AdversaryPair` bundles the generator and discriminator together, merging their parameter lists and learning-rate scalers so the rest of the harness sees one model. The `AdversaryCost2` object computes both players' losses from a single shared computation: it draws a batch of generator samples alongside the real minibatch, pushes both through the discriminator's dropout forward pass, and forms `d_obj` as the average of the discriminator's cross-entropy against label 1 on the real batch and label 0 on the fake batch — exactly `-log D(x)` and `-log(1 - D(G(z)))` — while `g_obj` is the cross-entropy of the same fake batch against label 1, which is `-log D(G(z))`, the non-saturating loss I derived above. Because the generator's and discriminator's parameters are disjoint, `T.grad` differentiates `d_obj` and `g_obj` against their own parameter sets independently and hands back two separate gradient updates; an optional norm-based rescaling on the generator's gradient keeps it from blowing up when the discriminator is very confident. Training runs a `split_sgd_epoch` loop that calls the discriminator's update function on every minibatch and only calls the generator's update function every `discriminator_steps` batches — the k-step schedule made concrete, with k = 1 the cheapest setting that still keeps D tracking the moving optimum.

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
