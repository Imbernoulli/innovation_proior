## Research Question

I want an unsupervised density model for high-dimensional continuous data such as images. The target is a coordinate system in which the transformed data distribution is easy to model. The setting is to find a deterministic transform `h = f(x)` whose coordinates are independent under a simple prior,

```text
p_H(h) = product_d p_{H_d}(h_d).
```

The model should support exact maximum-likelihood training, exact data-to-code inference, and unbiased ancestral sampling, with a representation having the same dimensionality as the data.

## Probability Tooling Already On The Table

The exact route starts from a smooth bijection between spaces of equal dimension. If `h = f(x)` and `f` is invertible, conservation of probability mass gives

```text
p_X(x) = p_H(f(x)) * |det df(x)/dx|,
log p_X(x) = log p_H(f(x)) + log |det df(x)/dx|.
```

With a factorial prior the first term is just a coordinate-wise sum. The determinant term corrects the scale effect of an invertible preprocessing. If `f` contracts a neighborhood around data, the prior density at the transformed point may rise, but the log-determinant falls by the corresponding local volume change. In the data-to-code direction this term penalizes contraction and rewards expansion around high-density regions.

Composition is also available. If `f = f_L o ... o f_1`, the inverse composes the layer inverses in reverse order, and the Jacobian determinant is the product of the layer determinants. The open design problem is therefore to find elementary bijective layers whose inverse and determinant are cheap while leaving enough room for deep nonlinear functions.

## Existing Routes

Undirected graphical models such as RBMs and DBMs define flexible densities, with training and sampling handled via MCMC, and likelihood evaluation via approximations such as annealed importance sampling.

Variational autoencoders use a stochastic recognition model and optimize a lower bound on the likelihood.

Autoregressive density estimators write `p(x)` as an ordered product of one-dimensional conditionals. The dependency structure is triangular, and sampling proceeds sequentially in the dimension count under a fixed ordering.

Adversarial generators sample in parallel from a simple noise source.

Earlier learned-transform models explore related ideas. Linear independent-component methods apply linear constraints; Gaussianization uses a greedy layered transform; and various neural transform architectures explore bijective structures.

## Data Protocol And Fair Evaluation

The natural evaluation target is held-out log-likelihood on dequantized image data. Discrete pixel values cannot be treated naively as samples from a continuous density: a continuous model can place arbitrarily narrow spikes on the finite grid and drive likelihood upward without learning a meaningful density. The standard fix is to add uniform noise within one quantization bin and rescale to the continuous data interval.

For 8-bit images in `[0, 1]`, dequantization corresponds to replacing a normalized grid value `k/255` by `(k + u)/256` with `u ~ Uniform(0, 1)`. For data stored in `[-1, 1]`, the same 256-level operation has bin width `2/256 = 1/128`. When reporting bits per dimension, the discrete 256-level constant must be accounted for by adding `D log 256` to the negative log-likelihood before dividing by `D log 2`, plus any constant log-determinant from fixed linear preprocessing when the score is for the original data coordinates.

## Starting Scaffold

The implementation only needs ordinary neural-network components plus a structured bijection. The unknown piece is the transform that can run in both directions and expose its log-Jacobian determinant.

```python
class FactorialPrior:
    def log_prob(self, h):
        # per-coordinate log density, summed by the density model
        raise NotImplementedError

    def sample(self, shape):
        raise NotImplementedError


class StructuredBijection:
    def forward(self, x):
        # data -> latent, returning (h, log_abs_det_df_dx)
        raise NotImplementedError

    def inverse(self, h):
        # latent -> data
        raise NotImplementedError


class DensityModel:
    def __init__(self, prior, bijection):
        self.prior = prior
        self.bijection = bijection

    def log_prob(self, x):
        h, log_det = self.bijection.forward(x)
        return self.prior.log_prob(h).sum(axis=-1) + log_det

    def sample(self, n):
        h = self.prior.sample((n, self.bijection.dim))
        return self.bijection.inverse(h)
```
