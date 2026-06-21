I want a generative model for continuous data that gives me everything at once: exact probability density for any input I might later encounter, exact sampling from that density, and an exact deterministic encoder that maps data back to the latent noise that generated it. Most model families force a choice. Variational autoencoders give fast approximate inference but only optimize a lower bound, and their Gaussian decoders blur fine detail. Autoregressive models give an exact likelihood, yet sampling is sequential in the dimensionality of the data and the chosen ordering is baked into the model. Adversarial networks sample in parallel and look sharp, but they offer no tractable density and no encoder. Energy-based models are flexible but require MCMC for both training and sampling, so evaluation is approximate and mixing can be slow. What I need is a deterministic, invertible map between data space and a simple latent space, because that is the only construction that simultaneously yields exact density, exact sampling, and exact inference from a single object.

The method that achieves this is the family of Normalizing Flows. The core idea is the change-of-variables formula from elementary probability. Suppose I have a smooth bijection f that maps a data point x to a latent code z = f(x), with a smooth inverse g = f^{-1} so that x = g(z). If I place a simple, factorized prior p_Z(z) on the latent space, then the density of x under the pushed-forward distribution is p_X(x) = p_Z(f(x)) |det(∂f(x)/∂x^T)|. In log form this is

log p_X(x) = log p_Z(f(x)) + log |det(∂f(x)/∂x^T)|.

This identity is doing three jobs at once. Evaluating p_X(x) is one forward pass through f plus the Jacobian determinant. Sampling is one draw z ~ p_Z followed by one inverse pass x = g(z). Inference is exactly z = f(x). There is no variational approximation, no sequential chain, and no discriminator. The only remaining problem is architectural: a generic neural network f has an O(D^3) determinant and a hard-to-invert mapping, so it cannot be used directly. The entire field of normalizing flows is therefore the search for bijections whose Jacobians are triangular or diagonal by construction, making the determinant a cheap sum and the inverse a closed-form recursion.

The two dominant structural solutions are coupling layers and autoregressive layers, both of which produce triangular Jacobians without ever inverting a neural network. In a coupling layer I split the coordinates into two blocks, leave the first block untouched, and apply an affine transformation to the second block whose scale and shift are arbitrary functions of the first block. Writing y_{1:d} = x_{1:d} and y_{d+1:D} = x_{d+1:D} ⊙ exp(s(x_{1:d})) + t(x_{1:d}), the Jacobian is block lower triangular: the top-left block is the identity, the top-right block is zero because the copied coordinates do not depend on the transformed ones, and the bottom-right block is diagonal with entries exp(s(x_{1:d})). Its determinant is therefore exp(∑_j s(x_{1:d})_j), a one-line sum. The inverse is equally cheap because I read off x_{1:d} = y_{1:d}, recompute s and t, and undo the affine map. The functions s and t can be deep convolutional networks or MLPs: their derivatives appear only in the off-diagonal block, so they contribute nothing to the determinant.

An autoregressive layer achieves the same triangularity through an explicit ordering rather than a partition. It defines z_i = (x_i - μ_i(x_{1:i-1})) / σ_i(x_{1:i-1}), where μ_i and σ_i are arbitrary functions of all previous coordinates. Because z_i does not depend on x_j for j > i, the Jacobian ∂z/∂x is lower triangular, and its determinant is the product of the reciprocals of the scales. Conditioning on the previous data coordinates makes density evaluation one parallel masked pass, while sampling becomes sequential because each x_i must be generated before it can be used to compute the next conditional. Reversing the conditioning—using previous latent coordinates instead of previous data coordinates—flips the trade-off: sampling becomes one parallel pass and density evaluation becomes sequential. This duality is the distinction between Masked Autoregressive Flows and Inverse Autoregressive Flows. Coupling layers can be viewed as a special case in which the first block of coordinates is completely frozen, so autoregressive flows are strictly more flexible per layer at the cost of either sampling or scoring speed.

To build a deep and expressive normalizing flow I compose many of these layers. Composition preserves the two properties I care about: the overall map is invertible because inverses reverse order, and the total log-determinant is the sum of the per-layer log-determinants by the chain rule. I also insert permutation or squeeze operations between layers so that every coordinate is transformed in every role over the depth of the network. Batch normalization can be folded in as a flow layer because it is an elementwise affine map with a diagonal Jacobian. For image data I use multi-scale architectures that factor out half the latent variables at each resolution, giving direct training signals at every scale and building a coarse-to-fine latent hierarchy. The prior is usually a standard Gaussian, though any tractable density works. Because pixel data is discrete, I dequantize by adding uniform noise inside each bin before feeding the model the continuous value, and I account for the bin width when reporting bits per dimension. Training is then ordinary maximum likelihood: I minimize the negative log p_X(x) averaged over a minibatch, computed exactly in one forward pass.

Normalizing flows sit at a useful boundary between the flexibility of autoregressive models and the compact inference of variational autoencoders. They give up some of the raw expressiveness per parameter of an unrestricted neural network, but they do so in exchange for exact, cheap density evaluation and a latent space of the same dimension as the data. This makes them natural for likelihood-free inference, learning priors over parameters, anomaly detection, variational inference with flexible posteriors, and any application where the ability to score an externally provided point matters as much as generating new ones. The canonical name for this family is Normalizing Flows.

```python
import numpy as np


def relu(x):
    return np.maximum(0.0, x)


class AffineCoupling:
    """Simple affine coupling layer with analytic scale/shift networks."""
    def __init__(self, dim, split, Ws, bs, Wt, bt):
        self.dim = dim
        self.split = split
        self.Ws = Ws
        self.bs = bs
        self.Wt = Wt
        self.bt = bt

    def forward(self, x):
        x1 = x[:, :self.split]
        x2 = x[:, self.split:]
        s = np.tanh(x1 @ self.Ws + self.bs)
        t = x1 @ self.Wt + self.bt
        y1 = x1
        y2 = x2 * np.exp(s) + t
        y = np.concatenate([y1, y2], axis=1)
        logdet = s.sum(axis=1)
        return y, logdet

    def inverse(self, y):
        y1 = y[:, :self.split]
        y2 = y[:, self.split:]
        s = np.tanh(y1 @ self.Ws + self.bs)
        t = y1 @ self.Wt + self.bt
        x1 = y1
        x2 = (y2 - t) * np.exp(-s)
        return np.concatenate([x1, x2], axis=1)


class Swap:
    def forward(self, x):
        return x[:, ::-1].copy(), np.zeros(x.shape[0])

    def inverse(self, y):
        return y[:, ::-1].copy()


class NormalizingFlow:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        logdet = np.zeros(x.shape[0])
        for layer in self.layers:
            x, ld = layer.forward(x)
            logdet += ld
        return x, logdet

    def inverse(self, z):
        x = z
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x

    def log_prob(self, x):
        z, logdet = self.forward(x)
        d = z.shape[1]
        log_prior = -0.5 * np.sum(z ** 2, axis=1) - 0.5 * d * np.log(2.0 * np.pi)
        return log_prior + logdet


np.random.seed(0)
dim = 4
split = dim // 2
n_layers = 3

layers = []
for _ in range(n_layers):
    Ws = 0.1 * np.random.randn(split, dim - split)
    bs = np.zeros(dim - split)
    Wt = 0.1 * np.random.randn(split, dim - split)
    bt = np.zeros(dim - split)
    layers.append(AffineCoupling(dim, split, Ws, bs, Wt, bt))
    layers.append(Swap())

flow = NormalizingFlow(layers)

z0 = np.random.randn(1000, dim)
x = flow.inverse(z0)
roundtrip, _ = flow.forward(x)
print("Invertibility max error:", np.max(np.abs(roundtrip - z0)))

logp = flow.log_prob(x)
print("Mean log prob of inverse-sampled data:", logp.mean())
print("Std log prob:", logp.std())
```
