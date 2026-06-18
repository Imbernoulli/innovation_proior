# Spectral Normalization for GANs

## Problem

GAN training depends on the discriminator's input gradients. If the discriminator score can become arbitrarily sharp, then in the nearly disjoint-support regime it can separate real and generated samples while giving the generator a useless or vanishing gradient. The fix is to restrict the discriminator score network to a controlled Lipschitz family, cheaply and without reducing it to a low-rank critic.

## Method

For a differentiable map, local stretch is controlled by the spectral norm of the Jacobian. For a linear layer `g(h) = W h`,

    ||g||_Lip = sigma(W),

where `sigma(W)` is the largest singular value. With 1-Lipschitz activations,

    ||f||_Lip <= product_l sigma(W^l).

Normalize each weight matrix by its own spectral norm:

    W_bar_SN = W / sigma(W).

For matrix layers this sets `sigma(W_bar_SN) = 1`; for convolutional kernels the implementation uses the proxy matrix `d_out x (d_in * k_h * k_w)`, with stride/padding effects absorbed into architecture constants.

## Power Iteration

A full SVD per layer per update is too expensive. Store a persistent estimate `u_l` for each layer and run one or more warm-started power iterations:

    v_l <- (W^l)^T u_l / ||(W^l)^T u_l||_2
    u_l <- W^l v_l / ||W^l v_l||_2
    sigma(W^l) ~= u_l^T W^l v_l.

The layer persists `u`; `v` is recomputed during the `W_bar` calculation.

## Gradient

For a simple top singular value,

    d sigma(W) / dW = u_1 v_1^T,

so

    d W_bar_SN / d W_ij
      = (1/sigma(W)) (E_ij - [u_1 v_1^T]_ij W_bar_SN).

With `delta := (dV / d(W_bar_SN h))^T`,

    dV/dW = (1/sigma(W)) ( Ehat[delta h^T] - lambda u_1 v_1^T ),
    lambda = Ehat[delta^T W_bar_SN h].

The correction term has a negative sign in the leading singular direction. It is an adaptive pressure against over-growing that direction, not a guarantee that every training trajectory avoids rank collapse.

## Canonical-Code Shape

The canonical code is Chainer layer code, not a PyTorch hook. A faithful minimal artifact follows the same structure: raw `W` stays trainable; a persistent `u` tracks the top left singular vector; `W_bar` computes `sigma` and calls the underlying layer with `W / sigma`.

```python
import chainer
import chainer.functions as F
import numpy as np
from chainer import cuda
from chainer.functions.array.broadcast import broadcast_to
from chainer.functions.connection import convolution_2d, linear
from chainer.links.connection.convolution_2d import Convolution2D
from chainer.links.connection.linear import Linear


def _l2normalize(x, eps=1e-12):
    xp = cuda.get_array_module(x)
    return x / (xp.sqrt((x * x).sum()) + eps)


def max_singular_value(W, u=None, n_power_iterations=1):
    if n_power_iterations < 1:
        raise ValueError("n_power_iterations must be positive")
    xp = cuda.get_array_module(W.data)
    if u is None:
        u = xp.random.normal(size=(1, W.shape[0])).astype(xp.float32)
    u_hat = u
    for _ in range(n_power_iterations):
        v_hat = _l2normalize(xp.dot(u_hat, W.data))
        u_hat = _l2normalize(xp.dot(v_hat, W.data.T))
    sigma = F.sum(F.linear(u_hat, F.transpose(W)) * v_hat)
    return sigma, u_hat, v_hat


class SNLinear(Linear):
    def __init__(self, in_size, out_size, nobias=False, initialW=None,
                 initial_bias=None, use_gamma=False, n_power_iterations=1,
                 factor=None):
        self.n_power_iterations = n_power_iterations
        self.use_gamma = use_gamma
        self.factor = factor
        super().__init__(in_size, out_size, nobias, initialW, initial_bias)
        self.u = np.random.normal(size=(1, out_size)).astype("f")
        self.register_persistent("u")

    @property
    def W_bar(self):
        sigma, u_hat, _ = max_singular_value(
            self.W, self.u, self.n_power_iterations
        )
        if self.factor is not None:
            sigma = sigma / self.factor
        self.u[:] = u_hat
        sigma = broadcast_to(sigma.reshape((1, 1)), self.W.shape)
        W_bar = self.W / sigma
        if hasattr(self, "gamma"):
            W_bar = broadcast_to(self.gamma, self.W.shape) * W_bar
        return W_bar

    def _initialize_params(self, in_size):
        super()._initialize_params(in_size)
        if self.use_gamma:
            _, s, _ = np.linalg.svd(self.W.data)
            with self.init_scope():
                self.gamma = chainer.Parameter(s[0], (1, 1))

    def __call__(self, x):
        if self.W.data is None:
            self._initialize_params(x.size // x.shape[0])
        return linear.linear(x, self.W_bar, self.b)


class SNConvolution2D(Convolution2D):
    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 nobias=False, initialW=None, initial_bias=None,
                 use_gamma=False, n_power_iterations=1, factor=None):
        self.n_power_iterations = n_power_iterations
        self.use_gamma = use_gamma
        self.factor = factor
        super().__init__(
            in_channels, out_channels, ksize, stride, pad,
            nobias, initialW, initial_bias
        )
        self.u = np.random.normal(size=(1, out_channels)).astype("f")
        self.register_persistent("u")

    @property
    def W_bar(self):
        W_mat = self.W.reshape(self.W.shape[0], -1)
        sigma, u_hat, _ = max_singular_value(
            W_mat, self.u, self.n_power_iterations
        )
        if self.factor is not None:
            sigma = sigma / self.factor
        if chainer.config.train:
            self.u[:] = u_hat
        sigma = broadcast_to(sigma.reshape((1, 1, 1, 1)), self.W.shape)
        W_bar = self.W / sigma
        if hasattr(self, "gamma"):
            W_bar = broadcast_to(self.gamma, self.W.shape) * W_bar
        return W_bar

    def _initialize_params(self, in_size):
        super()._initialize_params(in_size)
        if self.use_gamma:
            W_mat = self.W.data.reshape(self.W.shape[0], -1)
            _, s, _ = np.linalg.svd(W_mat)
            with self.init_scope():
                self.gamma = chainer.Parameter(s[0], (1, 1, 1, 1))

    def __call__(self, x):
        if self.W.data is None:
            self._initialize_params(x.shape[1])
        return convolution_2d.convolution_2d(
            x, self.W_bar, self.b, self.stride, self.pad
        )
```

## Optional Reparametrization

The relaxation

    W_tilde = gamma W_bar_SN

learns a scalar scale `gamma`. It gives up the per-layer fixed 1-Lipschitz matrix scale, so it should be paired with another Lipschitz control such as a gradient penalty.
