In a generative adversarial network the discriminator is not merely an evaluator; it is the only teacher the generator ever has. The generator never sees the data distribution directly — it moves because $D$ assigns scores to its samples and the gradient of those scores flows back through $G$. This makes the conditioning of $D$ the conditioning of the whole game, and it makes a stronger classifier not automatically a better one. The sharp failure is a support argument: real images and generated images lie near low-dimensional sets in a huge ambient space, and early in training those sets need not overlap, so a discriminator can separate them perfectly while being essentially flat on the generated side, where its input derivative vanishes exactly in the region the generator is asking about. The optimal-discriminator analysis gives the same diagnosis from the other side. For fixed $G$ the original GAN objective is maximized by $D^*_G(x) = q_{\text{data}}(x)/(q_{\text{data}}(x)+p_G(x)) = \mathrm{sigmoid}(f^*(x))$ with $f^*(x) = \log q_{\text{data}}(x) - \log p_G(x)$, and the input derivative $\nabla_x f^*(x) = \nabla_x q_{\text{data}}(x)/q_{\text{data}}(x) - \nabla_x p_G(x)/p_G(x)$ can blow up or become undefined near the edges of support. The dangerous object is therefore the input sensitivity of the score, not the classification accuracy, and what we need is the best discriminator inside a smoother function class rather than the best unconstrained one.

The right family is the $K$-Lipschitz one: if $\lVert f\rVert_{\mathrm{Lip}} \le K$ then $\lVert f(x)-f(x')\rVert \le K\lVert x-x'\rVert$, so no part of the score network can develop an arbitrarily steep transition. This is an upper control on sensitivity, not a lower bound on signal — it does not promise nonzero generator gradients everywhere — but it removes precisely the family of arbitrarily sharp separators that drives the most brittle behavior. The existing ways of enforcing it all miss the target in characteristic ways. Weight clipping, $w \leftarrow \mathrm{clip}(w,-c,c)$, constrains entries in a box rather than the operator stretch of the layer, and a critic can buy sensitivity back by lining its weights up into a low-dimensional effective map. The gradient penalty $\lambda\,\mathbb{E}_{\hat x}[(\lVert\nabla_{\hat x}D(\hat x)\rVert_2-1)^2]$ regularizes the right quantity but only locally at sampled interpolation points on the current supports, and it forces a heavier gradient-of-gradient computation. Weight or Frobenius normalization fixes a budget on the whole spectrum — with rows normalized to unit norm, $\sum_t \sigma_t(W)^2 = \mathrm{tr}(WW^\top) = d_{\text{out}}$ — and under a fixed sum of squared singular values the cheapest way to be sensitive in one direction is to dump the budget into the top singular value, pushing the layer toward low effective rank. Orthonormal regularization $\lVert W^\top W - I\rVert_F^2$ over-corrects the opposite way, forcing every direction to share one singular value even when the discriminator has no use for that direction. A soft spectral-norm penalty estimates the right matrix quantity but, being a penalty, only nudges the largest singular value through a coefficient that competes with the task loss rather than fixing the layer scale.

I propose spectral normalization. The reduction that makes it work is that for a differentiable map the local stretch is the largest singular value of the Jacobian, $\lVert g\rVert_{\mathrm{Lip}} = \sup_h \sigma(\nabla g(h))$; for a linear layer $g(h)=Wh$ the Jacobian is just $W$, so its Lipschitz constant is exactly the spectral norm $\sigma(W) = \max_{h\neq 0}\lVert Wh\rVert_2/\lVert h\rVert_2$. Since Lipschitz constants are submultiplicative under composition and ReLU/leaky-ReLU with slope at most one are $1$-Lipschitz, the whole score network obeys
$$\lVert f\rVert_{\mathrm{Lip}} \le \prod_l \sigma(W^l).$$
So I need not touch every entry, fix a Frobenius budget, or sample input gradients — controlling the top singular value of each weight controls this standard product bound. The operation is then almost embarrassingly direct:
$$\bar W_{\mathrm{SN}} = W/\sigma(W),$$
which gives $\sigma(\bar W_{\mathrm{SN}})=1$ by homogeneity of the spectral norm while leaving every ratio $\sigma_2/\sigma_1, \sigma_3/\sigma_1,\dots$ untouched. This is the decisive contrast with the budget methods: I fix only the top scale and leave the rest of the spectrum free, so the layer can stay high rank. Applied to every matrix layer, the product bound becomes one; a different global Lipschitz target can be obtained by allocating constants across layers, but the normal case is simply to use one.

Two obstacles remain. Convolutional kernels are tensors $W\in\mathbb{R}^{d_{\text{out}}\times d_{\text{in}}\times k_h\times k_w}$ whose true operator norm depends on stride, padding, and input size; I reshape the kernel into the matrix $d_{\text{out}}\times(d_{\text{in}}k_h k_w)$ and control the largest singular value of that proxy, treating the residual difference as a predefined architecture constant. So for fully connected layers I am normalizing the matrix operator norm exactly, while for convolutions I am controlling the standard flattened-kernel proxy that bounds the layer scale up to architecture constants. The second obstacle is cost: a full SVD per layer per discriminator update is unacceptable inside a GAN loop, but I only need the top singular value, which is exactly what power iteration delivers. Starting from a vector $u$, I alternate $v \leftarrow W^\top u/\lVert W^\top u\rVert_2$ and $u \leftarrow Wv/\lVert Wv\rVert_2$; this repeatedly applies $WW^\top$ to $u$, multiplying the component along $u_t$ by $\sigma_t^2$ each round, so the top component dominates when the largest singular value is isolated and the initialization is not orthogonal to it, after which $\sigma(W)\approx u^\top W v$. Restarting from a fresh random vector every step would waste this: the weight moves by one optimizer step at a time, so its leading singular vector moves gradually too. I therefore store the current $u$ with the layer and warm-start from it, which turns the routine into a cheap tracker for which a single power-iteration step per update suffices in practice — not as a theorem that one step always converges, but as the algorithmic reason the method is cheap.

The gradient that flows through the normalization is what distinguishes this reparameterization from a penalty, and it comes out with the right sign for free. For a simple top singular value, $\partial\sigma(W)/\partial W = u_1 v_1^\top$, so entrywise
$$\frac{\partial \bar W_{\mathrm{SN}}}{\partial W_{ij}} = \frac{1}{\sigma(W)}\Big(E_{ij} - [u_1 v_1^\top]_{ij}\,\bar W_{\mathrm{SN}}\Big).$$
Writing $\delta := (\partial V/\partial(\bar W_{\mathrm{SN}}h))^\top$ for the backpropagated output signal and $h$ for the layer input, the chain rule gives
$$\frac{\partial V}{\partial W} = \frac{1}{\sigma(W)}\Big(\hat{\mathbb{E}}[\delta h^\top] - \lambda\, u_1 v_1^\top\Big),\qquad \lambda = \hat{\mathbb{E}}[\delta^\top \bar W_{\mathrm{SN}}h].$$
The first term is the ordinary minibatch weight gradient; the second is a correction that is negative in the leading singular direction, with a data-dependent coefficient $\lambda$ equal to the average alignment between the backprop signal and the layer output. When that alignment is positive, the correction pushes back against growing the dominant singular component — an adaptive pressure, not an absolute prevention of rank collapse, since the fixed-point condition only requires the ordinary gradient to align with $u_1 v_1^\top$ up to a scalar, which is far weaker than driving the matrix to rank one. The general normalizer identity makes the design choice explicit: for $\bar W = W/N(W)$,
$$\frac{\partial V}{\partial W} = \frac{1}{N}\Big(\nabla_{\bar W}V - \mathrm{tr}\big((\nabla_{\bar W}V)^\top \bar W\big)\,\nabla_W N\Big),$$
so a Frobenius $N$ has $\nabla_W N = \bar W$ and corrects along the whole matrix direction, whereas a spectral $N$ has $\nabla_W N = u_1 v_1^\top$ and corrects only the leading singular pair — exactly the goal of controlling the top stretch while leaving the rest of the spectrum free.

One optional relaxation is worth keeping in the interface. I can write $\tilde W = \gamma\,\bar W_{\mathrm{SN}}$ with a learned scalar $\gamma$, which restores a trainable overall scale: the direction and spectrum shape stay normalized by the top singular value, but the layer is no longer pinned to a $1$-Lipschitz factor on its own, so some other mechanism such as a gradient penalty must then handle the global Lipschitz target. The method is implemented as a layer computation, not as an optimizer step that overwrites weights after training. Each layer keeps the raw trainable $W$ and a persistent non-parameter $u$; on each forward it flattens $W$ if convolutional, runs the configured power iterations to update $u$, computes $\sigma = u^\top W_{\text{mat}} v$, and calls the underlying linear or convolution operation with $W/\sigma$, optionally scaled by $\gamma$ or divided by a `factor` for a different target scale. The one subtle point of implementation is gradient flow: the power-iteration updates for $u$ and $v$ may use the raw array value of $W$ as bookkeeping, but $\sigma$ must then be recomputed with the current $W$ as a differentiable variable so that the division by $\sigma$ still contributes the adaptive correction above.

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
