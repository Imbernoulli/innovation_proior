# Spectral Normalization for GANs

## Problem

In a GAN the generator learns only through the discriminator's gradient, so the
discriminator's function class controls training stability. An unconstrained discriminator can
become arbitrarily sharp; in the (generic) case of nearly disjoint supports between the data
and generator distributions, the optimal discriminator separates them perfectly and has zero
input-gradient almost everywhere, which kills the generator's learning signal. The cure is to
restrict the discriminator to K-Lipschitz functions — but cheaply, with essentially no tuning,
and without crippling the discriminator's capacity.

## Key idea

The Lipschitz constant of a linear layer g(h) = Wh equals the **spectral norm** σ(W) — the
largest singular value of W. By sub-multiplicativity of the Lipschitz constant under
composition and the fact that ReLU/leaky-ReLU activations are 1-Lipschitz, the whole network
satisfies

    ‖f‖_Lip ≤ Π_l σ(W^l).

**Spectral normalization** sets each layer's spectral norm to 1 by dividing the weight by its
own spectral norm,

    W̄_SN = W / σ(W),    so    σ(W̄_SN) = 1   and   ‖f‖_Lip ≤ 1.

Because only the largest singular value is touched, the rest of the spectrum (the rank, the
number of usable features) is left free — unlike weight clipping, weight normalization, or
Frobenius normalization, all of which constrain Σ_t σ_t^2 and thereby push the weight toward
rank one ("capacity underuse"), and unlike orthonormal regularization, which forces all
singular values to 1 and erases the spectrum.

## Fast σ(W) by power iteration

A full SVD per layer per step is too expensive. Only the top singular value is needed, so it
is estimated by power iteration. Keeping a persistent left vector u̅ per layer and running a
single iteration per update (warm-started from the previous step, since W moves little under
SGD) is enough:

    v̅ ← W^T u̅ / ‖W^T u̅‖_2,    u̅ ← W v̅ / ‖W v̅‖_2,    σ(W) ≈ u̅^T W v̅.

Cost is two matrix–vector products per layer — negligible versus the forward/backward pass,
and far cheaper than WGAN-GP's gradient-of-gradient penalty. For a convolutional weight
W ∈ R^{d_out×d_in×kh×kw}, reshape to a 2-D matrix d_out × (d_in·kh·kw) before taking σ.

## Gradient of the normalized weight

Using ∂σ(W)/∂W = u_1 v_1^T,

    ∂W̄_SN/∂W_{ij} = (1/σ)( E_{ij} − [u_1 v_1^T]_{ij} W̄_SN ),

and with δ := (∂V/∂(W̄_SN h))^T the loss gradient with respect to the raw weight is

    ∂V/∂W = (1/σ(W)) ( Ê[δ h^T] − λ u_1 v_1^T ),    λ := Ê[δ^T W̄_SN h].

The first term is the ordinary unnormalized gradient; the second is an adaptive penalty (λ>0
when δ and W̄_SN h align) that prevents the column space from concentrating into one direction
— a built-in guard against the rank collapse the method was designed to avoid.

## Algorithm (per update, per layer l)

1. v̅_l ← (W^l)^T u̅_l / ‖(W^l)^T u̅_l‖,   u̅_l ← W^l v̅_l / ‖W^l v̅_l‖.
2. σ(W^l) = u̅_l^T W^l v̅_l;   W̄_SN^l = W^l / σ(W^l).
3. Update raw W^l by SGD/Adam on the loss using W̄_SN^l in the forward pass.

The persistent u̅_l (and v̅_l) carry across updates; one power iteration per step suffices. The
only hyper-parameter is the target Lipschitz constant, which is normally just 1.

## Code

```python
import torch
import torch.nn as nn
from torch.nn.functional import normalize

class SpectralNorm:
    """Forward pre-hook: replace `weight` by weight / sigma(weight) each forward pass,
    with sigma estimated by one warm-started power-iteration step."""
    def __init__(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12):
        self.name = name
        self.dim = dim
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def reshape_weight_to_matrix(self, weight):
        w = weight
        if self.dim != 0:
            w = w.permute(self.dim, *[d for d in range(w.dim()) if d != self.dim])
        return w.reshape(w.size(0), -1)

    def compute_weight(self, module, do_power_iteration):
        W     = getattr(module, self.name + '_orig')   # raw trainable weight
        u     = getattr(module, self.name + '_u')        # persistent buffers (warm start)
        v     = getattr(module, self.name + '_v')
        W_mat = self.reshape_weight_to_matrix(W)
        if do_power_iteration:
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    v = normalize(torch.mv(W_mat.t(), u), dim=0, eps=self.eps, out=v)
                    u = normalize(torch.mv(W_mat,     v), dim=0, eps=self.eps, out=u)
                if self.n_power_iterations > 0:
                    u = u.clone(memory_format=torch.contiguous_format)
                    v = v.clone(memory_format=torch.contiguous_format)
        sigma = torch.dot(u, torch.mv(W_mat, v))        # sigma(W) ~= u^T W v
        return W / sigma                                 # gradient flows through sigma

    def __call__(self, module, inputs):
        setattr(module, self.name,
                self.compute_weight(module, do_power_iteration=module.training))

    @staticmethod
    def apply(module, name='weight', n_power_iterations=1, dim=0, eps=1e-12):
        fn = SpectralNorm(name, n_power_iterations, dim, eps)
        weight = module._parameters[name]
        with torch.no_grad():
            W_mat = fn.reshape_weight_to_matrix(weight)
            h, w = W_mat.size()
            u = normalize(weight.new_empty(h).normal_(0, 1), dim=0, eps=fn.eps)
            v = normalize(weight.new_empty(w).normal_(0, 1), dim=0, eps=fn.eps)
        delattr(module, name)
        module.register_parameter(name + '_orig', weight)
        setattr(module, name, weight.data)
        module.register_buffer(name + '_u', u)
        module.register_buffer(name + '_v', v)
        module.register_forward_pre_hook(fn)
        return fn

def spectral_norm(module, name='weight', n_power_iterations=1, dim=None, eps=1e-12):
    if dim is None:
        dim = 0
    SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
    return module


def D_conv(in_ch, out_ch, k, s, p):
    return spectral_norm(nn.Conv2d(in_ch, out_ch, k, s, p))

class Discriminator(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        self.net = nn.Sequential(
            D_conv(3,    ch,   3, 1, 1), nn.LeakyReLU(0.1),
            D_conv(ch,   ch*2, 4, 2, 1), nn.LeakyReLU(0.1),
            D_conv(ch*2, ch*4, 4, 2, 1), nn.LeakyReLU(0.1),
            D_conv(ch*4, ch*8, 4, 2, 1), nn.LeakyReLU(0.1),
        )
        self.fc = spectral_norm(nn.Linear(ch*8*4*4, 1))
    def forward(self, x):
        return self.fc(self.net(x).flatten(1))
```

An optional reparametrization W̃ = γ W̄_SN with a single learned scalar γ relaxes the strict
1-Lipschitz constraint by one degree of freedom; it is useful when combined with another
Lipschitz control (e.g. a gradient penalty) rather than relying on the normalization alone.
