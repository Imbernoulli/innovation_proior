# Orthogonal initialization (dynamical isometry), distilled

Initialize convolutional and linear weight tensors as random semi-orthogonal
matrices, then scale them by the gain appropriate to the nonlinearity. The goal
is not merely to preserve the expected norm of a random vector; it is to keep the
singular spectrum of the end-to-end Jacobian concentrated near an `O(1)` value so
gradients do not vanish or explode through depth.

## Core dynamics

For a three-layer linear net `y = W32 W21 x`, continuous-time batch gradient
descent gives

```text
tau dW21/dt = W32^T (Sigma31 - W32 W21 Sigma11)
tau dW32/dt = (Sigma31 - W32 W21 Sigma11) W21^T
```

With whitened inputs, `Sigma11 = I`, and `Sigma31 = U S V^T`, the change of
variables `W21 = W21bar V^T`, `W32 = U W32bar` decouples the learning modes. On
the decoupled manifold, one mode with strength target `s` obeys

```text
tau da/dt = b(s - ab)
tau db/dt = a(s - ab)
```

For the symmetric case `u = ab = a^2`,

```text
tau du/dt = 2u(s - u)
u(t) = s exp(2st/tau) / (exp(2st/tau) - 1 + s/u0)
t_learn = (tau/s) log(s/eps) = O(tau/s)
```

For `N_l` layers, with `N_l - 1` weight matrices and composite strength
`u = prod_i a_i`,

```text
tau du/dt = (N_l - 1) u^(2 - 2/(N_l - 1)) (s - u)
```

The top Hessian eigenvalue on the symmetric path at
`a_opt = s^(1/(N_l - 1))` is

```text
lambda1(a_opt) = (N_l - 1) s^((2N_l - 4)/(N_l - 1)) / tau
alpha_opt = O(1 / (N_l s^2))
```

With that depth-dependent learning rate, the infinite-depth delay over the
three-layer case remains finite when `u0 = O(1)`. If each layer starts with
scale `a0 < 1`, then `u0 = a0^(N_l - 1)` vanishes exponentially, and learning
time diverges with depth.

## Why scaled Gaussian is insufficient

For `W_ij ~ N(0, 1/N)`, `E[v^T W^T W v] = v^T v`, so the average squared norm is
preserved. But the squared singular values follow a Marchenko-Pastur spread.
Products of such matrices become highly anisotropic: most singular values shrink
toward zero while a small tail grows. The product is non-normal, with
non-orthogonal eigenvectors, so eigenvalues near the origin can coexist with
large singular values. Backpropagation through the transpose then crushes error
components in most directions.

Dynamical isometry asks for the full product of Jacobians to act like an
isometry, not just for an average norm calculation to be correct. A square
orthogonal matrix has every singular value exactly `1`, and products of such
matrices stay orthogonal. Rectangular tensors use the corresponding
semi-orthogonal form, with all nonzero singular values equal to `1`.

## Nonlinear gain

For `x^{l+1} = g W phi(x^l)` with `W` orthogonal,

```text
q^{l+1} = g^2 int Dz phi(sqrt(q^l) z)^2
```

For `tanh`, linearizing near `q = 0` gives slope `g^2`, so the critical gain is
`g_c = 1`. For ReLU,
`int Dz max(0, sqrt(q) z)^2 = q/2`, so the variance-preserving gain is
`sqrt(2)`. PyTorch's `calculate_gain("relu")` returns this value. Its
`calculate_gain("tanh") = 5/3` is a separate practical variance heuristic, not
the `tanh` edge-of-chaos value above.

## PyTorch Primitive

The core of `nn.init.orthogonal_` implements the Haar-corrected QR recipe:

```python
def orthogonal_core(tensor, gain=1, generator=None):
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")
    if tensor.numel() == 0:
        return tensor

    rows = tensor.size(0)
    cols = tensor.numel() // rows
    flattened = tensor.new(rows, cols).normal_(0, 1, generator=generator)
    if rows < cols:
        flattened.t_()

    q, r = torch.linalg.qr(flattened)
    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph

    if rows < cols:
        q.t_()

    with torch.no_grad():
        tensor.view_as(q).copy_(q)
        tensor.mul_(gain)
    return tensor
```

The `rows < cols` transpose produces the correct semi-orthogonal orientation for
wide matrices, and `ph = sign(diag(r)); q *= ph` removes the QR sign bias.

## Edit Body

```python
def initialize_weights(model, config):
    """Orthogonal initialization.

    Conv2d & Linear: orthogonal matrix (gain=sqrt(2) for ReLU).
    BatchNorm2d: weight=1, bias=0.
    """
    gain = nn.init.calculate_gain('relu')
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight, gain=gain)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
```
