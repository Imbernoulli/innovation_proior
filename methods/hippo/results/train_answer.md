The problem is to maintain a useful summary of an entire input history using only a fixed-size state that is updated one step at a time. A plain recurrent hidden state can in principle remember arbitrarily far back, but in practice backpropagation through time multiplies Jacobian matrices across long gaps, and that product almost always decays or explodes geometrically. The effective memory horizon is therefore short regardless of how large the state is. LSTMs and GRUs improve this with gated convex updates, and orthogonal or unitary RNNs constrain the recurrent spectrum, yet these are heuristics that patch the dynamics rather than state what the memory should optimally represent. Fourier-based running coefficients and sliding-window polynomial methods give cleaner pictures but still depend on knowing or fixing a timescale, and they either forget the distant past by construction or require an external window length.

A better starting point is to treat memory as online function approximation. The history up to time t is a function f_{≤t} living in an infinite-dimensional space, and a fixed-size state can only store coordinates in some N-dimensional subspace. If we also choose a measure μ^(t) on the past that says which instants matter, then the optimal summary is the orthogonal projection of f_{≤t} onto that subspace under the inner product defined by μ^(t). Polynomials are especially convenient because orthogonal polynomials give a closed-form basis: the best degree-(<N) approximation is obtained by reading off inner-product coefficients, with no optimization at each step. The summary state is therefore the coefficient vector c(t) of this projection.

The key observation is that this coefficient vector obeys a linear ODE driven by the input. Differentiating c_n(t) = ⟨f_{≤t}, g_n^{(t)}⟩_{μ^(t)} in time produces two kinds of terms. The basis derivative ∂_t g_n is a polynomial of lower degree, so its integral against f re-expresses as a linear combination of the existing coefficients. The measure derivative injects the current input value at the moving boundary of the window, via the Leibniz rule written distributionally as ∂_t 𝟙_{[α,β]} = β'δ_β − α'δ_α. Together these give d/dt c(t) = A(t)c(t) + B(t)f(t). Thus the memory update is not chosen by hand; it is forced by the requirement that the state represent the optimal projection.

The canonical method is HiPPO, which stands for High-order Polynomial Projection Operator. The central instantiation is HiPPO-LegS, obtained by taking the measure to be uniform over the entire history [0, t], so μ^(t) = (1/t)𝟙_{[0,t]}. Because the window grows with t, nothing ever scrolls out and there is no fixed window length or timescale hyperparameter. In the corresponding orthonormal Legendre basis the dynamics reduce to a single fixed matrix A modulated by 1/t: d/dt c(t) = −(1/t)Ac(t) + (1/t)Bf(t), where A is lower-triangular with entries (2n+1)^{1/2}(2k+1)^{1/2} below the diagonal and n+1 on the diagonal, and B_n = (2n+1)^{1/2}. Euler discretization gives c_{k+1} = (1 − A/k)c_k + (1/k)Bf_k. The step size cancels, so the recurrence is invariant to the sampling rate: dilating or contracting the input signal simply dilates or contracts the coefficient trajectory by the same factor.

HiPPO-LegS inherits strong guarantees from this construction. The unrolled sensitivity of a later state to an earlier input is a product of matrices (I − A/i). Because A has eigenvalues 1, …, N, this product telescopes and its norm decays as Θ(1/t_1), polynomial rather than exponential in the gap, which eliminates the vanishing-gradient problem. The triangular structure of A makes each update O(N) instead of O(N²): multiplication by I + δA reduces to diagonal scalings and a cumulative sum, and the implicit bilinear inverse can also be computed by a scalar cumsum recurrence. Finally, the approximation error of the degree-(<N) projection is O(tL/√N) for an L-Lipschitz input, sharpening to O(t^k N^{−k+1/2}) for smoother inputs, so larger states faithfully compress longer histories.

```python
import numpy as np
from scipy import linalg as la, special as ss
import torch
import torch.nn as nn
import torch.nn.functional as F


def transition(measure, N):
    """Return continuous (A, B) matrices for dc/dt = A c + B f."""
    if measure == 'legs':
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        # M has 2k+1 below diagonal, k+1 on diagonal, 0 above.
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None].copy()
    elif measure == 'legt':
        Q = np.arange(N, dtype=np.float64)
        R = (2 * Q + 1) ** .5
        j, i = np.meshgrid(Q, Q)
        A = R[:, None] * np.where(i < j, (-1.) ** (i - j), 1) * R[None, :]
        B = R[:, None]
        A = -A
    return A, B


class HiPPO_LegS(nn.Module):
    """Scale-invariant memory: optimal Legendre projection over [0, t]."""

    def __init__(self, N, max_length=1024, discretization='bilinear'):
        super().__init__()
        self.N = N
        A, B = transition('legs', N)
        B = B.squeeze(-1)
        A_stacked = np.empty((max_length, N, N), dtype=A.dtype)
        B_stacked = np.empty((max_length, N), dtype=B.dtype)
        for t in range(1, max_length + 1):
            At, Bt = A / t, B / t
            if discretization == 'forward':
                A_stacked[t - 1] = np.eye(N) + At
                B_stacked[t - 1] = Bt
            elif discretization == 'bilinear':
                A_stacked[t - 1] = la.solve_triangular(
                    np.eye(N) - At / 2, np.eye(N) + At / 2, lower=True)
                B_stacked[t - 1] = la.solve_triangular(
                    np.eye(N) - At / 2, Bt, lower=True)
            else:  # zero-order hold
                A_stacked[t - 1] = la.expm(A * (np.log(t + 1) - np.log(t)))
                B_stacked[t - 1] = la.solve_triangular(
                    A, A_stacked[t - 1] @ B - B, lower=True)
        self.register_buffer('A_stacked', torch.Tensor(A_stacked))
        self.register_buffer('B_stacked', torch.Tensor(B_stacked))
        vals = np.linspace(0.0, 1.0, max_length)
        self.eval_matrix = torch.Tensor(
            (B[:, None] * ss.eval_legendre(np.arange(N)[:, None], 2 * vals - 1)).T)

    def forward(self, inputs):
        L = inputs.shape[0]
        u = (inputs.unsqueeze(-1).transpose(0, -2) * self.B_stacked[:L]).transpose(0, -2)
        c = torch.zeros(u.shape[1:])
        cs = []
        for k in range(L):
            c = F.linear(c, self.A_stacked[k]) + u[k]
            cs.append(c)
        return torch.stack(cs, dim=0)

    def reconstruct(self, c):
        return (self.eval_matrix @ c.unsqueeze(-1)).squeeze(-1)
```
