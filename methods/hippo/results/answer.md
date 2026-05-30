# HiPPO: Recurrent Memory with Optimal Polynomial Projections

## Problem

Maintain, online and in fixed storage, a summary of the entire history f(x) for x ≤ t of a signal arriving one value at a time — rich enough to predict from, cheap to update, and not tied to any particular timescale. Recurrent states in principle have unbounded context but suffer vanishing/exploding gradients; gates, orthogonal recurrences, and sliding-window transforms patch the dynamics without saying what the state should optimally represent.

## Key idea

Define memory as **online function approximation**. Equip function space with an inner product from a time-varying measure μ^(t) on (−∞, t] that encodes how much each past instant matters. At every t, store the coefficients c(t) ∈ ℝ^N of the optimal projection of f_{≤t} onto degree-(<N) polynomials, using the orthonormal (orthogonal) polynomial basis {g_n^{(t)}} of μ^(t):

c_n(t) = ⟨f_{≤t}, g_n^{(t)}⟩_{μ^{(t)}} (closed form — no optimization).

Differentiating c(t) in t turns the projection into a **linear ODE driven by the input**,

d/dt c(t) = A(t) c(t) + B(t) f(t),

because (a) the derivative of an orthogonal polynomial is a lower-degree polynomial (so ∂_t g_n re-expands in the basis → linear in the c_k), and (b) the derivative of the (indicator) measure, via the Leibniz/Dirac-delta rule ∂_t 𝟙_{[α,β]} = β'δ_β − α'δ_α, injects the current value f(t) at the moving edge. Discretizing the ODE gives a step recurrence c_k = A_k c_{k−1} + B_k f_k.

## Instantiations

**HiPPO-LegT (translated Legendre)** — uniform measure on a sliding window [t−θ, t]. Yields a linear time-invariant ODE d/dt c = −(1/θ)Ac + (1/θ)Bf with, in the orthonormal basis,
A_{nk} = (2n+1)^{1/2}(2k+1)^{1/2} · {1 if k ≤ n; (−1)^{n−k} if k ≥ n}, B_n = (2n+1)^{1/2}.
Rescaling the basis by λ_n = (2n+1)^{1/2}(−1)^n reproduces the **Legendre Memory Unit** exactly: A_{nk} = (2n+1)·{(−1)^{n−k} if k ≤ n; 1 if k ≥ n}, B_n = (2n+1)(−1)^n. (The sliding update needs the dropped value f(t−θ), recovered from the current reconstruction — an extra approximation intrinsic to the window.) At order N = 1 with an exponentially decaying (Laguerre) measure, the discretized update is c ← (1−Δt)c + Δt f; with an adaptive, input-dependent Δt this is a **gated RNN** — so LSTM/GRU gating is the order-1 corner of the same framework.

**HiPPO-LegS (scaled Legendre)** — uniform measure over the *entire* history, μ^(t) = (1/t)𝟙_{[0,t]}, whose width grows with t so nothing is ever forgotten and there is no window/step-size prior. The dynamics are a single fixed matrix modulated by 1/t:

d/dt c(t) = −(1/t) A c(t) + (1/t) B f(t),
A_{nk} = (2n+1)^{1/2}(2k+1)^{1/2} (n > k); n+1 (n = k); 0 (n < k); B_n = (2n+1)^{1/2},

with Euler discretization c_{k+1} = (1 − A/k) c_k + (1/k) B f_k. Equivalently A = T(L+D)T^{−1} with T = diag((2n+1)^{1/2}) and L the all-ones lower-triangular matrix.

**Theoretical properties of LegS:**
- **Timescale equivariance:** if h(t) = f(αt) then hippo(h)(t) = hippo(f)(αt); the discrete recurrence is invariant to the step size Δt (it cancels because A and B share the 1/t factor). No timescale hyperparameter.
- **Bounded gradients:** ‖∂c(t_1)/∂f(t_0)‖ = Θ(1/t_1) — polynomial, not exponential, decay (the unrolled product ∏_{i=k+1}^{ℓ}(1−1/i) = k/ℓ telescopes, and A has eigenvalues 1,…,N).
- **O(N) updates:** the D·(L+D')·D' structure makes multiplication by I+δA a cumsum and (I−δA)^{−1} a scalar cumsum/cumprod recurrence.
- **Approximation error:** O(tL/√N) for L-Lipschitz f, sharpening to O(t^k N^{−k+1/2}) for order-k-smooth f (Parseval on the coefficient tail + repeated integration by parts).

## Code

```python
import numpy as np
from scipy import linalg as la, special as ss
import torch, torch.nn as nn, torch.nn.functional as F


def transition(measure, N):
    """(A, B) for the continuous update dc/dt = A c + B f (A in machine form,
    negative diagonal; the text's A has positive diagonal, i.e. this A negated)."""
    if measure == 'legt':                       # translated Legendre (sliding window) = LMU
        Q = np.arange(N, dtype=np.float64)
        R = (2 * Q + 1) ** .5
        j, i = np.meshgrid(Q, Q)
        A = R[:, None] * np.where(i < j, (-1.) ** (i - j), 1) * R[None, :]
        B = R[:, None]
        A = -A
    elif measure == 'legs':                     # scaled Legendre (whole history [0, t])
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)            # sqrt(2n+1)sqrt(2k+1) (n>k); n+1 (n=k); 0 (n<k)
        B = np.diag(T)[:, None].copy()          # (2n+1)^{1/2}
    return A, B


class HiPPO_LegT(nn.Module):
    """Optimal Legendre projection over a fixed sliding window of length ~1/dt."""
    def __init__(self, N, dt=1.0, discretization='bilinear'):
        super().__init__()
        self.N = N
        A, B = transition('legt', N)
        C, D = np.ones((1, N)), np.zeros((1,))
        from scipy import signal
        A, B, _, _, _ = signal.cont2discrete((A, B, C, D), dt=dt, method=discretization)
        self.register_buffer('A', torch.Tensor(A))           # (N, N)
        self.register_buffer('B', torch.Tensor(B.squeeze(-1)))
        vals = np.arange(0.0, 1.0, dt)
        self.eval_matrix = torch.Tensor(
            ss.eval_legendre(np.arange(N)[:, None], 1 - 2 * vals).T)

    def forward(self, inputs):                  # (L, ...) -> (L, ..., N)
        c = torch.zeros(inputs.shape[1:] + (self.N,))
        cs = []
        for f in inputs:
            c = F.linear(c, self.A) + self.B * f.unsqueeze(-1)
            cs.append(c)
        return torch.stack(cs, dim=0)

    def reconstruct(self, c):
        return (self.eval_matrix @ c.unsqueeze(-1)).squeeze(-1)


class HiPPO_LegS(nn.Module):
    """Scale-invariant memory: optimal Legendre projection over all of [0, t]."""
    def __init__(self, N, max_length=1024, discretization='bilinear'):
        super().__init__()
        self.N = N
        A, B = transition('legs', N)
        B = B.squeeze(-1)
        A_stacked = np.empty((max_length, N, N), dtype=A.dtype)
        B_stacked = np.empty((max_length, N), dtype=B.dtype)
        for t in range(1, max_length + 1):       # discretize the 1/t-scaled ODE; Delta t cancels
            At, Bt = A / t, B / t
            if discretization == 'forward':
                A_stacked[t - 1] = np.eye(N) + At
                B_stacked[t - 1] = Bt
            elif discretization == 'bilinear':
                A_stacked[t - 1] = la.solve_triangular(np.eye(N) - At / 2, np.eye(N) + At / 2, lower=True)
                B_stacked[t - 1] = la.solve_triangular(np.eye(N) - At / 2, Bt, lower=True)
            else:                                # zero-order hold
                A_stacked[t - 1] = la.expm(A * (np.log(t + 1) - np.log(t)))
                B_stacked[t - 1] = la.solve_triangular(A, A_stacked[t - 1] @ B - B, lower=True)
        self.register_buffer('A_stacked', torch.Tensor(A_stacked))
        self.register_buffer('B_stacked', torch.Tensor(B_stacked))
        vals = np.linspace(0.0, 1.0, max_length)
        self.eval_matrix = torch.Tensor(
            (B[:, None] * ss.eval_legendre(np.arange(N)[:, None], 2 * vals - 1)).T)

    def forward(self, inputs):                  # (L, ...) -> (L, ..., N)
        L = inputs.shape[0]
        u = (inputs.unsqueeze(-1).transpose(0, -2) * self.B_stacked[:L]).transpose(0, -2)
        c = torch.zeros(u.shape[1:])
        cs = []
        for k in range(L):                       # c_k = A_k c_{k-1} + B_k f_k
            c = F.linear(c, self.A_stacked[k]) + u[k]
            cs.append(c)
        return torch.stack(cs, dim=0)

    def reconstruct(self, c):
        return (self.eval_matrix @ c.unsqueeze(-1)).squeeze(-1)
```

The continuous operator is discretized with Euler / bilinear / ZOH; LegS is invariant to the step size by construction. Both modules implement the same interface — `forward` rolls the recurrence to produce coefficient vectors, `reconstruct` maps any coefficient vector back to an approximation of the input over the represented window.
