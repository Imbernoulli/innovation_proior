# S4: Structured State Space sequence model

## Problem

Build one sequence layer that handles dependencies over 10k+ steps with (i) principled unbounded memory, (ii) parallel training, (iii) constant-time-per-step recurrent inference, and (iv) ~linear cost in sequence length L and state size N. RNNs have memory but vanishing gradients and sequential training; CNNs are parallel but finite-context; Transformers are global but O(L²). A linear state space model (SSM) is simultaneously a recurrence and a convolution and, with the right state matrix, has principled memory — but the prior deep-SSM layer cost O(N²L) time / O(NL) memory to build its convolution kernel and was numerically unstable when sped up. S4 makes the SSM kernel computable in Õ(N+L) time and O(N+L) space, stably.

## Key idea

Start from the continuous SSM x'(t)=Ax(t)+Bu(t), y(t)=Cx(t) (the feedthrough Du is a skip connection). Initialize A to the HiPPO matrix (optimal continuous-time memory). Discretize with the bilinear transform to a recurrence/convolution. The convolution kernel K̄ = (C̄Ā^i B̄)_{i<L} naively needs L powers of Ā. S4 avoids the powers by:

1. **NPLR/DPLR parameterization.** HiPPO is *normal plus low-rank*: A = VΛV^* − PQ^*. A unitary change of basis (well-conditioned, unlike diagonalizing HiPPO directly, whose eigenvectors reach magnitude 2^{4N/3}) brings A to **diagonal plus low-rank** A = Λ − PQ^* over ℂ, with rank r = 1 (HiPPO-LegS) or 2 (LegT).
2. **Truncated generating function at roots of unity.** Compute K̂(z) = Σ_{i<L} C̄Ā^i B̄ z^i = C̄(I − Ā^L z^L)(I − Āz)^{-1} B̄ at z = ω_k = exp(−2πi k/L). Powers of Ā become one **inverse** per node; (I − Ā^L) becomes a constant folded into C̃ = (I − Ā^L)^*C; and recovering K̄ from {K̂(ω_k)} is a single **inverse FFT** (the evaluation is a DFT).
3. **Woodbury + Cauchy.** Back the resolvent onto the original A: K̂(z) = (2/(1+z)) C̃^*(g(z)I − A)^{-1}B with g(z) = (2/Δ)(1−z)/(1+z). Peel the low-rank term with the **Woodbury identity**, leaving the diagonal resolvent R(z) = (g(z)I − Λ)^{-1}. The resulting four bilinear forms u^*R(z)v = Σ_n u_n^*v_n/(g(z) − λ_n) over all nodes are **Cauchy matrix-vector products** — Õ(N+L), numerically stable (Fast-Multipole-style algorithms).

The same DPLR structure also gives an O(N)-per-step recurrence (the inverse of a DPLR matrix is DPLR by Woodbury), so inference is cheap too.

## Final algorithm (kernel)

Given DPLR parameters Λ, P, Q (=P for LegS), B, C̃ ∈ ℂ^N and step size Δ:

```
omega_k = exp(-2*pi*i*k/L),  k = 0..L-1
g(z)    = (2/Δ) (1 - z)/(1 + z)              # bilinear node
R(z)    = 1 / (g(z) - Λ)                      # diagonal resolvent (elementwise)
k_ab(z) = sum_n a_n^* b_n R_n(z)              # Cauchy multiply, for (a,b) in {(C̃,B),(C̃,P),(Q,B),(Q,P)}
K̂(z)   = (2/(1+z)) [ k_CB(z) - k_CP(z) k_QB(z) / (1 + k_QP(z)) ]   # rank-1 Woodbury
K̄      = iFFT( K̂ )                           # global convolution kernel, length L
y       = K̄ * u   (via FFT)                   # + D u skip
```

Discretization recurrence (bilinear):
Ā = (I − Δ/2·A)^{-1}(I + Δ/2·A), B̄ = (I − Δ/2·A)^{-1} ΔB, C̄ = C.
DPLR recurrence step: Ā = A_1 A_0 with A_0 = (2/Δ)I + (Λ − PQ^*), A_1 = (2/Δ)[D − DP(1+Q^*DP)^{-1}Q^*D], D = ((2/Δ) − Λ)^{-1}; both O(N) matrix-vector products.

Per 1-D SSM: ~5N parameters (Λ, P, B, C̃, Δ). A model uses H independent copies + position-wise linear channel mixing; nonlinearity, norm, residual between layers. Step size Δ is a learned per-feature timescale (init log-uniform in [10^{-3}, 10^{-1}]), enabling multi-timescale memory and test-time resolution change. A stored as conjugate pairs (half size); SSM parameters trained with lower lr and no weight decay.

## Code

```python
import torch
import torch.nn as nn
import numpy as np

def cauchy_naive(v, z, w):
    """Cauchy matrix-vector product: out_i = sum_n v_n / (z_i - w_n)."""
    # v, w: (..., N) ; z: (..., L)  ->  (..., L)
    cauchy = v.unsqueeze(-1) / (z.unsqueeze(-2) - w.unsqueeze(-1))  # (..., N, L)
    return cauchy.sum(dim=-2)

def hippo_legs_nplr(N):
    """HiPPO-LegS A, B, then its NPLR/DPLR form A = Lambda - P P^* (unitary V)."""
    q = np.arange(N, dtype=np.float64)
    col, row = np.meshgrid(q, q)
    r = 2 * q + 1
    M = -(np.where(row >= col, r, 0) - np.diag(q))         # the (signed) LegS matrix
    T = np.sqrt(np.diag(2 * q + 1))
    A = T @ M @ np.linalg.inv(T)                            # symmetrized HiPPO-LegS
    B = np.diag(T)[:, None]                                 # HiPPO B
    A = torch.as_tensor(A); B = torch.as_tensor(B)[:, 0]
    P = torch.sqrt(0.5 + torch.arange(N))                   # rank-1 correction: A + P P^* is skew
    AP = A + P[:, None] * P[None, :]                        # = (1/2) I + S, S skew-symmetric
    # eigen-decompose the skew part S (unitary V): A_P = V (i*Imag) V^*
    w, V = torch.linalg.eig(AP - 0.5 * torch.eye(N))        # eigenvalues pure imaginary
    Lambda = w - 0.5                                        # diagonal of DPLR A = Lambda - P P^*
    Vc = V.conj().transpose(-1, -2)
    P = Vc @ P.to(V.dtype); B = Vc @ B.to(V.dtype)          # rotate P, B into the V basis
    return Lambda, P, B, V

class S4Kernel(nn.Module):
    """Global convolution kernel for H independent DPLR SSMs (A = Lambda - P P^*)."""
    def __init__(self, H, N=64, dt_min=1e-3, dt_max=1e-1):
        super().__init__()
        Lambda, P, B, _ = hippo_legs_nplr(N)
        Lambda = Lambda.unsqueeze(0).expand(H, N).contiguous()  # (H, N)
        P = P.unsqueeze(0).expand(H, N).contiguous()
        B = B.unsqueeze(0).expand(H, N).contiguous()
        C = torch.randn(H, N, dtype=torch.cfloat)               # learns C-tilde directly
        log_dt = torch.rand(H) * (np.log(dt_max) - np.log(dt_min)) + np.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)
        self.Lambda = nn.Parameter(torch.view_as_real(Lambda))
        self.P      = nn.Parameter(torch.view_as_real(P))
        self.B      = nn.Parameter(torch.view_as_real(B))
        self.C      = nn.Parameter(torch.view_as_real(C))

    def forward(self, L):
        dt = torch.exp(self.log_dt)[:, None]                    # (H, 1)
        Lambda = torch.view_as_complex(self.Lambda)
        P = torch.view_as_complex(self.P); Q = P.conj()
        B = torch.view_as_complex(self.B); C = torch.view_as_complex(self.C)

        omega = torch.exp(-2j * np.pi * torch.arange(L // 2 + 1) / L)   # roots of unity (rfft half)
        g = (2.0 / dt) * (1 - omega) / (1 + omega)              # bilinear node g(z), (H, L//2+1)

        # four Cauchy multiplies = four bilinear forms u^* R(z) v
        k_CB = cauchy_naive(C.conj() * B, g, Lambda)
        k_CP = cauchy_naive(C.conj() * P, g, Lambda)
        k_QB = cauchy_naive(Q.conj() * B, g, Lambda)
        k_QP = cauchy_naive(Q.conj() * P, g, Lambda)

        K_hat = (2.0 / (1 + omega)) * (k_CB - k_CP * k_QB / (1 + k_QP))   # rank-1 Woodbury
        K = torch.fft.irfft(K_hat, n=L)                         # K-bar, (H, L)
        return K

class S4Layer(nn.Module):
    def __init__(self, H, N=64):
        super().__init__()
        self.kernel = S4Kernel(H, N)
        self.D = nn.Parameter(torch.randn(H))
        self.activation = nn.GELU()
        self.out = nn.Conv1d(H, H, 1)                           # position-wise channel mixing

    def forward(self, u):                                       # u: (B, H, L)
        L = u.size(-1)
        K = self.kernel(L)                                      # (H, L)
        K_f = torch.fft.rfft(K, n=2 * L)
        u_f = torch.fft.rfft(u, n=2 * L)
        y = torch.fft.irfft(u_f * K_f, n=2 * L)[..., :L]        # non-circular conv
        y = y + u * self.D[:, None]                             # D u skip
        return self.out(self.activation(y))

class S4Model(nn.Module):
    """Encoder -> stack of (S4Layer + LayerNorm + residual + dropout) -> mean pool -> decoder."""
    def __init__(self, d_input, d_output, d_model=128, n_layers=4, dropout=0.1):
        super().__init__()
        self.encoder = nn.Linear(d_input, d_model)
        self.layers = nn.ModuleList(S4Layer(d_model) for _ in range(n_layers))
        self.norms  = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(n_layers))
        self.drops  = nn.ModuleList(nn.Dropout(dropout) for _ in range(n_layers))
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):                                       # x: (B, L, d_input)
        x = self.encoder(x).transpose(-1, -2)                   # (B, H, L)
        for layer, norm, drop in zip(self.layers, self.norms, self.drops):
            z = drop(layer(x))
            x = z + x
            x = norm(x.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(-1, -2).mean(dim=1)                     # pool over L
        return self.decoder(x)
```

Optimizer: AdamW with cosine schedule; SSM parameters (Λ, P, B, Δ) get a smaller learning rate (~1e-3) and no weight decay; other parameters use the standard learning rate and weight decay. (A later refinement replaces Λ − PQ^* with Λ − PP^* to keep eigenvalues stable.)

## Complexity

| | Convolution | Recurrence | Attention | S4 |
|---|---|---|---|---|
| Parameters | LH | H² | H² | H² |
| Training | L̃H(B+H) | BLH² | B(L²H+LH²) | BH(H̃+L̃)+BL̃H |
| Space | BLH | BLH | B(L²+HL) | BLH |
| Parallel | Yes | No | Yes | Yes |
| Inference | LH² | H² | L²H+H²L | H² |

S4 combines the parallel training of convolutions with the O(H²) per-step inference of recurrences, at Õ(N+L) kernel cost.
