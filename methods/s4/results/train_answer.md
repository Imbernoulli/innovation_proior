We want a single sequence layer that can reach across very long contexts — tens of thousands of steps — while staying cheap to train and to run. The trouble is that every architecture I know trades away one of the three things I need. A recurrent model carries unbounded memory in its state, but learning it means backpropagating through $L$ applications of the same map, so the gradient is a product of $L$ Jacobians that generically decays or explodes geometrically; signal from step $0$ is washed out long before $L$ reaches the thousands, and the recurrence is sequential, so training is slow. Gating and orthogonal or unitary constraints push the eigenvalues toward magnitude one to slow the decay, but they are fighting an exponential and remain sequential. A convolution fixes the sequentiality — a fixed filter, applied everywhere, with stable gradients — but a filter has finite width, and stacking or dilating only grows the receptive field with depth, so no single layer is genuinely global. Attention is global, every position talking to every other, but that is $L^2$ pairs, hopeless at $L = 16{,}384$; the efficient-attention approximations buy back the $L^2$ at a cost in quality and still offer no cheap stepwise inference. So I want, simultaneously, the recurrent dream of principled unbounded memory, the convolutional dream of parallel training, and the recurrent dream again of constant-time-per-step inference — and these pull in opposite directions for every model on the table.

There is one object that is at once a recurrence and a convolution: a linear, time-invariant state space model. I propose to build the sequence layer from it, and the method is S4 — a Structured State Space sequence model that makes the linear SSM both well-conditioned in its memory and computable in near-linear time. Start from the continuous system

$$x'(t) = A\,x(t) + B\,u(t), \qquad y(t) = C\,x(t) + D\,u(t),$$

with a 1-D input $u(t)$, an $N$-dimensional latent state $x(t)$, and a 1-D output $y(t)$; the feedthrough $D\,u$ is just a skip connection, so I set it aside. The linearity looks like a liability for expressivity, but I will stack many of these layers with nonlinearities between them — like a CNN stacks linear convolutions — and let depth supply the nonlinearity. The point of the linearity is that it makes the long-range behavior exactly analyzable and the recurrent and convolutional faces of the layer exact rather than approximate.

The first danger is that a linear ODE has precisely the exponential problem I was fleeing: $x'(t) = A\,x(t)$ solves to $x(t) = e^{At}x(0)$, and the spectrum of $A$ decides whether that grows or decays. A random $A$ reproduces vanishing/exploding behavior and performs terribly. The escape is to not pick $A$ at random. There is a continuous-time memorization theory — HiPPO — that solves for the $A$ (and $B$) making the state $x(t)$ hold the coefficients of the optimal polynomial approximation of the entire input history, an honest bounded summary that slides forward in time. The canonical scaled-Legendre matrix is $A_{nk} = -(2n+1)^{1/2}(2k+1)^{1/2}$ for $n>k$, $-(n+1)$ on the diagonal, and $0$ above; swapping a random $A$ for this one takes a state space layer on sequential MNIST from about $60\%$ to about $98\%$. So $A$ is initialized to HiPPO and the memory problem is, in principle, solved; what remains is to make the thing run.

The data is sampled, $u_k = u(k\Delta)$, so the continuous system must be discretized. The crude forward-Euler choice gives $\bar A = I + \Delta A$, but nothing keeps its eigenvalues inside the unit disk, so the discrete recurrence can blow up even though the continuous dynamics are stable. I want the discretization to map the stable continuous region (eigenvalues in the left half-plane) into the stable discrete region (eigenvalues in the unit disk). The bilinear/trapezoidal rule does exactly that: integrating $x'$ over one step with the trapezoid gives $(I - \tfrac{\Delta}{2}A)\,x_k = (I + \tfrac{\Delta}{2}A)\,x_{k-1} + \Delta B\,u_k$, hence

$$x_k = \bar A\,x_{k-1} + \bar B\,u_k,\quad y_k = \bar C\,x_k,\qquad \bar A = (I - \tfrac{\Delta}{2}A)^{-1}(I + \tfrac{\Delta}{2}A),\ \ \bar B = (I - \tfrac{\Delta}{2}A)^{-1}\Delta B,\ \ \bar C = C.$$

The map $A \mapsto (I - \tfrac{\Delta}{2}A)^{-1}(I + \tfrac{\Delta}{2}A)$ is the Cayley transform, which sends the left half-plane into the open unit disk; that is why bilinear and not Euler. Worth noting for later: as $\Delta \to 0$, $\bar A \to I$.

This discrete recurrence is RNN-shaped, perfect for stepwise inference but sequential. For parallel training I unroll it from $x_{-1}=0$, which gives $y_k = \sum_{j=0}^{k} \bar C\,\bar A^{\,k-j}\bar B\,u_j$ — a convolution $y = \bar K * u$ with a single global filter

$$\bar K = \big(\bar C\bar B,\ \bar C\bar A\bar B,\ \bar C\bar A^2\bar B,\ \dots,\ \bar C\bar A^{\,L-1}\bar B\big) \in \mathbb{R}^L.$$

With $\bar K$ in hand the convolution is one FFT-based product in $O(L\log L)$, fully parallel. So the whole problem collapses to producing the vector $\bar K$ — and that is the wall. Building it naively powers $\bar A$ up to $L-1$ times, $O(N^2 L)$ time and $O(NL)$ memory, orders of magnitude past the $\tilde O(N+L)$ one should hope for. The entire game is computing $\bar K$ without ever forming those powers.

The SSM is invariant under conjugation: replacing $(A,B,C)$ by $(V^{-1}AV,\,V^{-1}B,\,CV)$ leaves the input-output map unchanged, and conjugation commutes with the discretization, so I am free to pick the most convenient basis for $A$. The most convenient would be the one that diagonalizes it, $A = V\Lambda V^{-1}$, since then $\bar A^k$ is elementwise in $\Lambda$ and each kernel entry becomes a Vandermonde form in the eigenvalues, a near-linear computation. But diagonalizing HiPPO directly is hopeless: its eigenvector matrix has a $(3i,i)$ entry $\binom{4i}{2i} \approx 2^{4i}$, so $V$ carries entries up to $2^{4N/3}$ and the change of basis is pure floating-point noise. (The alternative "fast" route through the characteristic polynomial has the same disease — for $\bar A$ near the identity, which is the typical small-$\Delta$ regime, $(1-x)^N$ has coefficients up to $\binom{N}{N/2}\approx 2^N$ and its inverse mod $x^L$ is larger still.) Diagonalization is the right idea; diagonalizing *this* matrix by *this* $V$ is the problem. The real constraint, then, is to conjugate only by well-conditioned $V$, and the perfectly conditioned case is unitary $V$. By the spectral theorem $A$ is unitarily diagonalizable exactly when it is normal, and HiPPO is not — but it is close. Adding $\tfrac{1}{2}(2n+1)^{1/2}(2k+1)^{1/2}$ to every entry turns the off-diagonal part antisymmetric and the diagonal into $-(n+1)+\tfrac{1}{2}(2n+1) = -\tfrac12$, so the result is $-\tfrac12 I$ plus a skew-symmetric matrix — which is normal — and the rank-one piece I added is exactly $PP^*$ with $P_n = \sqrt{n+\tfrac12}$. Thus HiPPO is **normal plus low-rank**, $A = (\text{normal}) - PQ^*$, rank $1$ here (rank $2$ for some variants). Writing the normal part as $V\Lambda V^*$ and conjugating by the unitary $V$ gives, over $\mathbb{C}$,

$$A = \Lambda - PQ^*,\qquad \Lambda\ \text{diagonal},\ P,Q \in \mathbb{C}^{N\times r},\ r=1,$$

the **diagonal-plus-low-rank** (DPLR) form, with the conditioning problem gone because the only basis change was unitary.

DPLR is not diagonal, though, and the kernel still wants powers of $\bar A$, which $\Lambda - PQ^*$ does not give in closed form. The lever is to stop computing $\bar K$ in the time domain and instead compute its truncated generating function, the polynomial whose coefficients are the kernel entries, evaluated at points $z$:

$$\hat K(z) = \sum_{i=0}^{L-1} \bar C\,\bar A^{\,i}\,\bar B\,z^i = \bar C\,(I - \bar A^{L} z^{L})(I - \bar A z)^{-1}\bar B.$$

The matrix *power* has become a matrix *inverse*, and inverses of DPLR matrices are tractable while powers are not. I am free to choose the evaluation nodes, so I take the $L$-th roots of unity $z = \omega_k = \exp(-2\pi i\,k/L)$: then $z^L = 1$, the factor $(I - \bar A^L)$ becomes the same constant at every node and folds into $\bar C$ as $\tilde C^* = \bar C^*(I - \bar A^L)$ (learned directly in practice), and evaluating the generating function at the roots of unity is exactly a DFT of the kernel, so a single inverse FFT recovers $\bar K$ in $O(L\log L)$. The core object is now the resolvent $\tilde C^*(I - \bar A\omega)^{-1}\bar B$ of the discretized matrix. To use $A = \Lambda - PQ^*$ I push it back to the original $A$: substituting the bilinear $\bar A$ and $\bar B$, the two $(I - \tfrac\Delta2 A)$ factors cancel cleanly, and factoring out $(1-z)$ and clearing the resulting scalar gives

$$\hat K(z) = \frac{2}{1+z}\,\tilde C^*\,\big(g(z)I - A\big)^{-1} B, \qquad g(z) = \frac{2}{\Delta}\,\frac{1-z}{1+z}.$$

The $L$ powers of $\bar A$ have become one resolvent of the original $A$ per node. Splitting off the low-rank term, $g I - A = (gI - \Lambda) + PQ^*$ with $gI - \Lambda$ diagonal, the **Woodbury identity** handles the diagonal-plus-low-rank inverse: writing $R(z) = (g(z)I - \Lambda)^{-1}$ (an elementwise reciprocal),

$$\tilde C^*(gI - A)^{-1}B = \tilde C^* R B - \tilde C^* R P\,(I + Q^* R P)^{-1}\,Q^* R B,$$

and for rank $1$ the middle factor is a scalar, so

$$\hat K(z) = \frac{2}{1+z}\Big[\, \tilde C^* R(z) B \;-\; \frac{(\tilde C^* R(z) P)(Q^* R(z) B)}{1 + Q^* R(z) P}\,\Big].$$

Every term is a bilinear form $u^* R(z) v = \sum_n u_n^* v_n / (g(z) - \lambda_n)$, and across all the nodes the matrix with entries $1/(g(\omega_i) - \lambda_n)$ is a **Cauchy matrix**. Each bilinear form is therefore one Cauchy matrix-vector product against the weights $u_n^* v_n$, a classical and numerically stable problem solvable in $\tilde O(N+L)$ by Fast-Multipole-style algorithms. Four such multiplies — for $(\tilde C,B),(\tilde C,P),(Q,B),(Q,P)$ — scale by $2/(1+z)$, then inverse-FFT, and the $O(N^2 L)$ kernel is $\tilde O(N+L)$, with no ill-conditioned matrix or exponentially large coefficient anywhere on the path.

The recurrence survives too. To step cheaply I split $\bar A$: the forward factor $I + \tfrac\Delta2 A = \tfrac\Delta2[(2/\Delta)I + \Lambda - PQ^*]$ is diagonal-plus-low-rank, an $O(N)$ apply, and the backward factor $(I - \tfrac\Delta2 A)^{-1}$ is again DPLR by Woodbury, so one recurrence step is $O(N)$. Both faces — convolution for training, recurrence for inference — come off the same parameters. The trainable objects of one 1-D SSM are the length-$N$ vectors $\Lambda, P, Q, B, \tilde C$ (the public half-state kernel stores half the conjugate pairs and ties $Q = P^*$ in its stabilized path) plus a learnable step size $\Delta$, a per-feature timescale initialized log-uniform in $[10^{-3},10^{-1}]$ so features specialize to different memory horizons and the input sampling rate can be changed at test time without retraining. $A$ is initialized to HiPPO and immediately conjugated to DPLR — a random $A$ would bring back the exponential decay, so this is load-bearing, not cosmetic — and because the eigenvalues come in complex-conjugate pairs, storing half and taking twice the real part halves the cost. A full model runs $H$ independent copies of this 1-D SSM, producing $H$ global filters implicitly, then mixes channels with a position-wise linear layer (the SSM never mixes channels itself) — a depthwise-separable convolution whose per-channel filters are global and generated rather than stored. Stacking these with norm, residual, and a pointwise nonlinearity supplies the nonlinearity the linear core lacks; the DPLR/SSM parameters train with a smaller learning rate and no weight decay, since decaying a continuous dynamical system toward zero would corrupt the HiPPO structure, while the mixing layers train normally.

```python
import torch
import torch.nn as nn
import numpy as np

def cauchy_naive(v, z, w, conj=True):
    """Sum_n v_n / (z_i - w_n), with S4's conjugate-pair expansion."""
    if conj:
        v = torch.cat([v, v.conj()], dim=-1)
        w = torch.cat([w, w.conj()], dim=-1)
    return (v.unsqueeze(-1) / (z.unsqueeze(-2) - w.unsqueeze(-1))).sum(dim=-2)

def hippo_legs_nplr(N):
    """HiPPO-LegS -> half-state DPLR parameters, following state-spaces/s4."""
    q = np.arange(N, dtype=np.float64)
    col, row = np.meshgrid(q, q)
    r = 2 * q + 1
    M = -(np.where(row >= col, r, 0) - np.diag(q))
    T = np.sqrt(np.diag(2 * q + 1))
    A = torch.as_tensor(T @ M @ np.linalg.inv(T), dtype=torch.float64)
    B = torch.as_tensor(np.diag(T).copy(), dtype=torch.float64)

    P = torch.sqrt(0.5 + torch.arange(N, dtype=torch.float64))
    AP = A + P[:, None] * P[None, :]                 # -1/2 I + skew
    w_re = torch.diagonal(AP).mean()
    skew = AP - w_re * torch.eye(N, dtype=torch.float64)
    w_im, V = torch.linalg.eigh((-1j * skew).to(torch.cdouble))
    Lambda = w_re.to(torch.cdouble) + 1j * w_im

    idx = torch.argsort(Lambda.imag)
    Lambda, V = Lambda[idx][: N // 2], V[:, idx][:, : N // 2]
    B = V.conj().T @ B.to(torch.cdouble)
    P = V.conj().T @ P.to(torch.cdouble)
    return Lambda.to(torch.cfloat), P.to(torch.cfloat), B.to(torch.cfloat)

class S4Kernel(nn.Module):
    """Global S4 kernel for H independent half-state DPLR SSMs."""
    def __init__(self, H, N=64, dt_min=1e-3, dt_max=1e-1):
        super().__init__()
        Lambda, P, B = hippo_legs_nplr(N)
        Lambda = Lambda.unsqueeze(0).expand(H, -1).contiguous()
        P = P.unsqueeze(0).expand(H, -1).contiguous()
        B = B.unsqueeze(0).expand(H, -1).contiguous()
        C_tilde_star = torch.randn(H, N // 2, dtype=torch.cfloat)
        log_dt = torch.rand(H) * (np.log(dt_max) - np.log(dt_min)) + np.log(dt_min)

        self.Lambda = nn.Parameter(torch.view_as_real(Lambda))
        self.P = nn.Parameter(torch.view_as_real(P))
        self.B = nn.Parameter(torch.view_as_real(B))
        self.C = nn.Parameter(torch.view_as_real(C_tilde_star))
        self.log_dt = nn.Parameter(log_dt)

    def forward(self, L):
        Lambda = torch.view_as_complex(self.Lambda)
        P = torch.view_as_complex(self.P)
        B = torch.view_as_complex(self.B)
        C = torch.view_as_complex(self.C)       # stored as C_tilde^*
        Q = P.conj()                            # stabilized public-code convention
        dt = torch.exp(self.log_dt)[:, None]

        omega = torch.exp(
            -2j * torch.pi * torch.arange(L // 2 + 1, device=Lambda.device) / L
        )
        z = 2 * (1 - omega) / (1 + omega)
        A = dt * Lambda                         # equivalent to using g(z)

        r00 = cauchy_naive(dt * C * B, z, A)
        r01 = cauchy_naive(dt * C * P, z, A)
        r10 = cauchy_naive(dt * Q * B, z, A)
        r11 = cauchy_naive(dt * Q * P, z, A)
        K_hat = (2 / (1 + omega)) * (r00 - r01 * r10 / (1 + r11))
        return torch.fft.irfft(K_hat, n=L)

class S4Layer(nn.Module):
    def __init__(self, H, N=64):
        super().__init__()
        self.kernel = S4Kernel(H, N)
        self.D = nn.Parameter(torch.randn(H))
        self.activation = nn.GELU()
        self.out = nn.Conv1d(H, H, 1)                           # position-wise channel mixing

    def forward(self, u):                                       # u: (B, H, L)
        L = u.size(-1)
        K = self.kernel(L)                        # (H, L)
        K_f = torch.fft.rfft(K, n=2 * L)
        u_f = torch.fft.rfft(u, n=2 * L)
        y = torch.fft.irfft(u_f * K_f, n=2 * L)[..., :L]
        y = y + u * self.D[:, None]
        return self.out(self.activation(y))

    def step(self, u_k, state):
        # Reference recurrent mode uses A_bar = A1 @ A0 and B_bar = 2 A1 B,
        # where both A0 and A1 are DPLR/Woodbury matrix-vector applies.
        ...

class S4Model(nn.Module):
    """Encoder -> stack of (S4Layer + LayerNorm + residual + dropout) -> mean pool -> decoder."""
    def __init__(self, d_input, d_output, d_model=128, n_layers=4, dropout=0.1):
        super().__init__()
        self.encoder = nn.Linear(d_input, d_model)
        self.layers = nn.ModuleList([S4Layer(d_model) for _ in range(n_layers)])
        self.norms  = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.drops  = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])
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
