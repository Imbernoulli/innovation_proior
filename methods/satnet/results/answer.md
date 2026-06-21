# SATNet: a differentiable (smoothed) MAXSAT layer

## Problem

Deep networks cannot learn hard, global, discrete logical constraints from data: the natural objective (count satisfied clauses) is discrete and non-differentiable, so the structure of a logical/constraint problem cannot be learned by gradient descent. SATNet makes the *logical structure itself* a learnable parameter by embedding an approximate MAXSAT solver as a differentiable layer, so a network can learn the rules of a problem end to end — even behind a perception front-end — from weak supervision.

## Key idea

Represent the unknown logic as a (learnable) MAXSAT instance and replace the discrete problem by its **low-rank semidefinite relaxation**, which is continuous and differentiable:

- Lift each boolean variable `ṽ_i ∈ {−1,1}` to a unit vector `v_i ∈ R^k`, with a special unit "truth direction" `v_⊤`; a variable is "true" by its angle to `v_⊤`.
- The MAXSAT (= MIN-UNSAT) objective relaxes to a unit-diagonal SDP `min ⟨S^T S, V^T V⟩, ‖v_i‖=1`, where the scaled clause matrix `S` (rows = variables incl. truth, columns = clauses) **is the layer's learnable weight**.
- Solve the SDP by **Mixing-method coordinate descent**: `v_i = −g_i/‖g_i‖`, `g_i = V S^T s_i − ‖s_i‖² v_i`, which converges to the global SDP optimum for `k > √(2n)`.
- Move between probabilities and vectors with the Goemans–Williamson randomized-rounding law `Pr[same side] = arccos(−v_i^T v_⊤)/π`, so probabilistic inputs/outputs are exact and differentiable.
- **Backpropagate by differentiating the fixed point** of the coordinate-descent update (implicit differentiation), not by unrolling the solver. The resulting linear system has the same structure as the forward solve, so the *same* coordinate descent computes the gradients in `O(nmk)` with no stored Jacobians.

Two design levers make it generalize: keep the number of clauses `m` (the rank) small — low rank is the regularizer that forces a compact rule set instead of memorization — and add **auxiliary variables** (CNF "register memory") to raise representational power without raising `m`.

## The method, precisely

**Setup.** `n` variables (indices `1..n`) plus truth variable `⊤`; `m` clauses. `I` = indices with known assignment, `O = {1..n}\I` unknown. Inputs `z_i ∈ [0,1]`, `i∈I`; outputs `z_o ∈ [0,1]`, `o∈O`. Weights = scaled clause matrix `S = [s̃_⊤ s̃_1 … s̃_n]·diag(1/√(4|s_j|))`. Rank `k = √(2n)+1`.

**SDP relaxation (forward objective).**
```
minimize_{V ∈ R^{k×(n+1)}}  ⟨S^T S, V^T V⟩    subject to  ‖v_i‖ = 1,  i = ⊤,1,…,n.
```

**Forward pass.**
1. *Relax inputs.* For each `i∈I`, build `v_i` with `v_i^T v_⊤ = −cos(π z_i)`:
   `v_i = −cos(π z_i) v_⊤ + sin(π z_i)(I_k − v_⊤ v_⊤^T) v_i^rand`.
2. *Solve* over output columns only by coordinate descent:
   `g_o = V S^T s_o − ‖s_o‖² v_o`,  `v_o = −g_o/‖g_o‖`,
   maintaining `Ω = V S^T` with rank-one updates (`O(nmk)` per sweep).
3. *Read out* `z_o = arccos(−v_o^T v_⊤)/π`. (Test time may instead round: assign true iff `sgn(r^T v_o)=sgn(r^T v_⊤)` for random `r`, repeat, keep the max-objective assignment.)

**Backward pass (Theorem).** Given `∂ℓ/∂z_o`:
- Readout: `∂ℓ/∂v_o = (∂ℓ/∂z_o) v_⊤ / (π sin(π z_o))`.
- With `P_o = I_k − v_o v_o^T`, `P = diag(P_o)`, `C = S_O^T S_O − diag(‖s_o‖²)`, `D = diag(‖g_o‖)`, define `U` by `U_I=0` and
  `vec(U_O) = ( P((D+C)⊗I_k)P )^† vec(∂ℓ/∂V_O)`,
  solved by the **same** coordinate descent: `dg_o = Ψ s_o − ‖s_o‖² u_o − ∂ℓ/∂v_o`, `u_o = −P_o dg_o/(‖g_o‖+λ)`, maintaining `Ψ = U_O S_O^T`.
- Gradients:
  `∂ℓ/∂V_I = −(Σ_{o∈O} u_o s_o^T) S_I`,
  `∂ℓ/∂S  = −(Σ_{o∈O} u_o s_o^T)^T V − (S V^T) U`.
- Input chain: `∂ℓ/∂z_i = ∂ℓ/∂z_i^direct + (∂v_i/∂z_i)^T(∂ℓ/∂v_i)`, with `∂v_i/∂z_i = π( sin(π z_i)v_⊤ + cos(π z_i)(I_k − v_⊤ v_⊤^T)v_i^rand )`.

`λ` (a small diagonal increment, `prox_lam`) keeps the backward system well-conditioned when some `‖g_o‖→0`.

## Code (PyTorch, faithful to the canonical implementation)

The released solver implements `init / forward / backward` in C++/CUDA behind a `torch.autograd.Function`; the inner coordinate-descent kernel is shared between the forward (maintaining `W = V^T S`) and backward (maintaining `Φ = U^T S`) passes. The Python interface is:

```python
import torch
import torch.nn as nn
from torch.autograd import Function

import satnet._cpp
if torch.cuda.is_available():
    import satnet._cuda


def get_k(n):
    return int((2 * n) ** 0.5 + 3) // 4 * 4          # rank >= sqrt(2n)+1, mult. of 4


class MixingFunc(Function):
    """Apply the Mixing method (low-rank SDP coordinate descent) to input probs.

    The SATNet module is a wrapper handling init and the auxiliary/truth columns.
    """
    @staticmethod
    def forward(ctx, S, z, is_input, max_iter, eps, prox_lam):
        B, n, m, k = z.size(0), S.size(0), S.size(1), 32
        ctx.prox_lam = prox_lam
        dev = 'cuda' if S.is_cuda else 'cpu'

        ctx.g, ctx.gnrm = torch.zeros(B, k, device=dev), torch.zeros(B, n, device=dev)
        ctx.index = torch.zeros(B, n, dtype=torch.int, device=dev)
        ctx.is_input = torch.zeros(B, n, dtype=torch.int, device=dev)
        ctx.V = torch.zeros(B, n, k, device=dev).normal_()       # random unit init
        ctx.W = torch.zeros(B, k, m, device=dev)                 # W = V^T S
        ctx.z = torch.zeros(B, n, device=dev)
        ctx.niter = torch.zeros(B, dtype=torch.int, device=dev)
        ctx.S = torch.zeros(n, m, device=dev)
        ctx.Snrms = torch.zeros(n, device=dev)

        ctx.z[:] = z.data
        ctx.S[:] = S.data
        ctx.is_input[:] = is_input.data

        perm = torch.randperm(n - 1, dtype=torch.int, device=dev)
        impl = satnet._cuda if S.is_cuda else satnet._cpp
        # init: relax inputs into unit vectors (truth angle), set update order
        impl.init(perm, is_input, ctx.index, ctx.z, ctx.V)
        for b in range(B):
            ctx.W[b] = ctx.V[b].t().mm(ctx.S)
        ctx.Snrms[:] = S.norm(dim=1) ** 2

        # forward coordinate descent: v_o = -g_o/||g_o||, rank-1 update of W
        impl.forward(max_iter, eps,
                     ctx.index, ctx.niter, ctx.S, ctx.z,
                     ctx.V, ctx.W, ctx.gnrm, ctx.Snrms, ctx.g)
        return ctx.z.clone()

    @staticmethod
    def backward(ctx, dz):
        B, n, m, k = dz.size(0), ctx.S.size(0), ctx.S.size(1), 32
        dev = 'cuda' if ctx.S.is_cuda else 'cpu'
        ctx.dS = torch.zeros(B, n, m, device=dev)
        ctx.U = torch.zeros(B, n, k, device=dev)                 # backward vectors
        ctx.Phi = torch.zeros(B, k, m, device=dev)              # Phi = U^T S
        ctx.dz = torch.zeros(B, n, device=dev)
        ctx.dz[:] = dz.data

        impl = satnet._cuda if ctx.S.is_cuda else satnet._cpp
        # backward: solve the fixed-point linear system by the SAME coord. descent,
        # then assemble dS = U W + V Phi and the input-probability gradients.
        impl.backward(ctx.prox_lam,
                      ctx.is_input, ctx.index, ctx.niter, ctx.S, ctx.dS, ctx.z, ctx.dz,
                      ctx.V, ctx.U, ctx.W, ctx.Phi, ctx.gnrm, ctx.Snrms, ctx.g)
        ctx.dS = ctx.dS.sum(dim=0)
        return ctx.dS, ctx.dz, None, None, None, None


def insert_constants(x, pre, n_pre, app, n_app):
    one = x.new(x.size()[0], 1).fill_(1)
    seq = []
    if n_pre != 0:
        seq.append((pre * one).expand(-1, n_pre))
    seq.append(x)
    if n_app != 0:
        seq.append((app * one).expand(-1, n_app))
    r = torch.cat(seq, dim=1)
    r.requires_grad = False
    return r


class SATNet(nn.Module):
    """A differentiable MAXSAT layer; the clause matrix S is the learned logic.

    Args:
        n: number of input variables.
        m: rank of the clause matrix (number of clauses) -- low for generalization.
        aux: number of auxiliary ("register memory") variables.
        max_iter: max coordinate-descent iterations (default 40).
        eps: relative stopping threshold (default 1e-4).
        prox_lam: diagonal increment stabilizing the backward solve (default 1e-2).
    """
    def __init__(self, n, m, aux=0, max_iter=40, eps=1e-4, prox_lam=1e-2,
                 weight_normalize=True):
        super().__init__()
        S = torch.FloatTensor(n + 1 + aux, m).normal_()         # +1 = truth vector
        if weight_normalize:
            S = S * ((0.5 / (n + 1 + aux + m)) ** 0.5)          # small init
        self.S = nn.Parameter(S)
        self.aux = aux
        self.max_iter, self.eps, self.prox_lam = max_iter, eps, prox_lam

    def forward(self, z, is_input):
        B = z.size(0)
        dev = 'cuda' if self.S.is_cuda else 'cpu'
        m = self.S.shape[1]
        if dev == 'cpu' and m % 4 != 0:
            raise ValueError('m must be a multiple of 4 on CPU (SSE). Got ' + str(m))
        # prepend always-true truth variable; append unconstrained aux variables
        is_input = insert_constants(is_input.data, 1, 1, 0, self.aux)
        z = torch.cat([torch.ones(B, 1, device=dev), z,
                       torch.zeros(B, self.aux, device=dev)], dim=1)
        z = MixingFunc.apply(self.S, z, is_input,
                             self.max_iter, self.eps, self.prox_lam)
        return z[:, 1:self.S.size(0) - self.aux]                # strip truth + aux
```

Usage (e.g. learning 9×9 Sudoku from examples): one SATNet layer with `n = 9³ = 729`, `m = 600`, `aux = 300`, trained with Adam (lr `2e-3`) under digit-wise NLL / binary-cross-entropy. For visual Sudoku, feed cell-wise softmax outputs of a LeNet digit classifier as the probabilistic input `z` and train the whole stack end to end (separate learning rates for the perception and logic parts). For parity of a length-`L` string, chain `L−1` tied SATNet layers (each computing one XOR) with single-bit supervision at the end.

```python
class SudokuSolver(nn.Module):
    def __init__(self, boardSz, aux, m):
        super().__init__()
        self.sat = SATNet(boardSz ** 6, m, aux)     # boardSz=3 -> n = 729

    def forward(self, y_in, mask):
        return self.sat(y_in, mask)
```
