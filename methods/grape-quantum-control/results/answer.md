# GRAPE — Gradient Ascent Pulse Engineering

## Problem

Design time-dependent control amplitudes u_k(t) for a quantum system with Hamiltonian
H(t) = H0 + Σ_k u_k(t) H_k that, over a fixed duration T, either steer an initial state
ρ0 to maximize overlap with a target operator C (coherence/polarization transfer) or
realize a target propagator U_F (quantum gate). The controls live over a continuum of
times; using any local optimizer requires the gradient of the performance index with
respect to a large number of control amplitudes, and computing that gradient by finite
differences costs on the order of m·N+1 full propagations (m controls, N time slices),
which forces tiny pulse parameterizations.

## Key idea

Discretize [0,T] into N slices of width Δt = T/N with piecewise-constant controls, so
the evolution is a product of slice propagators

  U_j = exp{ -i Δt ( H0 + Σ_k u_k(j) H_k ) },   U(T) = U_N … U_1.

Maximizing the overlap with the target requires ∂Φ/∂u_k(j) for all m·N amplitudes. Cut
the propagator product at slice j and use cyclic invariance of the trace: the objective
becomes the overlap of a **forward-propagated** state with a **backward-propagated**
target (the optimal-control adjoint / co-state). Perturbing only slice j, with the
matrix-exponential derivative δU_j = -iΔt δu_k(j) H_k U_j (valid for Δt ≪ ‖H‖⁻¹), gives
a per-slice gradient whose every ingredient is available from **one forward sweep plus
one backward sweep** — so the entire m·N gradient costs **two** propagations, not m·N+1.
Then take gradient-ascent steps on the amplitudes.

## Performance indices and their gradients

State transfer, hermitian ρ0, C:  Φ0 = ⟨C|ρ(T)⟩, with ρ_j forward-propagated, λ_j the
backward-propagated target (λ_N = C):

  Φ0 = ⟨λ_j | ρ_j⟩,   ∂Φ0/∂u_k(j) = -⟨ λ_j | iΔt [H_k, ρ_j] ⟩.

Non-hermitian state transfer:
  Φ1 = Re⟨C|ρ(T)⟩,  ∂Φ1/∂u_k(j) = -⟨λ_j^x|iΔt[H_k,ρ_j^x]⟩ - ⟨λ_j^y|iΔt[H_k,ρ_j^y]⟩
  Φ2 = |⟨C|ρ(T)⟩|²,  ∂Φ2/∂u_k(j) = -2 Re{ ⟨λ_j|iΔt[H_k,ρ_j]⟩ ⟨ρ_N|C⟩ }
(x/y are hermitian and skew-hermitian parts.)

Relaxation (Liouville space, ρ̇ = L̂ρ, L̂_j = exp{L̂Δt}): identical structure,
  Φ0 = ⟨λ_j|ρ_j⟩,  ∂Φ0/∂u_k(j) = -⟨λ_j | iΔt[H_k, ρ_j]⟩.

Unitary (gate) synthesis, U(T) = U_N…U_1, X_j = U_j…U_1, P_j = U_{j+1}†…U_N† U_F:
  phase-sensitive  Φ3 = Re⟨U_F|U(T)⟩ = Re⟨P_j|X_j⟩,  ∂Φ3/∂u_k(j) = -Re⟨P_j | iΔt H_k X_j⟩
  phase-insensitive Φ4 = |⟨U_F|U(T)⟩|² = ⟨P_j|X_j⟩⟨X_j|P_j⟩,
       ∂Φ4/∂u_k(j) = -2 Re{ ⟨P_j | iΔt H_k X_j⟩ ⟨X_j|P_j⟩ }.
Φ4 quotients out the unobservable global phase and is the natural gate fidelity (often
normalized by the Hilbert-space dimension d).

rf-power penalty Φrf = α Σ_{j,k} u_k(j)² Δt adds -2α u_k(j)Δt to the gradient; a hard
amplitude ceiling is enforced by clipping after each update. Robustness over a parameter
range ω_p: maximize Φtot = Σ_p Φ(ω_p), gradient = Σ_p of per-system gradients.

## Algorithm

1. Guess controls u_k(j).
2. Forward-propagate the state/identity to get ρ_j (or X_j) for all j.
3. Backward-propagate the target to get λ_j (or P_j) for all j.
4. Form ∂Φ/∂u_k(j) for the whole m×N grid from these two sweeps; update
   u_k(j) ← u_k(j) + ε ∂Φ/∂u_k(j) (optionally with conjugate-gradient / L-BFGS and
   adaptive step size; add noise to escape critical points).
5. Repeat until the performance index converges.

For a controllable system with adequate control resources the fidelity landscape is
benign (critical points tend to be global optima or saddles), so local ascent reaches
near-optimal pulses in practice.

## Code

```python
import numpy as np
from scipy.linalg import expm

def overlap(A, B):
    # Hilbert–Schmidt inner product tr(A† B), normalized by dimension d
    return np.trace(A.conj().T @ B) / A.shape[0]

def grape_unitary(U_target, H0, H_ops, times, R, eps,
                  phase_sensitive=False, alpha=None, u_start=None):
    """Shape piecewise-constant controls so U(T)=U_N...U_1 realizes U_target.
       H_ops: list of m control operators H_k. times: slice grid (length M)."""
    M, J = len(times), len(H_ops)
    dt = times[1] - times[0]
    d = U_target.shape[0]
    u = np.zeros((R, J, M))
    if u_start is not None:
        for k, u0 in enumerate(u_start):
            u[0, k, :] = u0

    for r in range(R - 1):
        # slice propagators U_j = expm(-i dt (H0 + Σ_k u_k(j) H_k))
        def H_idx(idx):
            return H0 + sum(u[r, k, idx] * H_ops[k] for k in range(J))
        U_list = [expm(-1j * H_idx(idx) * dt) for idx in range(M - 1)]

        # forward sweep -> X_j ; backward sweep -> map giving P_j (the adjoint trick)
        X_list, Pb_list = [], []
        X = np.eye(d); Pb = np.eye(d)
        for n in range(M - 1):
            X = U_list[n] @ X
            X_list.append(X)
            Pb_list.insert(0, Pb)
            Pb = U_list[M - 2 - n].conj().T @ Pb

        for k in range(J):
            for j in range(M - 1):
                P = Pb_list[j] @ U_target           # P_j = backward-propagated target
                Q = 1j * dt * H_ops[k] @ X_list[j]  # iΔt H_k X_j

                if phase_sensitive:                 # Φ3 gradient
                    g = -overlap(P, Q)
                else:                               # Φ4 gradient
                    g = -2 * overlap(P, Q) * overlap(X_list[j], P)

                if alpha:                           # rf-power penalty gradient
                    g += -2 * alpha * u[r, k, j] * dt

                u[r + 1, k, j] = u[r, k, j] + eps * g.real
            u[r + 1, k, -1] = u[r + 1, k, -2]

    return u[-1]
```

This is the bare steepest-ascent form. In a production setting the same analytic
gradient is handed to a quasi-Newton optimizer (L-BFGS-B) for superlinear convergence,
and the slice propagators / propagator gradients are computed via eigendecomposition or
the Fréchet derivative for efficiency, but the load-bearing content is the
forward/backward (adjoint) gradient above.
