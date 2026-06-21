We are asked to shape time-dependent control amplitudes for a quantum system whose Hamiltonian splits into a fixed part and tunable rf terms, H(t) = H0 + Σ_k u_k(t) H_k, so that over a fixed duration T the system either reaches a target state or implements a target unitary propagator. The central difficulty is not the physics but the cost of optimization: a freely shaped pulse must be discretized into many time slices, giving thousands of control parameters, and obtaining the gradient of the performance index with respect to every amplitude by finite differences requires on the order of m·N+1 full time evolutions. That expense forces practitioners into tiny hand-chosen pulse families with only a few dozen parameters, which may never contain the truly optimal shape.

The existing ideas therefore fall short in the same way. Finite-difference gradients treat the propagator as a black box and scale linearly with the number of controls, composite pulses and Fourier ansätze trade expressiveness for tractability, and exact eigendecomposition of the total propagator is tied to special performance forms. What is missing is a generic, per-slice gradient formula whose cost does not grow with the number of parameters.

The method that solves this is GRAPE, Gradient Ascent Pulse Engineering. The first step is to discretize the interval into N equal slices of width Δt and hold each control amplitude constant on its slice, so the evolution over a slice is a single matrix exponential U_j = exp{ -i Δt (H0 + Σ_k u_k(j) H_k) } and the total evolution is the ordered product U_N … U_1. The objective is an overlap with a target, either a state target C or a propagator target U_F. When the propagator product is cut at slice j and the trace is regrouped using cyclic invariance, the objective becomes the overlap of a forward-propagated quantity with a backward-propagated target. The backward-propagated target is the optimal-control adjoint, or co-state, and it satisfies a terminal-value equation fixed by the cost. This regrouping is the key structural insight: the derivative with respect to u_k(j) localizes to slice j and can be computed from the forward and backward sweeps alone.

To first order in the small-Δt regime, perturbing slice j gives δU_j = -i Δt δu_k(j) H_k U_j. For state transfer the gradient is ∂Φ/∂u_k(j) = -⟨λ_j | iΔt [H_k, ρ_j]⟩, where ρ_j is the state propagated forward to slice j and λ_j is the target propagated backward from T to slice j. For gate synthesis the analogous gradient is ∂Φ/∂u_k(j) = -2 Re{ ⟨P_j | iΔt H_k X_j⟩ ⟨X_j|P_j⟩ }, where X_j is the propagator built up to slice j and P_j is the target propagated backward past slice j; the second factor quotients out the unobservable global phase so that the optimizer drives the realized gate to the target up to an irrelevant overall phase. In both cases all m·N partial derivatives are obtained from one forward sweep and one backward sweep, so the gradient costs two propagations regardless of how finely the pulse is discretized. The amplitudes are then updated by gradient ascent, optionally with an rf-power penalty or amplitude clipping, and the benign topology of quantum-control landscapes means local ascent typically reaches high-fidelity pulses.

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
