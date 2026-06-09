# Context: shaping control pulses to steer coupled quantum spin dynamics

## Research question

Given a quantum system whose Hamiltonian splits into a fixed part and a set of
externally tunable control terms,

    H(t) = H0 + Σ_k u_k(t) H_k,

we want to find the time-dependent control amplitudes u_k(t) over a fixed duration
T that steer the system to a desired outcome. Two flavors of "outcome" matter in
practice:

- **State / coherence transfer.** Start from an initial density operator ρ(0)=ρ0 and
  reach, at time T, a state ρ(T) with maximum overlap onto a target operator C. In
  coupled-spin NMR this is exactly polarization or coherence transfer between spins.
- **Propagator (gate) synthesis.** Realize, over [0,T], a desired unitary propagator
  U_F — i.e. implement a specified logical operation on the whole system, independent
  of input state.

The dynamics are not free to choose: the state obeys the Liouville–von Neumann
equation and the propagator obeys the operator Schrödinger equation, both driven by
H(t). The control vector u(t)=(u_1,…,u_m) can in principle be any function of time.

The pain point is **dimensionality of the control search**. To use any local
optimizer one needs the gradient of the performance index with respect to *every*
control amplitude at *every* instant. If the pulse is discretized into many time
slices, that is a very large number of partial derivatives. The question is whether
this gradient can be obtained cheaply enough to optimize freely-shaped pulses with
thousands of parameters, rather than being forced into a handful of hand-tuned knobs.

## Background

**The dynamics being controlled.** For a spin system the density operator evolves by
the Liouville–von Neumann equation

    ρ̇(t) = -i [ H0 + Σ_k u_k(t) H_k , ρ(t) ],

and the propagator of the closed system evolves by

    U̇(t) = -i ( H0 + Σ_k u_k(t) H_k ) U(t),    U(0) = 1.

Overlap between operators is measured by the Hilbert–Schmidt inner product
⟨A|B⟩ = tr{A† B}. For two hermitian operators this overlap is real; for general
(non-hermitian) operators it is complex, which will matter when choosing a
performance index.

**Optimal control theory and the adjoint/co-state.** Steering a dynamical system
ẋ=f(x,u) to maximize a terminal cost is the classical problem of optimal control
(Pontryagin's maximum principle; Bryson & Ho, *Applied Optimal Control*, 1975;
Krotov, *Global Methods in Optimal Control*, 1996). The central device is a
**co-state / adjoint variable** λ that obeys a *backward* differential equation whose
terminal condition is fixed by the cost; in the classical theory it accompanies the
forward state x and enters the optimality conditions through inner products of the form
⟨λ, ∂f/∂u⟩. These principles are textbook material in applied optimal control; how they
bear on the *coupled* quantum-spin gradient — where the dynamics is a product of slice
propagators rather than a scalar cost integral — is open.

**The cost of finite-difference gradients.** Before this, gradient-based optimization
of coupled-spin pulse sequences almost exclusively used the **difference method**:
to estimate ∂Φ/∂u_k(j) one perturbs each control amplitude in turn and re-evaluates
the performance index by re-propagating the system from 0 to T. With m controls and N
time slices that is on the order of m·N+1 full time evolutions per gradient. As a
concrete scale: for N=500 slices and m=4 controls, the difference method needs 2001
full propagations of the system to produce a single gradient. This is why
coupled-spin optimizations were typically restricted to small **pulse families** —
composite pulses with a few flip/phase angles, Gaussian pulse cascades, spline
functions, or low-order Fourier expansions — with at most a few dozen free
parameters. One exception in the prior art: Levante et al. computed analytic
derivatives by expressing the performance in terms of the eigenvalues and
eigenfunctions of the *total* propagator.

**Where prior pulse-design optimal control stopped.** Optimal-control-style pulse
design had been applied in laser spectroscopy and, within NMR, to **uncoupled** spin
systems governed by the Bloch equations: band-selective pulses, and robust broadband
excitation/inversion pulses. The coupled-spin case — the physically rich one, where
J-couplings mediate transfer — had not been opened up to many-parameter gradient
optimization, precisely because of the difference-method cost above.

**Validity scale for a per-slice linearization.** If a slice is short enough that
Δt ≪ ‖H0 + Σ_k u_k H_k‖^{-1}, then across that slice a control operator barely rotates
under the evolution. This sets the discretization scale at which any first-order,
linearized treatment of a slice propagator can be trusted.

**The shape of the optimization landscape.** For a controllable finite-level quantum
system with adequate (essentially unconstrained) control resources, the fidelity as a
function of the control field is known to have a benign topology: its critical points
tend to be global optima or saddle points rather than genuine local traps (Rabitz and
co-workers, 2004). This is the empirical/analytical reason a humble *local* ascent on
the control amplitudes can be expected to reach near-optimal pulses in practice, even
without any global-search machinery.

## Baselines

- **Finite-difference (difference-method) gradient optimization.** Perturb each of the
  m·N control amplitudes, re-propagate, and form (Φ(u+δ)−Φ(u))/δ. Core idea: treat the
  propagation as a black box and probe it numerically. Gap: cost grows linearly with
  the number of control parameters (≈ m·N+1 full propagations per gradient), so the
  parameter space must be kept tiny.

- **Restricted pulse families.** Composite pulses (a few flip and phase angles),
  Gaussian pulse cascades, spline-parameterized shapes, low-order Fourier expansions.
  Core idea: hand-pick a low-dimensional ansatz so that the (often finite-difference)
  optimization stays tractable. Gap: expressive power is capped by the ansatz; the
  truly time-optimal or relaxation-optimal shape may simply not lie in the family.

- **Analytic derivatives via total-propagator eigendecomposition (Levante et al.).**
  Express the performance index through eigenvalues/eigenfunctions of the full
  propagator and differentiate that. Core idea: avoid finite differences by an exact
  derivative. Gap: tied to the eigendecomposition of the *total* propagator and to the
  particular performance form; less direct than a per-slice forward/backward scheme,
  and not framed as a generic time-sliced control-amplitude gradient.

- **Bloch-equation (uncoupled-spin) optimal control.** Optimal-control pulse design
  for single, uncoupled spins: band-selective, broadband excitation/inversion pulses.
  Core idea: shape pulses for one spin's Bloch dynamics. Gap: does not address coupled
  spins, where the interesting transfer (mediated by J-couplings) and gate synthesis
  live.

## Evaluation settings

The natural testbeds are small coupled-spin systems where an analytical optimum is
independently known, so a numerical pulse-design method can be checked against a
benchmark rather than judged in a vacuum:

- **Heteronuclear two-spin system**, free-evolution Hamiltonian H0 = 2π J I_z S_z with
  heteronuclear coupling constant J, controls being the x/y rf fields on each spin
  (H_1=2πI_x, H_2=2πI_y, H_3=2πS_x, H_4=2πS_y, so m=4). Coherence-order-selective
  in-phase transfer I⁻→S⁻ with ρ0=I⁻=I_x−iI_y, target C=S⁻=S_x−iS_y; figure of merit
  the normalized transfer amplitude η(T)=|⟨S⁻|ρ(T)⟩| / (‖I⁻‖‖S⁻‖). Time discretized in
  steps of order Δt∼10⁻³ s over total durations T up to ∼1.5 s (thousands of control
  parameters). The independent yardstick is the analytic time-optimal transfer curve
  from geometric control.
- **Relaxation-optimized transfer** in an isolated two-spin pair (e.g. in-phase to
  anti-phase, I_x → 2I_zS_x) under dipolar relaxation, evolved in Liouville space with
  a relaxation superoperator; benchmarked against analytically derived
  relaxation-optimized pulse elements and against standard INEPT-type transfer.
- **Unitary (gate) synthesis** on a network of coupled spins: realize a specified
  target propagator U_F in fixed time T; figure of merit a phase-insensitive overlap
  |⟨U_F|U(T)⟩| / d (normalized by the Hilbert-space dimension d).

Protocol: discretize T into N equal slices; initialize controls (e.g. random values on
a coarse grid, smoothed by a cubic spline, to give smooth but unbiased starting
pulses); iterate the optimizer to convergence of the performance index; optionally
sample a range of chemical shifts / rf amplitudes to build in robustness.

## Code framework

The primitives that already exist: a way to build operators on the spin Hilbert space,
a matrix exponential for the slice propagators, the Hilbert–Schmidt inner product, and
an ordinary gradient-ascent / quasi-Newton loop. What is missing is the body that turns
a discretized control array into a performance value and, crucially, into a cheap
gradient. Empty stubs for those slots:

```python
import numpy as np
from scipy.linalg import expm

def overlap(A, B):
    # Hilbert–Schmidt inner product tr(A† B), optionally normalized by dimension
    return np.trace(A.conj().T @ B) / A.shape[0]

def slice_propagators(H0, H_ctrl, u, dt):
    """u: shape (m, N) control amplitudes; H_ctrl: list of m control operators.
    Return the N per-slice propagators U_j = expm(-i dt (H0 + Σ_k u[k,j] H_ctrl[k]))."""
    pass  # TODO: one matrix exponential per slice

def performance(U_list, target):
    """Combine the slice propagators into U(T) and score it against `target`."""
    pass  # TODO: form U(T) = U_N ... U_1 and an overlap-based fidelity

def gradient(U_list, H_ctrl, target, dt):
    """Partial derivative of `performance` w.r.t. every control amplitude u[k,j].
    The whole point is to get all m*N derivatives without re-propagating per control."""
    # TODO: compute the m*N partial derivatives
    pass

def optimize(H0, H_ctrl, target, T, N, m, iters, eps):
    u = np.zeros((m, N))            # initial controls
    dt = T / N
    for _ in range(iters):
        U_list = slice_propagators(H0, H_ctrl, u, dt)
        g = gradient(U_list, H_ctrl, target, dt)
        u = u + eps * g            # ascent on the performance index
    return u
```
