The problem is to explain why a metal below a critical temperature loses all electrical resistance, expels magnetic fields, and develops a gap in its low-energy electronic excitations. The normal-state picture of independent Bloch electrons cannot account for any of this: it has gapless single-particle states arbitrarily close to the Fermi surface, and merely reducing scattering would improve conductivity but would not produce a sharp thermodynamic transition or perfect diamagnetism. A first hint comes from Cooper's two-electron calculation, which shows that two electrons above an inert Fermi sea can form a bound state if their interaction is weakly attractive. That bound state, however, is enormous in real space, so in any real superconductor an enormous number of pairs overlap. A simple dilute-gas picture of independent molecules is therefore the wrong next step.

What is needed is a many-electron state that can exploit the attractive matrix elements coherently without violating Fermi statistics. The key obstacle is that random occupation configurations make the off-diagonal pair-scattering terms cancel. The way out is to restrict the allowed configurations: states are occupied in time-reversed pairs, so that if the single-particle state with momentum k and spin up is occupied, the state with momentum -k and spin down is also occupied. With this pairing constraint, the scattering terms connect different configurations with a common sign and the interaction energy can be lowered collectively.

The method is Bardeen-Cooper-Schrieffer (BCS) theory. It constructs a variational ground state in which every near-Fermi pair state is a superposition of empty and doubly occupied amplitudes, u_k and v_k. The trial state is the product over one representative of each time-reversed pair,

|Ψ⟩ = ∏_k (u_k + v_k c†_{k↑} c†_{-k↓}) |0⟩,

with u_k² + v_k² = 1. The reduced Hamiltonian keeps only the kinetic energy and the attractive pair-scattering channel,

H_red = ∑_{k,σ} ε_k c†_{kσ} c_{kσ} − ∑_{k,k′} V_{kk′} c†_{k↑} c†_{-k↓} c_{-k′↓} c_{k′↑}.

Minimizing the expectation value of this Hamiltonian gives the self-consistent gap equation. For a constant attraction V inside a shell |ε| < ℏω around the Fermi surface, the quasiparticle spectrum becomes E_k = √(ε_k² + Δ²), and the gap Δ is determined by

1 = N(0)V ∫_0^{ℏω} tanh(√(ε² + Δ²)/(2 k_B T)) / √(ε² + Δ²) dε.

The same variational parameter Δ that mixes the pair occupancy also serves as the order parameter. In the weak-coupling limit the transition temperature and zero-temperature gap satisfy

k_B T_c ≈ 1.14 ℏω exp(−1/(N(0)V)),
2Δ(0)/(k_B T_c) ≈ 3.50.

Because the lowest quasiparticle energy is Δ, there is a minimum energy cost to create an excitation, which explains the observed gap in single-particle and pair-breaking probes. Finally, the paired state is rigid against long-wavelength electromagnetic perturbations: the paramagnetic current no longer cancels the diamagnetic term, giving the London rigidity at short distances and the nonlocal Pippard response at finite coherence lengths. This closes the loop from a weak microscopic attraction to the macroscopic electrodynamics of superconductivity.

```python
import numpy as np

# Compatibility shim for different NumPy versions.
_trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))

def bcs_integral(delta, T, omega, n=50000):
    """Evaluate the BCS gap-equation integral for a given Delta and T."""
    eps = np.linspace(1e-12 * omega, omega, n)
    if delta <= 0:
        # Delta -> 0 limit: integral of tanh(eps / 2T) / eps.
        if T == 0.0:
            return float('inf')
        return _trapz(np.tanh(eps / (2.0 * T)) / eps, eps)
    denom = np.sqrt(eps**2 + delta**2)
    if T == 0.0:
        # At T = 0 the integral reduces to asinh(omega / Delta).
        return _trapz(1.0 / denom, eps)
    return _trapz(np.tanh(denom / (2.0 * T)) / denom, eps)

def solve_gap(T, omega, N0V, n=50000, tol=1e-9):
    """Solve the BCS gap equation for Delta(T) by bisection."""
    target = 1.0 / N0V
    delta0 = omega / np.sinh(target)  # exact zero-temperature gap
    if T == 0.0:
        return delta0
    # At T > T_c the only solution is Delta = 0.
    f_lo = bcs_integral(0.0, T, omega, n) - target
    if f_lo <= 0.0:
        return 0.0
    # I(Delta) decreases with Delta, so f(0) > 0 and f(delta0) < 0.
    lo, hi = 0.0, delta0
    f_hi = bcs_integral(hi, T, omega, n) - target
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        f_mid = bcs_integral(mid, T, omega, n) - target
        if abs(f_mid) < tol or (hi - lo) < tol * omega:
            return mid
        if f_mid * f_lo > 0.0:
            lo, f_lo = mid, f_mid
        else:
            hi, f_hi = mid, f_mid
    return 0.5 * (lo + hi)

def find_tc(omega, N0V, n=50000):
    """Find the critical temperature by setting Delta -> 0."""
    target = 1.0 / N0V
    lo, hi = 1e-6 * omega, omega
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        eps = np.linspace(1e-12 * omega, omega, n)
        I = _trapz(np.tanh(eps / (2.0 * mid)) / eps, eps)
        if I > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

if __name__ == "__main__":
    omega = 1.0          # phonon cutoff in energy units
    N0V = 0.3            # dimensionless coupling

    Tc = find_tc(omega, N0V)
    print(f"BCS T_c / (hbar omega) = {Tc:.4f}")
    print(f"Weak-coupling analytic  = {1.14 * np.exp(-1.0 / N0V):.4f}")

    for frac in [0.0, 0.3, 0.5, 0.9]:
        T = frac * Tc
        delta = solve_gap(T, omega, N0V)
        print(f"T/T_c = {frac:.1f}, Delta/(k_B T_c) = {delta / Tc:.4f}")
```
