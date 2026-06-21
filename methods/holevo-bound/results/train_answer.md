A classical channel has a clean limit: per use it transmits at most the maximized mutual information, and a symbol with m distinguishable levels carries log m bits. When the carrier shrinks to a quantum system, the picture blurs. A single pure qubit lives on the Bloch sphere, so there are uncountably many preparation directions and it looks as if one could encode a real number, hence infinitely many classical bits, in one shot. That cannot be right, and the resolution lies in measurement. Two non-orthogonal quantum states cannot be told apart with certainty, so the apparent continuum in preparation is not a continuum in readout. The honest operational question is: for an ensemble E = {p_x, ρ_x}, what is the largest classical mutual information I(X;Y) that any measurement can extract? The accessible information Acc(E) is the maximum of I(X;Y) over all possible POVMs. We need a finite, computable ceiling on that maximum, and in particular we want to know whether an n-qubit carrier caps out at exactly n classical bits.

A first guess is S(ρ̄), the von Neumann entropy of the average state ρ̄ = Σ_x p_x ρ_x. That works for pure signal states, because then all the entropy comes from the spread over messages. But it over-counts badly when the signal states themselves are mixed. If every ρ_x is the maximally mixed state I/d, the carrier reveals nothing about x, so Acc(E) = 0, yet S(ρ̄) = log d. The internal noise inside each signal state must be subtracted. The corrected quantity is χ(E) = S(ρ̄) − Σ_x p_x S(ρ_x). For identical mixed states the subtracted term cancels the first term and χ = 0; for pure states the subtracted term vanishes and χ = S(ρ̄). This is the Holevo quantity, and the Holevo bound states that Acc(E) ≤ χ(E). Because χ(E) ≤ S(ρ̄) ≤ log d, an n-qubit carrier conveys at most n classical bits. Orthogonal signal states achieve equality: projecting onto their supports recovers x perfectly, and the average state is block diagonal, so χ = H(X) = Acc.

The conceptual heart of the bound is that χ equals the quantum mutual information I(X;A) of the classical-quantum state ρ_XA = Σ_x p_x |x⟩⟨x| ⊗ ρ_x. The measurement that produces Y is just a processing of the carrier A, so the correlation between X and Y cannot exceed the correlation between X and A. Formally, I(X;A) reduces to the average relative entropy Σ_x p_x D(ρ_x ‖ ρ̄), while for a fixed POVM {E_y} the induced classical mutual information is I(X;Y) = Σ_x p_x D(q_x ‖ q̄) with q_x(y) = Tr(E_y ρ_x) and q̄(y) = Tr(E_y ρ̄). The proof then needs only the measurement contraction D({Tr(E_y ρ)} ‖ {Tr(E_y σ)}) ≤ D(ρ ‖ σ), which follows from the variational formula for quantum relative entropy and Naimark dilation. Applying it to each (ρ_x, ρ̄) pair and averaging gives I(X;Y) ≤ χ(E). Since the POVM was arbitrary, the same inequality holds for the accessible information.

```python
import numpy as np
from scipy.linalg import logm, sqrtm

def von_neumann_entropy(rho):
    """S(rho) = -Tr(rho log2 rho)."""
    eps = 1e-12
    rho = (rho + rho.conj().T) / 2.0
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > eps]
    return -np.sum(eigvals * np.log2(eigvals))

def quantum_relative_entropy(rho, sigma):
    """D(rho || sigma) = Tr(rho (log2 rho - log2 sigma))."""
    eps = 1e-12
    rho = (rho + rho.conj().T) / 2.0
    sigma = (sigma + sigma.conj().T) / 2.0
    log_rho = logm(rho + eps * np.eye(rho.shape[0])) / np.log(2)
    log_sigma = logm(sigma + eps * np.eye(sigma.shape[0])) / np.log(2)
    return np.real(np.trace(rho @ (log_rho - log_sigma)))

def holevo_quantity(ensemble):
    """
    ensemble: list of (probability, density_matrix) tuples.
    Returns chi(E) = S(rho_bar) - sum p_x S(rho_x).
    """
    rho_bar = sum(p * rho for p, rho in ensemble)
    avg_signal_entropy = sum(p * von_neumann_entropy(rho) for p, rho in ensemble)
    return von_neumann_entropy(rho_bar) - avg_signal_entropy

def classical_mutual_information(px, py_given_x):
    """
    px: P(X=x)
    py_given_x: P(Y=y | X=x), shape (|X|, |Y|)
    """
    pxy = px[:, None] * py_given_x
    py = pxy.sum(axis=0)
    py = py[py > 0]
    pxy = pxy[pxy > 0]
    hxy = -np.sum(pxy * np.log2(pxy))
    hx = -np.sum(px * np.log2(px))
    hy = -np.sum(py * np.log2(py))
    return hx + hy - hxy

def apply_povm(ensemble, povm):
    """
    ensemble: list of (p, rho)
    povm: list of positive operators summing to identity
    Returns induced channel p(y|x).
    """
    outcomes = len(povm)
    states = len(ensemble)
    py_given_x = np.zeros((states, outcomes))
    for i, (_, rho) in enumerate(ensemble):
        for j, e in enumerate(povm):
            py_given_x[i, j] = np.real(np.trace(e @ rho))
    return py_given_x

# Example 1: orthogonal qubit states -> one bit is perfectly accessible.
zero = np.array([[1.0, 0.0], [0.0, 0.0]])
one = np.array([[0.0, 0.0], [0.0, 1.0]])
orthogonal_ensemble = [(0.5, zero), (0.5, one)]
chi_orth = holevo_quantity(orthogonal_ensemble)
print(f"Orthogonal ensemble chi = {chi_orth:.4f} bits (expected 1.0)")

# Example 2: three symmetric pure qubit states at 120 degrees on Bloch equator.
angles = np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])
states = []
for theta in angles:
    psi = np.array([np.cos(theta / 2), np.sin(theta / 2)])
    states.append(psi[:, None] * psi[None, :].conj())
tricode_ensemble = [(1/3, rho) for rho in states]
chi_tri = holevo_quantity(tricode_ensemble)
print(f"Tricode ensemble chi = {chi_tri:.4f} bits")

# Example 3: induced mutual information under a simple POVM for the tricode ensemble.
povm_z = [np.array([[1.0, 0.0], [0.0, 0.0]]), np.array([[0.0, 0.0], [0.0, 1.0]])]
py_given_x = apply_povm(tricode_ensemble, povm_z)
px = np.array([1/3, 1/3, 1/3])
mi = classical_mutual_information(px, py_given_x)
print(f"Tricode I(X;Y) under Z measurement = {mi:.4f} bits")
print(f"Holevo bound satisfied: {mi <= chi_tri + 1e-9}")

# General dimensional corollary: n qubits -> at most n bits.
n = 3
print(f"Max Holevo information for {n} qubits = {np.log2(2**n):.4f} bits")
```
