The problem is to write a single-electron wave equation that respects special relativity while keeping the probability structure of ordinary quantum mechanics. In the non-relativistic theory, Schrödinger's equation is first order in time, so the density rho = |psi|^2 is manifestly non-negative and a continuity equation conserves total probability. The natural relativistic starting point is the Klein-Gordon equation, obtained by quantizing E^2 = p^2 c^2 + m^2 c^4. It is second order in time, and the only conserved density one can build from its solutions contains a time derivative of psi. Because psi and its time derivative are independent initial data, that density can be negative, so it cannot be interpreted as a probability. The Klein-Gordon equation also describes a spinless particle and therefore cannot account for the electron's measured spin and magnetic moment. Pauli's two-component spin theory adds spin by hand, but it is non-relativistic and needs an empirically adjusted spin-orbit factor. What is needed is an equation that is first order in time like Schrödinger's, Lorentz invariant, and capable of carrying spin.

The new method is the Dirac equation. The decisive move is to insist on an equation that is first order in both time and space. Relativity treats space and time on an equal footing, so if the equation is first order in the time derivative it must also be first order in the spatial derivatives. One therefore writes a linear equation in the four momentum operators and demands that squaring it reproduce the relativistic energy-momentum relation E^2 = p^2 c^2 + m^2 c^4. Squaring a linear expression produces cross terms that must vanish, which forces the coefficients to anticommute rather than commute. Ordinary numbers cannot satisfy alpha_i alpha_j + alpha_j alpha_i = 0 while also squaring to one, so the coefficients must be matrices. The smallest algebra that works needs four mutually anticommuting 4x4 matrices, constructed from two copies of the Pauli spin matrices. Consequently the wave function psi has four components. This matrix structure is not put in to describe spin; it is forced by the requirement that a first-order relativistic equation square to the correct dispersion relation.

The free Dirac equation can be written in Hamiltonian form as i hbar d psi/dt = (c alpha dot p + beta m c^2) psi, with p = -i hbar grad, and 4x4 matrices alpha_i and beta satisfying alpha_i alpha_j + alpha_j alpha_i = 2 delta_{ij} I, beta^2 = I, and alpha_i beta + beta alpha_i = 0. In four-index notation it becomes (i sum_mu gamma_mu p_mu + m c) psi = 0, where the gamma matrices satisfy the Clifford algebra gamma_mu gamma_nu + gamma_nu gamma_mu = 2 delta_{mu nu} I. The standard Minkowski form is (i hbar gamma^mu partial_mu - m c) psi = 0 with gamma^0 = beta and gamma^i = beta alpha_i, and the anticommutator becomes gamma^mu gamma^nu + gamma^nu gamma^mu = 2 g^{mu nu} I for the metric diag(1, -1, -1, -1).

Because the equation is first order in time, the probability density is simply rho = psi^dagger psi, a sum of the squared moduli of the four components, and it is non-negative. The associated current is j = c psi^dagger alpha psi, and they obey the continuity equation d rho/dt + div j = 0. When the electron is placed in an electromagnetic field by minimal coupling, p -> p + (e/c) A, the squared equation acquires an extra term proportional to sigma dot B. This gives the electron an intrinsic magnetic moment of one Bohr magneton with gyromagnetic ratio g = 2. Orbital angular momentum alone is not conserved, but the total angular momentum M = x cross p + (hbar/2) sigma is conserved, identifying the electron as a spin-1/2 particle. The hydrogen fine structure then emerges automatically, without the hand-inserted Thomas factor required by Pauli's theory.

The equation also retains both signs of the energy, E = plus or minus sqrt(p^2 c^2 + m^2 c^4). The negative-energy solutions cannot simply be discarded, because perturbations would drive transitions into them. Interpreting the vacuum as a filled sea of negative-energy states and an empty state as a hole gives a particle with positive energy, positive charge, and the same mass as the electron: the anti-electron. The Dirac equation therefore predicts antimatter, with electron-positron annihilation and pair creation contained in the same framework.

```python
import numpy as np

# Pauli matrices: sigma_r^2 = I and sigma_r sigma_s + sigma_s sigma_r = 0 for r != s.
I2 = np.eye(2, dtype=complex)
sigma = [
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
]
Z = np.zeros((2, 2), dtype=complex)

# Two commuting copies of the Pauli algebra in 4x4 block form.
rho1 = np.block([[Z, I2], [I2, Z]])
rho2 = np.block([[Z, -1j * I2], [1j * I2, Z]])
rho3 = np.block([[I2, Z], [Z, -I2]])
sigma4 = [np.block([[s, Z], [Z, s]]) for s in sigma]

# Dirac-Pauli alpha_i and beta matrices.
alpha = [rho1 @ s for s in sigma4]
beta = rho3

# Standard Minkowski gamma matrices: gamma^0 = beta, gamma^i = beta alpha_i.
gamma0 = beta
gammai = [beta @ a for a in alpha]
G = [gamma0] + gammai

I4 = np.eye(4, dtype=complex)
g = np.diag([1, -1, -1, -1]).astype(complex)

def anticommutator(A, B):
    return A @ B + B @ A

# Verify the Dirac/Clifford algebra {gamma^mu, gamma^nu} = 2 g^{mu nu} I.
for mu in range(4):
    for nu in range(4):
        assert np.allclose(anticommutator(G[mu], G[nu]), 2 * g[mu, nu] * I4)

# Verify the Hamiltonian algebra as well.
for i in range(3):
    assert np.allclose(alpha[i] @ alpha[i], I4)
    assert np.allclose(anticommutator(alpha[i], beta), Z)
    for j in range(3):
        assert np.allclose(anticommutator(alpha[i], alpha[j]), 2 * (i == j) * I4)
assert np.allclose(beta @ beta, I4)

print("Dirac-Clifford algebra verified.")
```
