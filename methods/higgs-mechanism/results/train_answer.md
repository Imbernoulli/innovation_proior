The problem is that massive vector bosons are needed for short-range forces like the weak interaction, but a direct Proca mass term for a gauge field violates local gauge invariance. Meanwhile, the ordinary global version of spontaneous symmetry breaking in a complex scalar theory does give a mass scale, but it pays for it by producing a physical massless Goldstone boson for every broken continuous generator. That is the wrong spectrum for particle physics: we want massive vectors, not extra long-range scalars. The two difficulties are actually linked. If the broken symmetry is local rather than global, the scalar phase that would have become the Goldstone mode is already mixed with the gauge field through the covariant derivative, so it can be absorbed into the vector's longitudinal polarization instead of surviving as an independent particle.

The solution is the Higgs mechanism. It keeps the local gauge symmetry exact in the Lagrangian, gives a complex scalar field a Mexican-hat potential whose minimum has nonzero modulus, and lets the gauge field acquire mass from the gauge-invariant scalar kinetic term. No explicit mass term for the vector is ever added, and no physical massless scalar remains. The mechanism works by recognizing that the phase of the scalar is not an observable particle once the symmetry is gauged; it is the gauge redundancy that supplies the extra longitudinal polarization required by a massive vector.

In the simplest Abelian prototype, take a complex scalar with the covariant derivative D_mu = partial_mu - i e A_mu and the U(1) transformations phi -> exp(i alpha(x)) phi and A_mu -> A_mu + (1/e) partial_mu alpha. The gauge-invariant Lagrangian is L = (D_mu phi)^* D^mu phi - lambda(phi^*phi - v^2/2)^2 - (1/4) F_{mu nu} F^{mu nu}. Writing phi in polar form as (rho/sqrt(2)) exp(i theta), the kinetic term becomes (1/2)(partial rho)^2 + (1/2) rho^2 (partial_mu theta - e A_mu)^2. The combination B_mu = A_mu - (1/e) partial_mu theta is gauge invariant, and because the field strength is unchanged by adding a gradient, F_{mu nu}(A) = F_{mu nu}(B). Thus the Lagrangian can be written entirely in physical variables as L = (1/2)(partial rho)^2 + (1/2) e^2 rho^2 B_mu B^mu - lambda(rho^2 - v^2)^2/4 - (1/4) B_{mu nu} B^{mu nu}. Expanding around rho = v + h gives a quadratic mass term (1/2) e^2 v^2 B_mu B^mu for the vector and a mass term lambda v^2 h^2 for the radial scalar. The masses are m_B = e v and m_h = sqrt(2 lambda) v.

The degree-of-freedom counting confirms the consistency. Before the mechanism there are two transverse polarizations of the massless vector plus two real components of the complex scalar, for four degrees of freedom. After the mechanism there are three polarizations of the massive vector plus one real scalar h, still four. The would-be Goldstone mode theta has not disappeared arithmetically; it has become the longitudinal polarization of B_mu. In a non-Abelian theory the same structure generalizes: the gauge-boson mass matrix is g^2 (T^a v)^dagger (T^b v), where v is the scalar vacuum value. Generators that leave v invariant correspond to massless gauge bosons, and broken generators correspond to massive gauge bosons whose longitudinal modes come from the scalar directions they rotate.

A concrete implementation can verify the spectrum numerically by building the scalar potential, the covariant kinetic term, and the mass matrices in a chosen vacuum.

```python
import numpy as np

# Abelian Higgs mechanism: compute physical masses from the quadratic expansion.

def abelian_higgs_masses(e, lam, v):
    """
    Parameters:
      e    : U(1) gauge coupling
      lam  : quartic coupling (> 0)
      v    : scalar vacuum modulus (the minimum of V at |phi| = v/sqrt(2))
    Returns:
      m_h : radial Higgs scalar mass
      m_B : massive gauge boson mass
    """
    m_h = np.sqrt(2 * lam) * v
    m_B = e * v
    return m_h, m_B

# Example: e = 0.3, lambda = 0.1, v = 246 GeV (up to units)
e = 0.3
lam = 0.1
v = 246.0

m_h, m_B = abelian_higgs_masses(e, lam, v)
print(f"Radial Higgs mass m_h = {m_h:.4f}")
print(f"Gauge boson mass m_B  = {m_B:.4f}")

# Non-Abelian gauge-boson mass matrix for a scalar multiplet with VEV v_vec.
# Generators T^a are Hermitian matrices; mass matrix is
# (M^2_V)_{ab} = g^2 v^\dagger T^a T^b v (up to normalization conventions).

def gauge_boson_mass_matrix(g, generators, vev):
    """
    Parameters:
      g           : gauge coupling
      generators  : list of Hermitian generator matrices T^a
      vev         : scalar vacuum expectation value vector
    Returns:
      M2 : gauge-boson squared-mass matrix
    """
    n = len(generators)
    M2 = np.zeros((n, n), dtype=complex)
    vev = np.asarray(vev, dtype=complex)
    for a in range(n):
        T_a_v = generators[a] @ vev
        for b in range(n):
            T_b_v = generators[b] @ vev
            M2[a, b] = g**2 * (T_a_v.conj() @ T_b_v)
    # The result is real for Hermitian generators and a real VEV direction.
    return np.real(M2)

# Example: SU(2) with a VEV along the third component.
# Use Pauli matrices as generators.
sigma1 = np.array([[0, 1], [1, 0]], dtype=complex)
sigma2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma3 = np.array([[1, 0], [0, -1]], dtype=complex)
generators = [0.5 * sigma1, 0.5 * sigma2, 0.5 * sigma3]
g = 0.65
vev = np.array([0.0, 1.0], dtype=complex)  # arbitrary normalization

M2 = gauge_boson_mass_matrix(g, generators, vev)
masses_squared = np.linalg.eigvalsh(M2)
print("SU(2) gauge-boson squared masses:", masses_squared)
# With this VEV, T^3 leaves the vacuum invariant -> one massless boson.
```
