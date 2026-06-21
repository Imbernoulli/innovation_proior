I would call this the Kosterlitz-Thouless transition. It describes how a two-dimensional system with a continuous symmetry, such as the XY magnet, a neutral superfluid film, or a two-dimensional crystal, can still undergo a sharp phase transition even though ordinary long-range order is forbidden at any finite temperature. The key is to stop looking for a local order parameter and instead look at the topology of the phase field.

In two dimensions, the low-energy excitations of a continuous-symmetry system are long-wavelength modes that cost very little energy. For the XY model I write the spin as s = (cos θ, sin θ), and at low temperature the energy is well approximated by the Gaussian spin-wave form H ≈ (J/2) ∫ d²r |∇θ|². These spin waves already show that the magnetization must vanish, because thermal phase fluctuations grow logarithmically with distance and destroy any nonzero expectation value of e^{iθ}. The same infrared mechanism rules out true long-range positional order in a two-dimensional crystal and true condensate order in a two-dimensional Bose fluid. So a Landau-style symmetry-breaking transition is impossible from the start.

But the spin-wave theory is not the whole story. It treats θ as a smooth single-valued field, which misses the fact that the target space of θ is a circle. Around a closed loop the phase can wind by an integer multiple of 2π, q = (1/2π) ∮ ∇θ·dℓ ∈ ℤ. That integer cannot be removed by small local fluctuations; it is a topological defect, a vortex. A unit vortex centered at the origin has θ = φ and |∇θ| = 1/r, so its energy is E_v = π J ln(L/a) up to a core energy E_c. The number of places the core can sit gives an entropy S_v ≈ 2 k_B ln(L/a). The free energy of an isolated vortex is therefore ΔF_v = E_c + (π J − 2 k_B T) ln(L/a). At low temperature the logarithmic term is positive and an isolated vortex is expelled in the thermodynamic limit; at high temperature it is negative and entropy favors free vortices. That sign change is already the seed of the transition.

A finite system with neutral boundary conditions cannot contain a single vortex, so the elementary neutral excitation is a vortex-antivortex pair. A pair separated by distance r has energy E_pair(r) ≈ 2 E_c + 2π J ln(r/a), which is exactly the potential energy of two opposite charges in a two-dimensional Coulomb gas. Thus the vortex sector is a neutral Coulomb gas with fugacity y = exp(−E_c/k_B T) and dimensionless stiffness K = J/(k_B T). At low temperature this gas consists of small bound dipoles; at high temperature the dipoles unbind and free vortices proliferate.

The crude energy-entropy estimate uses the bare stiffness all the way to the system size, which is not self-consistent. Smaller bound pairs screen the interaction between larger pairs, so the effective stiffness must be renormalized scale by scale. I therefore consider an RG step in which I integrate out neutral vortex-antivortex pairs whose separations lie between a and a e^{dl}. A thin shell of small dipoles acts as a dielectric medium, reducing the stiffness and giving the flow dK^{-1}/dl = 4π³ y² to leading order in fugacity. At the same time, rescaling the cutoff changes the vortex fugacity according to the free-energy balance; a vortex weight gets an area factor e^{2dl} and is suppressed by the interaction energy e^{−πK dl}, so dy/dl = (2 − πK)y to leading order.

The resulting phase portrait explains the transition. The line y = 0 is a fixed line corresponding to the Gaussian spin-wave theory. It is stable only where πK > 2. In that region a small fugacity flows back to zero, vortices remain bound in neutral pairs, and the long-distance theory is a spin-wave theory with a finite renormalized stiffness K_R. The spin-spin correlation function decays as a power law, ⟨s(0)·s(r)⟩ ∼ r^{-η(T)} with η(T) = 1/(2π K_R(T)). There is no spontaneous magnetization, but the susceptibility is infinite because algebraic decay is not integrable. At the endpoint πK_R = 2, the exponent reaches η(T_c) = 1/4.

For πK < 2 the fugacity is relevant. It grows under RG, vortex-antivortex pairs unbind, and free vortices screen the logarithmic interaction. The correlation length becomes finite and correlations cross over from algebraic to exponential decay. The transition is the separatrix that flows into the fixed point (K = 2/π, y = 0). Because the relevant variable is marginal at this fixed point, the correlation length above T_c diverges as an essential singularity, ξ(T > T_c) ∼ a exp(b / √((T − T_c)/T_c)), rather than as a power law. Consequently the singular free-energy density, which scales like ξ^{-2}, is weaker than any power-law singularity.

The most striking experimental signature follows from the universal endpoint condition. Just below the transition the renormalized stiffness satisfies K_R(T_c^-) = 2/π, or equivalently J_R(T_c^-) = 2 k_B T_c / π for the XY model. For a neutral superfluid film this becomes the Nelson-Kosterlitz jump in the areal superfluid density, ρ_s^R(T_c^-)/T_c = 2 m² k_B / (π ħ²). The transition is therefore not marked by the onset of a nonzero local order parameter, which remains zero on both sides, but by a universal jump in the renormalized stiffness caused by the sudden relevance of free vortices.

The planar nature of the order parameter is essential. If the spins had three components, as in the Heisenberg model, a would-be vortex could escape through the third dimension, and the topological barrier would be finite rather than logarithmic in system size. Only the circle-valued phase field of the planar model gives vortices with a logarithmic energy that can compete with entropy and protect a topological low-temperature phase. In a two-dimensional crystal the analogous defects are dislocations, and the same unbinding transition destroys quasi-long-range positional order above the melting temperature.

In short, the Kosterlitz-Thouless transition replaces Landau's order-parameter picture with a topological mechanism. Spin waves destroy ordinary long-range order, but vortices provide integer winding sectors whose unbinding drives a sharp change in response. The transition is controlled by the RG flow of a logarithmic Coulomb gas, predicts a universal stiffness jump, and produces an essential singularity in the correlation length above T_c.

```python
import numpy as np
import matplotlib.pyplot as plt

"""
Kosterlitz-Thouless RG flow in the (K, y) plane.
K is dimensionless stiffness and y is vortex fugacity.
Flow equations: dK^{-1}/dl = 4*pi^3*y^2 and dy/dl = (2 - pi*K)*y.
"""

def rg_flow(state, ell):
    K, y = state
    y = max(y, 0.0)
    if K <= 0.0:
        return np.array([0.0, 0.0])
    dK_inv_dl = 4.0 * np.pi**3 * y**2
    dK_dl = -K**2 * dK_inv_dl
    dy_dl = (2.0 - np.pi * K) * y
    return np.array([dK_dl, dy_dl])

def rk4_step(state, dl):
    k1 = rg_flow(state, 0.0)
    k2 = rg_flow(state + 0.5*dl*k1, 0.0)
    k3 = rg_flow(state + 0.5*dl*k2, 0.0)
    k4 = rg_flow(state + dl*k3, 0.0)
    return state + (dl/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def integrate(K0, y0, l_max=10.0, n_steps=20000):
    dl = l_max / n_steps
    K = np.zeros(n_steps + 1)
    y = np.zeros(n_steps + 1)
    ell = np.zeros(n_steps + 1)
    K[0], y[0] = K0, y0
    state = np.array([K0, y0])
    for i in range(n_steps):
        state = rk4_step(state, dl)
        K[i+1], y[i+1] = state
        ell[i+1] = ell[i] + dl
    return ell, K, y

ell1, K1, y1 = integrate(K0=0.9, y0=0.05)
ell2, K2, y2 = integrate(K0=0.5, y0=0.05)
ell3, K3, y3 = integrate(K0=0.7, y0=0.02)

fig, ax = plt.subplots()
ax.plot(ell1, K1, label="K low-T (K0=0.90)")
ax.plot(ell1, y1, label="y low-T", linestyle="--")
ax.plot(ell2, K2, label="K high-T (K0=0.50)")
ax.plot(ell2, y2, label="y high-T", linestyle="--")
ax.axhline(2.0/np.pi, color="gray", linestyle=":", label="K = 2/pi")
ax.set_xlabel("RG scale l = ln(r/a)")
ax.set_ylabel("K(l), y(l)")
ax.set_ylim(-0.05, 1.0)
ax.set_title("Kosterlitz-Thouless RG flow")
ax.legend()
plt.tight_layout()
plt.savefig("kt_rg_flow.png", dpi=150)
print("Saved kt_rg_flow.png")
print("Critical K =", 2.0/np.pi)
print("Final low-T  K, y:", K1[-1], y1[-1])
print("Final high-T K, y:", K2[-1], y2[-1])
```
