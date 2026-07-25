# Context

## Research question

Kohn–Sham density functional theory reduces the many-electron ground-state problem to self-consistent single-particle equations in which the *only* unknown is the exchange–correlation energy as a functional of the electron spin densities, written $E_{\rm xc}[n_\uparrow,n_\downarrow]$. Everything in a calculation of atoms, molecules, and solids — total energies, atomization energies, bond lengths, energy barriers — is only as good as the approximation used for this one functional. The question is how to write down a *usable, transferable* approximation to $E_{\rm xc}$ that is accurate across the wildly different density regimes of real matter (the slowly varying interior of a metal, the rapidly varying tail of an atom, the bond between two atoms) — and, ideally, to do so without tuning parameters to experimental data, so the same functional can be applied to any system the user throws at it.

A solution has to be cheap (a local or semilocal integral over the density, not a many-body wavefunction), it has to give a smooth, well-behaved exchange–correlation potential $\delta E_{\rm xc}/\delta n_\sigma(\mathbf r)$ (the potential is what enters the self-consistent equations and what pseudopotentials are built from), and it has to respect the exact properties of the true functional that actually control energies.

## Background

**Kohn–Sham DFT.** The interacting problem is mapped onto non-interacting electrons in an effective potential, with the energy written $E = T_s[n] + \tfrac12\!\iint\! \frac{n(\mathbf r)n(\mathbf r')}{|\mathbf r-\mathbf r'|} + \int v\,n + E_{\rm xc}[n_\uparrow,n_\downarrow]$. $T_s$ is the non-interacting kinetic energy, the double integral is the classical Hartree term, and $E_{\rm xc}$ collects everything else: exchange, correlation, and the kinetic correction. Minimizing gives single-particle equations with $v_{\rm eff} = v + v_{\rm H} + v_{\rm xc}$, $v_{\rm xc}=\delta E_{\rm xc}/\delta n_\sigma$. The whole apparatus is exact *if* $E_{\rm xc}$ were known.

**The local spin density (LSD) approximation.** The first and most popular approximation takes the exchange–correlation energy per particle to be that of a uniform electron gas of the same local density,
$$E_{\rm xc}^{\rm LSD}[n_\uparrow,n_\downarrow]=\int d^3r\; n\,\epsilon_{\rm xc}^{\rm unif}(n_\uparrow,n_\downarrow).$$
The uniform-gas energy per particle $\epsilon_{\rm xc}^{\rm unif}(n_\uparrow,n_\downarrow)$ is itself essentially exactly known (quantum Monte Carlo for the gas, plus the known exchange $\epsilon_x^{\rm unif}=-3e^2k_F/4\pi$ with $k_F=(3\pi^2 n)^{1/3}$). LSD is constructed to be exact for a slowly varying density. It is remarkably useful far outside that regime, a fact rationalized by the *exchange–correlation hole*: $E_{\rm xc}=\tfrac12\int d^3r\,n(\mathbf r)\int d^3u\, n_{\rm xc}(\mathbf r,\mathbf r+\mathbf u)/u$, the Coulomb interaction of each electron with the depletion of other electrons around it. The exact hole obeys sum rules — $\int d^3u\,n_x(\mathbf r,\mathbf r+\mathbf u)=-1$ (exchange hole holds exactly one missing electron), $\int d^3u\,n_c=0$ (correlation merely reshapes it), and a negativity condition $n_x\le 0$ — and the LSD model of the hole, borrowed from the real uniform gas, respects all of them. That is why LSD is far better than its slowly-varying derivation has any right to be: the energy depends only on the *spherical, system average* of the hole, and a model that gets the sum rules right gets the energy roughly right.

**The gradient expansion (GEA).** The systematic next correction adds the leading gradient terms,
$$E_{\rm xc}^{\rm GEA}=E_{\rm xc}^{\rm LSD}+\sum_{\sigma,\sigma'}\int d^3r\; C_{\rm xc}^{\sigma\sigma'}(n_\uparrow,n_\downarrow)\,\frac{\nabla n_\sigma\!\cdot\!\nabla n_{\sigma'}}{n_\sigma^{2/3}n_{\sigma'}^{2/3}},$$
with coefficients known from many-body theory in the slowly varying / high-density limit. Counterintuitively this is *worse* than LSD for real atoms and molecules: the truncated gradient series is not the hole of any physical system, and in particular it violates the hole sum rules and negativity. Its large-gradient behavior diverges; its exchange hole develops an undamped long-range oscillation. So one cannot simply add gradient terms.

**The generalized gradient approximation (GGA).** The fix is to allow a general function of the density and its gradient,
$$E_{\rm xc}^{\rm GGA}[n_\uparrow,n_\downarrow]=\int d^3r\; f(n_\uparrow,n_\downarrow,\nabla n_\uparrow,\nabla n_\downarrow).$$
Relative to LSD, well-constructed GGAs improve total energies, atomization energies, energy barriers, and structural energy differences; they expand and soften bonds; they favor density inhomogeneity more than LSD does. The natural dimensionless variables are the reduced gradients $s=|\nabla n|/(2k_F n)$ (the gradient on the scale of the local Fermi wavelength, relevant for exchange) and $t=|\nabla n|/(2\phi k_s n)$ (the gradient on the scale of the Thomas–Fermi screening length $1/k_s$, $k_s=\sqrt{4k_F/\pi a_0}$, relevant for correlation), with $\phi(\zeta)=\tfrac12[(1+\zeta)^{2/3}+(1-\zeta)^{2/3}]$ the spin-scaling factor and $\zeta=(n_\uparrow-n_\downarrow)/n$.

**Exact constraints the true functional obeys** (knowable independent of any approximation, and the load-bearing facts here): the hole sum rules and negativity above; the uniform-scaling relations — under $n(\mathbf r)\to\lambda^3 n(\lambda\mathbf r)$, exchange scales as $E_x[n_\lambda]=\lambda E_x[n]$ exactly, and the correlation energy approaches a constant as $\lambda\to\infty$ (the high-density limit); the spin-scaling relation for exchange $E_x[n_\uparrow,n_\downarrow]=(E_x[2n_\uparrow]+E_x[2n_\downarrow])/2$; the high-density logarithmic form of the uniform-gas correlation energy per particle $\epsilon_c^{\rm unif}\to (e^2/a_0)\phi^3[\gamma\ln(r_s/a_0)-\omega]$ (Gell-Mann–Brueckner); and the **Lieb–Oxford bound** on the exchange–correlation energy, $E_{\rm xc}[n_\uparrow,n_\downarrow]\ge E_x[n_\uparrow,n_\downarrow]\ge -1.679\,e^2\!\int d^3r\, n^{4/3}$, a rigorous lower bound that any approximation should not undershoot.

## Baselines

**LSD.** Energy per particle borrowed from the uniform gas (above). Core idea: respect the uniform-gas physics and the hole sum rules locally. *Where it stalls:* it is built for a slowly varying density and washes out inhomogeneity; it overbinds atoms and molecules and gives atomization energies that are systematically too large; it cannot distinguish the rapidly varying tail of a finite system from the gas interior.

**Gradient expansion approximation (GEA).** Adds the leading gradient terms with first-principles coefficients. Core idea: systematically include the leading inhomogeneity correction. *Where it stalls:* it is *less* accurate than LSD for real systems because the truncated series violates the hole sum rules and negativity, diverges at large reduced gradient, and produces an exchange hole with an unphysical undamped long-range oscillation; it cannot be used as-is.

**Real-space-cutoff numerical GGA and its analytic fits (Langreth–Mehl; Becke 1988; Perdew–Wang 1986/1991).** Core idea: start from the gradient-expanded exchange–correlation *hole* around an electron, then sharply cut off its spurious large-separation parts to restore the exact sum rules and negativity, and integrate the repaired hole to get an energy; the result is a numerical $f$ which is then fitted to a closed analytic form. Perdew–Wang 1986 used sharp cutoffs of the GEA hole; the Perdew–Wang 1991 (PW91) refinement is an analytic fit to this numerical GGA designed to satisfy several further exact conditions, and it is the leading non-empirical GGA. Becke 1986/1988 instead proposed a compact exchange enhancement of the form $F_x(s)=1+\kappa-\kappa/(1+\mu s^2/\kappa)$ but fixed its two coefficients ($\kappa=0.967,\mu=0.235$) by fitting atomic exchange energies. *Where the non-empirical fits stall:* the construction is long and depends on a mass of detail; the fitted analytic function is complicated and over-parametrized; its parameters are not seamlessly joined, so the exchange–correlation potential $\delta E_{\rm xc}/\delta n_\sigma$ develops spurious wiggles at small and large reduced gradient, which bedevils GGA-based pseudopotentials; the analytic parametrization fails to keep the correlation energy finite under uniform scaling to the high-density limit even though the underlying numerical functional behaves correctly; and because it reduces to the second-order gradient expansion whenever the density variation is either slowly varying or small, it describes the linear response of the uniform gas *less* well than LSD does. The empirical-fit route (Becke/LYP) buys accuracy near the systems it was fitted to at the cost of transferability far away.

## Evaluation settings

The natural yardstick for an exchange–correlation functional at this time: atomization energies of small molecules (a standard set such as H$_2$, LiH, CH$_4$, NH$_3$, H$_2$O, HF, CO, N$_2$, O$_2$, F$_2$, etc.), evaluated on self-consistent densities at experimental geometries, in kcal/mol against experimental values with zero-point vibration removed; total energies and exchange/correlation energies of atoms; the linear response of the uniform electron gas; lattice constants and cohesive properties of solids; and the smoothness and transferability of the resulting exchange–correlation potential (relevant for pseudopotential construction). Quantum-chemistry codes (e.g. CADPAC-class programs) and uniform-gas reference data supply the comparison numbers; quantum Monte Carlo provides the uniform-gas correlation energy benchmark.

## Code framework

Existing primitives: a Kohn–Sham SCF loop, a grid over real space with weights, and the uniform-gas exchange–correlation energy-per-particle routine. A GGA exchange–correlation functional is a *kernel* that, given the spin densities and their gradients on each grid point, returns the energy density and the derivatives needed to assemble $v_{\rm xc}$. The slot to be filled is the kernel itself.

```python
import numpy as np

# --- given / already-existing pre-method machinery ---

def eps_x_unif(n):
    """Exchange energy per particle of the uniform electron gas (spin-unpolarized).
    eps_x_unif = -3 e^2 kF / (4 pi),  kF = (3 pi^2 n)^(1/3).  Hartree atomic units."""
    kF = (3.0 * np.pi**2 * n) ** (1.0/3.0)
    return -3.0 * kF / (4.0 * np.pi)

def eps_c_unif(rs, zeta):
    """Correlation energy per particle of the uniform electron gas (known, fitted to QMC).
    Returns eps_c^unif(rs, zeta) in Hartree.  Treated as a black box here."""
    pass  # established uniform-gas correlation parametrization

def reduced_gradients(n, grad_n, zeta):
    """Dimensionless gradients on the two natural length scales."""
    kF = (3.0 * np.pi**2 * n) ** (1.0/3.0)
    ks = np.sqrt(4.0 * kF / np.pi)                      # Thomas-Fermi screening wavenumber (a0=1)
    phi = 0.5 * ((1.0+zeta)**(2.0/3.0) + (1.0-zeta)**(2.0/3.0))
    s = np.abs(grad_n) / (2.0 * kF * n)                 # gradient on Fermi-wavelength scale
    t = np.abs(grad_n) / (2.0 * phi * ks * n)           # gradient on screening-length scale
    return s, t, phi

# --- the slot the method will fill ---

def xc_energy_density(n_up, n_dn, grad_n_up, grad_n_dn):
    """Return the semilocal exchange-correlation energy density at a grid point,
    as a function of the spin densities and their gradients.
    # TODO: the functional we will construct here.
    """
    pass

def Exc_GGA(n_up, n_dn, grad_n_up, grad_n_dn, grid_weights):
    e_dens = xc_energy_density(n_up, n_dn, grad_n_up, grad_n_dn)
    return np.sum(e_dens * grid_weights)
```
