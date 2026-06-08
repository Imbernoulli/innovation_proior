# The Cooper instability and the BCS gap equation

## Problem

Explain superconductivity microscopically: a second-order transition at $T_c$, an energy gap (exponential specific heat $\exp(-T_0/T)$), the Meissner effect, infinite conductivity, and the isotope effect $T_c M^{1/2} = \text{const}$ — given that the condensation energy ($\sim10^{-8}$ eV/electron) is eight orders of magnitude below the uncertainty in the total energy of the electron–phonon system.

## Key idea

Near the Fermi surface, virtual phonon exchange gives a *net attraction* between electrons (it dominates the screened Coulomb repulsion when the electronic energy transfer is below the phonon energy $\hbar\omega_D$). Against this attraction the filled Fermi sea is **unstable**: two electrons added just above the surface bind into a "Cooper pair" with energy below $2\mathcal{E}_F$ for *any* attraction strength, however weak. The true ground state develops coherent zero-momentum, opposite-spin pair amplitudes across the near-surface shell; this opens an energy gap and is the superconducting state. The two-electron binding and the many-body gap are non-analytic in the coupling, respectively $e^{-2/N(0)V}$ and $e^{-1/N(0)V}$, so both are invisible to finite-order perturbation theory — which is why earlier self-energy theories failed.

## The Cooper instability (two electrons above a frozen Fermi sea)

Pair wavefunction $|\psi\rangle = \sum_{k>k_F} a_\mathbf{k}|\mathbf{k}\uparrow,-\mathbf{k}\downarrow\rangle$, energies $\epsilon_\mathbf{k}$ measured from $\mathcal{E}_F$, and pair energy $E$ measured relative to $2\mathcal{E}_F$. The Schrödinger equation is

$$ (E - 2\epsilon_\mathbf{k})\,a_\mathbf{k} = \sum_{k'>k_F} V_{\mathbf{k}\mathbf{k}'}\,a_{\mathbf{k}'}. $$

With the model interaction $V_{\mathbf{k}\mathbf{k}'} = -V$ for $0<\epsilon_\mathbf{k},\epsilon_{\mathbf{k}'}<\hbar\omega_D$ (else 0), $a_\mathbf{k} \propto 1/(2\epsilon_\mathbf{k}-E)$ and summing gives the eigenvalue condition

$$ 1 = V\!\!\sum_{0<\epsilon_\mathbf{k}<\hbar\omega_D}\!\!\frac{1}{2\epsilon_\mathbf{k}-E} = N(0)V\!\int_0^{\hbar\omega_D}\!\frac{d\xi}{2\xi - E} = \frac{N(0)V}{2}\ln\frac{2\hbar\omega_D - E}{-E}. $$

The logarithm diverges as $E\to0^-$, so a bound state ($E<0$ relative to $2\mathcal{E}_F$) exists for every $V>0$ — the divergence comes from the *finite* density of states $N(0)$ that the filled sea provides at the bottom of the available shell. Weak coupling gives the absolute pair energy:

$$ \boxed{\,E_{\text{pair}} = 2\mathcal{E}_F - 2\hbar\omega_D\,e^{-2/N(0)V}\,}\qquad(\text{below } 2\mathcal{E}_F). $$

The factor $e^{-2/N(0)V}$ has an essential singularity at $V=0$: non-perturbative.

## The BCS ground state and gap equation

Pair operators $b_\mathbf{k} = c_{-\mathbf{k}\downarrow}c_{\mathbf{k}\uparrow}$, with $[b_\mathbf{k},b_{\mathbf{k}'}^\dagger]=(1-n_{\mathbf{k}\uparrow}-n_{-\mathbf{k}\downarrow})\delta_{\mathbf{k}\mathbf{k}'}$ and $b_\mathbf{k}^2=0$ (hard-core, *not* bosons). Reduced Hamiltonian (zero-momentum singlet pairs):

$$ H_{\text{red}} = 2\!\sum_{k>k_F}\!\epsilon_\mathbf{k} b_\mathbf{k}^\dagger b_\mathbf{k} + 2\!\sum_{k<k_F}\!|\epsilon_\mathbf{k}| b_\mathbf{k} b_\mathbf{k}^\dagger - \sum_{\mathbf{k}\mathbf{k}'} V_{\mathbf{k}\mathbf{k}'} b_{\mathbf{k}'}^\dagger b_\mathbf{k}. $$

Variational ground state (Hartree-like in occupation amplitudes; $u_\mathbf{k}^2+v_\mathbf{k}^2=1$, $h_\mathbf{k}\equiv v_\mathbf{k}^2$):

$$ |\Psi\rangle = \prod_\mathbf{k}\big(u_\mathbf{k} + v_\mathbf{k}\,b_\mathbf{k}^\dagger\big)|0\rangle. $$

Minimizing $W_0=\langle\Psi|H_{\text{red}}|\Psi\rangle = 2\sum_{k>k_F}\epsilon_\mathbf{k}h_\mathbf{k} + 2\sum_{k<k_F}|\epsilon_\mathbf{k}|(1-h_\mathbf{k}) - \sum_{\mathbf{k}\mathbf{k}'}V_{\mathbf{k}\mathbf{k}'}[h_\mathbf{k}(1-h_\mathbf{k})h_{\mathbf{k}'}(1-h_{\mathbf{k}'})]^{1/2}$ over $h_\mathbf{k}$ gives

$$ h_\mathbf{k} = \frac{1}{2}\!\left(1 - \frac{\epsilon_\mathbf{k}}{\sqrt{\epsilon_\mathbf{k}^2+\epsilon_0^2}}\right),\qquad [h_\mathbf{k}(1-h_\mathbf{k})]^{1/2} = \frac{\epsilon_0}{2\sqrt{\epsilon_\mathbf{k}^2+\epsilon_0^2}}, $$

with $\epsilon_0 = V\sum_{\mathbf{k}'}[h_{\mathbf{k}'}(1-h_{\mathbf{k}'})]^{1/2}$. Self-consistency yields the **gap equation**

$$ \frac{1}{V} = \sum_\mathbf{k}\frac{1}{2\sqrt{\epsilon_\mathbf{k}^2+\epsilon_0^2}} \;\Longrightarrow\; \frac{1}{N(0)V} = \int_0^{\hbar\omega_D}\!\frac{d\xi}{\sqrt{\xi^2+\epsilon_0^2}} = \sinh^{-1}\!\frac{\hbar\omega_D}{\epsilon_0}, $$

$$ \boxed{\,\epsilon_0 = \frac{\hbar\omega_D}{\sinh[1/N(0)V]} \approx 2\hbar\omega_D\,e^{-1/N(0)V}\,}. $$

The exponent is $1$ (not $2$ as in the single-pair Cooper problem) because each electron is self-consistently paired: the bare $2\xi-E$ denominator becomes $\sqrt{\xi^2+\epsilon_0^2}$, turning the $\tfrac12\ln$ into $\sinh^{-1}$.

## Consequences (all five facts as outputs)

- **Energy gap.** Quasiparticle dispersion $E_\mathbf{k}=\sqrt{\epsilon_\mathbf{k}^2+\epsilon_0^2}\ge\epsilon_0$; single-particle spectrum has a gap of width $2\epsilon_0$, density of states $dN/dE=N(0)E/\sqrt{E^2-\epsilon_0^2}$ (singular at the edge). Explains the exponential specific heat.
- **Condensation energy.** $W_0 = -2N(0)(\hbar\omega_D)^2 e^{-2/N(0)V}$; the exponential gives the right tiny magnitude, and since $W_0\sim N(0)(kT_c)^2$, the phonon scale $\hbar\omega_D\propto M^{-1/2}$ gives the isotope effect for $T_c$.
- **Meissner / persistent currents.** $b_\mathbf{k}^2=0$ gives a gap also against pair translation with the wrong momentum, locking the condensate phase rigid over macroscopic distances.
- **Finite temperature / second-order transition.** $\dfrac{1}{N(0)V}=\displaystyle\int_0^{\hbar\omega_D}\dfrac{d\xi}{\sqrt{\xi^2+\epsilon_0^2}}\tanh\!\Big[\tfrac12\beta\sqrt{\xi^2+\epsilon_0^2}\Big]$; $\epsilon_0(T)\to0$ continuously at

$$ kT_c = 1.14\,\hbar\omega_D\,e^{-1/N(0)V}, \qquad \frac{2\epsilon_0(0)}{kT_c} \approx 3.5 \;\;(\text{parameter-free, universal}). $$

## Worked numerical check

```python
import numpy as np
from scipy import integrate, optimize

# Energies in units of the phonon cutoff hbar*omega_D = 1.
N0V = 0.3   # dimensionless coupling N(0)V

# Cooper instability: 1 = N0V * int_0^1 dxi/(2xi - E_rel), with E_rel < 0 for any N0V>0
E_rel = optimize.brentq(
    lambda E: N0V*integrate.quad(lambda xi: 1/(2*xi-E), 0, 1)[0] - 1, -10, -1e-12)
print(E_rel, -2*np.exp(-2/N0V))                     # pair energy relative to 2E_F vs weak-coupling form

# Gap equation: 1/N0V = arcsinh(1/eps0) -> eps0 = 1/sinh(1/N0V)
eps0 = optimize.brentq(
    lambda D: integrate.quad(lambda xi: 1/np.sqrt(xi**2+D**2), 0, 1)[0] - 1/N0V, 1e-12, 10)
print(eps0, 1/np.sinh(1/N0V), 2*np.exp(-1/N0V))     # gap vs closed form vs weak-coupling form

# T_c: 1/N0V = int_0^1 (dxi/xi) tanh(xi/2kTc)
kTc = optimize.brentq(
    lambda kT: integrate.quad(lambda xi: np.tanh(xi/(2*kT))/xi, 1e-9, 1)[0] - 1/N0V, 1e-6, 1)
print(kTc, 1.14*np.exp(-1/N0V), 2*eps0/kTc)         # kTc vs weak-coupling form; gap/Tc ~ 3.5
```

Output: pair energy relative to $2\mathcal{E}_F$ is $-2.5\times10^{-3}$ (vs $-2e^{-2/N(0)V}=-2.5\times10^{-3}$); $\epsilon_0=0.0714$ (vs $1/\sinh(1/N(0)V)=0.0714$, $2e^{-1/N(0)V}=0.0713$); $kT_c=0.0404$ (vs $1.14\,e^{-1/N(0)V}=0.0407$); ratio $2\epsilon_0/kT_c = 3.53$.
