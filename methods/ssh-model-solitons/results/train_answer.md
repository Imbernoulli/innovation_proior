Undoped trans-(CH)$_x$ refuses to behave like a band insulator. It carries a dilute population — a few hundred parts per million — of paramagnetic centres whose electron-spin-resonance line is a single narrow Lorentzian, $g = 2.00263$ and $\Delta H \approx 1.65$ Oe, and that line stays narrow all the way down to 10 K. A narrow Lorentzian surviving to 10 K is the fingerprint of motional narrowing, so whatever carries this spin is intrinsic (no dopant put it there), neutral, spin-$\tfrac12$, and moving fast enough to average away the hyperfine and dipolar broadenings. Then doping with acceptors (AsF$_5$, I$_2$) or with donors raises the conductivity by orders of magnitude while the Curie-law susceptibility barely moves: the charge being added arrives without spin. A mobile neutral spin and spinless charge are precisely what a rigid-band picture cannot supply — an electron at the conduction-band edge or a hole at the valence-band edge costs the half-gap $\Delta$ and is both charged and spin-$\tfrac12$. In this material spin and charge are not merely decoupled, they are anticorrelated.

The available accounts each fail at an identifiable place. The shifted-Fermi-level semiconductor picture of doping predicts spin-$\tfrac12$ carriers appearing in step with the conductivity, which the flat susceptibility contradicts, and it charges every carrier the band-edge energy $\Delta$. Attributing the undoped ESR line to a bond-alternation defect quenched into the polymer during cis$\to$trans isomerization (Goldberg et al. 1976) names the object but gives it no energy, no width, no mass, and no reason to carry spin $\tfrac12$ rather than nothing. The sine-Gordon route — treat the dimerization as a charge-density wave and its low-lying excitations as $\pi$ phase slips (Rice 1979; Bishop, Krumhansl and Trullinger 1976) — has the merit of making the carrier a domain wall, but a $\pi$ phase slip of a charge-density wave carries charge $\pm e$ by construction, so it assigns charge to the wall itself and can never produce a neutral spin-$\tfrac12$ excitation. And the field-theoretic observation that a Dirac fermion in a kink background acquires a normalizable zero mode and a fractional fermion number $\pm\tfrac12$ (Jackiw and Rebbi 1976) concerns a *postulated* scalar background in a relativistic continuum, with no microscopic material attached to it. What is missing is a microscopic theory of the dimerized chain in which a wall's formation energy, width, mass, charge and spin all come out as numbers that can be checked against the measurements.

I propose the following model of the chain and, as its elementary excitation, an amplitude soliton — a $\varphi^4$-type kink of the bond-alternation order parameter carrying one nonbonding mid-gap electronic state. Keep only the degrees of freedom that live below $0.5$ eV. The $\sigma$ electrons sit in bonds with a $\sim 10$ eV gap to their antibonding partners, so they never get excited; they act purely as an elastic medium, and expanding their energy to second order about the undimerized geometry (the linear term vanishes there, since that is the $\sigma$ equilibrium) gives a harmonic spring $\tfrac12 K\sum_n (u_{n+1}-u_n)^2$ in the displacements $u_n$ of the CH units along the chain axis. The $\pi$ electrons are tight-binding, one $p_z$ orbital per site, nearest-neighbour hopping only, which is the minimal thing with a half-filled band. The one essential coupling is that the transfer integral depends on bond length — push two carbons together and their $\pi$ overlap grows — and for the tiny displacements at issue a linear dependence suffices. The CH units are heavy, so they are classical and the electrons follow adiabatically. That gives

$$H = -\sum_{n,s} t_{n+1,n}\,(c^\dagger_{n+1,s}c_{n,s} + \text{h.c.}) + \tfrac12 K\sum_n (u_{n+1}-u_n)^2 + \tfrac12 M\sum_n \dot u_n^2,\qquad t_{n+1,n} = t_0 - \alpha\,(u_{n+1}-u_n),$$

with $\alpha$ the electron–lattice coupling and $M$ the CH-unit mass. I deliberately leave the $\pi$–$\pi$ Coulomb repulsion out, absorbing it into a screened $t_0$ and $\alpha$; that is the model's known weak point, and if the on-site $U$ were dominant one would have to start from a large-$U$ description instead, but the small observed dimerization and the band-like optical gap say we are not in that regime.

Solve the perfectly dimerized chain first. With $u_n = (-1)^n u$ the hopping alternates as $t_0 \pm t_1$ with $t_1 = 2\alpha u$: dimerization literally means a strong bond and a weak bond. The alternating piece scatters $k$ into $k + \pi/a$, so in the reduced zone $-\pi/2a \le k \le \pi/2a$ the two folded states mix through a $2\times 2$ Hamiltonian with $\varepsilon_k = -2t_0\cos ka$ on the diagonal and $\Delta_k = 4\alpha u \sin ka$ off it, giving

$$E_k = \pm\sqrt{\varepsilon_k^2 + \Delta_k^2}.$$

The unperturbed bands cross zero at $k = \pm\pi/2a$, and $\Delta_k$ is *maximal* exactly there, so the gap opens precisely at the Fermi points, with band edges at $\pm 4\alpha u$ and a full gap

$$E_g = 2\Delta = 4t_1 = 8\alpha u .$$

Pinning $E_g \approx 1.4$ eV from the optical measurement fixes $t_1 = 0.35$ eV and the half-gap $\Delta = 0.70$ eV. Summing the doubly occupied valence band over the reduced zone and adding the elastic cost gives, with $z = t_1/t_0 = 2\alpha u/t_0$ and $E$ the complete elliptic integral,

$$E_0(u) = -\frac{4Nt_0}{\pi}\,E(1-z^2) + \frac{NKt_0^2 z^2}{2\alpha^2}.$$

The Peierls instability is now visible in a logarithm. For small $z$, $E(1-z^2) \cong 1 + \tfrac12\big(\ln(4/|z|) - \tfrac12\big)z^2$, so the electronic term contributes $-\,\propto z^2\ln(1/|z|)$ while the elastic cost is a plain $+z^2$; the derivative $z\,(2\ln(1/|z|)-1)$ of the former beats the linear derivative of the latter as $z \to 0$. Hence $u = 0$ is a local *maximum* and $E_0(u)$ is a symmetric double well with minima at $\pm u_0$ — the two degenerate bond-alternation vacua, which is the structural fact everything else hangs on. Minimizing with $K = 21$ eV/Å$^2$ and $4t_1 = 1.4$ eV gives $\alpha \approx 4.1$ eV/Å and $u_0 \approx 0.042$ Å, i.e. a bond-length change $\sqrt{3}\,u_0 \approx 0.073$ Å — a few percent of a bond, which retroactively justifies linearizing the hopping — and a condensation energy of only $-E_c/N \approx -0.015$ eV per CH. The shallowness of that well already suggests a cheap domain wall. For later use the perfect-chain density of states per spin is

$$\rho_0(E) = \frac{N}{\pi}\,\frac{|E|}{\sqrt{(4t_0^2-E^2)(E^2-\Delta^2)}},\qquad \Delta \le |E| \le 2t_0,$$

with square-root divergences at the band edges and a hard gap inside.

The right variable for a wall is not $u_n$ but the staggered order parameter $\psi_n = (-1)^n u_n$, which equals $-u_0$ in one phase, $+u_0$ in the other, and passes through zero at the centre of a wall. I take the one-parameter trial kink

$$\psi_n = u_0\tanh(n/\ell),$$

with $\ell$ the width in lattice units. The choice of $\tanh$ is not cosmetic: $E_0(u)$ *is* a symmetric double well in a real amplitude, i.e. a $\varphi^4$ potential, and the static kink of a real scalar in a $\varphi^4$ well is exactly $\tanh$, the profile that balances gradient energy against potential energy. This is also where the sine-Gordon reading has to be rejected. Interchanging double and single bonds looks like shifting a charge-density-wave phase by $\pi$, and a $\pi$ phase slip carries charge, but the order parameter here is a *real amplitude* sitting in one of two discrete minima, not the phase of a complex field that can wind. A real scalar with two degenerate minima is $\varphi^4$, not sine-Gordon; its kink is an amplitude distortion with no winding number and no built-in charge. The soliton is therefore intrinsically neutral, and whatever charge it ends up carrying must come from how electrons fill the levels it creates. That distinction is the hinge of the whole problem: it is what converts a would-be charged phase soliton into a neutral spin-$\tfrac12$ object.

To get the formation energy without diagonalizing an entire chain I treat the wall as a local change $\hat V$ of the hopping, $\hat V_{n,n+1} = t_0 + (-1)^n\alpha(\psi_{n+1}+\psi_n)$, relative to a perfectly dimerized reference, and use the Green's-function determinant for the shift of the filled sea,

$$\Delta E = \frac{2}{\pi}\int_{-\infty}^{0} \operatorname{Im}\ln\det\!\big[1 - G^0(\omega)\hat V\big]\,d\omega ,$$

with the chemical potential at mid-gap. The determinant has dimension equal only to the spatial extent of the perturbation — a $(2\nu+1)\times(2\nu+1)$ block with $2\nu+1$ of order $41$–$61$ — and converges fast. The reference Green's function follows in closed form from the Bloch states already found, e.g. $G^0_{nn}(\omega) = -i\omega/\sqrt{(4t_0^2-\omega^2)(\omega^2-\Delta^2)}$ in the band, and the three-segment reference (perfect A on one side, perfect B on the other, soliton region between) is built by decoupling the segments with an infinite diagonal potential and taking the limit. One trap has to be sidestepped here: a *single* wall on a finite chain leaves the two ends in different phases, so the computed energy is contaminated by an end-effect that depends on the boundary. I therefore compute a soliton–antisoliton pair, which returns both ends to the same phase, and halve the result once the two are far enough apart to be non-interacting. Minimizing over $\ell$ gives a shallow minimum at

$$\ell \approx 7,\qquad E_s \approx 0.4\ \text{eV}\quad (E_g = 1.4\ \text{eV}),$$

with $\ell \approx 9$ for a $1.0$-eV gap and $\ell \approx 5$ for a $2.0$-eV gap. The wall is therefore *diffuse*, spanning about $2\ell \approx 14$ sites rather than a single bond — which matters twice over, because a wide smooth wall is barely pinned by the discrete lattice and because it makes the mass small.

Now the electronic structure, where the spin and charge live. At the wall the gap parameter passes through zero, and a state is pulled out of the bands into the gap. At $E = 0$ the eigenvalue equation decouples the two sublattices (the diagonal vanishes and hopping connects only neighbours), leaving the two-step recursion $t_{n+1,n}\phi_0(n) + t_{n+1,n+2}\phi_0(n+2) = 0$; the normalizable branch lives entirely on one sublattice, $\phi_0(\text{odd}) = 0$ for a wall centred on an even site, and telescoping the ratio $\phi_0(n+2) = -(t_{n+1,n}/t_{n+2,n+1})\phi_0(n)$ of weak-side to strong-side hoppings makes it decay exponentially into both phases. For the smooth $\tanh$ wall the envelope is

$$\phi_0(n) \;\cong\; \frac{1}{\ell}\,\operatorname{sech}\!\Big(\frac{n}{\ell}\Big)\cos\!\Big(\frac{\pi n}{2}\Big),$$

peaked at the centre, decaying as a sech, with the $\cos(\pi n/2)$ factor simply selecting the even sublattice and putting nodes on the odd sites. That this level sits at *exactly* mid-gap is not an accident of the trial shape: the bipartite chain obeys $C^{-1}HC = -H$, a chiral/sublattice symmetry that pairs every state at $+E$ with one at $-E$, so the odd state out — the one trapped on the wall — has no partner and must sit at the self-conjugate point $E = 0$. It is a genuinely nonbonding level, neither bonding nor antibonding.

The charge and spin then follow from counting. The number of one-electron states is fixed by the number of sites, so the soliton can only redistribute them: $\int \Delta\rho(E)\,dE = 0$, and the chiral symmetry makes $\Delta\rho(E) = \Delta\rho(-E)$. The mid-gap level is a new $\delta$-function of unit weight per spin, so it must have been drawn *half from the valence band and half from the conduction band*. Locally the same bookkeeping is sharper: differencing the completeness sum rule site by site gives

$$2\int_{-\infty}^{-\Delta}\!\Delta\rho_{nn}(E)\,dE + |\phi_0(n)|^2 = 0,$$

i.e. at every site the electron density missing from the valence band is exactly compensated by the density of the mid-gap state. So one electron in $\phi_0$ makes the wall neutral *locally as well as globally*, and that electron is unpaired because the valence band is spin-paired. Filling $\phi_0$ with $0$, $1$ or $2$ electrons therefore gives

$$(Q,s) = (+e,0),\quad (0,\tfrac12),\quad (-e,0),$$

the reverse of the usual electron assignment: the neutral soliton carries spin, the charged solitons do not. The apparent conflict with Kramers' theorem — adding a spin-$\tfrac12$ object without changing the electron number — is resolved by the same pairing used for the energy: on a ring a soliton must be accompanied by an antisoliton, each removes half a state, and the two together remove one complete state, so the spin counting stays integral. The half-state per soliton is real but only resolves into integers globally.

The mass comes from letting the wall translate rigidly, $\psi_n(t) = u_0\tanh[(na - v_s t)/\ell a]$. Time-reversal symmetry says any change in the wall's *shape* under motion is $O(v_s^2)$ and cannot contribute at leading order, so a single width parameter suffices. Then $\tfrac12 M\sum_n \dot u_n^2 = \tfrac12 M (u_0^2 v_s^2/\ell^2a^2)\sum_n \operatorname{sech}^4(n/\ell)$, and with $\int \operatorname{sech}^4 y\,dy = 4/3$ the lattice sum is $\sum_n \operatorname{sech}^4(n/\ell) \approx (4/3)\ell$ for a wide wall, so that reading the result as $\tfrac12 M_s v_s^2$ gives

$$M_s = \frac{4}{3\ell}\Big(\frac{u_0}{a}\Big)^2 M \approx 5\,m_e .$$

A few electron masses — thousands of times lighter than a single carbon — because the mass is suppressed both by $(u_0/a)^2$, the displacements being tiny, and by $1/\ell$, the wall being wide so that few sites move much. This kinetic mass is the genuine dynamical mass; its equality with the inertial mass entering the soliton's equation of motion follows from a work–energy theorem for the wall. An object of a few $m_e$ must be treated quantum mechanically and will be extremely mobile, and sliding the rigid $\tanh$ across a lattice site changes the energy by only $\approx 0.002$ eV, a Peierls–Nabarro barrier so small that translation is nearly free down to $20$–$40$ K. That is the motionally narrowed ESR line, quantitatively.

The doping mechanism closes the loop. Adding a charge to the chain can be done two ways: put a carrier at a band edge, costing $\Delta$, or make a charged soliton, costing $E_s$. Since $E_s \approx 0.4$ eV (and $0.3$–$0.6$ eV across reasonable parameters) is below $\Delta = 0.70$ eV, soliton formation wins, so each dopant delivers its charge as a spinless charged soliton and the conductivity rises with no matching Curie contribution. In the dilute alloy that soliton stays bound to its parent dopant by the Coulomb attraction between its charge density $e|\phi_0(n)|^2$ and the impurity a distance $d$ off the chain,

$$\Delta E_I(n_s) = -\frac{e^2}{\varepsilon}\sum_n \frac{|\phi_0(n-n_s)|^2}{[(na)^2 + d^2]^{1/2}},$$

which with $\varepsilon \approx 10$ and $d \approx 2.4$ Å gives $E_b \approx 0.3$ eV — comparable to the measured low-temperature conductivity activation energy, as it should be if conduction requires unbinding the soliton from its dopant. Two further signatures follow without extra assumptions: the soliton oscillating against that impurity has $\hbar\omega_s = \hbar (K_b/M_s)^{1/2} \approx 0.07$ eV, sizable precisely because $M_s$ is so small, and it carries a large dipole oscillator strength since a full electronic charge is sloshing, which is a natural candidate for the $\sim 0.1$ eV infrared feature; and because $\psi_n \to 0$ at the wall centre, the C–C bonds there are intermediate between single and double, giving a stretch frequency between the two, consistent with the strong mode near $1370$ cm$^{-1}$ in lightly doped material.

Finally, the half-state bookkeeping is not a lattice artifact. In the continuum the $\pi$ electrons near $\pm k_F$ become a $1+1$-dimensional Dirac fermion whose mass is set by the order parameter, and the soliton is a kink across which that mass changes sign. The same conjugation symmetry $C^{-1}HC = -H$ pairs positive- and negative-energy states, leaving exactly one unpaired self-conjugate zero mode bound to the kink — the continuum echo of $\phi_0$ — and the regularized fermion number of a single isolated soliton is $\pm\tfrac12$. On the finite lattice the eigenvalues remain integer, $Q = 0, \pm e$; the $\tfrac12$ is an expectation value that sharpens into a fraction only when soliton and antisoliton are taken infinitely far apart and each keeps half of the one removed state. Fermion fractionalization, postulated in a field theory, is realized here by a domain wall in a real polymer.

Here is the worked evaluation of the final formulas for the standard parameter set — a check of the derived expressions, not a simulation: the gap-centre wavefunction built from its closed form and shown to live on the even sublattice, the lattice sum $\sum_n \operatorname{sech}^4(n/\ell) \to (4/3)\ell$, the soliton mass of about five electron masses, the doping comparison $E_s < \Delta$, and the charge–spin table read off the occupation of the mid-gap level.

```python
import numpy as np

# parameters fixed from independent data
t0   = 2.5            # eV  (pi bandwidth W = 4 t0 ~ 10 eV)
Eg   = 1.4            # eV  optical gap = 4 t1
t1   = Eg / 4.0       # eV  -> 0.35
Delta = 2 * t1        # eV  half-gap = 0.70
a    = 1.22           # Ang CH spacing along chain axis
u0   = 0.042          # Ang dimerization amplitude (from minimizing E0(u))
ell  = 7              # soliton half-width (sites), from minimizing E(ell)
M    = 13 * 1836.0    # CH-unit mass in electron masses (13 amu)

# mid-gap state  phi0(n) = (1/ell) sech(n/ell) cos(pi n/2)
def phi0(n, ell):
    return (1.0/ell) * (1.0/np.cosh(n/ell)) * np.cos(0.5*np.pi*n)

n   = np.arange(-60, 61)
phi = phi0(n, ell); phi /= np.sqrt(np.sum(phi**2))
print("weight on odd sites :", round(float(np.sum(phi[n % 2 == 1]**2)), 6))  # ~0

# lattice sum that sets the mass:  sum sech^4(n/ell) -> (4/3) ell
S = float(np.sum((1.0/np.cosh(n/ell))**4))
print("sum sech^4          :", round(S, 3), " vs (4/3)ell =", round(4*ell/3, 3))

# soliton mass  M_s = (4/3 ell)(u0/a)^2 M
M_s = (4.0/(3.0*ell)) * (u0/a)**2 * M
print("soliton mass M_s    :", round(M_s, 2), "m_e")

# doping channel
E_s = 0.42
print("E_s < Delta ?       :", E_s, "<", round(Delta, 2), "->", E_s < Delta)

# charge/spin from occupation of the mid-gap level
for occ, label in [(0, "empty"), (1, "single"), (2, "double")]:
    Q = 1 - occ; spin = 0.5 if occ == 1 else 0.0
    print(f"phi0 {label:6s}: Q = {Q:+d} e, s = {spin}")
```
