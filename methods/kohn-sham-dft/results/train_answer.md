What I want is mundane to state and brutal to deliver: the ground-state energy and density of $N$ interacting electrons sitting in an external potential $v(\mathbf r)$, accurate enough for real chemistry — bond lengths to about $0.01$ Å, binding energies to about $0.1$ eV — for molecules, solids, and alloys, not just for hydrogen. The honest object is the many-electron wavefunction $\Psi(\mathbf r_1,\dots,\mathbf r_N)$ and the honest method is Rayleigh–Ritz, $E = \min_\Psi \langle\Psi|H|\Psi\rangle$ with $H = -\tfrac12\sum_i\nabla_i^2 + \sum_i v(\mathbf r_i) + \tfrac12\sum_{i\neq j}1/|\mathbf r_i-\mathbf r_j|$. For H$_2$ it is glorious. But $\Psi$ lives in $3N$-dimensional configuration space, and to pin it to the accuracy I need takes on the order of $p^{3N}$ parameters with $p\approx3$–$10$. That exponent is the entire problem: invert it and the largest tractable system grows only logarithmically with computing power, so I reach $N\approx10$ optimistically, perhaps $N\approx20$ being clever, and for $N=100$ I would be minimizing in a space of dimension $\sim10^{150}$ — a $\Psi$ I could not even record, since storing it would take more bits than there are baryons in the universe. Past $N$ of order ten the wavefunction is simply not a usable variable. This is an exponential wall, and it is not a software problem; it is the dimensionality of configuration space itself.

The quantities I actually care about depend on only a few coordinates. The density $n(\mathbf r) = N\int|\Psi(\mathbf r,\mathbf r_2,\dots)|^2\,d\mathbf r_2\cdots d\mathbf r_N$ is a function of three variables no matter how large $N$ is, and $\int n\,d\mathbf r = N$. If I could write the energy as a functional of $n$ and minimize *that*, the wall would vanish. The crude version of this dream already sits on the table: take the kinetic, exchange, and electrostatic energies of a *uniform* electron gas, evaluate their densities at the local $n(\mathbf r)$, and integrate, giving the Thomas–Fermi(–Dirac) functional $E[n] = C_F\int n^{5/3} + \int v\,n + \tfrac12\iint n\,n'/|\mathbf r-\mathbf r'| - C_x\int n^{4/3}$. It is pure density, but its track record is fatal: it gives rough atomic totals and then predicts *no chemical binding at all* — the dissociated atoms always come out below the molecule, so it describes no bonds. The orbital-based self-consistent schemes do better but each is flawed in its own way: Hartree binds atoms and shows shell structure but omits exchange and correlation and is never tied to the true energy; Hartree–Fock treats exchange exactly but carries a non-local exchange operator, contains no correlation, and scales badly; Slater's $X\alpha$ replaces that non-local exchange by a cheap local potential $-3\alpha(3/8\pi\cdot n)^{1/3}$ but fixes $\alpha$ by an averaging *choice* rather than any energy principle. None is simultaneously exact-in-principle, cheap, and anchored to a variational statement of the true energy. The crucial clue is *where* Thomas–Fermi fails: the electrostatic term is exact classical electrostatics, the external term $\int v\,n$ is exact, and the one structural difference between Thomas–Fermi and the orbital schemes that bind is how the kinetic energy is treated — a local $n^{5/3}$ versus a Laplacian acting on orbitals. So the kinetic energy is the term whose local modeling is lethal.

I propose Kohn–Sham density functional theory. It rests on three load-bearing moves. First, the density legitimately contains enough information — this is not a hope but a theorem. Suppose two external potentials $v_1$ and $v_2$, differing by more than a constant, gave the same ground-state density $n$, with ground states $\Psi_1,\Psi_2$ and energies $E_1,E_2$. Use $\Psi_2$ as a trial function for $H_1$; since $\Psi_1$ is the nondegenerate ground state of $H_1$, strict Rayleigh–Ritz gives $E_1 < \langle\Psi_2|H_1|\Psi_2\rangle = E_2 + \int[v_1-v_2]\,n\,d\mathbf r$, because $H_1-H_2$ is the multiplicative operator $v_1-v_2$ whose expectation in any state of density $n$ is $\int(v_1-v_2)n$. Doing the symmetric thing, $E_2 \le E_1 + \int[v_2-v_1]\,n\,d\mathbf r$. The two density integrals are exact negatives, so adding the inequalities yields $E_1+E_2 < E_1+E_2$ — a contradiction. The density therefore fixes $v$ up to a constant, hence $H$, hence $\Psi$ and every observable. So there is a *universal* functional $F[n] = \langle\Psi[n]|T+U|\Psi[n]\rangle$ with no $v$ in it, and an exact variational principle $E_v[n] = \int v\,n\,d\mathbf r + F[n] \ge E_0$, with equality at the true ground-state density. The $3N$-dimensional minimization has become a three-dimensional one.

The catch is that $F[n]$ is unknown, and unpacking it sends me back to $3N$-dimensional wavefunctions. The naive escape — model $T[n]$ locally as $C_F\int n^{5/3}$ — rebuilds Thomas–Fermi and its missing bonds. So the second move is to refuse to model the kinetic energy locally and instead compute its dominant part exactly. There exists a fictitious system of *non-interacting* electrons that I can tune to have the *same* density $n$; for it the kinetic energy is exact and easy — solve single-particle equations, fill the lowest states, sum the orbital kinetic energies. Call it $T_s[n]$, an implicit functional of $n$ that nonetheless carries the Laplacian honestly. The true interacting kinetic energy differs by $T_c[n] = T[n]-T_s[n]$, which is *small* compared to $T_s$, where in Thomas–Fermi the *whole* kinetic energy was being mangled. I reorganize $F[n]$ to split off the two terms I control exactly or classically — $T_s[n]$ and the Hartree energy $U_H[n] = \tfrac12\iint n\,n'/|\mathbf r-\mathbf r'|$ — and shovel everything else into a single remainder,
$$F[n] = T_s[n] + U_H[n] + E_{xc}[n], \qquad E_{xc}[n] \equiv (T[n]-T_s[n]) + (U[n]-U_H[n]),$$
so the exchange-correlation energy $E_{xc}$ is the kinetic correlation plus the non-classical part of the electron–electron interaction (exchange from antisymmetry, plus Coulomb correlation). This is still *exact* — I have only named things — but the unknown is now isolated in a small term while the two large terms are under control. That is the structural bet: model a small term crudely rather than a large one crudely.

Minimizing $E_v[n] = \int v\,n + T_s[n] + U_H[n] + E_{xc}[n]$ at fixed $\int n = N$ with multiplier $\mu$ gives the Euler–Lagrange condition $\delta T_s/\delta n + v(\mathbf r) + \int n(\mathbf r')/|\mathbf r-\mathbf r'|\,d\mathbf r' + \delta E_{xc}/\delta n = \mu$. Collect the three density-dependent pieces into one effective potential $v_{\rm eff}(\mathbf r) = v(\mathbf r) + \int n(\mathbf r')/|\mathbf r-\mathbf r'|\,d\mathbf r' + v_{xc}(\mathbf r)$ with $v_{xc} \equiv \delta E_{xc}/\delta n$, so the condition reads $\delta T_s/\delta n + v_{\rm eff} = \mu$. Now observe that a *genuinely* non-interacting system in some potential $v_s$ has energy functional $\int v_s\,n + T_s[n]$ and stationarity condition $\delta T_s/\delta n + v_s = \mu$ — *identical in form*. Choosing $v_s = v_{\rm eff}$ makes them literally the same equation with the same minimizing density. So the density of my interacting system is reproduced exactly by non-interacting electrons moving in $v_{\rm eff}$, and I never touch $\delta T_s/\delta n$: I just solve the auxiliary single-particle problem. These are the Kohn–Sham equations, in Hartree atomic units,
$$\left(-\tfrac12\nabla^2 + v_{\rm eff}(\mathbf r)\right)\varphi_i(\mathbf r) = \varepsilon_i\varphi_i(\mathbf r), \qquad n(\mathbf r) = \sum_i^{\rm occ}|\varphi_i(\mathbf r)|^2,$$
$$v_{\rm eff}(\mathbf r) = v(\mathbf r) + \int \frac{n(\mathbf r')}{|\mathbf r-\mathbf r'|}\,d\mathbf r' + v_{xc}(\mathbf r),\qquad v_{xc} = \frac{\delta E_{xc}}{\delta n},$$
with the $N$ lowest eigenstates occupied (the Pauli ground state of the auxiliary system). They are self-consistent — $v_{\rm eff}$ depends on $n$ through the Hartree and xc terms, $n$ depends on $v_{\rm eff}$ through the orbitals — so I guess $n$, build $v_{\rm eff}$, diagonalize, rebuild $n$, and iterate. This is exact in principle: with the exact $E_{xc}$ and $v_{xc}$ these single-particle equations deliver the exact interacting density and energy. Two sanity checks confirm the architecture. Drop $E_{xc}$ and $v_{xc}$ and $v_{\rm eff}$ collapses to nucleus-plus-Hartree, recovering the orbital-based self-consistent equations exactly — so those were never ad hoc; they are the no-exchange-no-correlation limit of a formally exact theory. Approximate even $T_s$ by the local gas value and I fall back to Thomas–Fermi. The total energy I take not as $\sum_i\varepsilon_i$, which double-counts, but from $\sum_i\varepsilon_i = T_s[n] + \int v_{\rm eff}\,n$; substituting $\int v_{\rm eff}\,n = \int v\,n + 2U_H + \int v_{xc}\,n$ into $E_v$ and cancelling gives
$$E = \sum_i \varepsilon_i - \tfrac12\iint \frac{n(\mathbf r)n(\mathbf r')}{|\mathbf r-\mathbf r'|}\,d\mathbf r\,d\mathbf r' - \int v_{xc}(\mathbf r)\,n(\mathbf r)\,d\mathbf r + E_{xc}[n],$$
the band sum minus the Hartree double-counting, minus the xc double-counting, plus $E_{xc}$.

The third move makes it computable: approximate only $E_{xc}$, and do it locally. From the one solved many-body system — the uniform electron gas with exchange-correlation energy per electron $e_{xc}(n)$ — pretend each little volume is locally a piece of uniform gas at the local density. This is the local density approximation, $E_{xc}^{\rm LDA}[n] = \int e_{xc}(n(\mathbf r))\,n(\mathbf r)\,d\mathbf r$ with $v_{xc}^{\rm LDA}(\mathbf r) = d[n\,e_{xc}(n)]/dn|_{n=n(\mathbf r)}$. The exchange part is analytic, since the uniform-gas exchange energy density is known in closed form:
$$E_x^{\rm LDA}[n] = -\tfrac34\left(\tfrac3\pi\right)^{1/3}\int n(\mathbf r)^{4/3}\,d\mathbf r,\qquad v_x^{\rm LDA}(\mathbf r) = -\left(\tfrac3\pi\right)^{1/3}n(\mathbf r)^{1/3}.$$
This pins down what Slater left floating: comparing $(3/\pi)^{1/3}$ against the whole-Fermi-sphere coefficient $3(3/8\pi)^{1/3}$, their ratio is $\tfrac13\cdot8^{1/3} = \tfrac23$, so $v_x^{\rm LDA} = \tfrac23\,v_x^{\rm Slater,\alpha=1}$ — the functional derivative *forces* $\alpha=2/3$ rather than leaving it a knob. The correlation part $e_c(n)$ is a known uniform-gas number from interpolation (e.g. Wigner's $e_c\approx-0.44/(r_s+7.8)$ in atomic units, $r_s = (3/4\pi n)^{1/3}$), improvable later, so $E_{xc}^{\rm LDA}$ has no free parameters once the gas data are parameterized.

I should justify why a uniform-gas model works for an argon atom, whose density swings over an order of magnitude within a Fermi wavelength and is about as far from slowly varying as it gets. The answer is in what $E_{xc}$ physically is. Put one electron at $\mathbf r$; Pauli and Coulomb repulsion deplete the other electrons nearby, and the exchange-correlation hole $n_{xc}(\mathbf r,\mathbf r') = g(\mathbf r,\mathbf r') - n(\mathbf r')$ measures that depletion. The electron screens itself, so exactly one electron's worth of charge is missing: $\int n_{xc}(\mathbf r,\mathbf r')\,d\mathbf r' = -1$, the sum rule. To connect $E_{xc}$ to the hole, use the adiabatic connection: scale the interaction $U\to\lambda U$ for $\lambda\in[0,1]$ while adjusting the confining potential $v_\lambda$ so the density stays equal to the physical $n$ for all $\lambda$ — at $\lambda=0$ this is the non-interacting auxiliary system, at $\lambda=1$ the physical one, all sharing one density. By the Hellmann–Feynman theorem $dE_\lambda/d\lambda = \langle\Psi_\lambda|dH_\lambda/d\lambda|\Psi_\lambda\rangle$, and since the potential's $\lambda$-dependence is a one-body multiplicative operator acting on a fixed density, integrating from $0$ to $1$ assembles $F[n] = T_s[n] + \int_0^1\langle U\rangle_\lambda\,d\lambda$. Subtracting $U_H$ and writing $\langle U\rangle_\lambda$ as the interaction of the density with its hole gives the formally exact
$$E_{xc}[n] = \tfrac12\int d\mathbf r\int d\mathbf r'\,\frac{n(\mathbf r)\,\bar n_{xc}(\mathbf r,\mathbf r')}{|\mathbf r-\mathbf r'|},\qquad \bar n_{xc} = \int_0^1 n_{xc}^\lambda\,d\lambda,$$
each electron interacting with its own coupling-averaged hole, and since every $\lambda$-hole integrates to $-1$ so does the average. Because the Coulomb kernel is isotropic, this integral depends mainly on the hole's *spherical average* and *normalization*, not its detailed angular shape. The LDA hole is the uniform-gas hole, which has exactly the right normalization even when its shape is wrong, and the Coulomb integral is forgiving about shape — that is why LDA gives good $E_{xc}$ for atoms it has no right to describe, and why its overestimate of exchange and underestimate of correlation cancel *systematically* rather than by luck. It is also a warning: any "improvement" that breaks the sum rule should be expected to do worse, so the fruitful refinements are the ones built to respect it.

To prove the equations actually drive a computer to a converged density, I take the simplest setting that exercises every piece: spin-paired electrons in a one-dimensional harmonic well $v(x)=x^2$, atomic units, exchange-only LDA (correlation slots in identically as another additive piece of $e_{xc}$ with its own potential). I represent $T_s$, i.e. $-\tfrac12\,d^2/dx^2$, by the $(-1,2,-1)/h^2$ second-difference stencil on a uniform grid; the real symmetric Hamiltonian gives real orthonormal Kohn–Sham orbitals and energies from `eigh`. Each orbital is normalized to one electron's worth, the lowest states are filled by Aufbau with occupation $2$, and $n(x)=\sum_n f_n|\varphi_n(x)|^2$. Exchange is $E_x=-\tfrac34(3/\pi)^{1/3}\int n^{4/3}$ with $v_x=-(3/\pi)^{1/3}n^{1/3}$. In 1D the bare $1/|x-x'|$ diverges on the diagonal, a numerical artifact, so I soften the Hartree kernel to $1/\sqrt{(x-x')^2+1}$; in 3D one uses the bare Coulomb kernel. I assemble $v_{\rm eff}=v_{\rm ext}+v_H+v_x$, diagonalize, mix old and new densities for stability, iterate, and evaluate the total energy as the band sum minus the double-counting exactly as derived, $E=\sum_n f_n\varepsilon_n - E_H - \int n\,v_x\,dx + E_x$. The loop converges to a stable energy and the density integrates to exactly $N$ electrons — the auxiliary non-interacting system faithfully reproduces the requested density, with the kinetic energy carried honestly by the orbitals and only the small exchange-correlation term approximated.

```python
import numpy as np

def build_kinetic(x):                         # T_s: -1/2 d^2/dx^2 (finite diff.)
    n = len(x); h = x[1] - x[0]
    lap = (np.diag(np.full(n, 2.0))
           + np.diag(np.full(n-1, -1.0), 1)
           + np.diag(np.full(n-1, -1.0), -1)) / h**2
    return 0.5 * lap

def density(psi_gn, occ, x):                  # n(x) = sum_n f_n |phi_n|^2
    h = x[1] - x[0]; n = np.zeros_like(x)
    for i, f in enumerate(occ):
        if f:
            psi = psi_gn[:, i]
            psi = psi / np.sqrt(np.sum(psi**2) * h)
            n += f * psi**2
    return n

def exchange_lda(n, x):                        # E_x = -(3/4)(3/pi)^(1/3) int n^(4/3)
    h = x[1] - x[0]; c = (3.0/np.pi)**(1.0/3.0)
    E_x = -(3.0/4.0) * c * np.sum(n**(4.0/3.0)) * h
    v_x = -c * n**(1.0/3.0)                     # v_x = -(3/pi)^(1/3) n^(1/3)
    return E_x, v_x

def hartree(n, x):                             # classical electrostatics (1D-softened)
    h = x[1] - x[0]
    K = 1.0 / np.sqrt((x[:, None] - x[None, :])**2 + 1.0)
    v_H = K @ n * h
    return 0.5 * np.sum(n * v_H) * h, v_H

def occupations(num_electrons, num_states):    # Aufbau: 2 electrons per state
    occ = np.zeros(num_states); e = num_electrons; i = 0
    while e > 0 and i < num_states:
        occ[i] = min(2, e); e -= occ[i]; i += 1
    return occ

def solve_ks(x, v_ext, num_electrons, iters=200, mix=0.3, tol=1e-8):
    T = build_kinetic(x); h = x[1] - x[0]
    n = np.zeros_like(x); occ = occupations(num_electrons, len(x)); E_old = None
    for _ in range(iters):
        E_x, v_x = exchange_lda(n, x)
        E_H, v_H = hartree(n, x)
        v_eff = v_ext + v_H + v_x                          # v_eff = v + v_H + v_xc
        eps, psi_gn = np.linalg.eigh(T + np.diag(v_eff))   # KS single-particle eqn
        n = (1 - mix) * n + mix * density(psi_gn, occ, x)  # density mixing
        # E = sum f_n eps_n - E_H - int n v_xc dx + E_xc  (double-counting fix)
        E = np.sum(occ * eps[:len(occ)]) - E_H - np.sum(n * v_x) * h + E_x
        if E_old is not None and abs(E - E_old) < tol:
            break
        E_old = E
    return E, eps, n

if __name__ == "__main__":
    x = np.linspace(-8, 8, 401); v_ext = x**2
    for Ne in (2, 6):
        E, eps, n = solve_ks(x, v_ext, Ne); h = x[1] - x[0]
        print(f"N={Ne}: E={E:.5f} Ha, N_check={np.sum(n)*h:.4f}, "
              f"eps[:3]={np.round(eps[:3],4)}")
```
