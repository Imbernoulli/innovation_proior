I can solve the mean-field problem exactly: a converged self-consistent field hands me a single Slater determinant $\Phi_0$ of orthonormal spin-orbitals, each electron gliding through the averaged Coulomb field of all the others. And that answer is wrong, always in the same direction — too high. The true Hamiltonian carries the *instantaneous* repulsion $\sum_{p<q}1/r_{pq}$, two electrons feeling each other's actual positions, while the mean field has replaced it with each electron feeling only the smeared-out average density of the rest. So the electrons in my determinant do not dodge one another the way real electrons do. That avoidance — the correlation hole — lowers the true energy below the mean-field energy, and the gap is the correlation energy. It is a sliver, perhaps a percent of the total, but it is the percent that decides whether a bond forms, how high a barrier sits, which isomer wins. I cannot leave it on the table, and I want it for an *arbitrary* molecule without launching a new bespoke calculation each time.

The routes I already know do not fit. The variational one — write $\Psi = c_0\Phi_0 + \sum c_i^a\Phi_i^a + \sum c_{ij}^{ab}\Phi_{ij}^{ab}+\cdots$ and diagonalize $\hat H$ in that determinant basis (configuration interaction) — is exact in the orbital basis and gives a rigorous upper bound, but the number of determinants explodes combinatorially with the electron count, and the moment I truncate it I lose size consistency: two non-interacting fragments computed together stop returning the sum of their separate energies. For a method whose whole reason to exist is to scale to many electrons, a tool that misbehaves as I add electrons is self-defeating. The Hylleraas route — put $r_{12}$ explicitly into the trial function and minimize the correlation-energy functionals — is genuinely accurate for helium and the two-electron ions, but it is hand-tailored to two electrons, with no turnkey many-electron version, and it is costly. What I actually want is a *general, non-iterative, transferable* correction that takes the mean-field answer I already paid for and simply corrects it.

The phrase "small correction to something I can already solve" is the definition of perturbation theory, so that is where I go. The method is **second-order Møller–Plesset perturbation theory (MP2)**, and its single load-bearing choice is the partition of the Hamiltonian. Rayleigh–Schrödinger theory splits $\hat H = \hat H_0 + \lambda\hat V$ with $\hat H_0$ soluble and $\hat V$ small, expands in powers of $\lambda$, and gives the second-order energy as a sum over the *other* eigenstates of $\hat H_0$, which is useless unless I can enumerate those eigenstates for a many-electron system. The escape is to stop inventing a model $\hat H_0$ by hand and use the soluble one-particle problem I already solved — the Fock equations $\hat F\psi_i = \varepsilon_i\psi_i$. I take the zeroth-order Hamiltonian to be the sum of one-electron Fock operators,

$$\hat H_0 = \sum_p \hat F(p),\qquad \hat V = \hat H - \hat H_0 = \sum_{p<q}\frac{1}{r_{pq}} - \sum_p\sum_i^{\rm occ}\big[\hat J_i(p)-\hat K_i(p)\big].$$

This choice does two things at once. First, because $\hat H_0$ is a one-electron operator diagonal in the canonical orbital set, *every* Slater determinant built from those spin-orbitals is an eigenfunction of it, with eigenvalue equal to the sum of its occupied orbital energies — the reference $\Phi_0$ has $E^{(0)} = \sum_i^{\rm occ}\varepsilon_i$, a single excitation $\Phi_i^a$ has $E^{(0)}-\varepsilon_i+\varepsilon_a$, a double $\Phi_{ij}^{ab}$ has $E^{(0)}-\varepsilon_i-\varepsilon_j+\varepsilon_a+\varepsilon_b$. The complete enumerable set of excited states that the second-order sum demanded is just the set of substituted determinants, with eigenvalues I can read off by inspection. Second, the perturbation $\hat V$ is forced to be *instantaneous repulsion minus its mean-field average* — the fluctuation of the true interaction about the field each electron already sits in. It is small in exactly the right sense: the bulk of the repulsion has been absorbed into $\hat H_0$, leaving only the fluctuation as the perturbation. The naive alternative of taking $\hat H_0 = \sum_p\hat h(p)$, the bare cores, would make $\hat V$ the entire $1/r_{pq}$ — enormous, no perturbation at all — and would not even leave $\Phi_0$ an eigenstate of $\hat H_0$. Putting the mean field into $\hat H_0$ is what both shrinks the perturbation and keeps the reference as the zeroth-order state.

Turning the crank confirms the partition is the good one. Zeroth order is $E^{(0)} = \sum_i\varepsilon_i$. First order is $E^{(1)} = \langle\Phi_0|\hat V|\Phi_0\rangle = -\tfrac12\sum_{ij}\langle ij\|ij\rangle$, so $E^{(0)}+E^{(1)} = \sum_i\varepsilon_i - \tfrac12\sum_{ij}\langle ij\|ij\rangle = E_{\rm HF}$ — through first order the expansion reproduces *only* the Hartree–Fock energy and not one bit of correlation. That is not a disappointment but the central feature: it means the mean-field answer is correct to first order, the expansion is anchored at the best single determinant, and the very first thing it does beyond that is correlate. The leading correlation correction is therefore the second-order term

$$E^{(2)} = \sum_{\mu\neq 0}\frac{|\langle\Phi_0|\hat V|\Phi_\mu\rangle|^2}{E^{(0)}-E_\mu},$$

and three facts collapse this apparently huge sum to a single channel. Since $\hat H_0$ is diagonal in determinants, $\langle\Phi_0|\hat V|\Phi_\mu\rangle = \langle\Phi_0|\hat H|\Phi_\mu\rangle$ for excited $\Phi_\mu$. Triples and higher then vanish identically: the Slater–Condon rules say a Hamiltonian carrying only one- and two-body operators cannot connect determinants differing in three or more spin-orbitals — physically, one application of $\hat H$ can disturb at most two electrons. The singles vanish too, because $\langle\Phi_0|\hat H|\Phi_i^a\rangle = \langle i|\hat h|a\rangle + \sum_j^{\rm occ}\langle ij\|aj\rangle$ is exactly the off-diagonal Fock matrix element $\langle i|\hat F|a\rangle$, which is zero in the canonical basis where $\hat F$ is diagonal — this is the variational stationarity of the HF determinant restated, the same fact that makes the reference optimal against single substitutions. Only the doubles survive, with matrix element equal to the antisymmetrized two-electron integral $\langle\Phi_0|\hat H|\Phi_{ij}^{ab}\rangle = \langle ij\|ab\rangle$ and denominator $E^{(0)}-E_{ij}^{ab} = \varepsilon_i+\varepsilon_j-\varepsilon_a-\varepsilon_b$, which is negative because occupied energies lie below virtual ones, so every term lowers the energy as a correlation correction must. Counting each distinct double once (an unordered occupied pair $\{i,j\}$ into an unordered virtual pair $\{a,b\}$), or equivalently letting all four indices run free and dividing by the four-fold overcounting, gives the closed spin-orbital correlation energy

$$E_{\rm MP2} = \frac14\sum_{ij}^{\rm occ}\sum_{ab}^{\rm vir}\frac{\big|\langle ij\|ab\rangle\big|^2}{\varepsilon_i+\varepsilon_j-\varepsilon_a-\varepsilon_b}\;=\;\sum_{i<j}\sum_{a<b}\frac{\big|\langle ij\|ab\rangle\big|^2}{\varepsilon_i+\varepsilon_j-\varepsilon_a-\varepsilon_b}\;\le\;0,$$

with $\langle pq|rs\rangle = \iint\psi_p^*(1)\psi_q^*(2)\,r_{12}^{-1}\,\psi_r(1)\psi_s(2)\,d1\,d2$ and $\langle pq\|rs\rangle = \langle pq|rs\rangle - \langle pq|sr\rangle$. Each term reads as a pair of electrons in occupied orbitals $i,j$ scattering into the empty pair $a,b$, weighted by how strongly they interact and how cheap the virtual excitation is — electrons borrowing excited-orbital character to get out of each other's way, the correlation hole written as virtual admixture. Because it is a sum of independent pair contributions rather than a truncated CI, it is size-extensive: separate a system into non-interacting fragments and the integrals connecting them vanish, leaving the energy additive.

For production I integrate out the spin. In a closed-shell molecule each spatial orbital is doubly occupied, and the two-electron integral $\langle ij|ab\rangle$ survives only when the spins match across it; enumerating the spin cases, opposite-spin pairs keep only the Coulomb term while same-spin pairs keep both Coulomb and exchange. Carrying the $\tfrac14$ and the spin sums through assembles, in chemists' notation $(ia|jb) = \iint\phi_i^*(1)\phi_a(1)\,r_{12}^{-1}\,\phi_j^*(2)\phi_b(2)$, into the closed-shell form

$$E_{\rm MP2} = \sum_{ij}^{\rm occ}\sum_{ab}^{\rm vir}\frac{(ia|jb)\,\big[\,2(ia|jb)-(ib|ja)\,\big]}{\varepsilon_i+\varepsilon_j-\varepsilon_a-\varepsilon_b},$$

over *spatial* orbitals — roughly an order of magnitude less work — where the "$2$" is the opposite-spin direct Coulomb and the "$-(ib|ja)$" is the same-spin exchange, the Pauli hole layered on the Coulomb hole. The energy contraction itself is $O(N^4)$, but the real cost lives upstream in transforming the two-electron integrals from the atomic-orbital basis $(\mu\nu|\lambda\sigma)$ that the SCF produces into the molecular-orbital basis $(ia|jb)$. Done as one nested contraction this is a hopeless $O(N^8)$; done as four successive quarter-transforms, each summing a single index against a four-index array, it is $4\,O(N^5)$, and that four-step transformation sets the formal $O(N^5)$ scaling of the whole method.

The code comes in two forms — first the transparent spin-orbital implementation that makes the boxed formula literal, then the production closed-shell version with the four-quarter transform and the spin-summed numerator, vectorized:

```python
import numpy as np

def antisymmetrized_spin_integrals(teimo, n_spatial):
    """<pq||rs> = <pq|rs> - <pq|sr> over spin orbitals. `teimo(p,q,r,s)` returns a
    SPATIAL MO integral in chemists' order; an integral survives only when spins
    line up across it."""
    dim = 2 * n_spatial
    V = np.zeros((dim, dim, dim, dim))
    for p in range(1, dim + 1):
        for q in range(1, dim + 1):
            for r in range(1, dim + 1):
                for s in range(1, dim + 1):
                    coulomb  = teimo((p+1)//2, (r+1)//2, (q+1)//2, (s+1)//2) \
                               * (p % 2 == r % 2) * (q % 2 == s % 2)
                    exchange = teimo((p+1)//2, (s+1)//2, (q+1)//2, (r+1)//2) \
                               * (p % 2 == s % 2) * (q % 2 == r % 2)
                    V[p-1, q-1, r-1, s-1] = coulomb - exchange
    return V

def mp2_spin_orbital(teimo, E_spatial, Nelec, n_spatial):
    V   = antisymmetrized_spin_integrals(teimo, n_spatial)
    dim = 2 * n_spatial
    eps = np.array([E_spatial[i // 2] for i in range(dim)])
    e2  = 0.0
    for i in range(Nelec):                  # occupied
        for j in range(Nelec):
            for a in range(Nelec, dim):     # virtual
                for b in range(Nelec, dim):
                    e2 += 0.25 * V[i, j, a, b] ** 2 / (eps[i] + eps[j] - eps[a] - eps[b])
    return e2
```

```python
import numpy as np

def mp2_closed_shell(I_ao, C, eps, ndocc):
    """Closed-shell MP2 correlation energy from a converged RHF solution.
    I_ao : (nbf,)*4 AO integrals (mu nu | la si); C : MO coeffs; eps : orbital
    energies (ascending); ndocc : doubly-occupied spatial orbitals."""
    Cocc, Cvir   = C[:, :ndocc], C[:, ndocc:]
    e_occ, e_vir = eps[:ndocc], eps[ndocc:]

    # AO -> MO as four O(N^5) quarter-transforms (not one O(N^8) step).
    t    = np.einsum('pi,pqrs->iqrs', Cocc, I_ao, optimize=True)
    t    = np.einsum('qa,iqrs->iars', Cvir, t,    optimize=True)
    t    = np.einsum('rj,iars->iajs', Cocc, t,    optimize=True)
    I_mo = np.einsum('sb,iajs->iajb', Cvir, t,    optimize=True)   # (ia|jb)

    d = (e_occ.reshape(-1,1,1,1) - e_vir.reshape(1,-1,1,1)
         + e_occ.reshape(1,1,-1,1) - e_vir.reshape(1,1,1,-1))      # e_i - e_a + e_j - e_b

    e_os = np.einsum('iajb,iajb,iajb->', I_mo, I_mo,                 1.0/d, optimize=True)
    e_ss = np.einsum('iajb,iajb,iajb->', I_mo, I_mo - I_mo.swapaxes(1,3), 1.0/d, optimize=True)
    return e_os + e_ss          # total MP2 correlation energy (<= 0)
```

On the smallest real test, HeH$^+$ in a minimal STO-3G basis — two electrons, two spatial orbitals — with HF orbital energies $\varepsilon = (-1.5238, -0.2676)$ and the spatial MO two-electron integrals, the spin-orbital routine reduces to the single double excitation of both electrons into the virtual pair and returns $E_{\rm MP2} = -0.00640$ Hartree: a small negative correlation correction added on top of the HF energy, evaluated in a single non-iterative pass that reuses the existing HF orbitals and integrals — exactly the profile I was after.
