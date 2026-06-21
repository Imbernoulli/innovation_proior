# Second-order Møller–Plesset perturbation theory (MP2)

## Problem

Hartree–Fock (HF) gives the best single Slater determinant, with each electron in the mean field of the others. It systematically misses the **correlation energy** — the energy lowering from electrons avoiding one another instantaneously, which the mean field averages away. MP2 supplies the leading correlation correction as a single, non-iterative, size-extensive formula built entirely from quantities the converged HF calculation already provides.

## Key idea

Apply Rayleigh–Schrödinger perturbation theory with the **sum of one-electron Fock operators** as the zeroth-order Hamiltonian:

$$\hat H_0 = \sum_p \hat F(p),\qquad \hat V = \hat H - \hat H_0 = \sum_{p<q}\frac{1}{r_{pq}} - \sum_p\sum_i^{\rm occ}\big[\hat J_i(p)-\hat K_i(p)\big].$$

Then $\hat H_0$ has *every* Slater determinant over the canonical HF spin-orbitals as an eigenfunction (eigenvalue = sum of occupied orbital energies), supplying the complete enumerable set that second-order PT requires. The perturbation $\hat V$ is the **fluctuation potential**: the instantaneous repulsion minus its mean-field average — precisely the correlation that HF discards.

## Derivation (summary)

- **Zeroth order:** $E^{(0)} = \sum_i^{\rm occ}\varepsilon_i$.
- **First order:** $E^{(1)} = \langle\Phi_0|\hat V|\Phi_0\rangle = -\tfrac12\sum_{ij}\langle ij\|ij\rangle$, and
$$E^{(0)} + E^{(1)} = \sum_i\varepsilon_i - \tfrac12\sum_{ij}\langle ij\|ij\rangle = E_{\rm HF}.$$
  Through first order the expansion reproduces only the HF energy; the first correlation correction is therefore **second order** (and the first-order correction to the charge density likewise vanishes).
- **Which determinants contribute to $E^{(2)} = \sum_{\mu\neq 0}|\langle\Phi_0|\hat V|\Phi_\mu\rangle|^2/(E^{(0)}-E_\mu)$:**
  - *Triples and higher* → 0, by the Slater–Condon rules (the Hamiltonian has only one- and two-body terms, so it cannot connect determinants differing in $\ge 3$ spin-orbitals).
  - *Singles* $\Phi_i^a$ → 0, because $\langle\Phi_0|\hat H|\Phi_i^a\rangle = \langle i|\hat h|a\rangle + \sum_j\langle ij\|aj\rangle = \langle i|\hat F|a\rangle = \varepsilon_a\delta_{ia} = 0$ in the canonical basis (HF stationarity).
  - *Doubles* $\Phi_{ij}^{ab}$ survive, with $\langle\Phi_0|\hat H|\Phi_{ij}^{ab}\rangle = \langle ij\|ab\rangle$ and $E^{(0)}-E_{ij}^{ab} = \varepsilon_i+\varepsilon_j-\varepsilon_a-\varepsilon_b < 0$.

Notation: $\langle pq|rs\rangle = \iint \psi_p^*(1)\psi_q^*(2)\,r_{12}^{-1}\,\psi_r(1)\psi_s(2)\,d1\,d2$ (physicists'), and the antisymmetrized integral $\langle pq\|rs\rangle = \langle pq|rs\rangle - \langle pq|sr\rangle$.

## Final formula

**Spin-orbital form** (occupied $i,j$; virtual $a,b$):

$$E_{\rm MP2} = \frac14\sum_{ij}^{\rm occ}\sum_{ab}^{\rm vir}\frac{\big|\langle ij\|ab\rangle\big|^2}{\varepsilon_i+\varepsilon_j-\varepsilon_a-\varepsilon_b}\;=\;\sum_{i<j}\sum_{a<b}\frac{\big|\langle ij\|ab\rangle\big|^2}{\varepsilon_i+\varepsilon_j-\varepsilon_a-\varepsilon_b}\;\le\;0.$$

**Closed-shell (RHF) spatial-orbital form** in chemists' notation $(ia|jb) = \iint\phi_i^*(1)\phi_a(1)\,r_{12}^{-1}\,\phi_j^*(2)\phi_b(2)$:

$$E_{\rm MP2} = \sum_{ij}^{\rm occ}\sum_{ab}^{\rm vir}\frac{(ia|jb)\,\big[\,2(ia|jb)-(ib|ja)\,\big]}{\varepsilon_i+\varepsilon_j-\varepsilon_a-\varepsilon_b},$$

splitting into an opposite-spin part $\sum (ia|jb)^2/(\cdots)$ (direct Coulomb only) and a same-spin part $\sum (ia|jb)[(ia|jb)-(ib|ja)]/(\cdots)$ (Coulomb plus exchange, the Pauli hole).

The energy contraction is $O(N^4)$; the cost is dominated by the AO→MO transformation of the two-electron integrals, done as **four successive $O(N^5)$ quarter-transforms** rather than one $O(N^8)$ contraction, giving MP2 its formal $O(N^5)$ scaling. Because the correction is a sum of independent pair contributions, it is **size-extensive**.

## Code

Transparent spin-orbital implementation (the boxed formula made literal):

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

Production closed-shell (RHF) implementation with the four-quarter integral transform and the spin-summed numerator, vectorized:

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

## Worked example

HeH$^+$ in a minimal STO-3G basis (2 electrons, 2 spatial orbitals): with HF orbital energies $\varepsilon = (-1.5238, -0.2676)$ and the spatial MO two-electron integrals, the spin-orbital routine reduces to the single double excitation of both electrons into the virtual pair and returns

$$E_{\rm MP2} = -0.00640\ \text{Hartree},$$

a small negative correlation correction added on top of the HF energy — illustrating the sign ($\le 0$), the non-iterative single-pass evaluation, and the reuse of the existing HF orbitals and integrals.
