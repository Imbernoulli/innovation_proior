# The coupled-cluster ansatz: exp(T) on a reference determinant

## Problem

Recover the electron-correlation energy `ΔE = E_exact − E_HF` missed by the Hartree–Fock mean field, **accurately, at usable cost, and — decisively — in a size-extensive way**: the energy must scale correctly with system size, so that `E(A···B) = E(A) + E(B)` for separated fragments and reaction energies are meaningful. Truncated configuration interaction (CISD) fails this: it cannot represent the *simultaneous* correlation of two separated electron pairs (a quadruple excitation that is really a *product* of two doubles), so its error grows with system size.

## Key idea

Parameterize the correlated wavefunction by an **exponential** of a connected cluster operator acting on the HF determinant,

```
|Ψ⟩ = e^{T} |Φ0⟩ ,    T = T1 + T2 + T3 + … ,
T1 = Σ_ia t_i^a a†_a a_i ,   T2 = ¼ Σ_ijab t_ij^ab a†_a a†_b a_j a_i ,  …
```

The exponential automatically generates the disconnected (factorizable) higher excitations as *products* of low ones — e.g. `½T2²` supplies the dominant disconnected quadruples with amplitude `t_ij^ab · t_kl^cd` — so the genuinely independent unknowns are only the **connected** cluster amplitudes, and size-extensivity is structural: `e^{T_A + T_B} = e^{T_A}e^{T_B}` factorizes for non-interacting fragments. The cluster–CI relations follow by matching `e^T = 1 + C`:

```
C1 = T1 ,   C2 = T2 + ½T1² ,   C3 = T3 + T1T2 + (1/3!)T1³ ,   C4 = T4 + ½T2² + … .
```

## Working equations (derivation)

Do **not** treat the ansatz variationally (`⟨Φ0|e^{T†}He^T|Φ0⟩` never terminates). Instead substitute into `H|Ψ⟩ = E|Ψ⟩`, left-multiply by `e^{−T}`, and project. With the **similarity-transformed Hamiltonian**

```
H̄ = e^{−T} H e^{T} ,
```

the energy and amplitude equations are

```
ΔE = ⟨Φ0| H̄ |Φ0⟩ ,        0 = ⟨Φ_ij…^ab…| H̄ |Φ0⟩   (one per amplitude).
```

**Termination (the crux).** By the Hausdorff/BCH expansion

```
H̄ = H + [H,T] + ½[[H,T],T] + (1/3!)[[[H,T],T],T] + (1/4!)[[[[H,T],T],T],T] ,
```

and this is **exact, with no further terms**, because (i) `T` is a pure excitation operator and commutes with itself, so each nested `T` must contract with `H` (not with another `T`); and (ii) `H` has at most a two-body operator (four legs), so at most four `T`'s can contract with it — the fifth commutator vanishes. The equations are thus a **finite polynomial** (degree ≤ 4) in the amplitudes even though `e^T` is infinite.

**Connectedness ⇒ extensivity.** The same commutator structure removes every term in which `H` and a `T` share no index, so `H̄ = (H e^T)_C` is **connected**; the amplitudes are connected, the wavefunction is linked, and the energy is size-extensive by construction (an algebraic re-derivation of the Brueckner–Goldstone linked-diagram theorem; Hubbard 1957).

**Coupled-cluster doubles (CCD) — the simplest complete model.** Keep `T = T2`. With the normal-ordered Hamiltonian `H_N = F_N + W_N` (canonical HF: `f_pq = ε_p δ_pq`), the energy is

```
ΔE = ¼ Σ_ijab ⟨ij||ab⟩ t_ij^ab ,
```

and the doubles equation `⟨Φ_ij^ab|(H_N e^{T2})_C|Φ0⟩ = 0`, with the quadratic `t·t` terms folded into intermediates, is

```
I_oo(m,i)     = f_mi + ½ Σ_nef ⟨mn||ef⟩ t_ef^in
I_vv(a,e)     = f_ae − ½ Σ_mnf ⟨mn||ef⟩ t_af^mn
I_voov(a,m,i,e) = ⟨am||ie⟩ + ½ Σ_nf ⟨mn||ef⟩ t_af^in
I_oooo(m,n,i,j) = ⟨mn||ij⟩ + ½ Σ_ef ⟨mn||ef⟩ t_ef^ij

R_ij^ab = ⟨ab||ij⟩
        + ½ I_vv(a,e) t_ij^eb  − ½ I_oo(m,i) t_mj^ab
        + I_voov(a,m,i,e) t_mj^eb
        + ⅛ ⟨ab||ef⟩ t_ij^ef + ⅛ I_oooo(m,n,i,j) t_mn^ab ,
   then antisymmetrize R over a↔b and i↔j; at convergence R_ij^ab = 0.
```

The linear pieces are the **particle–particle ladder** `½⟨ab||cd⟩t_ij^cd`, the **hole–hole ladder** `½⟨kl||ij⟩t_kl^ab`, and the **particle–hole ring** `P(ij)P(ab)⟨kb||cj⟩t_ik^ac`; the quadratic pieces (inside the intermediates) are the disconnected-quadruple contributions that make it an infinite-order resummation. Solve by Jacobi iteration off the orbital-energy denominators:

```
t_ij^ab ← R_ij^ab / D_ij^ab ,   D_ij^ab = ε_i + ε_j − ε_a − ε_b ,
```

starting from `t = 0` — the **first iteration is MP2** (`t_ij^ab = ⟨ij||ab⟩/D`), and the iteration resums the doubles to convergence. Cost is dominated by the particle–particle ladder at `∼ n²N⁴` — the same scaling as CISD, but size-extensive and summed to infinite order. Adding `T1` and projecting on singles gives the singles-and-doubles model (CCSD).

**Exactness check.** For a two-electron system there are no genuine triples/quadruples, so `T = T2` is the *complete* cluster operator and, with a canonical HF reference (`f_ai = 0`, so singles may be dropped), CCD = full CI exactly.

## Final artifact: a self-contained connected-cluster doubles solver

Grounded in the standard spin-orbital CCD residual (intermediate factorization à la Stanton–Bartlett; cf. the psi4numpy / miniccpy reference codes). Given the Fock matrix `f`, antisymmetrized integrals `g[p,q,r,s] = ⟨pq||rs⟩`, and occupied/virtual slices `o, v`, it iterates to the correlation energy. The worked example builds a consistent minimal two-orbital (four spin-orbital), two-electron model and confirms CCD reproduces full CI to machine precision.

```python
import numpy as np
from itertools import combinations

# ---- the connected-cluster doubles equation ----

def doubles_residual(t2, f, g, o, v):
    """⟨Φ_ij^ab| (H_N e^{T2})_C |Φ0⟩.  t2 indexed [a,b,i,j]; g = <pq||rs>."""
    I_oo   = f[o, o] + 0.5 * np.einsum("mnef,efin->mi", g[o, o, v, v], t2)
    I_vv   = f[v, v] - 0.5 * np.einsum("mnef,afmn->ae", g[o, o, v, v], t2)
    I_voov = g[v, o, o, v] + 0.5 * np.einsum("mnef,afin->amie", g[o, o, v, v], t2)
    I_oooo = g[o, o, o, o] + 0.5 * np.einsum("mnef,efij->mnij", g[o, o, v, v], t2)

    r  = 0.5 * np.einsum("ae,ebij->abij", I_vv, t2)             # particle-particle
    r -= 0.5 * np.einsum("mi,abmj->abij", I_oo, t2)             # hole-hole
    r += np.einsum("amie,ebmj->abij", I_voov, t2)              # particle-hole ring
    r += 0.125 * np.einsum("abef,efij->abij", g[v, v, v, v], t2)   # pp ladder
    r += 0.125 * np.einsum("mnij,abmn->abij", I_oooo, t2)         # hh ladder

    r -= np.transpose(r, (1, 0, 2, 3))   # antisymmetrize a<->b
    r -= np.transpose(r, (0, 1, 3, 2))   # antisymmetrize i<->j
    r += g[v, v, o, o]                   # driver <ab||ij>
    return r

def ccd_energy(t2, g, o, v):
    return 0.25 * np.einsum("ijab,abij->", g[o, o, v, v], t2)

def solve_ccd(f, g, o, v, maxit=200, tol=1e-12):
    eps = np.diagonal(f); n = np.newaxis
    inv_D = 1.0 / (-eps[v, n, n, n] - eps[n, v, n, n]
                   + eps[n, n, o, n] + eps[n, n, n, o])
    nv, no = f[v, v].shape[0], f[o, o].shape[0]
    t2 = np.zeros((nv, nv, no, no))      # t = 0  -> first step is MP2
    e_old = 0.0
    for _ in range(maxit):
        t2 = t2 + doubles_residual(t2, f, g, o, v) * inv_D
        e = ccd_energy(t2, g, o, v)
        if abs(e - e_old) < tol:
            break
        e_old = e
    return t2, e

# ---- worked example: 2 electrons, 4 spin-orbitals (CCD must equal full CI) ----

def build_model():
    nso, nocc = 4, 2
    spat = [0, 0, 1, 1]      # spatial index (g,g,u,u); 0,1 occ ; 2,3 vir
    spin = [0, 1, 0, 1]
    hg, hu = -1.2528, -0.4756                       # 1e MO energies
    chem = np.zeros((2, 2, 2, 2))                   # chemist (ab|cd), spatial
    Jgg, Juu, Jgu, Kgu = 0.6746, 0.6975, 0.6636, 0.1813
    chem[0,0,0,0]=Jgg; chem[1,1,1,1]=Juu; chem[0,0,1,1]=Jgu; chem[1,1,0,0]=Jgu
    for (p,q,r,s) in [(0,1,0,1),(1,0,1,0),(0,1,1,0),(1,0,0,1)]: chem[p,q,r,s]=Kgu
    def phys(p,q,r,s):
        return chem[spat[p],spat[r],spat[q],spat[s]] \
               if spin[p]==spin[r] and spin[q]==spin[s] else 0.0
    g = np.zeros((nso,)*4)
    for p in range(nso):
     for q in range(nso):
      for r in range(nso):
       for s in range(nso):
        g[p,q,r,s] = phys(p,q,r,s) - phys(p,q,s,r)   # <pq||rs>
    h = np.diag([hg, hg, hu, hu])
    occ = [0, 1]
    f = h.copy()
    for p in range(nso):
     for q in range(nso):
      f[p,q] += sum(g[p,i,q,i] for i in occ)          # Fock = h + mean field
    return f, g, h, slice(0,nocc), slice(nocc,nso)

def full_ci_corr(h, g, nso=4):
    dets = list(combinations(range(nso), 2))
    def elem(d1, d2):
        s1, s2 = set(d1), set(d2); diff = s1 ^ s2
        l1, l2 = sorted(d1), sorted(d2)
        if len(diff) == 0:
            return sum(h[i,i] for i in d1) + 0.5*sum(g[i,j,i,j] for i in d1 for j in d1)
        if len(diff) == 2:
            (m,)=s1-s2; (p,)=s2-s1; common=sorted(s1 & s2)
            e = h[m,p] + sum(g[m,j,p,j] for j in common)
            return (-1)**(l1.index(m)+l2.index(p)) * e
        if len(diff) == 4:
            a,b = sorted(s1-s2); r,s = sorted(s2-s1)
            return (-1)**(l1.index(a)+l1.index(b)+l2.index(r)+l2.index(s)) * g[a,b,r,s]
        return 0.0
    H = np.array([[elem(a,b) for b in dets] for a in dets]); H = (H+H.T)/2
    w = np.linalg.eigvalsh(H)
    e_ref = elem(dets[0], dets[0])
    return w[0] - e_ref

if __name__ == "__main__":
    f, g, h, o, v = build_model()
    _, e_ccd = solve_ccd(f, g, o, v)
    e_fci = full_ci_corr(h, g)
    print("CCD corr energy : %.10f" % e_ccd)     # -0.0205709294
    print("FCI corr energy : %.10f" % e_fci)     # -0.0205709294
    print("CCD == FCI (2e) :", abs(e_ccd - e_fci) < 1e-9)   # True
```

Running it prints `CCD corr energy = FCI corr energy = −0.0205709294` Hartree and `CCD == FCI (2e): True` — the connected-cluster doubles equations are exact wherever the cluster expansion is complete.
