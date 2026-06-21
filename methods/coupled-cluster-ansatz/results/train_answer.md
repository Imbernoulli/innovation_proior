Hartree–Fock hands me a single determinant $|\Phi_0\rangle$ and roughly ninety-nine percent of the electronic energy, but the missing one percent — the correlation energy $\Delta E = E_{\text{exact}} - E_{\text{HF}}$ — is where all the chemistry lives: bond energies, reaction barriers, the differences between molecules that I actually want to compute. The mean field misses the fact that electrons avoid each other instantaneously, and recovering that motion is the whole problem. The honest linear fix is configuration interaction: the exact state lives in the space of all determinants buildable from my finite orbital basis, so write $|\Psi_{\text{CI}}\rangle = (1 + C_1 + C_2 + \cdots)|\Phi_0\rangle$ where $C_1$ admixes all singles, $C_2$ all doubles, and so on, with coefficients fixed by making $\langle\Psi|H|\Psi\rangle$ stationary. Keep everything and this is full CI — exact, variational, but dead on arrival because the determinant count is $\sim (nN)^n$. So in practice I truncate to singles and doubles (CISD), since $H$ is only a two-body operator and so connects determinants differing by at most two orbitals.

And here is exactly where it breaks, and the precise failure is the whole clue. Take two closed-shell fragments $A$ and $B$ pulled infinitely far apart so nothing couples them; the right answer is $E(A\cdots B) = E(A) + E(B)$. But to correlate $A$ and $B$ *simultaneously* — the physically correct joint state — I need a determinant carrying a double excitation on $A$ *and at the same time* a double on $B$, which is a *quadruple* excitation of the combined system. CISD contains no quadruples, so it is forced to leave one fragment uncorrelated in any given determinant; the pair energy comes out strictly above $E(A) + E(B)$, and the error *grows with system size* and never vanishes. For $N$ non-interacting copies the per-copy correlation energy shrinks with $N$. This is not a numerical blemish — it means I cannot tabulate truncated-CI energies and subtract them to get a reaction energy. Truncated CI is not size-extensive, and size-extensivity is the one property I cannot do without. If I look at where the contamination enters the eigenvalue equations, I find renormalization terms of the form $-\Delta E\,\langle\Psi^{(1)}|\Psi^{(1)}\rangle$ in which $\Delta E$ scales like $n$ and $\langle\Psi^{(1)}|\Psi^{(1)}\rangle$ also scales like $n$, so a quantity that should scale like $n$ is poisoned by a term scaling like $n^2$. In full CI those bad terms are cancelled exactly by contributions descending from the higher excitations I threw away; truncate, and the cancellation dies. This is the linked-diagram theorem talking (Brueckner 1955 for the low-order cancellation, Goldstone 1957 to all orders): the exact correlation energy is a sum of *linked* diagrams only, and unlinked diagrams scale wrongly with $n$. Many-body perturbation theory respects this and is size-extensive at every order, but it is a *finite-order* snapshot — MBPT(2), MBPT(3), MBPT(4) — that does not resum a class of effects to infinite order, and for strong correlation a low order simply is not accurate enough. So I am stuck between two unsatisfactory poles: variational truncated CI that is not extensive, and extensive MBPT that is not resummed.

Let me stare at *what kind* of quadruples CISD was missing, because the failure points at the fix. Not some intricate genuine four-electron correlation — I needed exactly pair-on-$A$ *times* pair-on-$B$, a *product* of two double excitations whose amplitude is not an independent new number but just the product of the two pair amplitudes. This is general: for any molecule with two electron pairs in well-separated regions, the leading four-electron effect is the two pairs correlating independently and simultaneously, a *disconnected* quadruple. Counting confirms it — two pair amplitudes carry $\sim n^2N^2$ independent numbers each, and their product introduces nothing new, whereas a genuine connected four-electron cluster is $\sim n^4N^4$ independent numbers and is physically much smaller. The higher excitations that matter are *factorizable*. So I should not be adding quadruples as independent linear terms $C_4$; I should be *generating* them automatically as products. The mathematical object that turns a sum into automatic products of itself is the **exponential**.

I propose the coupled-cluster ansatz: parameterize the correlated wavefunction as the exponential of a connected cluster operator acting on the reference,
$$|\Psi\rangle = e^{T}|\Phi_0\rangle,\qquad T = T_1 + T_2 + T_3 + \cdots,$$
with
$$T_1 = \sum_{ia} t_i^a\, a_a^\dagger a_i,\qquad T_2 = \tfrac14\sum_{ijab} t_{ij}^{ab}\, a_a^\dagger a_b^\dagger a_j a_i,\ \ldots$$
Watch what the exponential does. Expanding $e^T = 1 + (T_1 + T_2 + \cdots) + \tfrac12(T_1 + T_2 + \cdots)^2 + \cdots$, the $\tfrac12 T_2^2$ term is a product of two double-excitation operators, so acting on $|\Phi_0\rangle$ it produces *quadruple* excitations with amplitude $t_{ij}^{ab}\,t_{kl}^{cd}$ — exactly the disconnected quadruple I needed for the two separated fragments, generated for free with no new unknowns. Matching to the CI picture by writing $e^T = 1 + C$ reads off the cluster decomposition of the CI coefficients, $C_1 = T_1$, $C_2 = T_2 + \tfrac12 T_1^2$, $C_3 = T_3 + T_1T_2 + \tfrac1{3!}T_1^3$, $C_4 = T_4 + \tfrac12 T_2^2 + T_1T_3 + \tfrac12 T_1^2 T_2 + \tfrac1{4!}T_1^4$: the genuinely independent unknowns are only the *connected* amplitudes $T_1, T_2, \ldots$, and everything disconnected is determined as products. The separated-fragments disaster now cannot happen even if I keep only $T_2$, because $e^{T_2} = 1 + T_2 + \tfrac12 T_2^2 + \cdots$ already contains the product quadruples, hextuples, and all higher even excitations, and $e^{T_A + T_B} = e^{T_A}e^{T_B}$ factorizes for non-interacting fragments since $T_A$ and $T_B$ commute (disjoint orbital sets). Size-extensivity is structural, forced before I write a single equation. This is the cluster idea the nuclear physicists had been circling — Coester 1958, Coester and Kümmel 1960 wrote the nuclear-matter wavefunction as an exponentiated correlation operator on a model state, an Ursell-type expansion — but they had only the representation, not closed finite equations for a real interacting many-fermion system, which is the gap I close.

The way to get equations is *not* variational: $\langle\Phi_0|e^{T^\dagger}H e^{T}|\Phi_0\rangle$ sandwiches $H$ between an infinite series of de-excitations and excitations and never terminates into a finite expression. Instead I project the Schrödinger equation. Left-multiplying $H e^T|\Phi_0\rangle = E\, e^T|\Phi_0\rangle$ by $e^{-T}$ defines the **similarity-transformed Hamiltonian**
$$\bar H = e^{-T} H\, e^{T},$$
which is non-Hermitian but preserves the spectrum, so it carries the same $E$. Projecting onto the reference gives the energy $\langle\Phi_0|\bar H|\Phi_0\rangle = E$, and projecting onto each excited determinant gives one amplitude equation $\langle\Phi_{ij\ldots}^{ab\ldots}|\bar H|\Phi_0\rangle = 0$ — exactly as many equations as unknowns, a closed nonlinear system. The crux is that this system is *finite*. Expand $\bar H$ by the Hausdorff/BCH series,
$$\bar H = H + [H,T] + \tfrac1{2!}[[H,T],T] + \tfrac1{3!}[[[H,T],T],T] + \tfrac1{4!}[[[[H,T],T],T],T],$$
and *no further terms appear*. The reason is twofold: $T$ is a pure excitation operator (only particle creators and hole annihilators), so it commutes with itself, which forces each nested $T$ in a surviving commutator to contract with $H$ rather than with another $T$; and $H$ has at most a two-body operator — four second-quantized legs — so at most four $T$'s can share an index with it, and the fifth commutator vanishes. The infinite exponential is tamed into a polynomial of degree at most four in the amplitudes, purely because the Hamiltonian is two-body. The same commutator structure also subtracts off every term in which $H$ and a $T$ have no index in common, so $\bar H = (H e^T)_C$ is *connected*; the amplitudes are connected, the wavefunction is linked, and extensivity is built in — an algebraic re-derivation of the Brueckner–Goldstone linked-diagram theorem (Hubbard 1957), now as an identity about a similarity transform.

To make the bookkeeping honest I work in second quantization with the Fermi vacuum $|\Phi_0\rangle$ as reference and the particle–hole reinterpretation ($a_a^\dagger, a_i$ create excitations; $a_a, a_i^\dagger$ destroy them), use normal ordering $\{\cdots\}$ so $\langle\Phi_0|\{\cdots\}|\Phi_0\rangle = 0$, and apply Wick's theorem (Wick 1950) to collapse vacuum expectation values to their fully contracted terms. Normal-ordering the bare Hamiltonian and subtracting $\langle\Phi_0|H|\Phi_0\rangle = E_{\text{HF}}$ gives the normal-ordered Hamiltonian $H_N = F_N + W_N$ with $F_N = \sum_{pq} f_{pq}\{a_p^\dagger a_q\}$ and $W_N = \tfrac14\sum_{pqrs}\langle pq||rs\rangle\{a_p^\dagger a_q^\dagger a_s a_r\}$, where the Fock matrix $f_{pq} = \langle p|h|q\rangle + \sum_i\langle pi||qi\rangle$ emerges as the contraction of the two-body term against the occupied orbitals — the mean field reappears rather than being assumed. For a canonical HF reference $f_{pq} = \varepsilon_p\delta_{pq}$ is diagonal, the occupied–virtual block $f_{ai}$ vanishes by Brillouin's theorem, and the projected equations become $\Delta E = \langle\Phi_0|(H_N e^T)_C|\Phi_0\rangle$ and $0 = \langle\Phi_{ij\ldots}^{ab\ldots}|(H_N e^T)_C|\Phi_0\rangle$.

The simplest complete realization keeps $T = T_2$ alone — coupled-cluster doubles (CCD). Singles describe orbital relaxation but, by Brillouin's theorem for a canonical HF reference, do not couple to the reference at first order, so the cheapest honest model drops them; with $n^2N^2$ amplitudes $t_{ij}^{ab}$ it is size-extensive because $\tfrac12 T_2^2$ supplies the disconnected quadruples and all higher even excitations. Working out the energy projection, the only surviving fully contracted term is $\langle\Phi_0|W_N T_2|\Phi_0\rangle$, giving the clean result
$$\Delta E = \tfrac14\sum_{ijab}\langle ij||ab\rangle\, t_{ij}^{ab}.$$
The amplitude projection $\langle\Phi_{ij}^{ab}|(H_N e^{T_2})_C|\Phi_0\rangle = 0$ has three groups of terms — $H_N$ alone (the driver $\langle ij||ab\rangle$ that makes $t$ nonzero), $H_N T_2$ (linear), and $\tfrac12 H_N T_2^2$ (quadratic, where the BCH series stops because two $T_2$'s already saturate the four legs of $W_N$). The linear terms are the **particle–particle ladder** $\tfrac12\sum_{cd}\langle ab||cd\rangle t_{ij}^{cd}$, the **hole–hole ladder** $\tfrac12\sum_{kl}\langle kl||ij\rangle t_{kl}^{ab}$, the **particle–hole ring** $P(ij)P(ab)\sum_{kc}\langle kb||cj\rangle t_{ik}^{ac}$, and the Fock pieces that for canonical HF become the orbital-energy denominator. The quadratic $t\cdot t$ terms are precisely the disconnected-quadruple contributions folded back into the doubles equation — they are what make this a true infinite-order resummation rather than a finite perturbation order, and what keeps it extensive. Folding the quadratic terms into reusable intermediates gives a compact residual,
$$
\begin{aligned}
I_{oo}(m,i) &= f_{mi} + \tfrac12\sum_{nef}\langle mn||ef\rangle t_{ef}^{in}, &
I_{vv}(a,e) &= f_{ae} - \tfrac12\sum_{mnf}\langle mn||ef\rangle t_{af}^{mn},\\
I_{voov}(a,m,i,e) &= \langle am||ie\rangle + \tfrac12\sum_{nf}\langle mn||ef\rangle t_{af}^{in}, &
I_{oooo}(m,n,i,j) &= \langle mn||ij\rangle + \tfrac12\sum_{ef}\langle mn||ef\rangle t_{ef}^{ij},
\end{aligned}
$$
$$R_{ij}^{ab} = \langle ab||ij\rangle + \tfrac12 I_{vv}(a,e)t_{ij}^{eb} - \tfrac12 I_{oo}(m,i)t_{mj}^{ab} + I_{voov}(a,m,i,e)t_{mj}^{eb} + \tfrac18\langle ab||ef\rangle t_{ij}^{ef} + \tfrac18 I_{oooo}(m,n,i,j)t_{mn}^{ab},$$
antisymmetrized over $a\leftrightarrow b$ and $i\leftrightarrow j$, with $R_{ij}^{ab} = 0$ at convergence. Because the intermediates each carry one $t$, multiplying them by another $t$ in $R$ reproduces the quadratic quadruple-product contributions while evaluating them once and reusing them, keeping the cost at $\sim n^2N^4$. I solve the nonlinear system not by quasi-Newton but by a fixed-point Jacobi iteration: pull the diagonal Fock part out as the orbital-energy denominator and update
$$t_{ij}^{ab} \leftarrow R_{ij}^{ab}/D_{ij}^{ab},\qquad D_{ij}^{ab} = \varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b,$$
starting from $t = 0$. The very first pass gives $R = \langle ab||ij\rangle$ and the update $t_{ij}^{ab} = \langle ij||ab\rangle/D$ — that is MP2 — so the method is a self-consistent resummation that starts at MP2 and iterates the doubles to infinite order, at the same $\sim n^2N^4$ scaling as CISD but size-extensive. Adding $T_1$ and projecting also on singles gives the singles-and-doubles model (CCSD). I accept that the projected equations are non-variational — $\bar H$ is non-Hermitian and the energy is not an upper bound — because that is the price of the BCH termination, and finite, connected, extensive equations are worth more for consistent energy differences than a variational bound. The decisive check: for a two-electron system there are no genuine triples or quadruples, so $T = T_2$ is the *complete* cluster operator and, with a canonical HF reference where singles may be dropped, CCD must equal full CI exactly — and a minimal two-orbital, four-spin-orbital model confirms the two correlation energies agree to machine precision, pinning down every sign and factor.

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
