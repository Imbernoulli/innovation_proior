I set out to find the field dependence of the magnetization inside a ferromagnetic domain. I have a lattice of atomic spins that line up below the Curie point because of the exchange interaction, I apply a static field $H$ along the magnetization, and I want $M(T,H)$ — in particular the intrinsic susceptibility $\partial M/\partial H$, carried all the way up to large fields where the dipolar physics should show through. The aligned state is trivial; everything interesting lives in the low-lying excitations, populated by ordinary statistical mechanics, with the field and the dipole–dipole term carried along. So the real object is the spectrum of the Heisenberg Hamiltonian $H = -J\sum_{\langle ij\rangle} \mathbf{S}_i\cdot\mathbf{S}_j + g\mu_B H\sum_i S_i^z$ (plus dipolar terms), with $J>0$.

The trouble is that the exchange term delocalizes excitations. Writing the dot product in ladder form, $\mathbf{S}_i\cdot\mathbf{S}_j = S_i^z S_j^z + \tfrac{1}{2}(S_i^+ S_j^- + S_i^- S_j^+)$, the fully aligned state $|\text{all }S\rangle$ is an exact eigenstate with energy $-\sum_{\langle ij\rangle} JS^2$, because every flip term $S^+S^-$ contains an $S^+$ acting on an already-maximally-raised spin and dies. But flip one spin down by a unit and the term $S_n^- S_{n+1}^+$ *moves* that flip to the neighbor: a localized flip is not stationary, it hops. Bloch saw that what hops with equal amplitude to each neighbor on a periodic lattice is diagonalized by a plane wave, $|k\rangle = N^{-1/2}\sum_n e^{i\mathbf{k}\cdot\mathbf{r}_n}|\text{flip at }n\rangle$, giving the single-excitation energy $\varepsilon_k = 2JS(1-\cos kd) \approx JS(kd)^2$ and, after statistical population, the famous $T^{3/2}$ fall-off of the spontaneous magnetization. That is exact and beautiful — but only for *one* excitation. It is a counting of an explicit one-flip eigenbasis, not an operator framework. Two excitations force hand bookkeeping of the on-site occupancy bound (a spin-$S$ can be reversed at most $2S$ times) and of two-magnon bound states; and the moment I add the Zeeman term and the dipole–dipole coupling — which is the actual problem — the convenient single-flip basis stops being the natural object and I am re-diagonalizing from scratch for each new term. There is no built-in expansion parameter and no canonical-coordinate structure: $\mathbf{S}_i\cdot\mathbf{S}_j$ is a product of non-commuting operators, not a set of oscillators I can fill with quanta. I want a framework, not a sequence of bespoke diagonalizations.

So I stare at the excitations themselves. The number of spin deviations from full alignment at a site, $n_i = S - S_i^z$, is a non-negative integer bounded above by $2S$, additive across sites, each unit carrying spin $-1$. Integer occupation, additive, non-negative — that is exactly the spectrum of a boson number operator $a^\dagger a$. This is the discovery I propose: the **Holstein–Primakoff transformation**, representing each spin by a boson and letting the su(2) algebra fix the rest. I set $S_i^z = S - a_i^\dagger a_i$ with $[a_i, a_j^\dagger] = \delta_{ij}$, identify the boson vacuum $n=0$ with the aligned ground state (counting deviations downward from the top, $m = S - n$, not upward from the bottom, so that the vacuum is the physical $+z$-aligned state and not the unphysical fully-reversed one), and then *derive* $S^\pm$ rather than guess them. The only non-negotiable constraint is that the spin commutators come out exactly. From $S^+|m\rangle = \sqrt{S(S+1)-m(m+1)}\,|m+1\rangle$, substituting $m = S-n$ and expanding $S(S+1)-(S-n)(S-n+1) = n(2S-(n-1))$, I get $S^+|n\rangle = \sqrt{n}\,\sqrt{2S-(n-1)}\,|n-1\rangle$. The $\sqrt{n}$ is precisely the boson annihilation $a|n\rangle = \sqrt{n}|n-1\rangle$; the leftover $\sqrt{2S-(n-1)}$ is the operator $\sqrt{2S - a^\dagger a}$ evaluated on the lowered state. Hence

$$S_i^z = S - a_i^\dagger a_i,\qquad S_i^+ = \sqrt{2S - a_i^\dagger a_i}\,\;a_i = \sqrt{2S}\,\sqrt{1 - \frac{a_i^\dagger a_i}{2S}}\;a_i,\qquad S_i^- = a_i^\dagger\,\sqrt{2S - a_i^\dagger a_i} = \sqrt{2S}\,a_i^\dagger\sqrt{1 - \frac{a_i^\dagger a_i}{2S}}.$$

Operator ordering matters: in $S^+$ the $a$ annihilates first, then the square root evaluates on the lowered state; $S^-$ is the Hermitian conjugate. That square-root factor $\sqrt{2S - a^\dagger a}$ does two jobs at once, and they are the same job. First, truncation: acting on the fully reversed state, $S^- |2S\rangle = a^\dagger\sqrt{2S-2S}\,|2S\rangle = 0$, so the boson Fock space is automatically cut off at $n=2S$ to match the $(2S+1)$-dimensional spin space — a bare $\sqrt{2S}\,a^\dagger$ would happily create bosons into states that do not exist for a real spin. Second, the algebra is *exact*. Writing $\hat n = a^\dagger a$, $f = \sqrt{2S-\hat n}$, and using $[\hat n, a] = -a$, $aa^\dagger = \hat n+1$: $[S^z, S^+] = -[\hat n, fa] = -f[\hat n, a] = fa = S^+$; and for the hard one, $[S^+, S^-] = [fa, a^\dagger f] = f a a^\dagger f - a^\dagger(2S-\hat n)a = (\hat n+1)(2S-\hat n) - \hat n(2S-\hat n+1) = 2(S-\hat n) = 2S^z$, holding for *all* $0\le n\le 2S$. So this is not an approximation but an exact rewriting of the non-commuting spin algebra in ordinary boson language, with the entire spin constraint packed into one square root.

The payoff is that the square root can be expanded. In the ordered phase at low temperature the deviation density is tiny, $\langle\hat n\rangle \ll S$ (and for large $S$ the ratio $\langle\hat n\rangle/2S$ is small anyway), so $\sqrt{1 - a^\dagger a/2S} \approx 1 - a^\dagger a/4S$, and to leading order $S^+ \approx \sqrt{2S}\,a$, $S^- \approx \sqrt{2S}\,a^\dagger$, $S^z = S - a^\dagger a$. Substituting into the Hamiltonian turns the product of spins into a quadratic form in $a, a^\dagger$ plus a controlled $1/S$ series of corrections. Organizing by powers of $S$: the $O(S^2)$ piece is the classical ground-state energy $-NqJS^2/2$ ($N$ sites, coordination $q$); the $O(S)$ piece is the free-magnon Hamiltonian, and the $O(S^0)$ leftovers (four-boson terms like $\hat n_i \hat n_j$) are the magnon–magnon interactions $H_2$. So

$$H = -\frac{NqJS^2}{2} + H_1 + H_2 + O(1/S),\qquad H_1 = JS\sum_{\langle ij\rangle}(a_i^\dagger - a_j^\dagger)(a_i - a_j) = -JS\sum_{\langle ij\rangle}(a_i^\dagger a_j + a_j^\dagger a_i) + qJS\sum_i a_i^\dagger a_i.$$

The difference form makes $H_1$ manifestly the cost of a *variation* in deviation between neighbors — a wave's stiffness — and manifestly positive. This is exactly what Bloch's counting could not provide: not just the leading spectrum but a controlled framework whose corrections are systematically smaller, in operator form, onto which the field and dipolar terms attach without redoing anything. Translational invariance means momentum is good, so Fourier transforming $a_j = N^{-1/2}\sum_k e^{i\mathbf{k}\cdot\mathbf{r}_j}a_k$ (which preserves $[a_k, a_{k'}^\dagger] = \delta_{kk'}$) diagonalizes $H_1$ into a free gas of bosonic magnons:

$$H_1 = \sum_k \varepsilon_k\, a_k^\dagger a_k,\qquad \varepsilon_k = qJS(1-\gamma_k),\qquad \gamma_k = \frac{1}{q}\sum_\delta \cos(\mathbf{k}\cdot\boldsymbol{\delta}).$$

For a 1D chain ($\boldsymbol{\delta} = \pm a$), $\gamma_k = \cos ka$ and $\varepsilon_k = 2JS(1-\cos ka) \approx JS(ka)^2$ — Bloch's dispersion, now dropping out as the diagonal form of a free-boson Hamiltonian rather than from a hand diagonalization. The leading interaction is $H_2 = (J/4)\sum_{\langle ij\rangle}[a_i^\dagger a_j^\dagger(a_i-a_j)^2 + (a_i^\dagger-a_j^\dagger)^2 a_i a_j]$. The linear map is accurate exactly where it is used: near the top of the ladder ($n$ small, $m\approx +S$) the matrix elements of $\sqrt{2S}\,a$ agree with the true $S^+$, and the error becomes $O(1)$ only near full occupancy $n\approx 2S$ — the highly excited sector that is thermally inaccessible at low $T$.

Now the external field, the actual problem. The Zeeman term is diagonal in the magnon basis: $g\mu_B H\sum_i S_i^z = g\mu_B H(NS - \sum_k a_k^\dagger a_k)$, so it just shifts the spectrum by a rigid, $k$-independent gap,

$$\varepsilon_k(H) = qJS(1-\gamma_k) + g\mu_B H,$$

with no re-diagonalization — the whole reason the boson framework is worth the trouble. Each magnon lowers the total spin by one unit, so with the Bose occupation $n_B(\varepsilon) = 1/(e^{\varepsilon/k_BT}-1)$,

$$M(T,H) = g\mu_B\Big(NS - \sum_k n_B(\varepsilon_k(H))\Big).$$

At $H=0$, low $T$, $\varepsilon_k \approx JS(ka)^2$ gives the saturation deficit $\sum_k n_B \propto T^{3/2}$ — Bloch's law recovered as a corollary. Turning $H$ on gaps out the cheapest small-$k$ magnons, suppressing the deficit and pulling $M$ back toward saturation; $\partial M/\partial H$ is the intrinsic susceptibility, whose high-field falloff (with the dipole–dipole term folded in) is the quantity I set out to find — and the long-range dipolar interaction, being bilinear in the moments, adds further direction-dependent quadratic terms to $\varepsilon_k$ under the same linearization, entering the same free-magnon framework rather than breaking it, giving an intrinsic volume susceptibility that falls off slowly ($\sim H^{-1/2}$) at high field. The framework even reaches the antiferromagnet: referencing deviations to *up* on sublattice A and *down* on sublattice B produces anomalous pair terms $a_i b_j + a_i^\dagger b_j^\dagger$ that number-conserving Fourier cannot diagonalize; a Bogoliubov rotation $\alpha_k = u_k a_k - v_k b_{-k}^\dagger$, $\beta_k = u_k b_{-k} - v_k a_k^\dagger$ with $u_k^2 - v_k^2 = 1$ and the off-diagonal-killing condition $\gamma_k(u_k^2+v_k^2)+2u_kv_k = 0$ gives $u_k^2 = \tfrac{1}{2}(1/\sqrt{1-\gamma_k^2}+1)$, $v_k^2 = \tfrac{1}{2}(1/\sqrt{1-\gamma_k^2}-1)$, and the energy $\varepsilon_k = qJS\sqrt{1-\gamma_k^2}$ (1D: $2JS|\sin ka|$, linear at small $k$) with a zero-point lowering signaling that the Néel state is not the true ground state.

I verify the construction numerically: build the spin-$S$ matrices and truncated boson matrices on the same $(2S+1)$-dimensional space, form $S^z = S - a^\dagger a$, $S^+ = \sqrt{2S - a^\dagger a}\,a$, $S^- = a^\dagger\sqrt{2S - a^\dagger a}$, and confirm the exact map reproduces the spin matrices and satisfies $[S^+,S^-] = 2S^z$ to machine precision, while the linear map $\sqrt{2S}\,a$ is exact at the top of the ladder and errs by $O(1)$ near full occupancy.

```python
import numpy as np

def spin_matrices(S):
    """Standard (2S+1)-dim spin-S matrices in the |S,m> basis, m = S,...,-S."""
    dim = int(round(2 * S + 1))
    m = np.array([S - k for k in range(dim)])
    Sz = np.diag(m).astype(complex)
    Sp = np.zeros((dim, dim), dtype=complex)
    for k in range(1, dim):                        # S^+|m> = sqrt(S(S+1)-m(m+1))|m+1>
        Sp[k - 1, k] = np.sqrt(S * (S + 1) - m[k] * (m[k] + 1))
    return Sz, Sp, Sp.conj().T

def boson_matrices(S):
    """Truncated bosons on the same (2S+1)-dim space, n=0..2S, |n> <-> m=S-n."""
    dim = int(round(2 * S + 1))
    a = np.zeros((dim, dim), dtype=complex)
    for n in range(1, dim):                         # a|n> = sqrt(n)|n-1>
        a[n - 1, n] = np.sqrt(n)
    return a, a.conj().T

def msqrt(M):                                        # matrix sqrt of a PSD operator
    w, V = np.linalg.eigh(M)
    return (V * np.sqrt(np.clip(w.real, 0, None))) @ V.conj().T

def hp_exact(S):
    """S^z = S - a^dag a ; S^+ = sqrt(2S - a^dag a) a ; S^- = a^dag sqrt(2S - a^dag a)."""
    a, ad = boson_matrices(S)
    n = ad @ a
    root = msqrt(2 * S * np.eye(n.shape[0]) - n)     # truncation + algebra factor
    return S * np.eye(n.shape[0]) - n, root @ a, ad @ root

def hp_linear(S):
    """Leading order: S^+ ~ sqrt(2S) a, S^- ~ sqrt(2S) a^dag (valid for <n> << 2S)."""
    a, ad = boson_matrices(S)
    n = ad @ a
    return S * np.eye(n.shape[0]) - n, np.sqrt(2 * S) * a, np.sqrt(2 * S) * ad

def comm(A, B):
    return A @ B - B @ A

if __name__ == "__main__":
    for S in [0.5, 1.0, 1.5, 2.0]:
        Sz, Sp, Sm = spin_matrices(S)
        ez, ep, em = hp_exact(S)
        repro = max(np.abs(Sz - ez).max(), np.abs(Sp - ep).max(), np.abs(Sm - em).max())
        alg = np.abs(comm(ep, em) - 2 * ez).max()     # exact: [S^+,S^-] = 2 S^z
        lz, lp, lm = hp_linear(S)
        dim = int(round(2 * S + 1))
        top = np.abs(lp[0, 1] - Sp[0, 1])             # n small: linear map exact
        bot = np.abs(lp[dim - 2, dim - 1] - Sp[dim - 2, dim - 1])  # n~2S: linear map fails
        print(f"S={S}: reproduce={repro:.1e} algebra={alg:.1e} "
              f"linear top={top:.1e} bottom={bot:.1e}")
```
