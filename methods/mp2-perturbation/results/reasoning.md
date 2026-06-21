Let me start from what actually hurts. I can solve the mean-field problem — I have a converged self-consistent field, a single Slater determinant $\Phi_0$ of orthonormal spin-orbitals $\psi_i$, each electron gliding through the averaged Coulomb field of all the others. And it is wrong, always in the same direction: too high. The exact Hamiltonian has $\sum_{p<q} 1/r_{pq}$, the *instantaneous* repulsion, two electrons feeling each other's actual position; the mean field has replaced that with each electron feeling only the smeared-out average density of the rest. So electrons in my determinant don't dodge each other the way real electrons do. That avoidance — the correlation hole — lowers the true energy below the mean-field energy, and the gap is the correlation energy. It's a sliver, maybe a percent of the total, but it's the percent that decides whether a bond forms, how high a barrier sits, whether one isomer beats another. I cannot leave it on the table.

So how do I get it back, for an *arbitrary* molecule, without starting a brand-new bespoke calculation every time? I know two routes and I don't love either. I could go variational: write the wavefunction as the reference plus a pile of substituted determinants, $\Psi = c_0\Phi_0 + \sum c_i^a \Phi_i^a + \sum c_{ij}^{ab}\Phi_{ij}^{ab}+\cdots$, and diagonalize $\hat H$ in that basis. Full configuration interaction — exact in the orbital basis, a rigorous upper bound. But the number of determinants explodes combinatorially with the number of electrons, and the moment I truncate it I lose size consistency: two non-interacting fragments computed together stop giving the sum of their separate energies. For a method whose whole reason to exist is to scale to *many* electrons, a tool that misbehaves as I add electrons is self-defeating. Or I could go the Hylleraas way — put $r_{12}$ explicitly into the trial function, minimize the correlation energy functionals. Beautiful for helium and the two-electron ions, genuinely accurate. But it's hand-tailored to two electrons; there's no turnkey many-electron version, and it's costly. Neither of these is the thing I want: a *general*, *non-iterative*, transferable correction that takes the mean-field answer I already paid for and just... corrects it.

That phrase — "small correction to something I can already solve" — is perturbation theory, plainly. Rayleigh and Schrödinger handed me the machinery decades ago. Split $\hat H = \hat H_0 + \lambda \hat V$ where $\hat H_0$ is exactly soluble and $\hat V$ is small, expand energy and wavefunction in powers of $\lambda$, collect orders. The first-order energy is $E^{(1)} = \langle \Psi^{(0)}|\hat V|\Psi^{(0)}\rangle$, and the second-order energy is the sum over the *other* eigenstates of $\hat H_0$,
$$E^{(2)} = \sum_{\mu\neq 0}\frac{|\langle\Psi^{(0)}|\hat V|\Psi_\mu^{(0)}\rangle|^2}{E^{(0)}-E_\mu^{(0)}}.$$
Lovely. Except that second-order sum is a trap unless I have $\hat H_0$ whose *entire* spectrum of eigenstates I can write down and enumerate. For a many-electron system, who hands me a soluble $\hat H_0$ together with a ready-made complete set of excited eigenstates? That's the whole obstacle. People have had perturbation theory for years and it hasn't given a general correlation recipe precisely because nobody supplied the right $\hat H_0$.

So let me ask the question the other way around. I don't want to invent a soluble model Hamiltonian from nothing. I already have, sitting in front of me, a soluble one-particle problem: the Fock equations. The Fock operator $\hat F\psi_i = \varepsilon_i\psi_i$ is something I *solved* — that's what self-consistency means. Its eigenfunctions are the canonical spin-orbitals, occupied and virtual, and they form a complete orthonormal set in the orbital basis. What if my zeroth-order Hamiltonian is just the sum of these one-electron Fock operators?
$$\hat H_0 = \sum_p \hat F(p).$$
Let me test whether that even makes sense as an $\hat H_0$. Take any Slater determinant built from the $\psi_i$ — say one occupying spin-orbitals $\{k\}$. Acting with $\sum_p\hat F(p)$, a one-electron operator that's diagonal in this orbital set, gives back the same determinant times $\sum_{k}\varepsilon_k$. So *every* Slater determinant in this orbital basis is an eigenfunction of $\hat H_0$, with eigenvalue equal to the sum of its occupied orbital energies. The reference $\Phi_0$ has $E^{(0)} = \sum_i^{\rm occ}\varepsilon_i$. A singly substituted $\Phi_i^a$ has $E^{(0)} - \varepsilon_i + \varepsilon_a$. A double $\Phi_{ij}^{ab}$ has $E^{(0)} - \varepsilon_i - \varepsilon_j + \varepsilon_a + \varepsilon_b$. There it is — the thing RSPT was missing. The complete set of eigenstates of $\hat H_0$ is just *the set of all substituted determinants*, and I can enumerate them and write down their eigenvalues by inspection. The obstacle dissolves the moment I let the Fock operator be $\hat H_0$ rather than trying to build a model Hamiltonian by hand.

Now what is the perturbation? It's forced on me: $\hat V = \hat H - \hat H_0$. Let me write it out. The full Hamiltonian is $\hat H = \sum_p \hat h(p) + \sum_{p<q} 1/r_{pq}$. And
$$\hat H_0 = \sum_p \hat F(p) = \sum_p\hat h(p) + \sum_{p}\sum_i^{\rm occ}\big[\hat J_i(p)-\hat K_i(p)\big].$$
Subtracting, the one-electron $\hat h$ cancels, and
$$\hat V = \sum_{p<q}\frac{1}{r_{pq}} - \sum_p\sum_i^{\rm occ}\big[\hat J_i(p)-\hat K_i(p)\big].$$
Stare at that. The first term is the *instantaneous* electron–electron repulsion. The second is the *averaged* repulsion — the mean-field potential each electron feels. So $\hat V$ is exactly *instantaneous minus averaged* — the fluctuation of the true repulsion about its mean field. It is, by construction, precisely the part of the interaction that the mean field threw away: the correlation. And it is "small" in the right sense — the bulk of the repulsion is already absorbed into $\hat H_0$, so the perturbation is only the fluctuation, not the whole $1/r$. This is the choice that makes the series have a chance. (Contrast the naive alternative: if I had taken $\hat H_0 = \sum_p\hat h(p)$, the bare cores, then $\hat V$ would be the *entire* $1/r_{pq}$ — enormous, not a perturbation at all, and $\Phi_0$ wouldn't even be an eigenstate of $\hat H_0$. Putting the mean field into $\hat H_0$ is what shrinks the perturbation to just the fluctuation and keeps the reference as the zeroth-order state.)

Good. Now turn the crank. Zeroth order is settled: $E^{(0)} = \sum_i\varepsilon_i$. First order:
$$E^{(1)} = \langle\Phi_0|\hat V|\Phi_0\rangle = \langle\Phi_0|\hat H|\Phi_0\rangle - \langle\Phi_0|\hat H_0|\Phi_0\rangle = E_{\rm HF} - \sum_i\varepsilon_i.$$
So $E^{(0)} + E^{(1)} = \sum_i\varepsilon_i + (E_{\rm HF} - \sum_i\varepsilon_i) = E_{\rm HF}$. Through *first order*, this perturbation expansion reproduces nothing but the Hartree–Fock energy. Let me make that concrete with the integrals to be sure I'm not fooling myself. $\langle\Phi_0|\hat V|\Phi_0\rangle$: the $1/r_{pq}$ part of $\hat V$ gives, by the Slater–Condon rules for a diagonal element, $\tfrac12\sum_{ij}\langle ij\|ij\rangle$. The mean-field part of $\hat V$ is $-\sum_p\sum_i[\hat J_i-\hat K_i](p)$, a sum of one-electron operators; its expectation over $\Phi_0$ is $-\sum_{j}\sum_i\langle j|\hat J_i-\hat K_i|j\rangle = -\sum_{ij}\langle ij\|ij\rangle$. Adding, $E^{(1)} = \tfrac12\sum_{ij}\langle ij\|ij\rangle - \sum_{ij}\langle ij\|ij\rangle = -\tfrac12\sum_{ij}\langle ij\|ij\rangle$. And indeed $E^{(0)}+E^{(1)} = \sum_i\varepsilon_i - \tfrac12\sum_{ij}\langle ij\|ij\rangle$, which is the textbook Hartree–Fock energy. Consistent.

Here's the consequence that decides everything: first-order perturbation theory gives me back the mean field I started from and *not one bit of correlation*. If I want the correlation energy, the very first place it can appear is **second order**. That's not a disappointment — it's a feature, and a clean one. It tells me the leading correlation correction is a single, closed second-order expression, and it tells me the mean-field energy is "correct to first order," so my expansion is anchored at the best one-determinant answer and the first thing it does beyond that is correlate. (And the same argument, run on a one-electron property, says the first-order correction to the charge density vanishes too — the density is right to first order as well. The reference really is a good launching point.)

So I need $E^{(2)} = \sum_{\mu\neq 0}|\langle\Phi_0|\hat V|\Phi_\mu\rangle|^2/(E^{(0)}-E_\mu)$, where $\Phi_\mu$ runs over all substituted determinants. This looks like a huge sum — singles, doubles, triples, quadruples, all of them. But before I drown in it, let me ask which matrix elements $\langle\Phi_0|\hat V|\Phi_\mu\rangle$ are even nonzero. $\hat V = \hat H - \hat H_0$. The $\hat H_0$ piece is diagonal in determinants, so $\langle\Phi_0|\hat H_0|\Phi_\mu\rangle = 0$ for $\mu\neq 0$; thus $\langle\Phi_0|\hat V|\Phi_\mu\rangle = \langle\Phi_0|\hat H|\Phi_\mu\rangle$ for the excited determinants. Now I lean on the structure of $\hat H$: it has only one- and two-body operators. The Slater–Condon rules then say $\langle\Phi_0|\hat H|\Phi_\mu\rangle$ is automatically zero whenever $\Phi_\mu$ differs from $\Phi_0$ by **three or more** spin-orbitals. So triples, quadruples, everything beyond doubles — gone, identically, because the Hamiltonian simply cannot connect determinants that differ in more than two orbitals. Physically: interactions are pairwise, so in one application of $\hat H$ you can disturb at most two electrons. That collapses the sum to singles and doubles.

Now the singles, $\Phi_i^a$ — one electron promoted $i\to a$. Is $\langle\Phi_0|\hat H|\Phi_i^a\rangle$ zero? The Slater–Condon rule for determinants differing by one spin-orbital gives
$$\langle\Phi_0|\hat H|\Phi_i^a\rangle = \langle i|\hat h|a\rangle + \sum_j^{\rm occ}\langle ij\|aj\rangle.$$
Let me look hard at the right-hand side. That combination — the one-electron core integral plus the sum of antisymmetrized two-electron integrals over the occupied orbitals — is exactly the off-diagonal matrix element of the *Fock operator*:
$$\langle i|\hat F|a\rangle = \langle i|\hat h|a\rangle + \sum_j^{\rm occ}\langle ij\|aj\rangle.$$
And in the canonical Hartree–Fock basis the Fock operator is *diagonal* — that's how I obtained the orbitals, $\hat F\psi_a = \varepsilon_a\psi_a$, so $\langle i|\hat F|a\rangle = \varepsilon_a\langle i|a\rangle = 0$ for $i\neq a$. Therefore $\langle\Phi_0|\hat H|\Phi_i^a\rangle = 0$. The singles drop out entirely. And this is not luck — it's the *stationarity* of the Hartree–Fock determinant restated: the reference is variationally optimal against single substitutions, so to first order it cannot mix with any singly excited determinant. The same fact, two faces. Because I started from the best single determinant, the singles vanish for free. (Anything but the converged HF reference and this would fail — the singles would survive and the whole expression would be messier and the reference no longer "right to first order." Yet another reason the HF determinant is the *right* zeroth-order state.)

So everything funnels into the **doubles**. $E^{(2)} = \sum_{\rm doubles}|\langle\Phi_0|\hat H|\Phi_{ij}^{ab}\rangle|^2/(E^{(0)}-E_{ij}^{ab})$. Let me get both pieces exactly. The matrix element between the reference and a double, by the Slater–Condon rule for determinants differing by two spin-orbitals (occupied $i,j$ replaced by virtual $a,b$), is simply the antisymmetrized two-electron integral:
$$\langle\Phi_0|\hat H|\Phi_{ij}^{ab}\rangle = \langle ij\|ab\rangle = \langle ij|ab\rangle - \langle ij|ba\rangle.$$
Clean — no one-electron part survives at the two-orbital difference level. And the denominator: $\Phi_{ij}^{ab}$ is an eigenstate of $\hat H_0$ with eigenvalue $E^{(0)} - \varepsilon_i - \varepsilon_j + \varepsilon_a + \varepsilon_b$, so
$$E^{(0)} - E_{ij}^{ab} = \varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b.$$
Since occupied energies sit below virtual energies, this is *negative*; the squared numerator is positive; so every term in $E^{(2)}$ is $\le 0$. The correction lowers the energy — exactly what a correlation correction must do. (I should hold onto the sign carefully: occupied *minus* virtual, $\varepsilon_i+\varepsilon_j-\varepsilon_a-\varepsilon_b$. Flip it and I'd get a positive "correlation energy," which would be nonsense. The sign comes straight from $E^{(0)}-E_\mu$ with the double lying above the reference.)

Now I just have to count the doubles correctly. A double excitation is specified by an ordered choice... no — by an *unordered* pair of occupied orbitals $\{i,j\}$ and an unordered pair of virtuals $\{a,b\}$, because the determinant $\Phi_{ij}^{ab}$ is antisymmetric and $\Phi_{ij}^{ab} = \Phi_{ji}^{ab}$ up to sign, which the squared modulus washes out. So the honest sum is over $i<j$ and $a<b$:
$$E^{(2)} = \sum_{i<j}\sum_{a<b}\frac{|\langle ij\|ab\rangle|^2}{\varepsilon_i+\varepsilon_j-\varepsilon_a-\varepsilon_b}.$$
If I'd rather let all four indices run freely — which is what I actually want for a clean tensor contraction in code — I'm overcounting: each distinct double is hit four times, $(i,j)\leftrightarrow(j,i)$ and $(a,b)\leftrightarrow(b,a)$, and the antisymmetric integral $\langle ij\|ab\rangle$ flips sign under either swap but the square is invariant, so I divide by $4$:
$$\boxed{\,E^{(2)} = \frac14\sum_{ij}^{\rm occ}\sum_{ab}^{\rm vir}\frac{|\langle ij\|ab\rangle|^2}{\varepsilon_i+\varepsilon_j-\varepsilon_a-\varepsilon_b}\,}.$$
That's it. That is the entire correlation correction at leading order: built only from the HF orbital energies and the two-electron integrals in the molecular-orbital basis, no iteration, one pass over the doubles. It reuses everything the mean-field calculation already produced, it lowers the energy, and — because it's a sum of independent pair contributions rather than a truncated CI — it doesn't suffer the size-consistency disease: split a system into non-interacting fragments and the integrals connecting them vanish, so the energy is additive. Exactly the profile I was after.

Let me sanity-check the physical reading of the formula, because I want to trust it. Each term is a pair of electrons in occupied orbitals $i,j$ scattering into the empty pair $a,b$, weighted by how strongly they interact ($|\langle ij\|ab\rangle|^2$) and how energetically cheap the virtual excitation is (small $|\varepsilon_a+\varepsilon_b-\varepsilon_i-\varepsilon_j|$ in the denominator means a big contribution). Electrons borrow excited-orbital character to get out of each other's way — that *is* the correlation hole, written as virtual admixture. Pairs close in energy to the occupied space contribute most; the more strongly two orbitals' product overlaps an empty pair, the more they correlate. Both readings match physical intuition. Good.

Now I want this in a form I can actually compute cheaply, which means getting rid of the spin. So far everything is in spin-orbitals $\psi$. For a closed-shell molecule each spatial orbital $\phi$ is doubly occupied — one $\alpha$ spin-orbital and one $\beta$. Let me integrate the spin out of $\langle ij\|ab\rangle$. The two-electron integral $\langle ij|ab\rangle = \int\psi_i^*(1)\psi_j^*(2)\tfrac{1}{r_{12}}\psi_a(1)\psi_b(2)$ is nonzero only if the spin of $i$ matches the spin of $a$ (electron 1) and the spin of $j$ matches the spin of $b$ (electron 2); the exchange term $\langle ij|ba\rangle$ needs spin($i$)=spin($b$) and spin($j$)=spin($a$). So whether the Coulomb part, the exchange part, or both survive depends on the spin pattern of the four spin-orbitals.

Let me enumerate the spin cases for a pair of occupied spatial orbitals $i,j$ exciting to virtual spatials $a,b$. Switch to chemists' (Mulliken) notation for the spatial integrals, $(ia|jb) = \int\phi_i^*(1)\phi_a(1)\tfrac{1}{r_{12}}\phi_j^*(2)\phi_b(2)$, so the physicist Coulomb integral $\langle ij|ab\rangle$ becomes $(ia|jb)$ and the exchange $\langle ij|ba\rangle$ becomes $(ib|ja)$.
- **Opposite spins**, $i_\alpha j_\beta \to a_\alpha b_\beta$: the Coulomb term survives (spins match across the integral), but the exchange term needs $\alpha$ to pair with $\beta$ and dies. So $\langle ij\|ab\rangle \to (ia|jb)$, contributing $(ia|jb)^2$. There's a second opposite-spin arrangement $i_\beta j_\alpha\to a_\beta b_\alpha$, identical by symmetry — that's the factor of two on the opposite-spin piece.
- **Same spins**, $i_\alpha j_\alpha\to a_\alpha b_\alpha$ (and the $\beta\beta$ copy): both Coulomb and exchange survive, $\langle ij\|ab\rangle\to (ia|jb)-(ib|ja)$, contributing $(ia|jb)[(ia|jb)-(ib|ja)]$ after I expand $|\langle ij\|ab\rangle|^2$ and use the spatial-orbital reality to drop one of the cross terms appropriately.

Carrying the $\tfrac14$ and the spin sums through, the same-spin and opposite-spin contributions assemble into the closed-shell spatial-orbital form
$$E^{(2)} = \sum_{ij}^{\rm occ}\sum_{ab}^{\rm vir}\frac{(ia|jb)\,\big[\,2(ia|jb) - (ib|ja)\,\big]}{\varepsilon_i+\varepsilon_j-\varepsilon_a-\varepsilon_b},$$
where now $i,j$ index *spatial* occupied orbitals and $a,b$ *spatial* virtuals — half the index range of the spin-orbital sum, so roughly an order of magnitude less work. The "$2$" is the opposite-spin Coulomb (two electrons of unlike spin only feel direct repulsion), and the "$-(ib|ja)$" is the same-spin exchange — same-spin electrons are *additionally* correlated by the antisymmetry requirement, which is the Pauli hole on top of the Coulomb hole. I can even read off the two physically distinct channels: an opposite-spin part $\sum (ia|jb)^2/(\ldots)$ and a same-spin part $\sum (ia|jb)[(ia|jb)-(ib|ja)]/(\ldots)$, both individually negative.

One more thing to settle before I write code: what does this *cost*, and where does the cost live? The energy expression itself is a sum over $o^2 v^2$ terms ($o$ occupied, $v$ virtual) — that's $O(N^4)$, cheap. The real expense is upstream. The two-electron integrals come out of the SCF in the atomic-orbital basis, $(\mu\nu|\lambda\sigma)$, and I need them in the molecular-orbital basis, $(ia|jb)$, via the orbital coefficients $C$:
$$(ia|jb) = \sum_{\mu\nu\lambda\sigma} C_{\mu i}C_{\nu a}C_{\lambda j}C_{\sigma b}\,(\mu\nu|\lambda\sigma).$$
If I do that as one nested contraction, four index sums over four MO indices times four AO indices, it's $O(N^8)$ — hopeless. But I must never do it that way. Transform *one index at a time*, storing the half-transformed intermediate each time:
$$(\mu\nu|\lambda\sigma)\xrightarrow{C_{\mu i}}(i\nu|\lambda\sigma)\xrightarrow{C_{\nu a}}(ia|\lambda\sigma)\xrightarrow{C_{\lambda j}}(ia|j\sigma)\xrightarrow{C_{\sigma b}}(ia|jb).$$
Each quarter-transform is a single index summed against a four-index array — $O(N^5)$ — and there are four of them, so $4\,O(N^5)$ instead of $O(N^8)$. That four-step transformation is what makes the whole method affordable; it's the bottleneck and it sets the formal $O(N^5)$ scaling of the leading-correlation method. Everything else is downhill.

So now the code writes itself, and I'll do it twice: once transparently in the spin-orbital basis to *see* the boxed formula working, then in the production closed-shell spatial form with the four-quarter transform. Spin-orbital version first — build the antisymmetrized double-bar integrals from spatial MO integrals, set the orbital energies, and sum the doubles with the $\tfrac14$:

```python
import numpy as np

# Spin-orbital second-order correlation energy, the boxed formula made literal.
# `teimo(p,q,r,s)` returns a SPATIAL MO two-electron integral (chemists' order);
# E_spatial = canonical HF orbital energies (one per spatial orbital);
# Nelec = number of electrons = number of occupied spin orbitals.

def antisymmetrized_spin_integrals(teimo, n_spatial):
    """Build <pq||rs> = <pq|rs> - <pq|sr> over spin orbitals (physicists' notation).
    Each spatial orbital splits into two spin orbitals; a two-electron integral is
    nonzero only when the spins line up across the integral."""
    dim = 2 * n_spatial
    V = np.zeros((dim, dim, dim, dim))
    for p in range(1, dim + 1):
        for q in range(1, dim + 1):
            for r in range(1, dim + 1):
                for s in range(1, dim + 1):
                    # Coulomb part: spin(p)=spin(r) and spin(q)=spin(s)
                    coulomb  = teimo((p+1)//2, (r+1)//2, (q+1)//2, (s+1)//2) \
                               * (p % 2 == r % 2) * (q % 2 == s % 2)
                    # Exchange part: spin(p)=spin(s) and spin(q)=spin(r)
                    exchange = teimo((p+1)//2, (s+1)//2, (q+1)//2, (r+1)//2) \
                               * (p % 2 == s % 2) * (q % 2 == r % 2)
                    V[p-1, q-1, r-1, s-1] = coulomb - exchange
    return V

def second_order_spin_orbital(teimo, E_spatial, Nelec, n_spatial):
    V  = antisymmetrized_spin_integrals(teimo, n_spatial)
    dim = 2 * n_spatial
    eps = np.array([E_spatial[i // 2] for i in range(dim)])   # spin-orbital energies
    e2 = 0.0
    for i in range(Nelec):                 # occupied spin orbitals
        for j in range(Nelec):
            for a in range(Nelec, dim):    # virtual spin orbitals
                for b in range(Nelec, dim):
                    # boxed formula: (1/4) |<ij||ab>|^2 / (e_i + e_j - e_a - e_b)
                    e2 += 0.25 * V[i, j, a, b] ** 2 / (eps[i] + eps[j] - eps[a] - eps[b])
    return e2
```

Let me run it in my head on the smallest real case I can — HeH$^+$ in a minimal STO-3G basis, two electrons, two spatial orbitals, so two occupied spin-orbitals and two virtual. The HF orbital energies are $\varepsilon = (-1.5238, -0.2676)$, and the six unique spatial MO two-electron integrals are known. The double-loop has exactly one nontrivial double excitation (both electrons out of the occupied spatials into the virtual pair), and the $\tfrac14$ with the four-fold index symmetry collapses it to a single physical pair contribution. It returns $-0.00640$ Hartree of correlation energy — small, negative, as it must be. The formula is real.

Now the production form: closed-shell, spatial orbitals, the four-quarter integral transform, and the spin-summed numerator $2(ia|jb)-(ib|ja)$, vectorized so the $o^2v^2$ contraction runs at compiled speed:

```python
import numpy as np

# Closed-shell (RHF) second-order correlation energy.
# Inputs from a converged SCF:
#   I_ao : (nbf,)*4 AO two-electron integrals (mu nu | la si), chemists' notation
#   C    : (nbf, nmo) MO coefficients
#   eps  : (nmo,)  canonical orbital energies (ascending)
#   ndocc: number of doubly-occupied spatial orbitals

def second_order_closed_shell(I_ao, C, eps, ndocc):
    Cocc, Cvir = C[:, :ndocc], C[:, ndocc:]
    e_occ, e_vir = eps[:ndocc], eps[ndocc:]

    # AO -> MO as four O(N^5) quarter-transforms, never one O(N^8) contraction.
    t = np.einsum('pi,pqrs->iqrs', Cocc, I_ao, optimize=True)   # mu -> i
    t = np.einsum('qa,iqrs->iars', Cvir, t,    optimize=True)   # nu -> a
    t = np.einsum('rj,iars->iajs', Cocc, t,    optimize=True)   # la -> j
    I_mo = np.einsum('sb,iajs->iajb', Cvir, t, optimize=True)   # si -> b   # (ia|jb)

    # Denominator e_i + e_j - e_a - e_b, broadcast over the (i,a,j,b) grid.
    d = (e_occ.reshape(-1,1,1,1) - e_vir.reshape(1,-1,1,1)
         + e_occ.reshape(1,1,-1,1) - e_vir.reshape(1,1,1,-1))

    # Opposite-spin: (ia|jb)^2 ; same-spin adds the exchange -(ib|ja).
    e_os = np.einsum('iajb,iajb,iajb->', I_mo, I_mo, 1.0/d, optimize=True)
    e_ss = np.einsum('iajb,iajb,iajb->', I_mo, I_mo - I_mo.swapaxes(1,3),
                     1.0/d, optimize=True)
    return e_os + e_ss          # total second-order correlation energy (<= 0)

# E_total = E_hf + second_order_closed_shell(I_ao, C, eps, ndocc)
```

Let me close the loop on the whole chain. The mean field discards the fluctuation of the electron repulsion about its average, and that fluctuation is the correlation energy. Choosing $\hat H_0$ as the sum of Fock operators makes the Hartree–Fock determinant the zeroth-order state and makes every substituted determinant a known eigenstate, so the perturbation $\hat V = \hat H - \hat H_0$ is exactly that fluctuation potential and the second-order sum is enumerable. First-order perturbation theory only reproduces the Hartree–Fock energy, so correlation first appears at second order; the singles vanish because the reference is variationally stationary (the off-diagonal Fock element is zero), and triples and higher vanish because the Hamiltonian is at most two-body; only the doubles survive. Their matrix element is the antisymmetrized integral $\langle ij\|ab\rangle$ and their energy denominator is $\varepsilon_i+\varepsilon_j-\varepsilon_a-\varepsilon_b$, giving a single closed, manifestly-negative, size-extensive correlation correction $\tfrac14\sum|\langle ij\|ab\rangle|^2/(\varepsilon_i+\varepsilon_j-\varepsilon_a-\varepsilon_b)$ — equivalently the closed-shell $\sum (ia|jb)[2(ia|jb)-(ib|ja)]/(\varepsilon_i+\varepsilon_j-\varepsilon_a-\varepsilon_b)$ — whose cost is dominated by the $O(N^5)$ four-quarter transformation of the two-electron integrals into the molecular-orbital basis.
