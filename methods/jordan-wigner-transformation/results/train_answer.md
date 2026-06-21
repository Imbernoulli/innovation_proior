We have already quantized two wave fields and watched the particles fall out of the operator algebra rather than being put in by hand. For light: an assembly of oscillators, creation and annihilation operators, a number operator with eigenvalues $0,1,2,\dots$, and photons appear as a consequence of the multiplication rules. For the Einstein matter gas: amplitudes $b_r$ with conjugate phases $\Theta_r$, the number $N_r = b_r^\dagger b_r$, the commutator $[b_r, b_r^\dagger]=1$, the same spectrum $0,1,2,\dots$, and the density fluctuation comes out proportional to $n_r(1+n_r)$ exactly as Einstein found. The whole appeal of second quantization is that the existence of discrete particles and their statistics are *consequences* of how the field amplitude on ordinary three-dimensional space multiplies, with the abstract $3N$-dimensional configuration space gone. What is still missing is the third case: a quantized matter-wave field whose quanta obey the Pauli exclusion principle, so that every mode occupation is capped at $0$ or $1$ and the many-particle states are antisymmetric under exchange.

The trouble is that the bosonic machine, copied verbatim, gives the wrong answer, and it fails structurally rather than for want of tuning. The relation $[b,b^\dagger]=1$ forces the spectrum of $N=b^\dagger b$ to be the non-negative integers — that is a theorem about the oscillator algebra, with no knob to turn — whereas electrons need occupation only $0$ or $1$. Worse, the bosonic amplitudes on different modes *commute*, $b_\alpha b_\beta - b_\beta b_\alpha = 0$, and commuting amplitudes generate a *symmetric* state space, whereas Pauli's principle in its original form is that the many-electron wave function is antisymmetric: swap two electrons and pick up a minus sign. So there are two failures from one source — the spectrum is wrong and the exchange sign is wrong. One could try to keep $[b,b^\dagger]=1$ and simply *declare* that physical states have $n_r \le 1$, but creation steps $1\to 2$, so the restriction is not preserved by the algebra and it still supplies no exchange sign; the cap and the antisymmetry remain two hand-imposed facts rather than one property of the field. And one could stay in configuration space with the Heisenberg–Dirac determinant, but that is precisely the abstract space the whole program is trying to leave, and the antisymmetry there is built in by hand rather than emerging. What I want is a single algebra of amplitudes $a_r, a_r^\dagger$ from which *both* the cap at $1$ and the antisymmetric exchange drop out automatically. A clean target to hit at the end is Pauli's fluctuation result, $\overline{\Delta^2}\propto n_r(1-n_r)$ — the same algebraic shape as the boson's $n_r(1+n_r)$ but with the sign in the parenthesis flipped, the $(1-n_r)$ being the fingerprint of single occupancy because it vanishes the instant the mode is full.

I propose the Jordan–Wigner transformation. Start with one mode, where the difficulty surely is not. A mode holding $0$ or $1$ particle is a two-level system, and the smallest realization is two-by-two: $b=\sigma^- = \big(\begin{smallmatrix}0&1\\0&0\end{smallmatrix}\big)$ and $b^\dagger = \sigma^+$, giving $b^\dagger b = N$ with eigenvalues $\{0,1\}$, $b b^\dagger = 1-N$, and $b^2=0$. Here $b^2=0$ *is* "you cannot put two particles in one mode," and $b b^\dagger = 1-N$ already carries the Fermi $(1-N)$ factor in place of the boson's $1+N$ — the single-mode problem solves itself. These matrices are the spin-1/2 ladder operators, and the parity $1-2N = \mathrm{diag}(1,-1) = -\sigma^z$ is $+1$ on the empty state and $-1$ on the full one, so a single Fermi mode and a single spin are the same two-level object. The entire difficulty is combining $K$ modes, and the obvious move — the one that worked for bosons — is to tensor: place the single-mode $b$ in slot $r$ as $b_r = 1\otimes\cdots\otimes b\otimes\cdots\otimes 1$. But for $r\ne s$ these act on disjoint slots and therefore *commute*, $b_r b_s = b_s b_r$, which is the symmetric boson exchange. The per-mode cap is local and easy; the exchange sign is global and is where the real content lives.

That missing sign is not in any single mode but in the antisymmetric many-electron determinant, which vanishes when two single-particle states coincide. To read off how the sign enters when one *adds or removes* a particle — what $a_r^\dagger$ and $a_r$ do — note the determinant has a sign ambiguity: to make it single-valued one must fix, once and for all, an ordering of the single-particle states, $1<2<\cdots<K$ (where $<$ is merely a chosen order). With the order fixed, the amplitude becomes a single-valued function of the occupations, $2^K$ basis states each with $N\in\{0,1\}$. To remove the particle in mode $\lambda$ from the determinant, one must commute its factor — through antisymmetry — past every occupied factor sitting to its left in the fixed ordering, and each hop is a transposition costing a minus sign. So removing a particle from mode $\lambda$ costs $(-1)$ raised to the number of occupied modes to its left. That count is the operator $\sum_{k<\lambda} N_k$, and the sign is

$$ v_\lambda = \prod_{k<\lambda}(-1)^{N_k} = \prod_{k<\lambda}(1-2N_k) = \prod_{k<\lambda}(-\sigma^z_k) = \exp\!\Big(i\pi\sum_{k<\lambda}N_k\Big), $$

using $(-1)^N = 1-2N$ for a two-state mode. This is the Jordan–Wigner string. Since each factor $1-2N_k$ has eigenvalues $\pm1$, the string is an involution, $v_\lambda^2 = 1$ — exactly what a sign-carrier should be. I therefore *define* the Fermi amplitude as the bare two-level lowering operator dressed with this string,

$$ a_\lambda = \Big[\prod_{k<\lambda}(1-2N_k)\Big]\,b_\lambda = v_\lambda\,\sigma^-_\lambda, \qquad a_\lambda^\dagger = b_\lambda^\dagger\,\Big[\prod_{k<\lambda}(1-2N_k)\Big] = v_\lambda\,\sigma^+_\lambda . $$

Why this works is one local fact. On its own mode, $Nb = b^\dagger b b = 0$ (because $b^2=0$) while $bN = bb^\dagger b = (1-N)b = b$, so $b(1-2N) = b-2b = -b$ and $(1-2N)b = b$; that is, $b$ *anticommutes* with $1-2N$ on its own mode, $b(1-2N) = -(1-2N)b$, and likewise for $b^\dagger$. This single anticommutation drives everything. The string $v_{q''}$ contains the factor on mode $q'$ exactly when $q'\le q''$, so moving it past the bare operator $b_p(q')$ flips a sign when $q'\le q''$ and commutes when $q'>q''$. Tracking those flips for two dressed amplitudes gives $a_p(q')a_p(q'') = -v(q')v(q'')\,b_p(q')b_p(q'')$ while the reversed product gives $+v(q'')v(q')\,b_p(q'')b_p(q')$; the strings commute among themselves and the bare operators on different modes commute, so $a_p(q')a_p(q'') = -a_p(q'')a_p(q')$. On the same mode $a^2 = 0$ because $b^2=0$ regardless of the string sign. With one dagger the off-diagonal orderings cancel by the same string-flip, and on the diagonal the string squares to $1$ and leaves the single-mode relation $b^\dagger b + b b^\dagger = N + (1-N) = 1$. So the dressed amplitudes satisfy the canonical anticommutation relations

$$ \{a_i,a_j\} = 0, \qquad \{a_i^\dagger,a_j^\dagger\}=0, \qquad \{a_i^\dagger,a_j\}=\delta_{ij}, $$

forced by the determinant's reordering sign — antisymmetry was not postulated as an extra rule but emerged from the string, which was itself nothing but the exchange sign written as an operator.

The decisive payoff is the converse: the algebra *implies* the physics. Suppose one knows only the canonical relations. From $\{a_r^\dagger,a_r\}=1$ (so $1-a_r^\dagger a_r = a_r a_r^\dagger$) and $a_r^2=0$,

$$ N_r(1-N_r) = a_r^\dagger a_r\,a_r a_r^\dagger = a_r^\dagger(a_r a_r)a_r^\dagger = 0 \;\Longrightarrow\; N_r^2 = N_r \;\Longrightarrow\; N_r\in\{0,1\}, $$

so the Pauli cap is derived, and $[N_r,N_s]=0$ follows from the anticommutators, so a joint occupation-number basis exists. The existence of corpuscular electrons and the validity of the exclusion principle are thus *consequences* of the multiplication rules of the wave amplitudes — the Fermi analogue of the Bose result. The fluctuation target is met too: in the corrected expansion $\psi = \sum_r a_r u_r$ the calculation runs with $\overline{a_r^\dagger a_r\,a_r a_r^\dagger} = \overline{N_r(1-N_r)}$, so $\overline{\Delta^2}\propto n_r(1-n_r)$, Pauli's Fermi form. The equivalence to the configuration-space theory is exact: the second-quantized one-body energy $\Omega = \sum_{\kappa\lambda}H_{\kappa\lambda}a_\kappa^\dagger a_\lambda$ is unitarily equivalent to the antisymmetric $V = V_1+\cdots+V_N$ on determinants, the matrix-element signs matching precisely because the string was built from the determinant's own reordering parity, with the basis change $a_\alpha(\beta') = \sum_{q'}\Phi_{\alpha p}(\beta',q')\,a_p(q')$ acting linearly and unitarily on the *dressed* amplitudes and leaving $N$ invariant. The only change from the boson structural relations is commutator $\to$ anticommutator; the whole of Fermi statistics is that single sign flip $[\,,\,]\to\{\,,\,\}$, achieved by the string. And the construction is essentially unique: forming the Hermitian Majorana combinations $\alpha_\kappa = a_\kappa+a_\kappa^\dagger$ and $\alpha_{K+\kappa} = (1/i)(a_\kappa-a_\kappa^\dagger)$, the canonical relations give the Clifford/Dirac algebra $\alpha_\kappa\alpha_\lambda+\alpha_\lambda\alpha_\kappa = 2\delta_{\kappa\lambda}$. The $2K$ matrices $\alpha$ together with $-1$ generate a group of order $2^{2K+1}$ with center $\{1,-1\}$ and $2^{2K}+1$ conjugacy classes, hence $2^{2K}$ one-dimensional representations plus exactly one of dimension $d$ with $2^{2K}\cdot 1 + d^2 = 2^{2K+1}$, i.e. $d = 2^K$. That unique $2^K$-dimensional faithful irreducible representation is the string construction, so the anticommutation relations fix $a, a^\dagger$ up to a unitary transformation — the algebra is the physics.

Read on a one-dimensional lattice, the same string is an exact converter between spins and fermions: with $S^z_j = N_j-1/2$, $S^+_j = a_j^\dagger e^{i\pi\sum_{l<j}N_l}$, $S^-_j = a_j e^{-i\pi\sum_{l<j}N_l}$, the spins on different sites commute while the fermions anticommute. The miracle is that on a nearest-neighbor bond the strings telescope: in $S^+_j S^-_{j+1}$ the shared factors over $l<j$ cancel, leaving only $e^{-i\pi N_j} = 1-2N_j$ on site $j$, and $a_j^\dagger(1-2N_j) = a_j^\dagger - 2a_j^\dagger a_j^\dagger a_j = a_j^\dagger$ since $a_j^{\dagger 2}=0$, so $S^+_j S^-_{j+1} = a_j^\dagger a_{j+1}$. Hence the $XX$ chain $H = -J\sum_j(S^x_j S^x_{j+1}+S^y_j S^y_{j+1})$ becomes the *quadratic* $-(J/2)\sum_j(a_j^\dagger a_{j+1}+a_{j+1}^\dagger a_j)$ — a free fermion gas — which a Fourier transform on a periodic chain diagonalizes to $H = \sum_k \omega_k\,\tilde a_k^\dagger \tilde a_k$ with dispersion $\omega_k = -J\cos k$, the ground state filling all $\omega_k<0$. An added $-J_z\sum_j S^z_j S^z_{j+1} = -J_z\sum_j(N_j-1/2)(N_{j+1}-1/2)$ is string-free but *quartic*, a genuine density-density interaction, so the freeness is special to the bilinear $XX$ / transverse-Ising lines while the map itself stays exact for any couplings; a longer-range $S^+_j S^-_{j+m}$ leaves a genuine uncollapsed string $\prod_{j\le l<j+m}(1-2N_l)$, so the telescoping is a property of nearest-neighbor terms specifically.

```python
import numpy as np

# single two-level mode  ==  one spin-1/2  ==  one Fermi mode
I  = np.eye(2)
sm = np.array([[0, 1], [0, 0]], dtype=complex)   # b = sigma^- (lowering)
sp = sm.conj().T                                 # b^dagger = sigma^+
n1 = sp @ sm                                     # single-mode N, eigenvalues {0,1}
mz = I - 2 * n1                                  # 1 - 2N = -sigma^z (the string's local factor)

def on_mode(local, j, K):
    m = np.array([[1]], dtype=complex)
    for k in range(K):
        m = np.kron(m, local if k == j else I)
    return m

def jw_string(j, K):                             # prod_{l<j} (1 - 2 N_l) = prod_{l<j} (-sigma^z_l)
    s = np.eye(2 ** K, dtype=complex)
    for l in range(j):
        s = s @ on_mode(mz, l, K)
    return s

def a(j, K):    return jw_string(j, K) @ on_mode(sm, j, K)    # a_j      = string . b_j
def adag(j, K): return on_mode(sp, j, K) @ jw_string(j, K)    # a_j^dag  = b_j^dag . string

# canonical anticommutation relations
K = 4
for i in range(K):
    for j in range(K):
        tgt = np.eye(2 ** K) if i == j else np.zeros((2 ** K, 2 ** K))
        assert np.allclose(adag(i, K) @ a(j, K) + a(j, K) @ adag(i, K), tgt)   # {a_i^dag,a_j}=d_ij
        assert np.allclose(a(i, K) @ a(j, K) + a(j, K) @ a(i, K), 0)           # {a_i,a_j}=0

# XX spin chain == free fermions (exact)
J = 1.0
H_spin = sum(-(J / 2) * (on_mode(sp, j, K) @ on_mode(sm, j + 1, K)
                         + on_mode(sm, j, K) @ on_mode(sp, j + 1, K)) for j in range(K - 1))
H_free = sum(-(J / 2) * (adag(j, K) @ a(j + 1, K) + adag(j + 1, K) @ a(j, K)) for j in range(K - 1))
assert np.allclose(H_spin, H_free)

# periodic dispersion
N = 8
omega = [-J * np.cos(2 * np.pi * m / N) for m in range(N)]   # omega_k = -J cos k
```
