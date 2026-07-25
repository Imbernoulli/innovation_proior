The one-dimensional isotropic Heisenberg antiferromagnet has looked like a settled kind of problem since Bethe solved the spin-$\tfrac12$ chain: a unique ground state, power-law correlations, excitations of arbitrarily low energy, no gap. The working expectation has been that every one-dimensional isotropic antiferromagnet is like that. Haldane's mapping of the spin-$s$ chain onto an O(3) nonlinear sigma model says otherwise. The effective action carries a topological term with $\theta = 2\pi s$; for integer $s$ that term is trivial and the sigma model is massive, so the chain should have a unique disordered ground state with a finite gap and exponentially decaying correlations, while for half-odd-integer $s$, $\theta = \pi$ and the model is critical. Integer and half-integer chains would then differ in kind. But that argument is a continuum approximation valid at large $s$, pushed down to $s = 1$. Neutron scattering on the quasi-one-dimensional $s=1$ antiferromagnet CsNiCl$_3$ (Buyers et al. 1986) does show a gap, and finite-chain numerics put the $s = 1$ Heisenberg ground-state energy near $-1.40$ per bond, but none of that is a proof. On the half-integer side there already is a theorem — the Affleck–Lieb extension of Lieb–Schultz–Mattis forbids a unique gapped ground state for a translation-invariant half-odd-integer chain, which must be gapless or break a symmetry — so the dichotomy has one rigorous half and one conjectural half. What I want is the missing half: a single concrete, isotropic, translationally invariant integer-spin Hamiltonian whose ground state can be written in closed form and then shown *rigorously* to be unique, symmetry-unbroken, exponentially correlated, and gapped. One such example converts the massive integer-spin phase from a conjecture into a theorem.

None of the available routes reach that. The Bethe ansatz does not extend to spin-1, and even if it did it would deliver a spectrum rather than a phase I can reason about cleanly. Anderson's resonating-valence-bond proposal offers a disordered non-Néel wavefunction but no parent Hamiltonian that makes it exact. The one place where exact solvability has actually been engineered is the Majumdar–Ghosh chain, $H = \sum_i [\,\mathbf S_i\cdot\mathbf S_{i+1} + \tfrac12 \mathbf S_i \cdot \mathbf S_{i+2}\,]$ at spin-$\tfrac12$, and its trick is the one worth stealing. Grouped over consecutive triples, $H \propto \sum_i [(\mathbf S_i + \mathbf S_{i+1} + \mathbf S_{i+2})^2 - \tfrac34]$ is a sum of projections onto total spin $\tfrac32$ of each triple; every term is $\ge 0$, so $H \ge 0$, and any state on which no triple carries spin $\tfrac32$ is annihilated by every term and is therefore an exact zero-energy ground state with no diagonalization at all. The nearest-neighbor dimer coverings do exactly that. But the *output* of that construction is the wrong physics for my purpose: two degenerate ground states that break translation symmetry from period 1 to period 2, at half-integer spin — the symmetry-breaking side of the dichotomy, precisely as Affleck–Lieb requires. So I keep the recipe (a local Hamiltonian built as a sum of positive projectors, plus a hand-built state in the kernel of every term) and demand something the recipe has never yet produced: a *unique*, period-1, unbroken, integer-spin kernel state.

What forced Majumdar–Ghosh to dimerize was not the trick but a counting mismatch. A spin-$\tfrac12$ site carries exactly one singlet end, and a chain site has two neighbors, so a singlet consumes the whole site and the bond pattern cannot be uniform. Spin magnitude is what buys the missing bond-ends: a spin-$s$ is the fully symmetric part of $2s$ spin-$\tfrac12$'s, which in Schwinger-boson language is $|s,m\rangle \propto (a^\dagger)^{s+m}(b^\dagger)^{s-m}|0\rangle$ with a singlet bond between sites $i$ and $j$ written $(a^\dagger_i b^\dagger_j - b^\dagger_i a^\dagger_j)$. So if the number of bond-ends per site, $2s$, is made equal to the coordination number, every internal $\tfrac12$ can be used exactly once by a single singlet per link, with the same pattern at every site. On a chain the coordination is $2$, which selects $s = 1$ — an integer spin, exactly the case in question.

The method I propose is the valence-bond solid. Represent each spin-1 as two spin-$\tfrac12$'s symmetrized; on every nearest-neighbor link lock one $\tfrac12$ from each of the two sites into a singlet, exactly one bond per link; then symmetrize at each site to restore a genuine spin-1. In Schwinger bosons the state is

$$|\psi\rangle \;=\; \prod_i \left( a^\dagger_i b^\dagger_{i+1} - b^\dagger_i a^\dagger_{i+1} \right)|0\rangle ,$$

the valence-bond pattern being a rigid copy of the lattice itself — a *solid*, not a resonating superposition. Because the pattern is period-1 and there is only one way to lay it down, nothing is broken and nothing is degenerate in the bulk.

The parent Hamiltonian follows from a spin-counting fact about a single bond. Between neighboring sites $i$ and $i+1$ there are four constituent spin-$\tfrac12$'s, and one pair of them is locked into a singlet, which carries spin $0$; the two remaining free $\tfrac12$'s can combine to at most spin $1$. So the pair of physical spins can have total spin $0$ or $1$ but *never* $2$, even though two spin-1's generically span $0 \oplus 1 \oplus 2$. The valence-bond solid annihilates the top channel of every bond, so I take the local term to be exactly the projector onto that channel,

$$H \;=\; \sum_i P_2(\mathbf S_i + \mathbf S_{i+1}),$$

which is a sum of projectors, hence $H \ge 0$, and which annihilates $|\psi\rangle$ term by term, so $H|\psi\rangle = 0$ and $|\psi\rangle$ is an exact zero-energy ground state. Projecting out spin $2$ rather than spin $0$ or $1$ is forced, not chosen: the valence-bond solid genuinely has weight in the spin-0 and spin-1 channels of a bond (the two free $\tfrac12$'s really do mix singlet and triplet), so projecting either of those out would kill the state itself. Using exactly one singlet per bond is what makes the top spin $2 = 2s$ the unique forbidden channel.

Written as a spin polynomial the model is an honest isotropic chain. With $x \equiv \mathbf S_i\cdot\mathbf S_{i+1}$ and $(\mathbf S_i + \mathbf S_{i+1})^2 = 4 + 2x$, the total-spin-$J$ subspaces have $x = -2, -1, +1$ for $J = 0, 1, 2$, so the projector onto $J = 2$ is the quadratic that vanishes at $x = -1, -2$ and equals one at $x = +1$:

$$P_2 \;=\; \frac{(x+1)(x+2)}{6} \;=\; \tfrac16 x^2 + \tfrac12 x + \tfrac13, \qquad H = \sum_i \Big[\tfrac12\,\mathbf S_i\cdot\mathbf S_{i+1} + \tfrac16 (\mathbf S_i\cdot\mathbf S_{i+1})^2 + \tfrac13\Big].$$

Building the $9\times 9$ matrix and diagonalizing it confirms this is a genuine orthogonal projector and not merely a polynomial agreeing at three points: the eigenvalues are $\{0,0,0,0,1,1,1,1,1\}$ and $P_2^2 - P_2$ vanishes to machine precision, five ones and four zeros matching $\dim(J{=}2) = 5$ and $\dim(J{=}0\oplus 1) = 4$. Note the biquadratic coefficient is fixed and nonzero: since $2P_2 = \mathbf S_i\cdot\mathbf S_{i+1} + \tfrac13(\mathbf S_i\cdot\mathbf S_{i+1})^2 + \tfrac23$, the model sits at $\beta = -\tfrac13$ in the bilinear-biquadratic family $\sum_i[\mathbf S_i\cdot\mathbf S_{i+1} - \beta(\mathbf S_i\cdot\mathbf S_{i+1})^2]$, displaced from the realistic Heisenberg point $\beta = 0$. That displacement is the price of exact solvability, and I return to it below.

The same construction written site-by-site gives a matrix-product form of bond dimension $D = 2$: composing the per-site symmetrization with the bond singlet yields, for each physical $\sigma \in \{+1,0,-1\}$, a $2\times 2$ matrix on the auxiliary index,

$$B^{+1} = \sqrt{\tfrac23}\begin{pmatrix}0&1\\0&0\end{pmatrix},\quad B^{0} = -\tfrac{1}{\sqrt3}\begin{pmatrix}1&0\\0&-1\end{pmatrix},\quad B^{-1} = \sqrt{\tfrac23}\begin{pmatrix}0&0\\-1&0\end{pmatrix},$$

right-normalized, $\sum_\sigma B^\sigma B^{\sigma\dagger} = \mathbb 1$, with $|\psi\rangle = \mathrm{Tr}\,(B^{\sigma_1}\cdots B^{\sigma_N})$ on a ring. This is the computational handle on everything that follows.

Uniqueness has to be argued, not assumed, and the finite chain is instructive because it is *not* unique. Cutting the chain open leaves the leftmost and rightmost constituent $\tfrac12$'s without partners, so the open-chain state carries two free spinor indices $\Omega^{\alpha\beta}$ and there are four candidates. Their overlaps follow from a purely diagrammatic sum: each site contributes two pairings, index lines either close into loops (each closed loop $= \delta^\alpha_\alpha = 2$) or run between free ends, and counting $m$-loop diagrams gives $\sum_m \binom{L}{m+1}2^m = (3^L-1)/2$, so that for even $L$

$$\Omega^{\dagger\,\alpha\beta}\!\cdot\Omega_{\gamma\delta} \;=\; \delta^\alpha_\gamma \delta^\delta_\beta \,\frac{3^L-1}{2} \;+\; \delta^{\alpha\beta}\delta_{\gamma\delta},$$

with ring normalization $3^L + 3$. The four states are nonzero and independent, carry $S_z = 0,0,+1,-1$, and organize into a spin-1 triplet plus a spin-0 singlet: a four-fold degeneracy. Diagonalizing the open-chain Hamiltonian for $N = 3,4,5$ returns a kernel of dimension exactly $4$ at every length, with the next eigenvalue up near $0.41$–$0.5$. That the degeneracy does not grow with $N$ identifies it as a boundary effect — the two unpaired edge spin-$\tfrac12$'s, fractionalized half-spins living at the surface of a spin-1 chain — and it is a fingerprint of nontrivial bulk rather than a defect. Closing the chain into a ring pairs every constituent $\tfrac12$ and should collapse it, which it does: for $N = 4$ the ring ground energy is $0$ with a single ground state and a gap of $0.333333$; for $N = 6$ it is again $0$ with gap $0.347866$ — not shrinking as $N$ grows. Analytically, a ring ground state is in particular an open-chain ground state (the ring Hamiltonian is the open one plus an extra positive term), so it lies in the four-dimensional span; a spin-1 ring ground state would have to be the $S_z = +1$ member $\Omega^{11}$, whose index structure forces a $+$ at both site $1$ and site $L$, and two adjacent-on-the-ring $+$'s carry a spin-2 component that the extra bond term forbids. Only the spin-0 state survives, so the ring ground state is unique.

Expanding $\Omega$ in the $S^z$ product basis makes the physics of that state visible and hands over the correlations. The contraction rules allow only strings in which the nonzero spins strictly alternate in sign, separated by arbitrary runs of zeros: $\ldots 0 \ldots + \ldots 0 \ldots - \ldots 0 \ldots + \ldots$. This is a *diluted* antiferromagnet — perfect Néel order once the zeros are stripped out — which predicts a nonlocal string order even where the ordinary two-point function is short-ranged. Acting with $S^a_0 S^b_r$ breaks the bonds at $0$ and $r$ and inserts Pauli matrices at the dangling indices; diagrams in which both insertions sit on a short line vanish through $\mathrm{Tr}\,\sigma^a\,\mathrm{Tr}\,\sigma^b = 0$, and the survivors have two lines stretching from $0$ to $r$ carrying $\mathrm{Tr}(\sigma^a\sigma^b) = 2\delta^{ab}$ with the loop series running between them. Summing that geometric series, each closed loop contributing $2$ and each bond contraction $3^{-1}$, gives

$$\langle S^a_0 S^b_r\rangle \;=\; \delta^{ab}\,(-1)^r\,\tfrac43\,3^{-r}, \qquad r > 0.$$

A dropped factor or sign in a diagram sum of that size would be invisible, so I check it independently with the transfer matrix $T = \sum_\sigma \bar B^\sigma \otimes B^\sigma$. Its spectrum is $\{1, -\tfrac13, -\tfrac13, -\tfrac13\}$: a nondegenerate leading eigenvalue $1$, confirming the normalization, and a three-fold second eigenvalue $-\tfrac13$ giving $\xi = 1/\ln 3 = 0.910239$, against the code's $0.910239$; and evaluating $\langle S^z_0 S^z_r\rangle$ on a long ring for $r = 1,\dots,6$ reproduces $-0.444444$, $+0.148148$, $-0.049383$, $+0.016461$, $-0.005487$, $+0.001829$, matching $\tfrac43(-\tfrac13)^r$ to six digits in sign and magnitude. Correlations therefore decay exponentially with $\xi = 1/\ln 3 \approx 0.91$ lattice spacings, staggered in sign, with no Néel long-range order. The string operator $\lim \langle S^z_l \exp(i\pi\sum_{l<k<l'} S^z_k) S^z_{l'}\rangle$, evaluated at separation $60$ — far past $\xi$ — gives $-0.444444$, i.e. $-\tfrac49 = -(\tfrac23)^2$, so the hidden order predicted by the diluted-Néel reading is genuinely nonzero while the ordinary correlator has already fallen to $\sim 10^{-3}$ by $r = 6$. Isotropy also fixes the bond energy, $\langle \mathbf S_0\cdot\mathbf S_1\rangle = 3\langle S^z_0 S^z_1\rangle = -\tfrac43$.

Uniqueness in infinite volume comes from a factorization. Defining $\omega(A) = \lim_{L\to\infty}\langle \Omega^{(L)}, A\,\Omega^{(L)}\rangle/\langle \Omega^{(L)},\Omega^{(L)}\rangle$ and cutting the chain into a left block, a middle block containing the support of $A$, and a right block, the overlap formula above factorizes through the blocks and the left and right block factors contribute the same scalar to numerator and denominator, where they cancel. What is left is a finite-volume expectation over the middle block, independent of the boundary indices $\alpha,\beta$ — so all four boundary choices give the *same* infinite-volume state. To upgrade that to uniqueness among all infinite-volume ground states I need the finite-volume lemma: any state annihilated by every bond term is a linear combination of the four $\Omega_{\alpha\beta}$. It follows by induction — on two sites the general no-spin-2 state $\psi^{\alpha\gamma}A_\gamma$ has exactly four parameters, matching $\dim(0\oplus1)$; on three sites the absence of spin 2 on both bonds forces the two representations to agree and reduces the state to valence-bond form with one free end-tensor; and the induction step glues a site the same way, using the descent property that a ground state of $H_{1,n+1}$ is a ground state of $H_{1,n}$. So the kernel on any open chain is exactly four-dimensional at every length, which is precisely the numerical fact the diagonalizations reported. Combined with the block factorization, any infinite-volume ground state restricts to combinations of the $\Omega_{\alpha\beta}$ and hence agrees with $\omega$: the infinite-volume ground state is unique. The same factorization bounds all truncated correlations, $|\omega(AB) - \omega(A)\omega(B)| \le 3^{-(d-2)}\|A\|\,\|B\|$ for supports separated by $d$ sites, not just the spin-spin one.

The gap is the hard property, and the danger is the standard one: a sum of $L$ local terms can have its first excited state spread over the whole chain, so that the gap closes like $1/L$. I need a constant $\varepsilon > 0$ independent of $L$ with $H_{1,L} \ge \varepsilon P_L$, where $P_L$ projects onto the orthogonal complement of the ground space. The structural fact that makes this possible is the descent property again: a ground state of $H_{1,n}$ is a ground state of $H_{1,n-1}$, so the ground-space projectors $Q_n$ are *nested*, $Q_n \ge Q_{n+1}$, and $P_n = 1 - Q_n$ increases with $n$. On the common $(n+1)$-site space $Q_n$ is the four-dimensional first-$n$ ground space tensored with the full three-dimensional last site, hence $12$-dimensional, while $Q_{n+1}$ is the genuine four-dimensional ground space sitting inside it, so each increment $Q_n - Q_{n+1}$ has rank $8$. Monotonicity is the lever: telescope

$$P_L \;=\; \sum_{n=l}^{L-1}\big(P_{n+1} - P_n\big) \;+\; P_l$$

for a fixed window size $l$ chosen later, and bound each increment locally. Three estimates do it. First, splitting $|\langle x, \phi^i\rangle|^2 \le 2|\langle x, Q\phi^i\rangle|^2 + 2|\langle x, (1-Q)\phi^i\rangle|^2$ by Cauchy–Schwarz, where $Q$ projects onto the ground space of the $l$-site window Hamiltonian and the $\phi^i$ span the increment, gives $Q_n - Q_{n+1} \le 2\epsilon(l)\sum_i P(\psi^i_{n+1}) + (2/\varepsilon_{l+1})H_{n-l+1,n+1}$, using $1 - Q \le H_{\text{window}}/\varepsilon_{l+1}$ because anything orthogonal to the window ground space costs at least the window's own gap. Second, window projectors more than $l$ apart are mutually orthogonal, so grouping the increments into $l+1$ families of mutually orthogonal projectors bounds $\sum_n\sum_i P(\psi^i_{n+1}) \le 8(l+1)$. Third, and this is the crux, $\epsilon(l) \le c\,3^{-l}$: a state orthogonal to the $(n+1)$-site ground space has a nearly traceless boundary coefficient tensor — that is what the overlap formula $\delta\delta(3^n-1)/2 + \delta\delta$ says — and tracing it over a window of length $l$ costs a factor $3^{-l}$. The very same $3$ that sets the correlation length $1/\ln 3$ sets the rate at which a local wrong state decouples from the bulk ground space. Assembling,

$$P_L \;\le\; 16(l+1)\epsilon(l) \;+\; \Big[\frac{2(l+1)}{\varepsilon_{l+1}} + \frac{1}{\varepsilon_l}\Big] H_{1,L},$$

and since $16(l+1)\epsilon(l) \le 16(l+1)c\,3^{-l} \to 0$, I can fix an $l$ *independent of $L$* with $16(l+1)\epsilon(l) = \delta < 1$. On the excited subspace this reads $1 \le \delta + C\,H_{1,L}$, so $H_{1,L} \ge (1-\delta)/C$ there, with $C$ finite and $L$-independent. I never needed to know the size of the finite-window gaps $\varepsilon_l$, only that they are strictly positive, which they are because a finite chain is a finite matrix with a four-dimensional kernel — the $\approx 0.33$–$0.41$ first eigenvalues the small-$N$ diagonalizations returned. The gap survives the thermodynamic limit, and with the unique infinite-volume state $\omega$ in hand it transfers there: for local $A$ with $\omega(A) = 0$, $A\Omega^{(L)}$ is asymptotically entirely excited, so $\omega(A^*[H,A]) \ge \varepsilon\,\omega(A^*A)$.

Two loose ends deserve honest treatment. The first is whether $\beta = -\tfrac13$ is in the same phase as the realistic Heisenberg point $\beta = 0$; a gap forbids level crossings under small perturbations, which makes the connection plausible, and a variational comparison makes it concrete. Per bond, $E_{\text{VBS}} = -\tfrac43 - 2\beta$, $E_{\text{dimer}} = -1 - \tfrac83\beta$, $E_{\text{N\'eel}} = -1 - 2\beta$. At $\beta = 0$ the valence-bond solid gives $-1.333$ against $-1.000$ for both alternatives, and the finite-chain estimate of the true energy is $\approx -1.40$, so the trial state is high by only about $0.07$ per bond; the Néel state is never the lowest of the three; and the valence-bond and dimerized lines cross exactly at $\beta = \tfrac12$, beyond which the dimerized state wins, placing a transition near $\beta = 1/2$ with the Bethe-integrable point $\beta = 1$ on the dimerized, critical side. So the valence-bond solid is a good picture of the realistic massive phase even though it is the exact ground state only at $\beta = -\tfrac13$. The second is dimension. The construction generalizes the moment the condition is stated abstractly — bond-ends per site $2s$ must equal the coordination $z$ — so on any coordination-$z$ bipartite lattice take $s = z/2$, put one singlet on every link, and let $H = \sum_{\langle ij\rangle} P_z(\mathbf S_i + \mathbf S_j)$ project out the top spin of each bond. In one dimension the valence-bond solid exists only for integer spin, which is the structural reason integer and half-integer chains differ; above one dimension the criterion becomes the lattice rather than the parity of $2s$, and on the hexagonal lattice $z = 3$ gives the half-integer value $s = \tfrac32$ with an exact valence-bond ground state. There the one-dimensional geometric series becomes a sum over self-avoiding walks connecting the two points, with per-step weight bounded by $1/\sqrt6$ (an empty bond off the walk carries full weight, so every two steps gain $6$ rather than $4$) and at most $2^\ell$ walks of length $\ell$ at coordination $3$; since $2/\sqrt6 \approx 0.816 < 1$ the sum converges and gives exponential decay with $\xi_0 = 1/\ln(\sqrt6/2) \approx 4.93$, improving to $\approx 3.54$ with the exact connective constant — a disordered exact ground state above one dimension. But the intuition "valence-bond solid implies disorder" is false in general, and the Cayley tree shows exactly where it breaks: with polarizing boundary conditions the recursion for the central staggered magnetization flows to zero for $z = 3$ but converges to a strictly positive value for $z \ge 5$ ($z = 4$ is marginal), so a high-coordination tree valence-bond solid has Néel order, and there it is correspondingly gapless and non-unique, with a whole family of infinite-volume ground states. Low coordination is what starves the boundary of the room it needs to sustain order. Finally, the same telescoping argument applied to Majumdar–Ghosh proves that chain has a gap above its two dimerized ground states, so the symmetry-breaking side of the dichotomy is gapped too — degenerate rather than unique, which is exactly what the half-integer theorem demands.

Here are the two independent computations that pin the one-dimensional result down: exact diagonalization of $H = \sum_i P_2$ on a small ring, giving ground energy exactly $0$ and a finite gap, and the $D = 2$ matrix-product transfer matrix, giving leading eigenvalue $1$, next eigenvalue $-1/3$, hence $\xi = 1/\ln 3$ and $\langle S^z_0 S^z_r\rangle = \tfrac43(-\tfrac13)^r$.

```python
import numpy as np

Sz = np.diag([1.0, 0.0, -1.0])
Sp = np.array([[0, np.sqrt(2), 0], [0, 0, np.sqrt(2)], [0, 0, 0]], float)  # S^+
Sm = Sp.T.conj()
Sx, Sy = 0.5 * (Sp + Sm), -0.5j * (Sp - Sm)

def two_site_SS():
    return (np.kron(Sx, Sx) + np.kron(Sy, Sy) + np.kron(Sz, Sz)).real      # S_i . S_{i+1}

def P2():
    SS = two_site_SS()                                                      # spin-2 projector
    return (1/6) * (SS @ SS) + 0.5 * SS + (1/3) * np.eye(9)

def H_ring(N):
    dim = 3**N
    H = np.zeros((dim, dim))
    p2 = P2().reshape(3, 3, 3, 3)
    for i in range(N):
        j = (i + 1) % N
        full = np.zeros((dim, dim))
        for a in range(dim):
            idx = np.unravel_index(a, (3,) * N)
            for si in range(3):
                for sj in range(3):
                    amp = p2[si, sj, idx[i], idx[j]]
                    if amp == 0.0:
                        continue
                    nidx = list(idx); nidx[i] = si; nidx[j] = sj
                    b = np.ravel_multi_index(tuple(nidx), (3,) * N)
                    full[b, a] += amp
        H += full
    return H

def ed_demo(N=6):
    w = np.linalg.eigvalsh(H_ring(N))
    print(f"[ED N={N}] ground energy = {w[0]:.10f}  (VBS => 0)")
    print(f"[ED N={N}] gap = {w[1] - w[0]:.6f}")

sq = np.sqrt
B = {+1: sq(2/3) * np.array([[0, 1], [0, 0]], float),
      0: -1/sq(3) * np.array([[1, 0], [0, -1]], float),
     -1: sq(2/3) * np.array([[0, 0], [-1, 0]], float)}
sigma = {+1: 1.0, 0: 0.0, -1: -1.0}

def transfer(op=None):
    T = np.zeros((4, 4))
    for s in (+1, 0, -1):
        c = 1.0 if op is None else op[s]
        T += c * np.kron(B[s], B[s])
    return T

def mps_demo(rmax=6):
    T, Tz = transfer(), transfer(op=sigma)
    ev = sorted(abs(np.linalg.eigvals(T)))
    print(f"[MPS] leading eig = {ev[-1]:.6f} (=1); xi = {1/np.log(ev[-1]/ev[-2]):.6f} (1/ln3={1/np.log(3):.6f})")
    L = 200
    for r in range(1, rmax + 1):
        num = np.trace(np.linalg.matrix_power(T, L - r) @ Tz @ np.linalg.matrix_power(T, r - 1) @ Tz)
        den = np.trace(np.linalg.matrix_power(T, L))
        print(f"[MPS] <S0z S{r}z> = {num/den:+.6f}   (4/3)(-1/3)^{r} = {(4/3)*(-1/3)**r:+.6f}")

if __name__ == "__main__":
    ed_demo(6)
    mps_demo()
```
