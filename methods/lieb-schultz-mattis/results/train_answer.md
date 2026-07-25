The one-dimensional Heisenberg antiferromagnet with a spin-½ on every site,

$$H = \sum_i \mathbf{S}_i \cdot \mathbf{S}_{i+1},$$

is called a solved model, and in a formal sense it is: Bethe (1931) wrote its eigenstates as superpositions of plane waves of overturned spins with amplitudes pinned by two-body scattering phases and a set of coupled transcendental equations for the rapidities, and Hulthén (1938) used that framework to get the ground-state energy. Yet the two properties I most want are precisely the ones the construction does not hand over. Is the ground state unique, or do several states crowd the lowest energy? And is there a finite energy gap above it as the chain grows, or does the spectrum reach down to $E_0$ as $N \to \infty$? These control everything physical — whether the chain orders, how it responds to a low-frequency probe, whether the thermodynamics is activated or power-law. The Bethe ansatz has never yielded the long-range order in closed form, and the way it presents the spectrum makes the $N \to \infty$ scaling of the lowest excitation opaque; since in *any* finite chain a nondegenerate ground state is automatically separated from the rest by *some* gap, only that scaling is physical, and it is exactly what the rapidity equations do not put in my hand. The approximate routes are worse. Spin-wave theory (Anderson 1952; Kubo 1953) expands about a classical Néel state it has presupposed, which is not trustworthy in one dimension. The variational tradition — Hulthén (1938), Kasteleijn (1952), Marshall (1955), Ruijgrok and Rodriguez (1959) — produces excellent energies and contradictory verdicts on order, including a kink in the short-range order at a critical anisotropy that Orbach's (1958) anisotropic calculation showed to be spurious in one dimension. The bitter lesson is that a trial state can reproduce $E_0$ to high accuracy and still misrepresent the order and the gap: energy alone is a weak diagnostic. Marshall (1955), sharpening an observation of Peierls, comes closest — for a bipartite antiferromagnet the ground state is a singlet and its Ising-basis amplitudes obey a fixed sign rule — but that leaves the door open to additional degenerate ground states and says nothing at all about the spectrum just above.

So I will not try to out-Bethe Bethe. What I propose instead is a pair of variational theorems about the *genuine* isotropic model, each an inequality rather than a solution, because a variational principle survives even where the exact spectrum is out of reach: a sign-rule argument that forces the ground state to be nondegenerate, and a slowly winding **twist state**

$$\Psi_k = \exp\Big(i k \sum_n n\, S^z_n\Big)\, \Psi_0 \equiv \mathcal{O}^k \Psi_0, \qquad k = \frac{2\pi}{N},$$

which is provably orthogonal to the ground state and lies within $2\pi^2/N$ of it in energy, hence forces the gap to close. Alongside these, as an independent corroborating bridge, I solve the transverse (XY) chain exactly by a Jordan–Wigner string plus a Bogoliubov rotation, to see what a unique-and-gapless answer looks like in a model where every step is checkable.

Take the bridge first. What blocks a direct solution of the Heisenberg chain is the longitudinal term: in $\mathbf{S}_i\cdot\mathbf{S}_{i+1} = S^z_i S^z_{i+1} + \tfrac12(S^+_i S^-_{i+1} + S^-_i S^+_{i+1})$, it is $S^z S^z$ that will make any fermionization quartic. Drop it and keep an anisotropy $\gamma$ as a dial,

$$H_\gamma = \sum_i \big[(1+\gamma) S^x_i S^x_{i+1} + (1-\gamma) S^y_i S^y_{i+1}\big],$$

with $\gamma = 0$ the isotropic transverse point and $\gamma \to 1$ the Ising limit. In terms of the lowering operators $a_i = S^x_i - iS^y_i$, with $S^z_i = a^\dagger_i a_i - \tfrac12$, this is quadratic — but the $a$'s obey no single algebra. On one site $\{a_i, a^\dagger_i\} = 1$ and $a_i^2 = 0$ (a spin can be flipped at most once), which is fermionic; on different sites $[a_i, a_j] = [a_i, a^\dagger_j] = 0$, which is bosonic. A linear canonical transformation preserves whatever algebra it is fed, and here there is none to preserve, so a principal-axis rotation of the $a$'s just makes more of these paulions. Declaring them bosons throws away the hard core; declaring them fermions gets the cross-site signs wrong. The mismatch is *only* in the relative statistics at different sites, so the repair is to attach to $a_i$ an object that changes sign once for every excitation to its left. Since $a^\dagger_j a_j \in \{0,1\}$, $\exp(\pi i\, a^\dagger_j a_j) = 1 - 2a^\dagger_j a_j = (-1)^{n_j}$ is exactly that object, and the Jordan–Wigner string (Jordan and Wigner 1928)

$$c_i = \exp\Big[\pi i \sum_{j=1}^{i-1} a^\dagger_j a_j\Big] a_i, \qquad c^\dagger_i = a^\dagger_i \exp\Big[-\pi i \sum_{j=1}^{i-1} a^\dagger_j a_j\Big]$$

does the job. The string omits site $i$ itself, so $c^\dagger_i c_i = a^\dagger_i a_i$ and the on-site physics is untouched; for $j > i$ the string of $c_j$ *does* contain site $i$, and $\{a_i, 1 - 2a^\dagger_i a_i\} = 0$ (immediate from $a_i^2 = 0$) supplies the extra minus that turns the bosonic commutation into anticommutation. The result is genuine fermions, $\{c_i, c^\dagger_j\} = \delta_{ij}$, $\{c_i,c_j\} = 0$, and for a free-ended chain the adjacent strings cancel bond by bond:

$$H_\gamma = \tfrac12 \sum_{i=1}^{N-1} \big[(c^\dagger_i c_{i+1} + \gamma\, c^\dagger_i c^\dagger_{i+1}) + \text{h.c.}\big], \qquad \mathfrak{N} = \sum_i c^\dagger_i c_i = \sum_i (S^z_i + \tfrac12),$$

free spinless fermions hopping, with a pairing term when $\gamma \neq 0$, and each fermion carrying one unit of up-spin. One subtlety deserves flagging: on a cyclic chain the bond from site $N$ to site $1$ carries the *full* string, so it comes with a factor set by the parity of $\mathfrak{N}$ — periodic spin boundary conditions map to periodic or antiperiodic fermion boundary conditions according to whether $\mathfrak{N}$ is odd or even. That boundary term is $O(1/N)$ in macroscopic quantities, so for the bulk spectrum I work with the plainly periodic ("c-cyclic") problem and carry the parity caveat along.

Diagonalizing is then a general question about a real quadratic Fermi form $H = \sum_{ij}[c^\dagger_i A_{ij} c_j + \tfrac12(c^\dagger_i B_{ij} c^\dagger_j + \text{h.c.})]$ with $A$ symmetric and $B$ antisymmetric. Seek normal modes $\eta_k = \sum_i (g_{ki} c_i + h_{ki} c^\dagger_i)$ satisfying $[\eta_k, H] = \Lambda_k \eta_k$; matching the coefficients of $c_i$ and $c^\dagger_i$ separately gives two coupled equations for $g$ and $h$, and their sum and difference, $\phi_k = g_k + h_k$ and $\psi_k = g_k - h_k$, decouple them into the symmetric pair

$$\phi_k (A - B) = \Lambda_k \psi_k, \qquad \psi_k (A + B) = \Lambda_k \phi_k \quad \Longrightarrow \quad \phi_k (A-B)(A+B) = \Lambda_k^2\, \phi_k.$$

Because $(A+B)^T = A - B$, the operator $(A-B)(A+B) = (A+B)^T(A+B)$ is symmetric positive semidefinite, so $\Lambda_k^2 \ge 0$ automatically — every mode energy is real and the $\phi_k$ can be chosen real and orthogonal. Fixing the constant from the invariance of $\operatorname{tr} H$ under a canonical transformation,

$$H = \sum_k \Lambda_k\, \eta^\dagger_k \eta_k + \tfrac12\Big(\sum_i A_{ii} - \sum_k \Lambda_k\Big).$$

For the c-cyclic XY chain $A$ is $\tfrac12$ on nearest neighbours and $B$ is $\tfrac12\gamma$ antisymmetric on nearest neighbours, the eigenvectors are the plane waves $\phi_{kj} = \sqrt{2/N}\,\sin kj$ or $\cos kj$ with $k = 2\pi m/N$, and

$$\Lambda_k^2 = 1 - (1-\gamma^2)\sin^2 k,$$

taken with $\Lambda_k \ge 0$ (the particle-hole convention, which makes the ground state the fermion vacuum $\eta_k \Psi_0 = 0$ with every excitation positive; the alternative bookkeeping is a filled Fermi sea for $|k| > \pi/2$ and gives the same physics). Then $E_0 = -\tfrac12\sum_k \Lambda_k$, and in the thermodynamic limit $E_0/N = -(1/\pi)\,\mathcal{E}(1-\gamma^2)$ with $\mathcal{E}$ the complete elliptic integral, interpolating smoothly from $-1/\pi$ at $\gamma = 0$ to $-1/2$ at $\gamma = 1$. The payoff is the gap: $\Lambda_k^2$ is minimized at $\sin^2 k = 1$, i.e. $k = \pm\pi/2$, where it equals $\gamma^2$, so the gap is $|\gamma|$ and closes *only* at the isotropic point, where $\Lambda_k = |\cos k|$ vanishes linearly, $\Lambda_{\pi/2+q} \approx |q|$. With $N$ even and not a multiple of four, so that no exact zero mode sits at the Fermi points, the ground state is nondegenerate. A unique ground state with a gapless linearly dispersing spectrum, in a fully soluble quantum model. But I must be honest about what this is not: reinstating $S^z_i S^z_{i+1} = (n_i - \tfrac12)(n_{i+1} - \tfrac12)$ turns the fermions into an interacting nearest-neighbour density-density problem and destroys the diagonalization outright. The XY chain corroborates; it does not prove.

For the genuine model, uniqueness comes first, and it is a variational argument about signs. Work in the $S^z_{\text{total}} = 0$ sector, legitimate because $[S^z_{\text{total}}, H] = 0$ and this is where the antiferromagnetic ground state lives, with the Ising configurations $\Phi_\mu$ ($N/2$ up, $N/2$ down) as a complete basis. Rotate every spin on the B sublattice by $\pi$ about $z$, so $S^{x,y}_j \to -S^{x,y}_j$ and $S^z_j \to S^z_j$. The diagonal part is untouched but the flip-flop changes sign,

$$H' = \sum S^z_i S^z_j - \tfrac12 \sum (S^+_i S^-_j + S^-_i S^+_j),$$

so in the Ising basis *every* off-diagonal matrix element is $\le 0$. Writing $\Psi_0 = \sum_\mu C_\mu \Phi_\mu$ with $C$ real (H is real), Schrödinger's equation reads $(E - \varepsilon_\mu) C_\mu = \tfrac12 \sum_{\mu'(\mu)} C_{\mu'}$, where $\varepsilon_\mu$ is the diagonal $S^zS^z$ energy and $\mu'(\mu)$ runs over the configurations the flip-flop connects to $\mu$. Suppose some ground state had $C_\mu = 0$ on a set $\{\mu_1,\dots,\mu_r\}$. At one such $\mu_p$ the connected amplitudes cannot all vanish — otherwise $H'$ would decompose into blocks, impossible because the flip-flop ultimately connects every $S^z = 0$ configuration to every other — so $0 = \sum_{\mu'(\mu_p)} C_{\mu'}$ forces nonzero amplitudes of *both* signs. Now compare $\Psi_0$ with the trial state $\Psi_0' = \sum_\mu |C_\mu| \Phi_\mu$. On one hand $\Psi_0'$ cannot be an eigenstate, since $|C_{\mu_p}| = 0$ while $\sum_{\mu'(\mu_p)}|C_{\mu'}| \neq 0$, so the variational principle gives strictly $E_0' > E_0$. On the other hand, because every off-diagonal element carries a minus sign and $|C_\mu||C_{\mu'}| \ge C_\mu C_{\mu'}$ term by term,

$$E_0' = \sum_\mu \varepsilon_\mu C_\mu^2 - \tfrac12 \sum_\mu \sum_{\mu'(\mu)} |C_\mu||C_{\mu'}| \;\le\; \sum_\mu \varepsilon_\mu C_\mu^2 - \tfrac12 \sum_\mu \sum_{\mu'(\mu)} C_\mu C_{\mu'} = E_0,$$

a flat contradiction. Hence no amplitude vanishes. The same inequality then delivers the sign rule: for a genuine ground state equality must hold, which requires every connected product $C_\mu C_{\mu'}$ to be positive, and connectivity propagates that to all of them — the Marshall–Peierls rule, now as a strict statement. Uniqueness follows at once. Two ground states in the $S^z = 0$ sector would both have all-same-sign amplitudes and therefore could not be orthogonal, so there is exactly one. Since at least one ground state is a singlet, and any *additional* ground state of any multiplicity would necessarily contain an $S^z = 0$ member, there can be no other. The ground state is nondegenerate — strictly stronger than the sign rule alone, which had left extra, possibly non-singlet, degenerate states open. Nothing in this argument used one-dimensionality or nearest-neighbour range; only bipartiteness and the sign of the bonds, so it holds on any bipartite lattice in any dimension.

The gap is the harder half, and it is where the twist state earns its keep. I need an *upper* bound on the energy of the first excited state, which means a trial state that is both low in energy and *provably* orthogonal to $\Psi_0$ — orthogonality is the whole trap, since a trial state with any overlap merely re-bounds $E_0$ and says nothing about a gap. A uniform rotation $\exp(i\theta\sum_n S^z_n)$ commutes with $H$ and returns $\Psi_0$ unchanged, so it is useless; what I want is the gentlest possible deformation that is *not* a symmetry. That is a rotation about $z$ whose angle winds slowly along the chain, by $kn$ at site $n$: neighbouring spins are then rotated by angles differing only by $k$, each bond is distorted by $O(k)$, and the total energy cost is $O(Nk^2)$, which for $k \propto 1/N$ is $O(1/N)$. Hence $\Psi_k = \mathcal{O}^k \Psi_0$ with $\mathcal{O}^k = \exp(ik\sum_n n S^z_n)$, the same slow global twist Bloch used for persistent currents.

Orthogonality comes from translation. Let $U_z$ shift every spin by one site cyclically, $U_z \mathbf{S}_i U_z^{-1} = \mathbf{S}_{i+1}$ with $\mathbf{S}_{N+1} = \mathbf{S}_1$. Since $[H, U_z] = 0$ and the ground state has just been proved nondegenerate, $U_z \Psi_0 = e^{i\alpha}\Psi_0$ for some phase, so inserting $U_z^{-1}U_z$ costs nothing: $\langle \Psi_0 | \Psi_k\rangle = \langle \Psi_0 | U_z \mathcal{O}^k U_z^{-1} | \Psi_0\rangle$. Shifting $n \to n+1$ inside the exponent produces the wrap-around term at site $N$ and an overall subtraction,

$$U_z \mathcal{O}^k U_z^{-1} = \mathcal{O}^k \exp\big(ikN S^z_1\big) \exp\Big(-ik\sum_n S^z_n\Big).$$

The last factor is harmless because $\Psi_0$ is a singlet, $\sum_n S^z_n \Psi_0 = 0$. What is left is the single-site factor $\exp(ikN S^z_1)$, and $S^z_1 = \pm\tfrac12$, so with $k = 2\pi m/N$ we get $kN S^z_1 = \pm\pi m$ and $\exp(ikNS^z_1) = (-1)^m$ on *both* eigenvalues. Choosing $m$ odd makes this exactly $-1$, whence

$$\langle \Psi_0 | \Psi_k\rangle = -\langle \Psi_0 | \Psi_k\rangle = 0.$$

That minus sign is the load-bearing element of the entire theorem, and it exists only because the spin is half-integer: for integer $S$ one has $kNS^z_1 = 2\pi \times \text{integer}$, the factor is $+1$, the orthogonality collapses, and there is no argument at all. Gaplessness here is not a generic property of spin chains — it is bolted to the half-integer spin.

The energy bound uses the same $k$. Under the twist the in-plane components rotate, $\mathcal{O}^{-k} S^x_n \mathcal{O}^k = S^x_n \cos kn + S^y_n \sin kn$ and $\mathcal{O}^{-k} S^y_n \mathcal{O}^k = -S^x_n \sin kn + S^y_n \cos kn$, while $S^z_n$ is untouched, so the longitudinal part of $H$ drops out of the deformation entirely and

$$\mathcal{O}^{-k} H \mathcal{O}^k = H + (\cos k - 1)\sum_n (S^x_n S^x_{n+1} + S^y_n S^y_{n+1}) + \sin k \sum_n (S^x_n S^y_{n+1} - S^y_n S^x_{n+1}).$$

In $\Psi_0$ the first term gives $E_0$. The second has $\cos k - 1 = -\tfrac12 (2\pi/N)^2 + O(N^{-4})$ multiplying a bounded in-plane bond sum of $O(N)$ terms, so it contributes at most $(2\pi/N)^2 \cdot (N/2) + O(N^{-3}) = 2\pi^2/N + O(N^{-3})$. The third vanishes identically: $\sum_n (S^x_n S^y_{n+1} - S^y_n S^x_{n+1})$ is proportional to the commutator $[\sum_n n S^z_n, H]$, whose expectation in an energy eigenstate is zero. Taking $m = 1$, which is odd and therefore serves the orthogonality argument at the same time,

$$\langle \Psi_k | H | \Psi_k \rangle \le E_0 + \frac{2\pi^2}{N}.$$

So for every even $N$ there exists a state orthogonal to the nondegenerate ground state whose energy lies within $2\pi^2/N$ of $E_0$; the first excitation energy is at most $2\pi^2/N$, and it vanishes in the thermodynamic limit. The two theorems together give the result I set out for: the isotropic spin-½ Heisenberg antiferromagnetic chain has a unique ground state, necessarily a singlet, and a gapless excitation spectrum above it — obtained without any spectrum, without the Bethe equations, from two variational inequalities and a half-integer minus sign. It is worth noting what the twist argument actually consumed: translation invariance, a nondegenerate singlet ground state, and a transverse coupling for $\cos k - 1$ to act on. Nothing about range or dimensionality. On a torus of $N$ sites along $x$ and $M = O(N^\nu)$ along $y$ with $0 < \nu < 1$, twisting every row, $\mathcal{O}^k = \exp(ik\sum_{n,m} n S^z_{n,m})$, the orthogonality goes through unchanged and the bound becomes $\langle \Psi_k|H|\Psi_k\rangle \le E_0 + 2\pi^2/N^{1-\nu}$, still vanishing. For a genuinely two-dimensional $N \times N$ lattice, however, the twist is too crude a probe to be decisive — the trial state is no longer close enough to a true low-lying excitation — so the clean statement is the one-dimensional one, with the anisotropic-torus extension flagged as suggestive rather than conclusive.

Finally, the soluble bridge in explicit form, since it is the one place where every number can be checked against a closed form: the single-particle energies $\Lambda_k^2 = 1 - (1-\gamma^2)\sin^2 k$, the gap $\min_k \Lambda_k = |\gamma|$ closing only at $\gamma = 0$, and $E_0/N = -\tfrac{1}{2N}\sum_k \Lambda_k \to -(1/\pi)\,\mathcal{E}(1-\gamma^2)$, with the limits $-1/\pi$ at the isotropic point and $-1/2$ in the Ising limit.

```python
import numpy as np
from scipy.special import ellipe

def xy_spectrum(N, gamma):
    """Exact single-particle energies Lambda_k of the c-cyclic spin-1/2 XY chain
    via Jordan-Wigner + Bogoliubov:  Lambda_k^2 = 1 - (1 - gamma^2) sin^2 k."""
    m = np.arange(-N // 2, N // 2)        # k = 2*pi*m/N
    k = 2 * np.pi * m / N
    Lambda = np.sqrt(1.0 - (1.0 - gamma**2) * np.sin(k)**2)   # particle-hole convention, >= 0
    return k, Lambda

def xy_ground_state_energy_per_site(N, gamma):
    """E_0 = -1/2 sum_k Lambda_k ; per site."""
    _, Lambda = xy_spectrum(N, gamma)
    return -0.5 * Lambda.sum() / N

def xy_gap(N, gamma):
    """Minimum excitation energy = |gamma|, closing only at the isotropic point gamma=0."""
    _, Lambda = xy_spectrum(N, gamma)
    return Lambda.min()

if __name__ == "__main__":
    N = 20000
    # Isotropic point: gapless, E_0/N -> -1/pi
    print("gamma=0  E0/N =", xy_ground_state_energy_per_site(N, 0.0),
          " (-1/pi =", -1/np.pi, "),  gap =", xy_gap(N, 0.0))
    # Ising limit: gapped, E_0/N -> -1/2
    print("gamma=1  E0/N =", xy_ground_state_energy_per_site(N, 1.0),
          " (-1/2),  gap =", xy_gap(N, 1.0))
    # Thermodynamic-limit ground energy from the elliptic integral: -(1/pi) E(1 - gamma^2)
    for g in (0.0, 0.5, 1.0):
        print(f"gamma={g}: -(1/pi)*ellipe(1-g^2) =", -(1/np.pi) * ellipe(1.0 - g**2))
```
