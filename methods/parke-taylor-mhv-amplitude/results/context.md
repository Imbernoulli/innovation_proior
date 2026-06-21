# Context: a closed form for tree-level multi-gluon scattering

## Research question

At hadron colliders the dominant high-energy processes are the scattering of
gluons into many gluons, $gg \to (n{-}2)g$. The cross sections for these
many-jet final states are the irreducible QCD background to essentially every
search for new physics at the $S\bar p p S$, the Tevatron, and the proposed
Superconducting Super Collider. To predict them one needs the tree-level
$n$-gluon scattering amplitude in $SU(N)$ Yang-Mills theory, squared and summed
over colors, as a function of the external momenta and helicities.

The obstacle is combinatorial explosion. The number of Feynman diagrams for
$gg \to ng$ grows faster than factorially:

| $n$ | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| # diagrams | 4 | 25 | 220 | 2485 | 34300 | 559405 | 10525900 |

because the non-Abelian theory has both cubic and quartic gluon
self-interactions. Each diagram is itself a string of Lorentz contractions of
polarization vectors with momentum-dependent vertices. By the mid-1980s the
four-, five-, and six-gluon amplitudes had been obtained by heroic direct
computation, but a general-$n$ result was out of reach, and four-jet production
in particular had no analytic answer. The goal: a formula for the squared,
color-summed $n$-gluon amplitude valid for arbitrary $n$ — ideally one simple
enough to evaluate numerically inside a jet Monte Carlo.

## Background

**Massless kinematics and the helicity basis.** For massless external states the
cross section is built by summing the *squares* of fixed-helicity amplitudes
incoherently (different helicity configurations do not interfere). The advantage
over the textbook spin-summed "Casimir trick" is decisive at high multiplicity:
squaring a generic amplitude produces $\sim(n!)^2$ diagram-times-diagram
interference terms, whereas one can instead compute each helicity amplitude as a
single complex number and then sum $|A|^2$ over the few nonzero helicity
assignments. This helicity-amplitude program was pioneered by Bjorken and Chen
and by Reading-Henry, and developed into a working tool by the CALKUL
collaboration and in the Gastmans-Wu textbook.

**Spinor variables.** A massless momentum $k^\mu$ ($k^2=0$) factorizes:
$k_{\alpha\dot\alpha} = k_\mu(\sigma^\mu)_{\alpha\dot\alpha} =
\lambda_\alpha\tilde\lambda_{\dot\alpha}$, a product of a two-component
left-handed spinor $\lambda$ and a right-handed spinor $\tilde\lambda$. From
these one forms the antisymmetric Lorentz invariants

$$\langle i\,j\rangle = \epsilon^{\alpha\beta}(\lambda_i)_\alpha(\lambda_j)_\beta,
\qquad [i\,j] = \epsilon_{\dot\alpha\dot\beta}(\tilde\lambda_i)^{\dot\alpha}(\tilde\lambda_j)^{\dot\beta}.$$

The basic identities (all pre-method, true of any massless momenta):
antisymmetry $\langle ij\rangle = -\langle ji\rangle$, $\langle ii\rangle=0$;
the squaring relation $\langle i\,j\rangle[j\,i] = s_{ij} = 2k_i\cdot k_j$;
for real momenta $[ij]=\langle ij\rangle^*$, so that $|\langle ij\rangle|^2 =
s_{ij}$ — the brackets are *complex square roots* of the momentum invariants,
carrying a phase $\langle ij\rangle = \sqrt{s_{ij}}\,e^{i\phi_{ij}}$. Momentum
conservation reads $\sum_j\langle i\,j\rangle[j\,k]=0$, and there is the Schouten
identity $\langle ij\rangle\langle kl\rangle - \langle ik\rangle\langle jl\rangle
= \langle il\rangle\langle kj\rangle$. All particles are taken outgoing, with
helicity labeled as if outgoing, so $\sum_i k_i^\mu = 0$.

**Polarization vectors with a reference momentum.** A massless vector of momentum
$k$ and helicity $\pm$ can be written with an arbitrary auxiliary null
"reference momentum" $q$ (Xu, Zhang & Chang; also in the CALKUL line):

$$\epsilon^\pm_\mu(k,q) = \pm\frac{\langle q\mp|\gamma_\mu|k\mp\rangle}{\sqrt2\,\langle q\mp|k\pm\rangle}.$$

This is transverse, $\epsilon^\pm\cdot k = 0$, for any $q$; changing $q$ shifts
$\epsilon$ by a multiple of $k^\mu$, which is the residual on-shell gauge freedom
and leaves any amplitude invariant. The freedom is exploitable: with a common
reference momentum, like-helicity polarizations satisfy
$\epsilon^+(k_i,q)\cdot\epsilon^+(k_j,q)=0$, and one can also arrange
$\epsilon^+(k_i,q)\cdot\epsilon^-(k_j,k_i)=0$. An earlier CALKUL representation
expressed $\epsilon$ through a *pair* of external charged-fermion momenta, which
works well in QED but is awkward for a pure-gluon process (no external fermions)
and forces heavy bookkeeping of relative phases between gauge-invariant subsets.

**Color decomposition.** A tree gluon amplitude can be expanded on a basis of
color traces,
$\mathcal{M}_n = \sum_{P'}\mathrm{tr}(\lambda^{a_1}\cdots\lambda^{a_n})\,
m(1,\dots,n)$, summed over the $(n{-}1)!$ non-cyclic permutations, where the
$\lambda^a$ are $SU(N)$ generators. The kinematic coefficients $m(1,\dots,n)$ —
the color-ordered partial amplitudes — are individually gauge invariant,
invariant under cyclic permutations of their arguments, satisfy a dual Ward
identity ($\sum$ over cyclic insertions of one leg vanishes), and factorize on
physical poles. At leading order in $N$ the partial amplitudes add
incoherently in the squared, color-summed amplitude. This isolates the genuine
kinematic object: a single cyclically-ordered partial amplitude.

**Universal factorization (the constraints any amplitude must obey).** Tree
amplitudes have only two kinds of singularity. As an emitted gluon $s$ becomes
*soft* ($k_s\to0$) between color-adjacent hard legs $a,b$, the amplitude
factorizes as $A_n \to \mathrm{Soft}(a,s^\pm,b)\,A_{n-1}$ with a universal
eikonal factor independent of the species and helicity of $a,b$. As two
color-adjacent legs become *collinear* ($k_a\parallel k_b$, with longitudinal
fractions $z$ and $1-z$), the amplitude factorizes as
$A_n \to \sum_{\lambda}\mathrm{Split}_{-\lambda}(a,b;z)\,A_{n-1}$, where the
splitting amplitudes are the (square roots of the) Altarelli-Parisi splitting
functions $P(z)$ that govern collinear parton evolution. In a massless gauge
theory the collinear pole is softened from $1/s_{ab}$ to $1/\sqrt{s_{ab}}$ and
acquires an azimuthal phase, because angular momentum along the collinear axis is
mismatched by the intermediate gluon's spin. These soft and collinear limits are
*data the answer must reproduce*, and they involve $\langle ab\rangle$ (or its
conjugate $[ab]$) in the denominator, not the bare invariant $s_{ab}$.

**Supersymmetry relations as a computational shortcut.** At tree level, pure-gluon
amplitudes are identical in ordinary Yang-Mills and in its supersymmetric
extension, because no scalar or fermionic state can propagate internally in a
pure-glue tree. Hence gluon amplitudes obey the supersymmetry Ward identities of
$\mathcal{N}{=}1$ super Yang-Mills (Grisaru, Pendleton & van Nieuwenhuizen;
Grisaru & Pendleton). The supersymmetry charge $Q(\eta)$ rotates a gluon
$g^\pm$ into a gluino $\Lambda^\pm$ and vice versa; demanding that
$\langle[Q,\,z_1z_2\cdots z_n]\rangle = 0$ gives linear relations among
amplitudes of different external spin. Combined with fermion helicity
conservation, these relations constrain which helicity configurations can be
nonzero and tie gluon amplitudes to easier-to-compute amplitudes containing a
gluino pair.

## Baselines

**Direct Feynman-diagram computation.** Write every diagram from the QCD Feynman
rules, contract, square, sum over colors and helicities. Correct in principle and
the source of the known four-, five-, and six-gluon results (Gottschalk & Sivers;
Berends, Kleiss, De Causmaecker, Gastmans & Wu; Kunszt). The limitation is the
diagram count above: beyond five gluons the bookkeeping of the non-Abelian
vertices is unmanageable, the intermediate expressions are enormous, and there is
no closed-form $n$-dependence — each multiplicity is a separate heroic
calculation.

**Helicity amplitudes with reference-momentum polarizations.** Fix the external
helicities, choose the auxiliary reference momenta cleverly so that many diagrams
vanish through $\epsilon_i\cdot\epsilon_j=0$ or $\epsilon_i\cdot k_j=0$, compute
each surviving diagram as a spinor expression, and sum $|A|^2$ over the nonzero
helicity assignments. This makes the individual four- and five-gluon amplitudes
strikingly compact, and it is the right language for the problem. But by itself it
is still a per-process, per-helicity diagram computation: it shrinks each
calculation without supplying a closed form valid for all $n$.

**Color (dual) decomposition.** Reduces the colored amplitude to gauge-invariant,
cyclically-ordered partial amplitudes that factorize cleanly and decouple at
leading $N$. This removes color from the kinematic problem and is essential
scaffolding. The gap it leaves is the whole problem: the partial amplitude
itself, for general $n$ and a given helicity configuration, still has to be
determined.

**Supersymmetry Ward identities.** Prove vanishing theorems and relate helicity
sectors across spins. They establish which configurations are forced to be zero
and convert a gluon amplitude into a fermionic one with fewer diagrams. What they
do not directly hand over is the explicit closed form of the simplest
nonvanishing sector.

**Altarelli-Parisi collinear behavior.** Fixes the residue of every collinear
pole of any amplitude. This is not a method for computing the amplitude; it is a
nonlinear constraint the answer must satisfy in every collinear limit — and a
demanding consistency check on any proposed general-$n$ formula.

## Evaluation settings

The natural yardsticks are the multi-jet processes that motivate the calculation:
$gg \to ng$ at the $S\bar p p S$ and Tevatron and (prospectively) the SSC, with
the squared color-summed amplitude feeding parton-level cross sections for
$n$-jet rates (the phenomenology framework of Eichten, Hinchliffe, Lane & Quigg).
Internal correctness checks available at the time: agreement with the
independently-computed four-, five-, and six-gluon amplitudes (the last
verifiable numerically at a phase-space point); correct mass dimension, little-group
weight, and Bose/cyclic symmetry; and — the sharpest test — consistency with the
Altarelli-Parisi collinear factorization in every pair of adjacent legs, for all
$n$.

## Code framework

The pre-method tooling is a numerical kinematics layer: generate null momenta,
factor each into spinors, and evaluate the basic spinor invariants. On top of it
sits an empty slot for the partial amplitude to be determined, plus the
consistency checks the answer will have to pass.

```python
import numpy as np

def random_null(n, rng):
    """n outgoing null four-momenta p=(E, px, py, pz), E=|vec p|."""
    out = []
    for _ in range(n):
        v = rng.normal(size=3)
        out.append(np.array([np.linalg.norm(v), *v]))
    return out

def mink(p, q):                       # (+---) Minkowski product
    return p[0]*q[0] - np.dot(p[1:], q[1:])

def spinors(p):
    """Factor the rank-1 bispinor p_{a adot} = lambda_a * lambdatilde_adot
       (lambdatilde = lambda^* up to phase, for real momenta)."""
    E, px, py, pz = p
    M = np.array([[E + pz, px - 1j*py],
                  [px + 1j*py, E - pz]], dtype=complex)
    lam = M[:, 0]/np.sqrt(M[0, 0]) if abs(M[0, 0]) > 1e-9 else M[:, 1]/np.sqrt(M[1, 1])
    a = 0 if abs(lam[0]) >= abs(lam[1]) else 1
    lamt = M[a, :]/lam[a]
    return lam, lamt

class Kinematics:
    def __init__(self, ps):
        self.ps = ps
        self.S = [spinors(p) for p in ps]
    def ang(self, i, j):              # <ij>
        li, _ = self.S[i]; lj, _ = self.S[j]
        return li[0]*lj[1] - li[1]*lj[0]
    def sq(self, i, j):               # [ij]
        _, ti = self.S[i]; _, tj = self.S[j]
        return ti[0]*tj[1] - ti[1]*tj[0]
    def s(self, i, j):                # s_ij = 2 p_i . p_j
        return 2*mink(self.ps[i], self.ps[j])

def partial_amplitude(K, n, neg):
    """Color-ordered tree partial amplitude for a given helicity assignment;
       neg = the legs carrying negative helicity."""
    # TODO: the closed form we are after, as a function of the spinor brackets
    pass

# Consistency checks the answer must satisfy:
#   - <ab>[ba] == 2 p_a . p_b   (brackets are square roots of invariants)
#   - the squared amplitude reduces to a ratio of momentum invariants
#   - the collinear limit factorizes:  m_n -> Split(z) * m_{n-1}
```
