# Context: closed-form tree amplitudes for many-gluon scattering

## Research question

At the energies of the CERN SppS collider and the Fermilab Tevatron — and at the
energies planned for a future Superconducting Super Collider — the dominant hard
processes are QCD parton scatterings that produce many energetic, widely separated
jets. To use multi-jet events as a probe of new physics (heavy-particle cascades,
W/Z pairs decaying to jets, supersymmetric-particle chains), one must first predict
the *standard* QCD multi-jet rates that form the background. That requires the
tree-level matrix elements for processes with many final-state partons, above all
the purely gluonic process gluon + gluon → (n−2) gluons.

The obstacle is combinatorial. The number of Feynman diagrams for gg → n g grows
explosively:

| n (final gluons) | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| # diagrams | 4 | 25 | 220 | 2485 | 34300 | 559405 | 10525900 |

The non-abelian three- and four-gluon vertices then multiply each diagram into many
terms, so symbolic or numerical evaluation chokes well before n is large. By the
mid-1980s the four-, five-, and six-gluon amplitudes had been obtained — at great
labor — but there were essentially no analytic results beyond, and four-jet
production (the n = 4 case from a 2 → 4 viewpoint) had no compact analytic form.

The precise goal: find an *on-shell, gauge-invariant, closed-form* expression — one
that holds for an arbitrary number of external gluons n — for at least one
nontrivial helicity configuration of the tree-level n-gluon amplitude, and do it to
leading order in the number of colors so the color algebra does not reintroduce the
combinatorial blow-up. Such a formula would be the first time a nontrivial
on-mass-shell amplitude in a non-abelian gauge theory is written down for arbitrary
external multiplicity, and it would immediately feed the multi-jet Monte Carlo
programs.

## Background

**Helicity amplitudes.** For massless external particles it is far cheaper to compute
amplitudes for definite external helicities than to spin-sum from the start.
Different helicity configurations do not interfere, so the cross section is the
incoherent sum of squared helicity amplitudes. Massless fermions have conserved
helicity (chirality = helicity, preserved by gauge couplings), so many configurations
vanish outright. This program was pioneered by Bjorken and Chen and by
Reading-Henry, and developed into a working technology by the CALKUL collaboration
(and reviewed in the Gastmans–Wu book). The CALKUL trick writes a photon
polarization contracted with γ in terms of the momenta of an external charged-fermion
pair, so that whole sets of terms vanish by gauge invariance and by helicity
conservation along the fermion line.

**Spinor variables.** A massless momentum k^μ (k²=0) factorizes into a pair of
two-component Weyl spinors, k_{a ȧ} = λ_a λ̃_ȧ. The Lorentz-invariant contractions
are the spinor products

  ⟨ij⟩ = ε^{ab} λ_{i,a} λ_{j,b},  [ij] = ε^{ȧḃ} λ̃_{i,ȧ} λ̃_{j,ḃ},

which are antisymmetric and satisfy ⟨ij⟩[ji] = s_ij = 2 p_i·p_j and the momentum-
conservation relation Σ_j ⟨ij⟩[jk] = 0, plus the Schouten identity
⟨ij⟩⟨kl⟩ − ⟨ik⟩⟨jl⟩ = ⟨il⟩⟨kj⟩. For real momenta [ij] = ⟨ij⟩*, so the spinor
products are complex square roots of the Lorentz invariants: ⟨ij⟩ = √s_ij e^{iφ},
[ij] = √s_ij e^{−iφ}. This square-root-plus-phase structure is exactly what a
collinear singularity looks like (the amplitude goes like 1/√s_ij with an
azimuthal phase as two momenta become parallel).

**Spinor-helicity polarization vectors.** The polarization vector of a massless
gauge boson of definite helicity can be built from two massless spinors and a single
arbitrary *reference momentum* k (a null vector, k·p ≠ 0):

  ε±_μ(p,k) = ± ⟨p∓|γ_μ|k∓⟩ / (√2 ⟨k∓|p±⟩).

This obeys ε·p = 0, ε·k = 0, and ε⁺(p,k)·ε⁺(p′,k) = 0,
ε⁺(p,k)·ε⁻(k,k′) = 0. The reference momentum encodes the residual on-shell gauge
freedom ε → ε + α p; it can be chosen *independently for each external gluon*.
Choosing the same reference momentum for all like-helicity gluons (and tying it to
an opposite-helicity gluon's momentum) makes most polarization dot products vanish,
so most Feynman diagrams drop out. This single-reference form was introduced by
Xu, Zhang and Chang and (independently) within the CALKUL line of work.

**Color decomposition.** Using [λ^a,λ^b] = i f^{abc} λ^c and tr(λ^aλ^b)=δ^{ab}, any
tree-level n-gluon amplitude can be written as a sum over (n−1)! non-cyclic
permutations of color traces times kinematic "partial" (dual) sub-amplitudes:

  M_n = Σ' tr(λ^{a1}λ^{a2}···λ^{an}) m(p_1,ε_1; … ; p_n,ε_n).

Each m(1,…,n) is gauge invariant, invariant under cyclic permutations, obeys
m(n,…,1) = (−1)ⁿ m(1,…,n), satisfies the dual (sub-cyclic) Ward identity
m(1,2,3,…,n) + m(2,1,3,…,n) + m(2,3,1,…,n) + ⋯ = 0, and factorizes on multi-gluon
poles. At leading order in N the color traces are orthogonal, so the color-summed
square is incoherent:

  Σ_colors |M_n|² = N^{n−2}(N²−1) Σ' |m(1,…,n)|² + O(N^{n−4}).

The kinematic problem thus reduces to one cyclically-ordered partial amplitude. This
dual structure mirrors the zero-slope limit of an open-string (Koba–Nielsen)
amplitude, with the color traces playing the role of Chan–Paton factors.

**Supersymmetry relations.** Tree-level n-gluon amplitudes are identical in ordinary
Yang–Mills and in its supersymmetric extension, because at tree level no fermion or
scalar can run in an internal line of a pure-gluon process. Grisaru, Pendleton and
van Nieuwenhuizen, and Grisaru and Pendleton, showed that supersymmetry imposes Ward
identities (SWI) on scattering amplitudes: acting with the SUSY charge Q(k) on a
string of vector/spinor operators and using ⟨vacuum| Q = 0 gives linear relations
among amplitudes with different external spins. With a judicious choice of the
fermionic parameter's reference momentum, these identities (i) force certain gluon
helicity amplitudes to vanish and (ii) relate a given gluon helicity amplitude to a
simpler amplitude with a gluino pair. Because the pure-gluon trees coincide with the
SUSY ones, these relations hold for ordinary QCD trees too.

**Collinear and soft factorization.** When two color-adjacent massless legs a, b
become parallel (k_a → z P, k_b → (1−z) P, P² → 0), any gauge-theory amplitude
factorizes,

  A_n(…,a,b,…) → Σ_{λ} Split_{−λ}(aλ_a, bλ_b; z) · A_{n−1}(…,Pλ,…),

with universal splitting amplitudes (the square roots of the Altarelli–Parisi
splitting functions). A crucial physical fact: in massless gauge theory the
collinear singularity is only 1/√s_ab (not 1/s_ab) and carries an azimuthal phase,
because the intermediate gluon's helicity ±1 cannot equal the sum of the two external
helicities, so angular momentum is mismatched by one unit. That mismatch is exactly
what a spinor product 1/⟨ab⟩ (or 1/[ab]) captures. The soft limit (one gluon's
momentum → 0) similarly factorizes off a universal eikonal factor
S(a,s,b) = ⟨ab⟩/(⟨as⟩⟨sb⟩) for a positive-helicity soft gluon. The Altarelli–Parisi
splitting functions are pre-existing objects, so collinear factorization is a fixed
constraint that any candidate amplitude must satisfy.

**The data points.** The explicit four-, five-, and (numerically) six-gluon
tree amplitudes were known by direct computation (Gottschalk and Sivers; Berends,
Kleiss, de Causmaecker, Gastmans and Wu; and others). Any proposed closed form must
reproduce them. The phenomenological motivation and parton-level setup are laid out
in Eichten, Hinchliffe, Lane and Quigg.

## Baselines

**Brute-force Feynman diagrams.** Apply the Yang–Mills Feynman rules, sum all
diagrams, square, sum/average over colors and helicities. Correct but it is the very
method whose cost explodes as in the diagram table above; the non-abelian vertices
generate enormous term inflation, and for n ≳ 6 it is intractable analytically. It
gives no insight into *why* the answers, when finally obtained, look simpler than the
intermediate steps.

**CALKUL helicity amplitudes.** Fix external helicities, use the fermion-pair
reference-momentum polarization to kill terms, and exploit gauge invariance. This
tamed QED and some QCD processes, but for *purely* gluonic processes there is no
external fermion pair to anchor the reference momenta, and tracking the relative
phases between different gauge-invariant subsets becomes heavy. It computes a fixed n
at a time; it does not produce an n-independent closed form.

**Single-reference spinor-helicity (Xu–Zhang–Chang).** The improved polarization
vector ε±(p,k) with one arbitrary reference momentum per gluon makes individual
diagrams collapse and is the right tool for pure-glue processes. By itself, though,
it still leaves one to compute and assemble the surviving diagrams for each n; it
shows that *individual* amplitudes are simpler than expected, but stops short of a
formula valid for all n.

**Color/dual decomposition.** Reduces the colored amplitude to gauge-invariant,
cyclically-ordered partial amplitudes obeying the dual Ward identity and known
factorization. This isolates the kinematic object and explains the leading-N
incoherence, but the partial amplitude m(1,…,n) itself still has to be determined —
the decomposition reorganizes the problem rather than solving it.

**Supersymmetry Ward identities (Grisaru et al.).** Prove that the all-plus and the
single-minus gluon helicity amplitudes vanish to all multiplicity, and relate the
two-minus gluon amplitude to gluino amplitudes. The gap: SWI delivers *vanishing
theorems* and *relations between helicity sectors*, not the explicit closed form of
the first nonvanishing sector.

**Altarelli–Parisi collinear behavior.** Fixes the residue of every collinear pole
of any tree amplitude in terms of a splitting function times a lower-point amplitude.
This is a strong constraint and a powerful check, but it constrains *limits* of the
amplitude, not its full functional form away from the poles.

## Evaluation settings

The natural yardsticks, all pre-existing:

- **Cross-checks against known explicit results** for n = 4 and n = 5 (analytic) and
  n = 6 (numerical comparison at sampled phase-space points).
- **Symmetry/structure checks**: correct mass dimension and Bose symmetry of the
  squared n-particle amplitude; cyclic invariance and the dual Ward identity of each
  partial amplitude; vanishing of the all-plus and single-minus configurations.
- **Factorization checks**: the candidate amplitude must reduce, in every
  color-adjacent collinear limit (two gluons made parallel), to the appropriate
  Altarelli–Parisi splitting amplitude times the corresponding (n−1)-gluon amplitude;
  and to the universal eikonal factor in every soft limit.
- **Kinematics**: massless on-shell external momenta (all taken outgoing, with
  momentum conservation Σ p_i = 0), evaluated at generic and at near-collinear
  phase-space points; metric (+−−−); coupling g; SU(N) with the leading-N limit.

No outcome numbers are quoted; these are the settings against which any proposed
formula would be measured.

## Code framework

The pre-existing computational primitives are: massless-spinor construction from a
null momentum, the spinor products ⟨ij⟩ and [ij], the Minkowski product, and the
identity ⟨ij⟩[ji] = 2 p_i·p_j. What does *not* yet exist is the closed-form
n-gluon amplitude itself — that is the slot to be filled. The scaffold below provides
the spinor toolbox and leaves an empty stub for the n-point amplitude and for the
factorization/consistency checks it must pass.

```python
import numpy as np

def random_null(n, rng):
    """n outgoing null four-momenta p=(E, px, py, pz) with E=|vec p|."""
    out = []
    for _ in range(n):
        v = rng.normal(size=3)
        out.append(np.array([np.linalg.norm(v), *v]))
    return out

def spinors(p):
    """Factor the rank-1 bispinor p_{a adot} = lambda_a * lambdatilde_adot."""
    E, px, py, pz = p
    M = np.array([[E + pz, px - 1j*py],
                  [px + 1j*py, E - pz]], dtype=complex)
    lam = M[:, 0]/np.sqrt(M[0, 0]) if abs(M[0, 0]) > 1e-9 else M[:, 1]/np.sqrt(M[1, 1])
    a = 0 if abs(lam[0]) >= abs(lam[1]) else 1
    lamt = M[a, :]/lam[a]
    return lam, lamt

def mink(p, q):                       # (+---) Minkowski product
    return p[0]*q[0] - np.dot(p[1:], q[1:])

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

def n_gluon_amplitude(K, n, helicity_data):
    """The closed-form n-gluon tree partial amplitude we are looking for."""
    # TODO: the expression we will derive
    pass

def consistency_checks(K, n, helicity_data):
    """Does the candidate satisfy the constraints any tree amplitude must obey?"""
    # TODO: the checks the derived form has to pass
    pass
```

The final code fills `n_gluon_amplitude` with the derived closed form and
`consistency_checks` with the collinear/soft/symmetry verifications described under
Evaluation settings.
