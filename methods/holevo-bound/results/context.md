# Context: how much classical information can a quantum carrier hold?

## Research question

All entropies are measured in bits, so logarithms are base 2.

A classical communication channel obeys a hard law: per use, it transmits at most
`C = max_P I(X;Y)` bits, and a physical symbol with `m` distinguishable levels carries
`log m` bits. As communication hardware shrinks toward the quantum scale — lasers, single
photons, optical fibers carrying near-single-quantum signals — the carrier of a message is
no longer a classical symbol but a quantum state. A single qubit, viewed as a pure state,
lives on a continuous two-sphere (the Bloch sphere): there is an uncountable continuum of
"directions" a qubit can point in. Taken at face value this is alarming: if every point on the
sphere is a distinct signal, one qubit could encode a real number — unboundedly many
classical bits.

The catch is that to *read* a message one must *measure*, and quantum measurement is
limited: two non-orthogonal states cannot be told apart with certainty. So the honest
question is operational, not kinematic. Suppose a sender encodes a classical random variable
`X` (value `x` with probability `p_x`) into quantum states `ρ_x`, sends the carrier, and a
receiver applies *any* measurement whatsoever — any POVM, with any number of outcomes,
possibly a collective measurement on many carriers at once — producing a classical outcome
`Y`. Over all encodings and all measurements, what is the maximum classical mutual
information `I(X;Y)` the receiver can extract? Call this the *accessible information* of the
ensemble. Is a qubit a free lunch (accessible information unbounded), or is there a tight,
computable ceiling — and in particular, does an `n`-qubit carrier cap out at `n` classical
bits, recovering the classical intuition exactly?

A solution would (i) define the right quantity to compare against, (ii) bound the accessible
information uniformly over the infinite set of possible measurements, (iii) say when the bound
is tight, and (iv) deliver the corollary that a `d`-dimensional carrier conveys at most
`log d` classical bits.

## Background

**Shannon's classical theory (Shannon 1948).** For a classical channel `p(y|x)`, the mutual
information `I(X;Y) = H(X) − H(X|Y) = H(X) + H(Y) − H(XY)`, where `H` is the Shannon
entropy, measures the bits about `X` revealed by `Y`. The capacity is `C = max_P I(X;Y)`.
Relative entropy `D(p‖q) = Σ_x p(x) log(p(x)/q(x))` measures the distinguishability of two
distributions (it governs how fast a wrong hypothesis can be ruled out) and is nonnegative,
with `I(X;Y) = Σ_x p(x) D(p(·|x)‖p(·))`. These objects are defined for a *fixed* channel
`p(y|x)`. The quantum complication: the effective channel `p(y|x) = Tr(E_y ρ_x)` depends on
both the chosen states `ρ_x` *and* the receiver's chosen measurement `{E_y}`, and the
measurement is a free variable to be optimized against, with no a priori bound.

**von Neumann entropy (von Neumann 1932).** `S(ρ) = −Tr ρ log ρ` is the quantum analogue
of the Shannon entropy. The load-bearing properties: it is concave,
`S(Σ_x p_x ρ_x) ≥ Σ_x p_x S(ρ_x)`; for a `d`-dimensional system `0 ≤ S(ρ) ≤ log d`, with the
maximum at the maximally mixed state; for any ensemble `{p_x,ρ_x}` realizing a density
operator `ρ`, the eigenvalue distribution `λ(ρ)` is majorized by `p`, hence `S(ρ) ≤ H(p)`
with equality only when the states are mutually orthogonal — so mixing non-orthogonal pure
states irretrievably loses distinguishability.

**Quantum relative entropy.** `D(ρ‖σ) = Tr ρ(log ρ − log σ)`.
By Klein's inequality `D(ρ‖σ) ≥ 0`, with equality iff `ρ = σ` on the relevant support. The
quantum mutual information of a bipartite state is
`I(A;B) = S(A) + S(B) − S(AB) = D(ρ_AB‖ρ_A ⊗ ρ_B) ≥ 0`. A variational characterization is
available, `D(ρ‖σ) = sup_K {Tr(ρK) − log Tr(2^K σ)}` over Hermitian `K`. Whether the deeper
entropy principles — unrestricted monotonicity of `D` under arbitrary quantum channels, and the
equivalent strong-subadditivity of von Neumann entropy — hold is, in their full generality, a
separate and harder matter; the classical analogues follow from convexity but the quantum
statements are not in hand here as general theorems.

**Quantum measurements / POVMs.** The most general measurement is a positive
operator-valued measure `{E_y}` with `E_y ≥ 0` and `Σ_y E_y = I`; on a state `ρ` it yields
outcome `y` with probability `Tr(E_y ρ)`. Equivalently (Stinespring/Naimark) it is an isometry
into a larger space followed by a projective measurement; equivalently it is a quantum channel
`A → AY`, `ρ ↦ Σ_y M_y ρ M_y† ⊗ |y⟩⟨y|` with `M_y†M_y = E_y`, which records the outcome in a
fresh classical register `Y`. This last form expresses "the receiver measures" as a CPTP map.

**Distinguishability facts that anchor tightness.** Mutually orthogonal states are perfectly
distinguishable: projecting onto the support of each `ρ_x` gives `p(y|x) = δ_{xy}`, hence
`H(X|Y) = 0` and `I(X;Y) = H(X)`. Non-orthogonal states are not perfectly distinguishable, so
`H(X|Y) > 0` and `I(X;Y) < H(X)` strictly. This is the diagnostic phenomenon the whole
question turns on: enlarging the signal alphabet on the Bloch sphere makes the signals less
distinguishable at exactly the rate that defeats the extra encoded information.

## Baselines

There is no prior *bound* on accessible information to react to; the relevant baselines are the
prior *methods of reasoning* about the carrier, and the one limiting case that is already
understood.

- **Treat the qubit classically (orthogonal-state coding).** Encode each bit by one of two
  orthogonal states `{|0⟩,|1⟩}`. A projective measurement in that basis recovers the bit with
  certainty: `I(X;Y) = 1` per qubit. Core idea: stay inside a classical, perfectly
  distinguishable alphabet. Gap: it leaves on the table the entire continuum of non-orthogonal
  states — the whole reason one suspected a qubit might carry *more*. It tells us `1` bit is
  *achievable* but says nothing about whether more is *possible*.

- **Stuff the continuum (dense non-orthogonal alphabet).** Encode a message by a state chosen
  from a large alphabet of pure single-qubit states spread over the Bloch sphere, hoping the
  continuous parameter carries many bits. Core idea: exploit the continuity of Hilbert space.
  Gap: as the alphabet grows the signals crowd together and become less distinguishable; the
  receiver cannot resolve them, and the apparent extra capacity is not extractable. This is
  precisely the failure mode a bound must explain quantitatively — there is no analysis here
  that says *how much* is lost, only the qualitative obstruction.

- **Per-measurement Shannon analysis.** Fix the
  states `ρ_x` and a particular measurement `{E_y}`; the induced channel `p(y|x) = Tr(E_y ρ_x)`
  is classical, so Shannon's `I(X;Y)` applies for *that* measurement. Core idea: reduce to a
  classical channel one measurement at a time. Gap: the argument must be lifted from a fixed
  induced channel to a quantity that is intrinsic to the ensemble, explicitly computable from the
  `ρ_x` and `p_x`, uniform over all POVMs, and tight enough to recover the `n`-qubit `⇒ n`-bit
  corollary.

## Evaluation settings

The natural yardsticks are analytic, not benchmark datasets:

- **The communication game.** Sender draws `x` with probability `p_x` from a finite alphabet
  `Σ`, prepares carrier `A` in `ρ_x ∈ D(C^d)`, sends it; receiver applies a POVM `{E_y}` and
  records `Y`; the figure of merit is the classical mutual information `I(X;Y)` of registers
  `X` and `Y`. The accessible information is its maximum over all POVMs (it suffices to consider
  finite outcome sets for the finite-dimensional carrier).
- **Carrier dimensions.** Single qubit (`d = 2`), `n` qubits (`d = 2^n`), and general
  finite-dimensional `d`. The question is posed for arbitrary finite `d`.
- **Canonical small ensembles for sanity checks.** A single qubit carrying the two orthogonal
  states `{|0⟩,|1⟩}` (the perfectly-distinguishable limit); a single qubit carrying three
  symmetric pure states at `120°` in a plane of the Bloch sphere, each with probability `1/3`
  (a non-orthogonal ensemble whose density operator is maximally mixed, `ρ = I/2`); two
  non-orthogonal states for the random-access setting.
- **Metrics.** Bits of mutual information `I(X;Y)`; a candidate intrinsic ensemble ceiling; the
  entropy `S(ρ̄)` of the ensemble's average state; the binary entropy `H(·)` for the small
  examples. Comparisons are exact analytic inequalities checked by direct computation on the
  small ensembles — no learning, no statistical estimate.

## Code framework

The natural computational support is a compact matrix-entropy harness: compute entropic
quantities, run a few representative measurements, and compare the extracted classical
information with a placeholder intrinsic ensemble ceiling. The ingredients are dense-matrix
linear algebra (eigendecomposition for `−Tr ρ log ρ`), Shannon-entropy and
mutual-information routines from classical information theory, and a way to apply a POVM to a
state and read off the induced classical joint distribution.

```python
import numpy as np

def von_neumann_entropy(rho):
    # S(rho) = -Tr rho log2 rho, via eigenvalues
    pass  # TODO

def shannon_entropy(p):
    # H(p) = -sum p log2 p
    pass  # TODO

def mutual_information(joint):
    # classical I(X;Y) from a joint distribution p(x,y)
    pass  # TODO

def apply_povm(states, priors, povm):
    # given {rho_x}, {p_x}, POVM {E_y}: return joint p(x,y) = p_x Tr(E_y rho_x)
    pass  # TODO

def candidate_ceiling(states, priors):
    # placeholder for the intrinsic ensemble quantity to compare against
    pass  # TODO

def sanity_check(states, priors, povms):
    # for each measurement: I(X;Y) from apply_povm, assert <= candidate_ceiling
    pass  # TODO
```
