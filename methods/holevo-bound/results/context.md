# Context: how much classical information can a quantum carrier hold?

## Research question

All entropies are measured in bits, so logarithms are base 2.

A classical communication channel obeys a hard law: per use, it transmits at most
`C = max_P I(X;Y)` bits, and a physical symbol with `m` distinguishable levels carries
`log m` bits. As communication hardware shrinks toward the quantum scale — lasers, single
photons, optical fibers carrying near-single-quantum signals — the carrier of a message is
no longer a classical symbol but a quantum state. A single qubit, viewed as a pure state,
lives on a continuous two-sphere (the Bloch sphere): there is an uncountable continuum of
"directions" a qubit can point in. Taken at face value, if every point on the sphere is a
distinct signal, one qubit could encode a real number — unboundedly many classical bits.

To *read* a message one must *measure*, and quantum measurement is limited: two
non-orthogonal states cannot be told apart with certainty. So the operative question is
operational, not kinematic. Suppose a sender encodes a classical random variable `X` (value
`x` with probability `p_x`) into quantum states `ρ_x`, sends the carrier, and a receiver
applies *any* measurement whatsoever — any POVM, with any number of outcomes, possibly a
collective measurement on many carriers at once — producing a classical outcome `Y`. Over all
encodings and all measurements, what is the maximum classical mutual information `I(X;Y)` the
receiver can extract? Call this the *accessible information* of the ensemble. Is a qubit a
free lunch (accessible information unbounded), or is there a tight, computable ceiling — and
in particular, does an `n`-qubit carrier cap out at `n` classical bits, recovering the
classical intuition exactly?

## Background

**Shannon's classical theory (Shannon 1948).** For a classical channel `p(y|x)`, the mutual
information `I(X;Y) = H(X) − H(X|Y) = H(X) + H(Y) − H(XY)`, where `H` is the Shannon
entropy, measures the bits about `X` revealed by `Y`. The capacity is `C = max_P I(X;Y)`.
Relative entropy `D(p‖q) = Σ_x p(x) log(p(x)/q(x))` measures the distinguishability of two
distributions (it governs how fast a wrong hypothesis can be ruled out) and is nonnegative,
with `I(X;Y) = Σ_x p(x) D(p(·|x)‖p(·))`. These objects are defined for a *fixed* channel
`p(y|x)`. In the quantum setting the effective channel `p(y|x) = Tr(E_y ρ_x)` depends on
both the chosen states `ρ_x` *and* the receiver's chosen measurement `{E_y}`, and the
measurement is a free variable to be optimized over.

**von Neumann entropy (von Neumann 1932).** `S(ρ) = −Tr ρ log ρ` is the quantum analogue
of the Shannon entropy. Its load-bearing properties: it is concave,
`S(Σ_x p_x ρ_x) ≥ Σ_x p_x S(ρ_x)`; for a `d`-dimensional system `0 ≤ S(ρ) ≤ log d`, with the
maximum at the maximally mixed state; for any ensemble `{p_x,ρ_x}` realizing a density
operator `ρ`, the eigenvalue distribution `λ(ρ)` is majorized by `p`, hence `S(ρ) ≤ H(p)`
with equality only when the states are mutually orthogonal — so mixing non-orthogonal pure
states loses distinguishability.

**Quantum relative entropy.** `D(ρ‖σ) = Tr ρ(log ρ − log σ)`.
By Klein's inequality `D(ρ‖σ) ≥ 0`, with equality iff `ρ = σ` on the relevant support. The
quantum mutual information of a bipartite state is
`I(A;B) = S(A) + S(B) − S(AB) = D(ρ_AB‖ρ_A ⊗ ρ_B) ≥ 0`. A variational characterization is
available, `D(ρ‖σ) = sup_K {Tr(ρK) − log Tr(2^K σ)}` over Hermitian `K`. The classical
analogues of the deeper entropy principles — monotonicity of `D` under channels and
subadditivity of entropy — follow from convexity; their full quantum generalizations are
separate matters.

**Quantum measurements / POVMs.** The most general measurement is a positive
operator-valued measure `{E_y}` with `E_y ≥ 0` and `Σ_y E_y = I`; on a state `ρ` it yields
outcome `y` with probability `Tr(E_y ρ)`. Equivalently (Stinespring/Naimark) it is an isometry
into a larger space followed by a projective measurement; equivalently it is a quantum channel
`A → AY`, `ρ ↦ Σ_y M_y ρ M_y† ⊗ |y⟩⟨y|` with `M_y†M_y = E_y`, which records the outcome in a
fresh classical register `Y`. This last form expresses "the receiver measures" as a CPTP map.

**Distinguishability facts.** Mutually orthogonal states are perfectly distinguishable:
projecting onto the support of each `ρ_x` gives `p(y|x) = δ_{xy}`, hence `H(X|Y) = 0` and
`I(X;Y) = H(X)`. Non-orthogonal states are not perfectly distinguishable, so `H(X|Y) > 0` and
`I(X;Y) < H(X)` strictly. Enlarging the signal alphabet on the Bloch sphere makes the signals
less distinguishable as it adds encoded information.

## Baselines

There is no prior *bound* on accessible information to react to; the relevant baselines are the
prior *methods of reasoning* about the carrier, and the one limiting case that is already
understood.

- **Treat the qubit classically (orthogonal-state coding).** Encode each bit by one of two
  orthogonal states `{|0⟩,|1⟩}`. A projective measurement in that basis recovers the bit with
  certainty: `I(X;Y) = 1` per qubit. Core idea: stay inside a classical, perfectly
  distinguishable alphabet. It tells us `1` bit is *achievable* per qubit.

- **Stuff the continuum (dense non-orthogonal alphabet).** Encode a message by a state chosen
  from a large alphabet of pure single-qubit states spread over the Bloch sphere, using the
  continuous parameter to carry many bits. Core idea: exploit the continuity of Hilbert space.

- **Per-measurement Shannon analysis.** Fix the states `ρ_x` and a particular measurement
  `{E_y}`; the induced channel `p(y|x) = Tr(E_y ρ_x)` is classical, so Shannon's `I(X;Y)`
  applies for *that* measurement. Core idea: reduce to a classical channel one measurement at a
  time.

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
- **Metrics.** Bits of mutual information `I(X;Y)`; the entropy `S(ρ̄)` of the ensemble's
  average state; the binary entropy `H(·)` for the small examples. Comparisons are exact
  analytic inequalities checked by direct computation on the small ensembles — no learning, no
  statistical estimate.
