# The Holevo bound

All logarithms are base 2.

**Problem.** A classical message `X` (value `x` with prior `p_x`) is encoded into quantum
states `ρ_x` and sent on a quantum carrier; a receiver applies an arbitrary measurement and
obtains a classical outcome `Y`. Although a qubit's Hilbert space is a continuum, the
accessible information, the maximum classical mutual information `I(X;Y)` over all
measurements, is finite. The Holevo bound gives the ceiling.

**Key idea.** The carrier entropy `S(ρ̄)` alone over-counts the internal noise of mixed signal
states. Subtracting the average signal entropy gives the Holevo quantity `χ`, which is exactly
the quantum mutual information `I(X;A)` between the classical message register and the quantum
carrier. The only nontrivial step is the measurement-specific data-processing fact:
measuring two states with a fixed POVM cannot increase their relative entropy.

## The Holevo χ quantity

For an ensemble `E = {p_x, ρ_x}` with average state `ρ̄ = Σ_x p_x ρ_x` and von Neumann
entropy `S(ρ) = −Tr ρ log ρ`,

  `χ(E) := S(ρ̄) − Σ_x p_x S(ρ_x) = S(Σ_x p_x ρ_x) − Σ_x p_x S(ρ_x).`

With the classical-quantum state `ρ_XA = Σ_x p_x |x⟩⟨x| ⊗ ρ_x`,

  `S(X) = H(p),   S(A) = S(ρ̄),   S(XA) = H(p) + Σ_x p_x S(ρ_x),`

so

  `χ(E) = I(X;A)_ρ = S(X) + S(A) − S(XA) = Σ_x p_x D(ρ_x ‖ ρ̄),`

where `D(ρ‖σ) = Tr ρ(log ρ − log σ)`.

## Theorem

Let `E = {p_x, ρ_x}` be an ensemble on a `d`-dimensional system, and let any POVM `{E_y}`
produce `Y` with

  `p(x,y) = p_x Tr(E_y ρ_x).`

Then

  `I(X;Y) ≤ χ(E),  hence  Acc(E) := max_{POVM} I(X;Y) ≤ χ(E).`

Since `Σ_x p_x S(ρ_x) ≥ 0` and `S(ρ̄) ≤ log d`,

  `Acc(E) ≤ χ(E) ≤ S(ρ̄) ≤ log d.`

For `n` qubits, `d = 2^n`, so `log d = n`: an `n`-qubit carrier conveys at most `n`
classical bits, even under collective measurement.

**Tightness.** If the signal states have orthogonal supports, the receiver projects onto those
supports and recovers `x` perfectly, so `I(X;Y) = H(X)`. The average state is block diagonal
across the same supports, hence `S(ρ̄) = H(p) + Σ_x p_x S(ρ_x)` and `χ = H(p) = H(X)`. Thus
`Acc = χ = H(X)` for orthogonal signals; with `d` equally likely orthogonal pure states this
attains the dimensional ceiling `log d`.

## Proof

The relative-entropy reductions are immediate from the cq form:

  `I(X;A) = D(ρ_XA ‖ ρ_X ⊗ ρ_A) = Σ_x p_x D(ρ_x ‖ ρ̄).`

For the measurement, define

  `q_x(y) = Tr(E_y ρ_x),   q̄(y) = Tr(E_y ρ̄).`

The induced classical mutual information is

  `I(X;Y) = Σ_x p_x D(q_x ‖ q̄).`

It remains to prove `D(q_x‖q̄) ≤ D(ρ_x‖ρ̄)` for each `x`. For a projective measurement
`{P_y}`, write `q_y = Tr(P_y ρ)` and `r_y = Tr(P_y σ)`. With the standard support convention,
the variational formula

  `D(ρ‖σ) = sup_K {Tr(ρK) − log Tr(2^K σ)}`

gives the contraction directly: choose `K = Σ_y log(q_y/r_y) P_y`, omitting zero-probability
terms. Then `Tr(ρK) = D(q‖r)` and `Tr(2^K σ) = Σ_y (q_y/r_y) r_y = 1`, so
`D(q‖r) ≤ D(ρ‖σ)`. A general POVM reduces to this case by Naimark dilation, and the isometry
used in the dilation preserves relative entropy. Applying the inequality to each
`(ρ_x, ρ̄)` pair and averaging proves `I(X;Y) ≤ χ(E)`.

The same proof can be organized by the chain

  `I(X;Y)_{ρ'} ≤ I(X;AY)_{ρ'} ≤ I(X;A)_ρ = χ(E),`

where `ρ'` is the state after the measurement instrument `A → AY`. The left inequality is the
discarding step, equivalently strong subadditivity:
`I(X;AY) − I(X;Y) = I(X;A|Y) ≥ 0`. The right inequality is data-processing for the local
instrument on `A`. The direct relative-entropy argument above is the cq measurement instance
needed for the theorem, without assuming general channel monotonicity.

In isometric form, dilate the classical-output measurement channel to
`V: A → Y⊗Z` and set `ξ = (1_X⊗V)ρ_XA(1_X⊗V)†`. Since `V` preserves spectra,

  `S(ξ_X) = H(p),`
  `S(ξ_{XYZ}) = H(p) + Σ_x p_x S(ρ_x),`
  `S(ξ_{YZ}) = S(ρ̄).`

Thus `I(X;YZ)_ξ = χ`. Discarding `Z` gives `I(X;Y)_ξ ≤ I(X;YZ)_ξ = χ`, and the `X`-`Y`
marginal is exactly the classical joint distribution produced by the POVM.

