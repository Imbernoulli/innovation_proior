# Changelog — simulated-annealing

## 2026-08-18 — svfix(W3_notes_unclear)
Quality-gate verdict: the decisive step (cost→energy mapping, Metropolis's
Boltzmann-preserving uphill-acceptance rule derived from detailed balance,
then slow annealing vs. quenching) is genuinely derived on the page and is
fully backed by the two on-disk primaries (Kirkpatrick et al. 1983,
`refs/kirkpatrick1983.pdf`; Metropolis et al. 1953, `refs/metropolis1953.pdf`).
No external self-account/explainer material is load-bearing here — outcome
for the sourcing question is sound_as_is, matching TRIAGE.

While verifying the derivation against the sources, found one factual
contradiction: `results/reasoning.md`'s toy two-state Metropolis test claimed
that skipping the "recount on rejection" step leaves the *low-energy* state
`s` "systematically over-occupied." Metropolis et al. 1953 (p.1089) state the
opposite directly: failing to recount "would unjustifiably reduce the number
in state s relative to r" — i.e. `s` becomes *under*-occupied and the
high-energy state `r` becomes relatively over-represented. Corrected the
sentence to match the source's stated direction; no other file referenced
the old wording.
