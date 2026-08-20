# Sources — decisive-step sourcing check (svfix, W3_ancestors_only)

## Decisive step (per TRIAGE, independently re-identified)
`results/reasoning.md` lines ~17-98. Two linked moves:
1. Exponential ansatz `|Ψ⟩ = e^T|Φ0⟩` (not linear CI `1+C`) — because the
   quadruple excitation that two well-separated correlated pairs need is a
   *product* of two double amplitudes, not an independent number; `e^T`
   generates that product automatically via `½T2²`, so CCD is size-extensive
   by construction while truncated CI is not.
2. Similarity-transform-and-project (`H̄ = e^{-T}He^T`, project onto `⟨Φ0|`
   and `⟨Φ_ij^ab|`), not variational (`⟨Ψ|H|Ψ⟩/⟨Ψ|Ψ⟩`) — because the
   variational expectation value with `e^T` never truncates (infinite
   series), whereas the BCH expansion of the similarity-transformed `H̄`
   provably terminates after the fourfold nested commutator, since `H` has
   at most two-body (four-index) legs and each surviving `T` in the nest
   must contract at least one of them.

## Search performed
- `grep -ril` in `methods/coupled-cluster-ansatz/{refs,src,notes}` for
  "self-account", "Cizek", "interview", "thesis", "retrospective" → nothing
  beyond what's already logged in notes/synthesis.md (primary paywalled;
  Bartlett-Musiał RMP review + Sahan CC lecture notes + Crawford CC notes +
  Landau CC lecture notes already on disk, all secondaries, no first-person
  Čížek/Paldus self-account obtained — Theor. Chim. Acta "Origins" piece and
  Paldus 2005 chapter are paywalled, Springer redirected to auth wall).
- `grep -i coupled-cluster\|cizek\|coupled cluster` `SELF_ACCOUNT_SOURCES.md`
  at repo root → no entry.
- This run: re-checked whether the *decisive* step (exponential-for-
  extensivity, BCH termination) needs any of that missing self-account, or
  whether it is already fully derived + backed on the page. It is the
  latter — see below. No new web search was warranted because the quality
  gate is satisfied by material already on disk (refs/bartlett-musial-2007.pdf),
  and per the fixer instructions grafting a source onto an already-sound
  step is a defect, not an improvement.

## Correct backing IS already on disk, in the secondary already used to cross-check the equations (not merely implied)
`refs/bartlett-musial-2007.pdf` (Bartlett & Musiał, *Coupled-cluster theory
in quantum chemistry*, Rev. Mod. Phys. 79, 291 (2007)) states, independently
of the trace, the exact same three facts the trace derives:

**(a) disconnected T2²/2 quadruples dominate over connected T4, and this is
what makes CCD size-extensive** (matches reasoning.md lines 17-21, 41-50):
> "the simultaneous correlation of two electrons in different parts of a
> molecule, as represented by T2²/2, is more important in the wave function
> than the true, connected four-particle cluster interactions associated
> with T4... Such disconnected products are responsible for the
> size-extensivity property of the method."
(pdftotext -layout, source lines ~592-604; reading-order extraction confirms
identical wording.)

**(b) the variational expectation-value form of exp(T) is an infinite,
non-terminating series** (matches reasoning.md line 54, the "variational
never closes into a finite number of terms" wall):
> "the expectation value is an infinite series and the Λ-based expression
> is always in closed form."
(pdftotext, no -layout, reading order; re: `⟨0|e^{T†}p̂†q̂e^T|0⟩/⟨0|e^{T†}e^T|0⟩`
not truncating in T.)

**(c) the similarity-transformed H̄ = e^{-T}He^T terminates after the
fourfold commutator because H is (at most) two-body** (matches reasoning.md
lines 82-90, the "fifth commutator vanishes" counting argument):
> "is a critical one in CC theory since it terminates after fourfold
> commutators. That is, because the Hamiltonian has only one- and
> two-particle operators, the maximum number of T operators that can lead
> to nonvanishing contributions to the CC amplitude equations is four,
> regardless of their excitation level."
(pdftotext, no -layout, reading order; verified present verbatim, contiguous,
via substring search after whitespace normalization.)

The trace's own combinatorial argument (each `T` in a surviving nested
commutator must share an index with one of H's four legs; a fifth `T` has
nothing left to contract) is the *mechanism underneath* this same claim — a
faithful, first-person re-derivation, not a name-drop, and it was arrived at
independently of the secondary (the secondary merely confirms the trace got
the physics right, which is exactly the trace's-own-honest-computation prong
of the quality gate).

## Quality-gate verdict
sound_as_is. Both prongs of the gate hold:
(a) the decisive step is genuinely derived on the page: the obvious first
    move (linear CI truncated to CISD) fails for a concrete, checkable
    reason worked out explicitly (two separated fragments need a
    simultaneous double-on-A + double-on-B, i.e. a quadruple CISD does not
    have, so `E(A···B) > E(A)+E(B)` and the error grows with N) — and the
    exponential resolution follows directly from identifying that the
    missing quadruples are *products*, not independent numbers. The second
    half (why project-not-variational) fails for an equally concrete reason
    (the variational expression in `e^T` does not terminate) and the BCH
    resolution follows from a correct, self-contained combinatorial count;
(b) that reasoning either matches the primary/secondary on record
    word-for-word (quotes (a)-(c) above, refs/bartlett-musial-2007.pdf,
    already read and cited in notes/synthesis.md for equation cross-check)
    or is the trace's own honest, checkable computation (the C1..C4
    cluster-decomposition algebra, the fourfold-commutator counting
    argument, the two-electron CCD=FCI sanity check).
No source needs to be grafted onto reasoning.md: the secondary already on
disk was used (per synthesis.md) only to cross-check the CCD amplitude
equations, never to force the ansatz or the BCH-termination insight, and it
independently corroborates rather than supplies that insight — bolting an
explicit citation onto an already-self-derived step would be decorative,
not grounding, per the quality gate's explicit warning against grafting
sources onto sound traces. No rewrite of reasoning.md was performed; no
factual errors found; only this file was added.
