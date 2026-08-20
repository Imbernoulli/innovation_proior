# Changelog

## 2026-08-19 — svfix(W3_notes_unclear): corrected Lagrange-resolvent quintic degree, 6 → 24

`reasoning.md`'s decisive obstacle ("Now the quintic. I form Lagrange's resolvent for
five roots and count its values, and they come to six") and `context.md`'s matching
Baseline mechanism (`ρ = x_1+ωx_2+⋯+ω^{n-1}x_n`, its n-th power's stabilizer gives the
resolvent degree; "For n≤4 the resolvent degree is below n; for n=5 it is 6") stated a
number that is wrong for the construction as literally given. Recomputed directly from
the stated formula: a permutation fixes ρ^n only by sending ρ to a power of ω times
itself, and the only permutations doing this are the n cyclic index-shifts, so the
stabilizer has order n and the orbit size (resolvent degree) is n!/n=(n-1)!. Verified
independently by brute-force enumeration over S_n with generic complex root values
(n=3 → 2, n=4 → 6, n=5 → 24), matching the trace's own n=3 claim exactly and
contradicting the "below n for n≤4" / "6 for n=5" claims (n=4's true value, 6, is
already not below 4). No source on disk or found via search (see notes/sources.md)
states this figure explicitly with a mechanism, so the corrected figure rests on the
trace's own honest, checkable computation applied to its own stated formula (quality
gate criterion b) rather than a citation.

Fixed: `results/reasoning.md` (decisive-step paragraph, now derives the stabilizer
directly instead of asserting "six"), `results/context.md` (Baseline mechanism
paragraph and the matching Background sentence), `results/train_answer.md` (matching
sentence in the opening paragraph). The qualitative conclusion — Lagrange's method
fails at the quintic because the resolvent degree exceeds the original, forcing Galois
to replace "count the values of one clever function" with "the permutation group
itself" as the object — is unchanged; the corrected number makes the failure sharper
(24 ≫ 5, not a mild 6 > 5).
