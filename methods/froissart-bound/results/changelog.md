# Changelog

## 2026-08-18 — svfix repair pass (W3_notes_unclear)
Decisive-step fix: an independent verifier rejected a prior `sound_as_is` submission for
mismatched quote pairing — the trace_quote offered ("phases hide cancellation") is
self-evident, while the passage that actually needed backing ("A series Σc_l P_l(z) with
positive c_l ... radius of convergence is governed by the boundary") was left uncited. On
inspection that passage genuinely is ASSERTED, not derived: it invokes a non-obvious
complex-analysis fact (a Pringsheim-type theorem for positive Legendre series) by bare
analogy, with no derivation.

`refs/harmony.txt` (Arkhipov, "Harmony of the Froissart Theorem," hep-ph/0208263) already on
disk and already logged in `notes/synthesis.md`'s derivation chain but unused at this exact
spot, walks the real historical route: Cauchy's integral formula for A(s,z) analytic inside
the Lehmann ellipse, expanded via Heine's kernel formula into Legendre functions of the second
kind Q_l, bounded by a standard max-modulus estimate using Q_l's known asymptotic decay (Q_l
decays exactly where P_l grows). `results/context.md`'s own Background section already lists
that Q_l asymptotic as a pre-established fact, confirming this was the intended route.

Rewrote `results/reasoning.md`'s decisive-step paragraph (~line 17) to run the actual
Cauchy/Heine contour-integral projection instead of the positive-series hand-wave, landing on
the identical inequality Im a_l(s) ≲ P(s)·(z_0+√(z_0²−1))^{−l} the rest of the derivation
already depends on. Updated two later references to "positive Legendre series" (causal-chain
summary, stress-test paragraph) to name the Cauchy/Heine mechanism for internal consistency.
Propagated the same swap to `results/answer.md` (Key idea + derivation step 2) and
`results/train_answer.md` (matching paragraph) so the landing doesn't repeat the shortcut just
removed from reasoning.md. No factual errors found beyond the missing derivation; final
formula, numbers, and code unchanged — only the derivation route to the coefficient-decay
bound changed. See `notes/sources.md` for the grounding quote.
