# Changelog — ape-x

## 2026-08-18 — svfix(W3_notes_unclear)
Fixed a unit-conflation error in `reasoning.md`'s decisive-step paragraph (the rate
arithmetic motivating actor-computed initial TD-error priorities). The passage stated
"actors produce at ~50K transitions/s" and derived a "~5×" production/consumption
mismatch versus the learner's ~10⁴ samples/s. Per the primary paper (`src/apex_paper.tex`,
line 229), ~50K is the aggregate *frames*/s across 360 actors (~139 FPS each); with a
fixed action repeat of 4, the actual production rate is ~12.5K *transitions*/s, and the
learner consumes ~9.7K transitions/s (19 batches/s × 512, paper line 229/295) — a ~1.3×
surplus, not ~5×. Rewrote the paragraph to use the paper's exact figures and to ground
the "insert-at-max collapses into training-on-newest" argument in the paper's own stated
cause (the large *number* of concurrently-producing actors keeping a large tied-max-priority
crowd replenished faster than one learner-step can refine it), rather than an overstated
rate ratio. No change to the landing (actors compute initial |n-step TD error| priorities
online) — that step, and its grounding in the primary paper (line 178 of apex_paper.tex),
was already correct; only the numeric justification was wrong.

Searched for a self-account beyond the primary (OpenReview forum H1Dy---0Z: 2 official
reviews, no author rebuttal notes indexed; younggyoseo/Ape-X GitHub issues; arXiv has only
v1, no revision diff available) — found nothing beyond the primary paper's own stated
rationale, which is what the corrected paragraph now uses verbatim-faithfully.
