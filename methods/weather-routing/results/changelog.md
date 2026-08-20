# Changelog

## 2026-08-19 — svfix(W3_notes_unclear)

Decisive step (isochrone-loops fix via floating great-circle-referenced grid) was grounded only
by a bare name-drop in refs/transnav-0183.pdf ("...isochrone loops... (Hagiwara 1989, Spaans
1986, Wisniewski 1991)", no mechanism); an independent verifier confirmed the quote existed but
was decorative — reasoning.md's derivation never depended on it. Hagiwara & Spaans 1987 and
Hagiwara's 1989 TU Delft PhD thesis are unreachable from this sandbox (paywalled / connection
refused / bot-walled across all tried repositories — see notes/sources.md search log). Found an
open-access 2024 paper (Chen, Tian & Mao, *Ships and Offshore Structures*) that reproduces
Hagiwara's (1989) thesis algorithm step-by-step and states the documented cause of "isochrone
loop": non-convexity of the ship's own achievable-speed-vs-heading curve (per Wisniewski 1991),
not spatial "shear" as the trace previously asserted. Rewrote the decisive passage in
results/reasoning.md (and the parallel passage in results/train_answer.md) so the self-crossing
failure is derived from the method's own speed-loss curve f(θ) — which is genuinely non-convex
in heading (floor-hugging with a squeezed dip at head sea) — rather than from an unexplained
"shear" claim, and sharpened the floating-grid resolution's wording ("parallel lines... at equal
spacing" flanking the great circle, one survivor per lane per stage) to match Hagiwara's actual
documented subsector construction. No change to the landing (method + code) — this was a
grounding/precision correction to the justification, not a change to what the method does. See
notes/sources.md for the verified quotes and full search log.
