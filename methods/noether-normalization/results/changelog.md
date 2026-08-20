# Changelog

## 2026-08-20 — svfix (W3_notes_unclear)

Fixed a formula error in the decisive step of `results/reasoning.md` (the exponent-based shift that repairs the linear-genericity trick over finite fields). The trace had set `y_i' = z_i + y~^(r^(i-1))` for `i = 1,...,m-1`, which puts the first shifted generator's exponent at `r^0` — the same base-`r` digit slot occupied by `a_m`, the untouched exponent of `y~` itself. Two monomials of `f` with the same `a_1 + a_m` (e.g. `a_1=2,a_m=0` vs. `a_1=0,a_m=2`) then collide at the same power of `y~` and can cancel, breaking the "the coefficient of the top power survives untouched" claim the whole argument rests on.

Verified against `en.wikipedia.org/wiki/Noether_normalization_lemma`'s proof of the arbitrary-field case (citing Nagata's method via Eisenbud, *Commutative Algebra with a View Toward Algebraic Geometry*, GTM 150, Thm 13.3), which reserves `r^0` for the untouched variable and starts the shifted variables at `r^1`; and independently confirmed by direct symbolic expansion (the buggy shift collapses `f = y1'^2 - y^2` from the expected degree 2*r to degree 1 in `y` at `r=3`, while the corrected shift reaches the expected degree 6 with leading coefficient 1).

Fixed: shift exponent `r^(i-1)` -> `r^i` for `i = 1,...,m-1`; the downstream leading-degree formula `a_1 + a_2*r + ... + a_{m-1}*r^{m-2} + a_m` -> `a_1*r + a_2*r^2 + ... + a_{m-1}*r^{m-1} + a_m`. One sentence added explaining why `r^0` must stay reserved for `y~`'s own exponent. `answer.md` and `train_answer.md` do not state this exponent formula (they only carry the infinite-field / linear-genericity version and the rank-two hyperbola example), so no changes were needed there.

Source added: `refs/wikipedia_noether_normalization_lemma_proof.txt` (explainer; see `notes/sources.md`).
