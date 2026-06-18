## Problem Setting

A subset of the integers can avoid the local pattern `a, a+r, a+2r` for many initial values of `N`, yet the question is whether this can persist at fixed positive density. In finitary language, one studies subsets `A` of `[N]` with `|A| >= delta N` and asks whether all sufficiently large `N` force a nontrivial three-term arithmetic progression.

## Known Obstructions

There are large progression-free sets at sparse densities. Digit constructions and later sphere-based constructions show that avoiding three-term progressions is compatible with substantial size, so any proof must distinguish fixed positive density from densities that slowly vanish. These examples also rule out a naive proof that simply counts possible triples and assumes independence.

## Counting Baseline

For a random density-`delta` set, the expected number of three-term progressions is large. A deterministic dense set need not behave randomly, because its membership function may correlate with additive structure. The baseline count is therefore useful mainly as a benchmark: when the count fails, that failure should reveal a structured reason.

## Analytic Language

Three-term progressions are governed by a linear equation: the coefficients in `n, n+r, n+2r` satisfy `alpha - 2 alpha + alpha = 0`. Linear Fourier characters are therefore the natural probes for additive structure, and the balanced function `1_A - delta 1_[N]` measures deviation from random density on the ambient interval. If every nonzero linear Fourier coefficient is small, the progression count should remain close to the random prediction.

## Success Criterion

The desired proof should turn failure of the random count into a concrete certificate of structure. That certificate has to be local enough to survive passage to an affine subinterval, because three-term progression-freeness is preserved under affine rescaling. The endgame must also account for shrinking intervals: each smaller instance has to remain long enough for the same kind of obstruction test to restart.
