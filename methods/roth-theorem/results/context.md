## Problem Setting

A subset of the integers can avoid the local pattern `a, a+r, a+2r` for many initial values of `N`, yet the question is whether this can persist at fixed positive density. In finitary language, one studies subsets `A` of `[N]` with `|A| >= delta N` and asks whether all sufficiently large `N` force a nontrivial three-term arithmetic progression.

## Known Obstructions

There are large progression-free sets at sparse densities. Digit constructions and later sphere-based constructions show that avoiding three-term progressions is compatible with substantial size. These examples indicate that any argument must distinguish fixed positive density from densities that slowly vanish.

## Counting Baseline

For a random density-`delta` set, the expected number of three-term progressions is large. A deterministic dense set need not behave randomly, because its membership function may correlate with additive structure.

## Analytic Language

Three-term progressions are governed by a linear equation: the coefficients in `n, n+r, n+2r` satisfy `alpha - 2 alpha + alpha = 0`. Linear Fourier characters are therefore the natural probes for additive structure, and the balanced function `1_A - delta 1_[N]` measures deviation from random density on the ambient interval.

## Research Question

Given a subset `A` of `[N]` at fixed positive density `delta`, does `A` necessarily contain a nontrivial three-term arithmetic progression for all sufficiently large `N`?
