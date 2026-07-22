# Muffled Echo Sweeps

A prospecting rig has found a buried crystal: a straight row of N identical
atoms, each linked to its neighbours by identical springs, both ends clamped
to bedrock. The rig drives one end back and forth at a chosen frequency and
records the steady-state echo amplitude at the far end. Sweeping frequency
reveals sharp resonance peaks — the chain's normal modes — sitting on top of
a noisy, damped background. The equipment is muffled: any one sweep only
covers a **partial** frequency window, and only a **few small** chains were
ever dug up and tested.

Your job: from these partial sweeps, produce a closed-form formula predicting
the resonance frequency of mode `j` for a chain of size `N` — one that still
works for chains far larger than anything tested, and for modes whose true
frequency lies outside every window you were shown.

## Input (stdin)

```
t
k
N_1 p_1
omega_1_1  A_1_1
...
omega_1_p1 A_1_p1
N_2 p_2
...
```

`t` is the test id. `k` training chain sizes follow; for chain size `N_i`,
`p_i` rows give the driving frequency `omega` and the measured (noisy) echo
amplitude `A` at that frequency, swept over the SAME partial window for every
`N_i` in this instance. Resonances (echo peaks) appear as local maxima of
`A(omega)`; not every mode of a chain shows up — only those inside the swept
window. Modes are numbered `j = 1, 2, ...` in ascending frequency order for
each chain size.

## Output (stdout): one closed-form expression

Print exactly one line: a single Python-style arithmetic expression in the
two variables `j` (mode index) and `N` (chain size), using only
`+ - * / ** % //`, parentheses, numeric constants, unary `+/-`, the constant
`pi`, and the unary functions `sin cos tan sqrt exp log abs`. No other names,
calls, or statements.

**Illustrative FORM only — NOT the hidden law:**
```
0.5 * N + 3 * sqrt(j)
```
This only shows the syntax; the real resonance law has a different shape and
you must discover it from the data.

## Feasibility

The expression must parse under the grammar above and evaluate to a finite
number at every grading point. Any parse failure, disallowed name/call, or
non-finite value (during grading) scores `0`.

## Objective (minimise)

The grader evaluates your expression at held-out `(j, N)` pairs: chain sizes
far larger than any you were shown, and mode indices whose true frequency
mostly lies beyond the partial window you swept — genuine size *and*
frequency-band extrapolation. Let `MSE` be the mean squared error against the
true resonance frequencies there, and `nodes` the size of your expression
tree. The grader forms

```
F = MSE * (1 + LAMBDA * nodes)
B = MSE_of_constant_omega0 * (1 + LAMBDA * 1)   # internal baseline
Ratio = min(1000, 100 * B / F) / 1000
```

with a small fixed `LAMBDA` and a baseline `omega0` (the chain's own natural
frequency scale, not given to you). A constant predictor reproduces `B`
(Ratio ≈ 0.1). Getting the *shape* right lowers `MSE` a lot; a stray
complexity tax discourages padding the formula. Small crystals behave almost
ideally, but the same imperfection that is invisible in every chain you were
shown becomes material in much larger specimens, so even a structurally
correct formula keeps some residual error on the largest held-out chains;
report the best `Ratio` you can.

## Why the visible window is a trap

Inside the swept window you only ever see the *low-order* modes of *small*
chains — the regime where the resonance ladder looks almost perfectly
straight in the mode index. A curve that memorises that straight-line
spacing interpolates the training window beautifully but has no notion that
a finite chain's resonance ladder must **bend and saturate** as the mode
index approaches the chain size — it keeps climbing where the truth flattens
out, and it has no idea how the ladder's spacing should rescale for a chain
ten times bigger. Only a formula that captures the actual bounded, periodic
shape of the resonance ladder survives being asked about much larger chains
and unseen frequencies.

## Constraints

Time limit 5 s, memory 512 MB. Each training sweep has at most a few hundred
rows. Scoring is fully deterministic; no test involves any randomness beyond
the fixed hidden seed derived from `t`.
