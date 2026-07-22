# Word Machine

A "word machine" starts from the single letter `0` and repeatedly rewrites its
current word: every letter `c` in the word is simultaneously replaced by a
fixed string `sigma(c)` (the same string every time `c` occurs). Formally,
over the alphabet `{0,1,2}`, a hidden **substitution** `sigma` maps each
letter to a nonempty string of length `1..4` over `{0,1,2}`; the word after
`n` rewrites is `w_0 = "0"`, `w_{n+1} = sigma(w_n[0]) sigma(w_n[1]) ...`
(concatenation, in order). Different test cases use different hidden `sigma`,
always chosen so growth never stalls or explodes (its long-run growth
factor per rewrite is roughly between 1.2x and 3.6x).

You are shown the machine's own history — the exact words it produced for
the first several rewrites — and must recover `sigma` well enough to predict
what the word looks like **much later**, long after the printed history ends.

## Input (stdin)

```
t n_train
w_0
w_1
...
w_{n_train}
```

`t` is the test id. `n_train` rewrites are shown (`n_train+1` words total,
each a string over `0`,`1`,`2`). `sigma` is NOT printed — only its outputs.

## Output (stdout): your guessed substitution

Print exactly 3 lines: line `i` (for letter `i` = 0, 1, 2) is your guessed
`sigma(i)`, a nonempty string over `0,1,2` of length at most 12.

## Feasibility

Exactly 3 non-empty lines, each consisting only of the characters `0`,`1`,`2`,
each of length `1..12`. Any violation scores `0`.

## Objective (maximise)

The grader iterates YOUR guessed substitution (and, separately, the true
hidden one) from axiom `0` out to several horizons *far* beyond `n_train` —
tens of rewrites further than anything you were shown. At each horizon it
compares your predicted word against the true one on two things that are
computed **exactly** (via the substitution's letter-transition counts, never
by materialising astronomically long strings): (a) the word's length, on a
log scale, and (b) its letter-frequency mix (what fraction of the word is
`0`, `1`, `2`). Both must match for a high score — matching the length alone
while getting the letter mix wrong (or vice versa) is graded as a partial
answer, not a full one. Scores at the horizons are averaged into `F`, then
normalised against a fixed internal calibration: guessing "nothing grows"
lands around 0.1-0.2; recovering the exact hidden substitution tops out
around 0.85, leaving headroom above it.

**Illustrative FORM only — NOT the hidden mechanism:** imagine a toy 2-letter
machine with `sigma(0)="01"`, `sigma(1)="0"` (Fibonacci word). From axiom `0`
it produces `0, 01, 010, 01001, ...`; its word length grows by the golden
ratio ~1.618 per rewrite, and its long-run letter mix converges to a fixed
ratio of `0`s to `1`s — NOT computable by watching the length curve alone.
This shows the shape of the task; the real machines use 3 letters and a
different, harder-to-guess `sigma`.

## Why the growth curve is a trap

The word's length after `n` rewrites is governed by the substitution's
letter-transition counts — concretely, by the dominant eigenvalue of the
3x3 matrix `M[i][j]` = "how many times does letter `j` appear in `sigma(i)`".
Watching only the first handful of rewrites and fitting a curve to the raw
lengths mixes in the *sub-dominant* eigenvalues, which decay but are not
negligible yet on a short window — so a curve fit's implied rate is
systematically biased, and the bias compounds multiplicatively at long
horizons. Worse, a length-only curve says nothing about the letter mix at
all, which depends on the matrix's eigen*vector*, not just its top eigenvalue.
Matching the training words' own letter frequencies is also not enough: they
have not converged to their long-run values on a short prefix either.

## Constraints

Time limit 5s, memory 512MB. `n_train` is 5-8; each printed word has at most
a few thousand characters (`4^8` in the worst case). Scoring is fully
deterministic — same submission, same score, forever.
