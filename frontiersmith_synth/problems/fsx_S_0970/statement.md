# Prony Query Prospector: Mapping a Buried Sparse Seam from a Few Boreholes

A survey firm models a buried ore seam as a **sparse polynomial over a finite
field** `F_p`:

```
f(x) = sum_{i=0}^{t-1} c_i * x^{e_i}   (mod p)
```

Only `t` terms are nonzero, but the exponents (depths) could in principle be
anywhere in the huge ambient range `[0, p-2]` — the seam is thin compared to the
whole survey volume. Drilling a borehole at depth `x` and reading off `f(x)`
is expensive, so the crew always samples at **standard, geometrically-spaced**
depths following the field's fixed survey base `g` (a primitive root of
`F_p^*`): borehole `k` is drilled at `x_k = g^k mod p`, for `k = 0, 1, ..., Q-1`.
You are handed the `Q` resulting readings `s_k = f(g^k) mod p`. `Q` is small —
far below `p-1`, and on some instances even below what a **structure-unaware**
sparse-recovery routine would need (see below). Your job: report the buried
seam exactly.

**The one thing the geologists already know**: the `t` exponents are **evenly
spaced** — they form an arithmetic progression `e_i = a + i*d` for a secret
start depth `a` and secret spacing `d` (`1 <= d <= d_max`, `d_max` public). This
halves the unknowns that matter for recovering the *roots* of the underlying
signal from `t` independent values down to just **two** scalars `(a, d)` —
because `g^{e_i} = g^a * (g^d)^i` is a geometric progression once `d` is fixed.
A recovery method that ignores this and treats the `t` roots as generic
unknowns needs `2t` samples (the classical Ben-Or-Tiwari / Prony requirement
for an order-`t` linear recurrence); an AP-aware method needs only `t` samples
to pin down the `t` coefficients **once a valid `(a,d)` hypothesis is fixed**,
plus a handful more to verify the hypothesis is right. On several instances
below, `Q` sits in the gap between these two requirements: enough for the
structured approach, not enough for the generic one.

## Public instance (stdin JSON)

```json
{
  "p": 499,               // prime modulus
  "g": 7,                 // primitive root of F_p^* (the survey base)
  "t": 5,                 // number of nonzero terms (sparsity)
  "d_max": 10,             // secret spacing d satisfies 1 <= d <= d_max
  "Q": 8,                  // number of readings given
  "s": [ .. Q ints .. ]    // s[k] = f(g^k) mod p, k = 0..Q-1
}
```

## Answer (stdout JSON)

```json
{"terms": [[e_0, c_0], [e_1, c_1], ...]}   // at most t pairs
```

`e_j` must be a distinct integer in `[0, p-2]`; `c_j` any integer (interpreted
mod `p`). Submitting more than `t` pairs, a duplicate exponent, an
out-of-range exponent, or a non-finite/non-integral value is **invalid** and
scores 0 on that instance.

## Scoring

Reconstruct `fhat(x) = sum_j c_j * x^{e_j} mod p` from your answer and count
`fit_count` = the number of the `Q` given readings it reproduces exactly:
`fhat(g^k) mod p == s_k` for `k = 0..Q-1` (`fit_count` in `[0, Q]`). Every
instance is constructed so that a `<= t`-term explanation matching **all** `Q`
readings is unique — so `fit_count == Q` means exact recovery of the seam.

The evaluator's baseline is the "guess a single constant term equal to the
first reading" construction (`terms = [[0, s_0]]`), which by construction
matches exactly the `k=0` reading (`fit_count = 1`) and — on these
instances — nothing else. For a valid answer with `fit_count = obj`:

```
r = min(1, 0.1 * obj / 1)
```

so the constant-guess baseline scores exactly `0.1`, and reproducing `k` of
the `Q` readings scores `min(1, 0.1*k)`. The reported `Ratio` is the mean of
`r` over 10 deterministic, seeded instances (small and larger `t`, tight and
generous `Q` budgets, some held out for generalization). Infeasible or
malformed answers score `0` on that instance.

There is no shortcut around actually finding the seam: with `Q` far below
`p-1`, no dense/dense-ish interpolation strategy can work, and on tight
instances even the standard Ben-Or-Tiwari/Prony recipe (order-`t` recurrence
from `2t` samples) is provably underdetermined — the arithmetic-progression
structure is not a bonus, it is *required* to close the gap.

Your program reads one public instance JSON from stdin and writes one answer
JSON to stdout. It runs in an **isolated subprocess** and only ever sees the
public instance.
