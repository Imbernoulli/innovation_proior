# Context: Heilbronn triangle problem at n=11, Goldberg's exact 1/27 configuration (endpoint)

## Research question

Place `n = 11` points in `[0,1]^2` to maximize the **minimum** triangle area over all
`C(11,3) = 165` triples. The constructor emits 11 points; the exact minimum triangle area is the
score (higher better).

This is the record rung. The search-and-polish frontier (heavy SA + soft-min L-BFGS-B polish) reaches
`0.037032`, five parts in a million below the conjectured optimum `1/27 = 0.0370370...`. That residual
is pure optimizer tolerance, not a geometric gap. The question: can the last `5×10^-6` be closed
*exactly* by reproducing Goldberg's structured rational configuration instead of approximating it?

## How the score is defined

```
score(P) = min over all 165 triples of  |(b-a) x (c-a)| / 2,   all points in [0,1]^2.
```

## Record and yardstick

Unit-square record at `n = 11`: `Δ(11) = 1/27 = 0.0370370370` (Goldberg 1972, **conjectured optimal**;
listed "horizontally symmetric" on Erich Friedman's Heilbronn-for-squares page; exact rational
coordinates tabulated in arXiv:2603.11107 Table 11, attributing n=11 to Goldberg). Prior rungs: 11-gon
`0.021456` (0.579); random `0.010872` (0.294); SA `0.035639` (0.962); SA + soft-min polish `0.037032`
(0.99986). This rung REACHES the record exactly: `1/27` is believed optimal at `n = 11`, so the exact
value is the ceiling and the still-open part is the unproven optimality (only a proof certifies it).

Neighboring records for orientation: `Δ(10) ≈ 0.0465` (Comellas–Yebra), `Δ(12) ≈ 0.0326`
(Comellas–Yebra 2001), `Δ(13) ≈ 0.0270` (Karpov 2011).

## The editable interface

One editable function, `construct() -> (11,2) array in [0,1]^2`; evaluator and `n = 11` fixed. This
rung returns Goldberg's exact rational coordinates (denominators `3, 9, 6, 2`; eight points on the
boundary, three interior; mirror-symmetric about `x = 1/2`) and verifies, in **exact rational
arithmetic** over all `165` triples, that the smallest triangle is exactly `1/27` — with `28` triangles
binding at the minimum, a rigid over-determined web that pins the layout as a genuine max-min optimum.

```python
from fractions import Fraction as Fr
# Goldberg (1972) n=11 optimum: min triangle area = 1/27 (exact), verified in rational arithmetic.
GOLDBERG = [(Fr(1,3),Fr(0)),(Fr(2,3),Fr(0)),(Fr(0),Fr(2,9)),(Fr(1),Fr(2,9)),
            (Fr(1,3),Fr(4,9)),(Fr(2,3),Fr(4,9)),(Fr(0),Fr(2,3)),(Fr(1),Fr(2,3)),
            (Fr(1,2),Fr(7,9)),(Fr(1,6),Fr(1)),(Fr(5,6),Fr(1))]
```
