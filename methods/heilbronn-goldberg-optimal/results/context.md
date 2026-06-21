# Context: Heilbronn triangle problem at n=11 (record rung)

## Research question

Place `n = 11` points in `[0,1]^2` to maximize the **minimum** triangle area over all
`C(11,3) = 165` triples. The constructor emits 11 points; the exact minimum triangle area is the
score (higher better).

This is the record rung. The unit-square record at `n = 11` is `Δ(11) = 1/27 = 0.0370370...`
(Goldberg 1972, conjectured optimal). The question for this rung: produce a configuration whose minimum
triangle area equals the record value `1/27`.

## How the score is defined

```
score(P) = min over all 165 triples of  |(b-a) x (c-a)| / 2,   all points in [0,1]^2.
```

## Record and yardstick

Unit-square record at `n = 11`: `Δ(11) = 1/27 = 0.0370370370` (Goldberg 1972, conjectured optimal;
listed "horizontally symmetric" on Erich Friedman's Heilbronn-for-squares page; exact rational
coordinates tabulated in arXiv:2603.11107 Table 11, attributing n=11 to Goldberg). Prior rungs: 11-gon
`0.021456` (0.579); random `0.010872` (0.294); SA `0.035639` (0.962); SA + soft-min polish `0.037032`
(0.99986). The target value `1/27` is believed optimal at `n = 11`; whether it is provably optimal
remains open (only a proof certifies it).

Neighboring records for orientation: `Δ(10) ≈ 0.0465` (Comellas–Yebra), `Δ(12) ≈ 0.0326`
(Comellas–Yebra 2001), `Δ(13) ≈ 0.0270` (Karpov 2011).

## The editable interface

One editable function, `construct() -> (11,2) array in [0,1]^2`; the evaluator and `n = 11` are fixed.
The constructor returns the 11 point coordinates; the evaluator computes the minimum triangle area over
all `165` triples and reports the score.
