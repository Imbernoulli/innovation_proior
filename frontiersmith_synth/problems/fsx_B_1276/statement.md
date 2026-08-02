# Comparable Set Defense: Pricing an Intercompany Margin for Audit

## Problem
A subsidiary ("the tested party") sells services to its parent. Its functional profile is
a triple `(f0, a0, r0)` describing how much it contributes on three axes: **F**unctions
performed, **A**ssets used, **R**isks assumed (each `0..10`). Tax law requires the
intercompany operating margin to sit inside an "arm's-length range" built from independent
**comparable** companies.

You are given `N` candidate comparables. Candidate `i` has a reported operating margin
`margin_i` (basis points), a functional profile `(f_i, a_i, r_i)`, and a `doc_cost_i`
(the effort needed to document it). You must choose which comparables to submit as your
support, how much documentation *depth* `d_i in {0,1,2,3}` to give each, and the margin
you will declare. Submitting a comparable at all — even at depth 0 — already costs
`doc_cost_i` (gathering and filing its data); each extra depth level costs `doc_cost_i`
again. So submitting comparable `i` at depth `d_i` costs `doc_cost_i * (d_i + 1)` against
a shared `BUDGET`, and a bigger support set is never free.

An audit is not a single test: it can arrive under any of several **postures**, each
weighting the F/A/R axes differently (e.g. one posture cares mostly about risk, another
mostly about assets) — you are not told the exact weights, only that they vary. Under a
given posture, a candidate's **true functional distance** from the tested party is a
weighted sum of `|f_i-f0|`, `|a_i-a0|`, `|r_i-r0|`. A candidate that is far enough from the
tested party under that posture is *challenged*; it survives only if the documentation
depth you gave it is deep enough to cover the shortfall (deeper documentation covers a
bigger shortfall, up to a cap). The **arm's-length range** itself is computed by the
auditor from the *entire* universe of `N` candidates under that posture's weighting (a
weighted interquartile range, closer candidates counting more) — submitting a narrow,
favourable subset cannot move this range, it only determines how much of *your own* support
survives scrutiny.

If, under a posture, your declared margin falls inside that posture's range **and** at
least `MIN_COMPS` of your *submitted* comparables survive scrutiny, you keep the full
margin. If it falls outside the range (but support is adequate), the margin is clipped to
the nearest bound and a penalty applies (larger for a larger clip, damped by how much
documentation you gave your support). If fewer than `MIN_COMPS` of your submitted
comparables survive, the position is treated as undocumented: your margin is reset to a
conservative fallback and a heavy penalty applies regardless of where your declared margin
sat.

## Input (stdin)
```
N
f0 a0 r0
REV BUDGET MIN_COMPS
margin_1 f_1 a_1 r_1 doc_cost_1
...
margin_N f_N a_N r_N doc_cost_N
```

## Output (stdout)
```
S
i_1 d_1
...
i_S d_S
M
```
`S` = number of comparables you submit; each `i_k` (1-indexed, distinct, `1..N`) with a
documentation depth `d_k in {0,1,2,3}`; `M` = your declared margin (basis points).

## Feasibility
- `1 <= S <= N`; the `i_k` are pairwise distinct indices in `[1,N]`; each `d_k in {0,1,2,3}`.
- `sum(doc_cost_i * (d_i + 1))` over submitted comparables `<= BUDGET`.
- `0 <= M <= 10000`.
Any violation scores `Ratio: 0.0`.

## Objective
Maximize the expected post-audit profit: averaged over the (hidden, fixed) battery of
audit postures, `(adjusted_margin - penalty) * REV / 10000`, floored at 0 per posture.

## Scoring
Let `B` be the checker's own internal construction (`MIN_COMPS` candidates picked at a
fixed stride through the input order, no documentation, `M` = their plain average margin),
scored the same way, giving baseline profit `B`. With `F` your construction's profit:
```
Ratio = min(1000.0, 100.0 * F / max(1e-9, B)) / 1000.0
```

## Constraints
- `20 <= N <= 45`, `1 <= REV,BUDGET <= 10^7`, `MIN_COMPS = 3` in every test.
- Time limit 3s, memory 256m.

## Example
`N=5`, tested party `(5,5,5)`, one candidate `margin=1500 f=5 a=5 r=5 doc_cost=1` (a
perfect match). Submitting just that one candidate, `d=0`, `M=1500` satisfies feasibility
(`S=1 <= MIN_COMPS` is allowed by the format, though `MIN_COMPS=3` support is needed to
avoid the fallback in every posture) — this illustrates the I/O shape only, not a
competitive strategy.
