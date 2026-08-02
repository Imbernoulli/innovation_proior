# Alert Budget: Flagging Layering Behind a Market-Making Curtain

## Problem
You are building a surveillance rule for an exchange's order-book event log.
The log covers `N` participants across `W` short trading windows. In each
`(window, participant)` cell, the participant emits a sequence of local
events: **P**lace an order, **C**ancel an order, or **T**rade aggressively
(cross the book and take liquidity), each tagged with a side, **B**uy or
**S**ell, and a local timestamp `t` (0, 1, 2, ... within that cell, its own
clock).

Two behaviors look almost identical on the surface: genuine market makers
churn through many rapid place/cancel cycles to keep two-sided quotes fresh,
and layering manipulators fire off a rapid burst of cancels too. Simply
flagging "lots of cancels, fast" cannot tell them apart — it flags every
market maker along with every manipulator, and your alert budget is limited:
you may flag at most `K` `(window, participant)` cells. Your job is to spend
that budget on the cells that are *actually* manipulative.

## Input (stdin)
```
testId
N W K
E
w_1 pid_1 t_1 side_1 action_1 size_1
...
w_E pid_E t_E side_E action_E size_E
```
`0 <= w < W`, `0 <= pid < N`, `action` in `{P, C, T}`, `side` in `{B, S}`,
`size` is a flavor integer (order size, not needed for scoring). Events for
a given `(w, pid)` are listed with non-decreasing local timestamp `t`, but
`t` resets to 0 for every new `(w, pid)` pair (each cell has its own clock —
timestamps are not comparable across cells).

## Output (stdout)
```
C
w_1 pid_1
...
w_C pid_C
```
`C` is how many `(window, participant)` cells you flag as manipulative;
list each flagged cell on its own line, `0 <= C <= K`.

## Feasibility
- `C` must be a non-negative integer `<= K` (the alert budget is a hard cap:
  exceeding it is infeasible, not merely penalized).
- Exactly `C` well-formed lines must follow, each two integers in range
  (`0<=w<W`, `0<=pid<N`).
- No `(w, pid)` pair may be flagged twice.
- Any violation scores `Ratio: 0.0`.

## Objective
Each cell has a hidden ground-truth label — genuinely manipulative or not —
that is **not** given to you; you must infer it from the event pattern. Let
`M` be the (unknown to you) true set of manipulative cells and let your
flagged set be `F`, with `TP = |F ∩ M|`. Your score is a Laplace-smoothed,
geometric-mean precision-weighted recall:
```
precision = (TP + 0.5) / (|F| + 1)
recall    = (TP + 0.5) / (|M| + 1)
objective = sqrt(precision * recall)
```
This rewards catching real manipulation (recall) but punishes drowning it in
false alarms (precision) — both matter, and the budget forces a choice. The
square root compresses the scale so a large improvement over a naive
baseline is rewarded without blowing past a fixed ceiling.

## Scoring
The checker also computes its own naive reference `B`: the top-`K` cells
ranked purely by raw total cancel count, ignoring side and timing (same
`objective` formula as above), floored against a small fixed fraction of the
best any submission could ever achieve (so an unlucky, near-zero baseline
draw never inflates your score past a sane cap). Your `objective` value `F`
is compared against it: the checker computes `sc = min(1000, 100*F/B)` and
prints `Ratio: sc/1000`. Matching the naive baseline scores ~0.10; a
materially better discriminator scores higher, up to a cap that always
leaves headroom above the reference solution.

## Constraints
`6 <= N <= 14`, `3 <= W <= 5`, `K` scales with the (hidden) manipulation
density, time limit 5s, memory 512MB.

## Example (worked, illustrative only — not real test data)
Say `K=3`, and (unknown to the solver) `|M|=4`. Suppose your output flags 3
cells, 2 of which are truly manipulative (`TP=2`):
`precision=(2+0.5)/(3+1)=0.625`, `recall=(2+0.5)/(4+1)=0.5`,
`F=sqrt(0.625*0.5)=0.559`. If the checker's naive top-K-by-cancel-count
baseline only catches `B=0.28` on this instance (it mostly grabs market
makers, who cancel just as much but on both sides), your normalized score
is `min(1000, 100*0.559/0.28)/1000 = min(1000, 199.6)/1000 = 0.1996`.
Flagging more market makers instead would raise `|F|` without raising
`TP`, shrinking precision and the score.
