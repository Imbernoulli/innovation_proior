# Seat Quotas for the Apprentice Match

## Problem
A guild places `N` apprentices into `M` workshops across several districts using a fixed,
public matching rule. **You do not choose the matching** — you choose the **seat quota**
(capacity) `c[j]` of every workshop `j`, subject to a room limit `c[j] <= cap_max[j]` and
a total seat budget `sum_j c[j] = T`. The guild then runs standard **apprentice-proposing
deferred acceptance** with your quotas and lives with whatever match results.

Each apprentice has a ranked list of acceptable workshops (lists can be different
lengths — some apprentices have far fewer fallback options than others). Each workshop
has a strict priority order over all apprentices. If apprentice `i` ends up matched at
position `k` (0-indexed) of their own list, the guild earns `W[k]` guild-points for that
apprentice (only the first `len(list)` entries of `W` ever apply); an unplaced apprentice
earns `0`. The weight vector is fixed and public: `W = [100, 45, 8, 3, 1]`. Landing your
first choice is worth far more than any later one; no list is longer than 5.

## Deferred acceptance (the exact fixed rule)
Repeat until no apprentice is rejected: each unassigned apprentice proposes to the most
preferred workshop on their own list that has not yet rejected them; each workshop
tentatively holds the **top `c[j]` proposers so far by its priority order** (`c[j] = 0`
rejects everyone) and rejects the rest — a newly held apprentice can bump a
lower-priority one, who then proposes onward. An apprentice rejected by every workshop
on their list stays unplaced. The outcome is unique and independent of proposal order.

## Input (stdin)
```
N M T
cap_max[0] cap_max[1] ... cap_max[M-1]
<N lines: apprentice i's list = "L_i  w_1 w_2 ... w_{L_i}", most preferred first>
<M lines: workshop j's priority order = a permutation of 0..N-1, highest priority first>
```
Workshop ids are `0..M-1`; apprentice ids are `0..N-1`. Some workshops may never appear
in any apprentice's list at all — real seats that simply attract no interest this cycle.

## Output (stdout)
`M` integers (whitespace-separated): the quota `c[j]` for workshop `j` in order.

## Feasibility
Valid iff exactly `M` integers are given, every `0 <= c[j] <= cap_max[j]`, and
`sum_j c[j] = T`. Any violation (wrong count, out-of-range seat, wrong total, a
non-integer/`nan`/`inf`) scores `Ratio: 0.0`.

## Objective (maximize)
Run apprentice-proposing deferred acceptance with your quotas `c`. The objective is the
total guild-points of the resulting matching (see weight vector `W` above).

## Scoring
The checker builds its own **uniform** quota (split `T` as evenly as possible across all
`M` workshops, remainders to the lowest-indexed ones — always feasible here), runs
deferred acceptance on it, and calls the result `B`. With maximization normalization:
```
sc = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = sc / 1000.0
```
Reproducing the uniform quota scores `Ratio = 0.1`; a quota that makes the match `10x`
better than uniform caps at `1.0`.

## Why quotas are subtle
The score is **non-local** in `c`. Two workshops sit on *every* apprentice's list as
their #1 choice, but each one's priority is dominated by a small, fixed, hidden elite —
so the huge first-choice demand for them overstates how much quota they can usefully
absorb; extra capacity there mostly upgrades apprentices who already had a solid
fallback. District workshops are chained in a ring (a resident rejected from their home
district re-proposes to the next district along the ring), and inside a district some
apprentices have **no fallback at all** and sit at the very bottom of that district's own
priority queue — they are rescued only once the district's quota clears its whole loyal
population first, a real per-district threshold. A flat, count-proportional split treats
every unit of quota as equally valuable — it is not.

## Constraints
- `26 <= M <= 90`, `1 <= L_i <= 5`, `40 <= N <= 600`, `0 <= T <= N`, `0 <= cap_max[j]`.
- Time limit 5s, memory 512m.

## Example
(Toy instance below the stated size range, purely to illustrate the mechanic.)
`N=3, M=2, T=3`, `cap_max=[3,3]`. Apprentices 0 and 1 both list `[0,1]`; apprentice 2
lists `[1,0]`. Workshop 0 prioritizes `1,0,2`; workshop 1 prioritizes `0,1,2`.
Quota `c=[1,2]`: DA seats apprentice 1 at workshop 0 (rank 0 -> 100) and apprentices 0, 2
at workshop 1 (ranks 1, 0 -> 45, 100), `F = 245`. Quota `c=[2,1]` instead seats 0, 1 at
workshop 0 and 2 at workshop 1: `F = 100+100+100 = 300`. The seat you move decides who
cascades.
