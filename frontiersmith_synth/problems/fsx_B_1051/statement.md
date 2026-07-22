# Parade of the Two-Ring Clans

## Problem
The realm's clans are indexed by a pair (h, g): a house h in {0,...,n1-1} and a
guild g in {0,...,n2-1}. There are n = n1*n2 clans in all, one for every pair.
You are the parade marshal: choose an order c_1, c_2, ..., c_n in which **every**
clan marches past the reviewing stand **exactly once**.

As clan c_i joins, the formation's cumulative drift updates componentwise modulo
(n1, n2): S_0 = (0,0), S_i = (S_{i-1} + c_i) mod (n1,n2). Also, between every two
consecutive marchers there is a banner-gap vector d_i = (c_{i+1} - c_i) mod
(n1,n2), for i = 1..n-1.

The review committee dislikes repetition. A drift value that has already been
shown means the formation just re-displayed an old configuration. A banner-gap
that has already been used means two consecutive marchers repeated a transition
the crowd has already seen. Your score rewards how many DISTINCT drifts you
reach and how many DISTINCT banner-gaps you use — not the size of any single
gap or drift, only how many different ones occur.

## Input (stdin)
One line: two integers `n1 n2` (n1, n2 >= 2).

## Output (stdout)
Exactly n = n1*n2 lines, each with two integers `a b` (0 <= a < n1, 0 <= b <
n2): the clan marching at that position. Every one of the n possible pairs
(a,b) must appear exactly once (the parade is a permutation of all clans).

## Feasibility
- Exactly n lines, each with exactly two finite integer tokens.
- 0 <= a < n1 and 0 <= b < n2 for every printed pair.
- The n printed pairs are pairwise distinct (every clan appears once).
Any violation (wrong token count, out-of-range value, duplicate clan,
non-integer/non-finite token) scores 0.

## Objective (maximize)
Let S_1, ..., S_n be the running drifts defined above; let P = the number of
DISTINCT values among S_1..S_n (1 <= P <= n). Let d_1, ..., d_{n-1} be the
consecutive banner-gaps defined above; let D = the number of DISTINCT values
among d_1..d_{n-1} (0 <= D <= n-1). The objective is

    F = P + D            (theoretical ceiling 2n-1, not always attainable)

## Scoring
The checker also marches its own reference parade — the clans in plain
lexicographic order of (a,b) — to get a baseline value B = P_base + D_base
(always positive, computed the same way from that fixed order). Your ratio is

    ratio = min(1.0, F / (10 * B))

printed as `Ratio: <ratio>`. Matching the baseline exactly gives ratio 0.1;
you need F to be several times B to approach 1.0. Whether F = 2n-1 is even
possible depends on the algebraic structure of the group Z_n1 x Z_n2 — for
some (n1,n2) the true best-achievable F is strictly below the ceiling, so a
ratio well under 1.0 may already be close to the best any parade can score.

## Constraints
2 <= n1, n2, and n1*n2 <= 1000. Time limit 5s, memory 512MB.

## Example
n1=3, n2=2 (n=6). Suppose you print the parade:
```
1 0
0 1
2 1
2 0
0 0
1 1
```
Running drifts mod (3,2): (1,0), (1,1), (0,0), (2,0), (2,0), (0,1) — the 4th
and 5th are both (2,0), so P = 5 distinct drifts. Banner-gaps: (2,1), (2,0),
(0,1), (1,0), (1,1) — all five are distinct, so D = 5. Hence F = P + D = 10.

The reference parade in lexicographic order — (0,0),(0,1),(1,0),(1,1),(2,0),
(2,1) — gives drifts (0,0),(0,1),(1,1),(2,0),(1,0),(0,1) (P = 5, since the
last repeats the 2nd) and gaps (0,1),(1,1),(0,1),(1,1),(0,1) (only 2 distinct
values, D = 2), so B = 5 + 2 = 7.

ratio = min(1.0, 10 / (10*7)) = 10/70 = 0.142857.
