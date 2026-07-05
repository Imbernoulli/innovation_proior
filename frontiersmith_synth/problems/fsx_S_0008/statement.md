# Glacier Sensor Net: Balanced Dual-Uplink Split

## Problem

A field team has drilled `n` autonomous sensors into a glacier. Every sensor talks to
some of the others over short-range radio; a link between sensors `u` and `v` carries a
**cross-check weight** `w` (higher = more valuable redundancy if the two sensors end up on
opposite sides of the analysis).

You must assign every sensor to exactly one of **two satellite uplink stations**, call them
station `0` and station `1`. To keep power draw and bandwidth symmetric, the assignment must
be **perfectly balanced**: each station serves exactly `n/2` sensors (`n` is always even).

A link is *cross-checked* when its two endpoints are served by **different** stations; its
weight then contributes to the mission's redundancy score. Links whose endpoints land on the
same station contribute nothing. Your goal is to choose the balanced split that puts as much
link weight as possible across the two stations.

This is a balanced maximum-bisection instance. The optimum is NP-hard to find, so any balanced
assignment is accepted and graded by how much cross-checked weight it captures.

## Input

- Line 1: two integers `n m` — the number of sensors and the number of radio links.
- Next `m` lines: three integers `u v w` — a link between sensors `u` and `v` with weight `w`.

Sensors are numbered `1..n`. Links are undirected; there are no self-loops and no duplicate
links. `n` is even.

## Output

Print `n` integers `s_1 s_2 ... s_n` (whitespace-separated), where `s_i` is the station
(`0` or `1`) assigned to sensor `i`.

## Feasibility

- Each `s_i` must be `0` or `1`.
- Exactly `n/2` of the sensors must be assigned to station `1` (and thus `n/2` to station `0`).

Any output violating these rules scores `0`.

## Objective

Maximize the cross-checked weight

```
F = sum of w over all links (u,v) with s_u != s_v.
```

## Scoring

Let `B` be the cross-checked weight of the **reference split** in which sensors `1..n/2` go to
station `0` and sensors `n/2+1..n` go to station `1` (the checker computes `B` itself; it is
always positive). With `F` your feasible split's cross-checked weight, the raw score is

```
score = min(1000, 100 * F / B)
```

and the reported ratio is `score / 1000` (so the reference split earns ratio `0.1`, and matching
ten times the reference weight caps the ratio at `1.0`). Higher is better.

## Constraints

- `2 <= n <= 600`, `n` even.
- `1 <= m <= n*(n-1)/2`.
- `1 <= w <= 100`.
- Time limit: 5 s. Memory limit: 512 MB.

## Example

Input:

```
4 4
1 2 5
2 3 9
3 4 4
4 1 7
```

The reference split (sensors 1,2 -> station 0; sensors 3,4 -> station 1) cross-checks links
(2,3) and (4,1): `B = 9 + 7 = 16`.

Suppose you output `0 1 0 1` (sensors 1,3 -> station 0; sensors 2,4 -> station 1). All four
links are then cross-checked: `F = 5 + 9 + 4 + 7 = 25`. The split is balanced (two per
station), so `score = min(1000, 100 * 25 / 16) = 156`, ratio `0.156`.
