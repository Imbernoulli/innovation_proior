# Balanced Tour of the Forbidden-Window de Bruijn Graph

## Problem

You are given an alphabet of size `k` (symbols `0..k-1`), a window length `L`, and a set
of **forbidden** length-`L` windows. A length-`L` window is *allowed* if it is not in the
forbidden set. Your task is to output a **cyclic string** `s` (length `n`, read circularly,
so its windows wrap around the end) such that:

1. Every one of the `n` cyclic windows of `s` (the substring of length `L` starting at each
   position `0..n-1`, wrapping past the end) is an *allowed* window — none of them may equal a
   forbidden window.
2. Every *allowed* window (i.e. every length-`L` string over the alphabet that is not
   forbidden) appears as at least one of the `n` cyclic windows of `s`.

Among all strings satisfying both conditions, **shorter is better** — you are scored on how
close your length `n` gets to the best achievable length.

## Input (stdin)

```
k L m
w_1
w_2
...
w_m
```
`k` is the alphabet size, `L` the window length, `m` the number of forbidden windows, and
each `w_i` is a forbidden window: a string of length `L` over digits `0..k-1`. It is
guaranteed that a valid cyclic string exists for the given instance.

## Output (stdout)

A single line containing the cyclic string `s` (only digit characters `0..k-1`, length
`n >= 1`). Nothing else.

## Feasibility

The checker rejects (score 0) any output that: is empty, contains characters outside
`0..k-1`, is longer than `5,000,000` characters, contains a forbidden window (cyclically),
or fails to contain some allowed window (cyclically) at least once.

## Scoring

The checker builds its own always-feasible reference string of length `B` using a
deliberately naive construction (it revisits each required window in a fixed order,
re-walking from scratch each time — lots of redundant travel). Writing `n` for the length
of your accepted output, the score is
```
score = min(1000, 100 * B / n) / 1000
```
so matching the naive reference scores about `0.1`, and cutting the length roughly in half
scores about `0.2`; the score saturates below `1.0` by design (there is always headroom
above the reference solutions shipped with this problem).

## Why this is not just "greedy append a new window"

Think of every length-`(L-1)` string as a node and every length-`L` window as a directed edge
from its length-`(L-1)` prefix to its length-`(L-1)` suffix (a de Bruijn graph). A cyclic
string covering every allowed window at least once, using only allowed windows, corresponds
exactly to a **closed walk** in this graph (with forbidden windows deleted as edges) that
uses every remaining edge at least once. Deleting forbidden edges can leave some nodes with
more incoming than outgoing edges (or vice versa) — such a graph has **no Eulerian circuit**,
so a walk that only ever appends a symbol producing a *new, unused* window (the obvious
recipe) will run out of room and be forced into long, blind backtracking. The way out is to
first restore the missing balance: find shortest allowed detours between deficient and
surplus nodes, add those detours to the tour plan, and only then trace a closed walk that
uses every edge (original and detour) — this is far shorter.

## Example

`k=2, L=2`, forbidden = `{"11"}`. Allowed windows: `00, 01, 10`. The cyclic string `001`
has windows (positions 0,1,2, wrapping): `00`, `01`, `10` — every allowed window appears
exactly once and `11` never occurs. Its length is 3, which here already matches the
smallest possible length (this small illustrative case has no imbalance to repair).

## Constraints

`2 <= k <= 4`, `2 <= L <= 5`, `0 <= m < k^L`, time limit 5s, memory 512MB.
