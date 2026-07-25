# Minimum inspection points to stab all corridors, and how many corridors get double-stamped

## Research question

A museum has `n` corridors. Corridor `i` spans the closed integer interval `[l_i, r_i]` along a
single hallway axis. A curator wants to place **inspection markers** at integer coordinates so that
**every corridor contains at least one marker** (a marker at coordinate `p` is inside corridor `i`
iff `l_i <= p <= r_i`). The curator uses the *canonical* greedy stabbing rule to decide where markers
go:

> Sort the corridors by **right endpoint** (ties broken by left endpoint). Sweep through them in that
> order. Whenever the corridor currently being examined does **not** already contain a
> previously-placed marker, place a new marker at that corridor's **right endpoint** `r_i`.

This rule uses the fewest possible markers. You must report two numbers for the markers it places:

1. `K` = the number of markers placed.
2. `M` = the number of corridors (out of the original `n`) that end up containing **two or more** of
   the placed markers — call these the *double-stamped* corridors.

## Input / output contract

- Input (stdin): the first token is `n` (`1 <= n <= 2*10^5`). Then follow `n` lines (or just
  whitespace-separated pairs), each `l_i r_i` with `-10^9 <= l_i <= r_i <= 10^9`. A corridor may be a
  single point (`l_i == r_i`), and corridors may coincide, nest, or overlap arbitrarily.
- Output (stdout): a single line with two integers separated by one space: `K M`.
- Time limit: 1 second. Memory: 256 MB.

Example: for the five corridors `[0,10] [1,2] [3,4] [5,6]` plus `[0,5] [7,12]` the markers land at
`{2, 4, 6, 12}` (`K = 4`), and the corridors `[0,10]` and `[0,5]` each contain at least two of them,
so `M = 2`; the program prints `4 2`.

## Background

"Minimum number of points so that every interval is hit" is a classic family of interval-scheduling
problems, and stabbing (as opposed to selection or scheduling) is one member of that family. The
curator's exact procedure is fully specified above; this problem also asks for a second, derived
quantity (the double-stamped count) once that procedure has been run.

## Evaluation settings

Judged on hidden tests covering: a single corridor (`n = 1`); many identical corridors; single-point
corridors (`l_i == r_i`); deeply nested corridors where one giant corridor swallows every marker;
chains of disjoint corridors; markers and boundaries that coincide exactly; negative coordinates; and
large `n = 2*10^5` with coordinates near `±10^9`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<pair<long long,long long>> iv(n); // (l, r)
    for (int i = 0; i < n; i++) cin >> iv[i].first >> iv[i].second;

    // TODO: run the canonical greedy stab (sort by right endpoint, place a
    // marker on the right endpoint of each still-unhit corridor); then count
    // how many corridors contain >= 2 of the placed markers.
    long long numPoints = 0, multi = 0;

    cout << numPoints << " " << multi << "\n";
    return 0;
}
```
