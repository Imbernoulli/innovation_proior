# Spacing transmitters on a tower to maximize the tightest gap (and emit the placement)

## Research question

A telecom crew is mounting `k` transmitters on a tall lattice tower. The tower offers `n`
candidate mounting brackets at distinct integer heights `h[0..n-1]` (centimetres above the base,
given in arbitrary order). To keep mutual interference predictable, the crew wants the chosen
brackets spread out as evenly as possible: they will pick exactly `k` of the `n` brackets so that
the **smallest pairwise distance** between any two chosen brackets is **as large as possible**.

Formally, choose a size-`k` subset `C` of the heights and let

```
gap(C) = min over distinct chosen heights x, y of |x - y|.
```

You must report `D = max over all size-k subsets C of gap(C)` — the best achievable tightest gap —
**and** output one concrete subset `C` of `k` distinct brackets that actually attains `gap(C) = D`.

This is a *binary-search-the-answer* problem whose deliverable is a **structure**: the answer is
not just the number `D`, it is also a witness placement realizing `D`. The number alone is easy to
defend; the danger lives in the witness and in the boundary of the search. A construction that
looks right on a four-bracket toy can quietly print only `k-1` brackets, or print a `D` one short
of optimal, on the large inputs the judge actually runs.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `k` (`2 <= k <= n <= 2*10^5`). The second
  line (whitespace may wrap arbitrarily) has the `n` candidate heights `h[i]`
  (`0 <= h[i] <= 10^9`), **all distinct**, in arbitrary order.
- Output (stdout):
  - Line 1: the single integer `D`, the maximum achievable tightest gap.
  - Line 2: `k` space-separated distinct heights drawn from the input, forming a subset whose
    minimum pairwise distance equals `D`. Any valid witness is accepted; the order you print them
    in does not matter.
- Time limit: 2 seconds. Memory: 256 MB.

Example: for `n = 6`, `k = 3`, heights `1 2 8 4 9 15`, a correct output is

```
7
1 8 15
```

The brackets `{1, 8, 15}` have pairwise distances `7, 7, 14`, so their tightest gap is `7`, and no
choice of three brackets does better, so `D = 7`.

## Background

The value `D` is monotone-checkable: define the predicate `feasible(d)` = "some `k` of the heights
are pairwise at distance at least `d`". If a placement survives spacing `d`, it certainly survives
spacing `d-1`, so `feasible` is true on a prefix `[1, D]` and false afterwards. That monotonicity is
exactly what makes the answer binary-searchable: find the largest `d` with `feasible(d)` true.

Two families of approach are on the table before committing:

- **Brute force over subsets.** Enumerate every size-`k` subset, compute its tightest gap, keep the
  maximum, and remember one maximizer as the witness. This is obviously correct but costs
  `C(n, k)` work — fine for hand-checking `n <= 9`, hopeless at `n = 2*10^5`.
- **Binary search on `d` with a greedy feasibility test.** For a fixed `d`, sort the heights and
  walk left to right placing greedily: always anchor at the smallest height, then take the next
  height that is at least `d` above the last one placed. The open questions are whether this greedy
  count is genuinely optimal for the predicate, what the search bounds should be, and how to turn
  the final `d = D` into a *valid* witness of exactly `k` brackets.

## Evaluation settings

Judged on hidden tests covering: `k = 2` (the answer is the full span `max - min`); `k = n` (you
must take every bracket, so `D` is the minimum consecutive gap of the sorted heights); tightly
clustered heights forcing `D = 1`; widely spread heights with values near `10^9` so distances and
search bounds exceed 32-bit range; and large `n = 2*10^5` with `k` near `n/2`, where an off-by-one
in the feasibility test that merely *undercounts placements* silently lowers the reported `D` and
prints a short witness. The witness is independently validated: it must list `k` distinct input
heights whose realized tightest gap equals the reported `D`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, k;
    if (!(cin >> n >> k)) return 0;
    vector<long long> p(n);
    for (auto &x : p) cin >> x;
    sort(p.begin(), p.end());

    // TODO: binary-search the maximum tightest gap D, then reconstruct one
    //       valid witness subset of exactly k heights realizing gap == D.
    long long D = 0;

    cout << D << "\n";
    // ... print the k chosen heights on the second line ...
    return 0;
}
```
