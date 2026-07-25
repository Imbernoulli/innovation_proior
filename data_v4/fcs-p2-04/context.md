# Equal-sum partition of a small-integer multiset

## Research question

You are given a multiset of `n` positive integers `a[0..n-1]`. Decide whether the multiset can be
split into **two subsets whose sums are equal** — that is, whether there is a way to colour each
element red or blue so that the red sum equals the blue sum. Every element must go to exactly one of
the two subsets; an element may not be left out, split, or duplicated. Output `YES` if such a split
exists and `NO` otherwise.

This is the decision form of the equal-sum (balanced) partition problem. It is the building block
behind load-balancing two machines, fair two-way splits, and many "can these items be divided
evenly" subproblems, so the exact YES/NO boundary has to be right.

## Input / output contract

- Input (stdin): the first token is `n` (`1 <= n <= 200`); then `n` integers `a[i]`
  (`1 <= a[i] <= 1000`), whitespace-separated (spaces and/or newlines, in any layout).
- Output (stdout): a single line containing `YES` or `NO`.
- Time limit: 1 second. Memory: 256 MB.

Example:

```
Input
5
4 9 10 12 15

Output
YES
```

(The split `{10, 15}` and `{4, 9, 12}` both sum to `25`.)

## Evaluation settings

Judged on hidden tests covering: odd-total inputs (immediate `NO`), single elements (`NO`), two equal
or unequal values, all-equal multisets with even and odd counts, value extremes (mixes of `1` and
`1000`), planted-YES instances where an exact equal split exists, and the size extreme `n = 200` with
values up to `1000` (so the half-target can reach `100000`).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<int> a(n);
    long long total = 0;
    for (auto &x : a) { cin >> x; total += x; }

    // TODO: decide whether the multiset splits into two subsets of equal sum.
    bool ok = false;

    cout << (ok ? "YES" : "NO") << "\n";
    return 0;
}
```
