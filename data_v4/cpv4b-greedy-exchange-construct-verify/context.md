# Distinct-echo repeater offsets: the smallest collision-free delay schedule

## Research question

A mesh of `n` radio repeaters relays a beacon. Each repeater `i` is assigned an integer **delay
offset** `x[i] >= 0` (in clock ticks). When the beacon passes through any *ordered or unordered
pair* of repeaters `{i, j}` (including a repeater paired with itself, `i = j`), the receiver sees a
combined echo at time `x[i] + x[j]`. To keep the echoes separable, the operator requires that **all
pairwise sums `x[i] + x[j]` (taken over `i <= j`) are distinct** — no two different pairs may land on
the same combined delay. A set with this "all pairwise sums distinct" property is a **Sidon set**
(equivalently a `B_2` set).

Among all valid assignments the operator wants a single **canonical** schedule for reproducible
firmware builds: output the **lexicographically smallest** valid assignment, listed in increasing
order. Formally, sort each candidate set ascending and compare sequences position by position; you
must output the set whose sorted sequence is smallest. The offsets must stay within a hardware cap
`M = 200000000`.

This is a construction task: you do not report a number, you must *emit a structure* and it must
satisfy the property exactly. The delicate part is that "all pairwise sums distinct" is a global
condition over `Theta(n^2)` pairs, and a construction that looks right on a handful of repeaters can
silently produce a colliding schedule once `n` grows.

## Input / output contract

- Input (stdin): a single integer `n` (`1 <= n <= 1000`).
- Output (stdout): one line with the `n` chosen offsets in **strictly increasing** order, separated
  by single spaces. The set must be a valid Sidon set, every offset must lie in `[0, M]` with
  `M = 200000000`, and the sorted sequence must be the lexicographically smallest such set.
- Time limit: 2 seconds. Memory: 256 MB.

Example: for `n = 6` the answer is `0 1 3 7 12 20`.

## Background

Two construction strategies are on the table before committing to one:

- **A closed-form formula.** Pick a slick algebraic family — squares `x[k] = k^2`, or an
  Erdos-Turan-style `2 p k + (k^2 mod p)` set — that is *provably* Sidon for the full family. The
  appeal is `O(n)` construction with no search. The open question is whether the chosen formula is
  actually collision-free for every `n` in range, and whether it is the *lexicographically smallest*
  valid set (a formula set is almost never lex-minimal).

- **Greedy by smallest extension.** Start from `x[0] = 0` and repeatedly append the smallest integer
  greater than the current maximum that keeps every pairwise sum distinct. The hope is that this
  greedy prefix is exactly the lexicographically smallest Sidon set. The open questions are (1)
  proving greedy = lex-min by an exchange argument, and (2) checking the *new* sums correctly and
  globally at each step rather than against only a local window of recent elements.

## Evaluation settings

Judged on hidden tests covering: the smallest sizes (`n = 1, 2, 3`), the sample (`n = 6`), sizes in
the regime where a naive formula or a local greedy first breaks (`n` around `6` to `30`), and the
largest size `n = 1000`, where the construction must be both valid over all `~500000` pairwise sums
and fast enough under the time limit. A schedule that is valid for tiny `n` but colliding (or not
lexicographically minimal) at larger `n` scores zero.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;

    vector<long long> x;
    x.reserve(n);

    // TODO: build the lexicographically smallest size-n Sidon set (all pairwise sums x[i]+x[j]
    // distinct), offsets in [0, 200000000], then print them in increasing order.

    for (int i = 0; i < n; i++) {
        cout << x[i];
        cout << (i + 1 < n ? ' ' : '\n');
    }
    return 0;
}
```
