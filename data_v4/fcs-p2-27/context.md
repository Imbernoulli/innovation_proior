# Counting subsets with a given sum

## Research question

You are given `n` non-negative integers `a[0..n-1]` and a target `T`. Count the number of
**subsets of the positions** `{0, 1, ..., n-1}` whose values sum to exactly `T`. Two subsets are
different if they differ in any chosen position, even when the chosen values are equal; the empty
subset is allowed and sums to `0`. Because the count can be astronomically large (up to `2^n`),
output it **modulo `1000000007`**.

This is the exact-target counting version of subset sum. It is the building block behind partition
counting, coin/combination counting, and many "how many ways" combinatorial DP problems, so getting
the corners exactly right — repeated values, zeros, `T = 0`, and unreachable `T` — matters.

## Input / output contract

- Input (stdin): the first line holds two integers `n` and `T`
  (`0 <= n <= 200`, `0 <= T <= 100000`). The second line holds `n` integers `a[i]`
  (`0 <= a[i] <= 1000`), whitespace-separated. When `n = 0` the value line is empty or absent.
- Output (stdout): a single line with the number of subsets summing to exactly `T`, taken modulo
  `1000000007`.
- Time limit: 2 seconds. Memory: 256 MB.

Example: for `n = 3`, `T = 3`, `a = [1, 2, 3]`, the answer is `2` (the subsets `{3}` and `{1, 2}`).

## Background

"Count subsets with a given sum" looks like it should reduce to something cheaper than touching every
sum value. The most tempting shortcut is a greedy / sorting heuristic: sort the values and try to
assemble `T` by selecting items in some order, counting "ways" as you go. Greedy is `O(n log n)` and
tempting, but counting is not an optimization — there is no single best object to be greedy toward,
and the number of ways is a global quantity over all combinations.

## Evaluation settings

Judged on hidden tests covering: distinct positive values; many repeated values (so distinct
positions yield distinct subsets); arrays containing zeros; `T = 0`; unreachable `T` larger than the
total; `n = 0`; and large `n = 200`, `T = 100000`, where counts can far exceed the modulus.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long T;
    if (!(cin >> n >> T)) return 0;

    const long long MOD = 1000000007LL;

    // TODO: count subsets of a[0..n-1] summing to exactly T, modulo MOD.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
