# Counting fixed-balance windows in a warehouse ledger

## Research question

A warehouse keeps a single signed ledger of stock movements for one SKU over a shift: position `i`
records `a[i]`, the net change in units at minute `i` (a positive number is a delivery, a negative
number a withdrawal). An auditor wants to know how many **contiguous time windows** `[l, r]` had a
net movement of exactly `S` units, where `S` is a fixed target the auditor is reconciling against.
Formally, count the pairs `(l, r)` with `0 <= l <= r <= n-1` such that `a[l] + a[l+1] + ... + a[r] = S`.
Output that count.

This is the classic "subarrays with a given sum" problem stated over a prefix sum. It looks small,
but the constraints are chosen so that two different quantities overflow a 32-bit integer: the running
prefix total, and the answer itself. Getting the data types right is the whole game.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `S` (`0 <= n <= 2*10^5`,
  `-2*10^14 <= S <= 2*10^14`). The second line has `n` integers `a[i]`
  (`-10^9 <= a[i] <= 10^9`), whitespace-separated. When `n = 0` the second line is empty or absent.
- Output (stdout): a single line with the number of contiguous windows whose sum equals `S`.
- Time limit: 1 second. Memory: 256 MB.

Example: for `S = 2` and `a = [3, -1, 1, 2, -2, 2]` the answer is `6`.

## Background

The standard tool is the prefix sum `P[0] = 0`, `P[k] = a[0] + ... + a[k-1]`. A window `[l, r]` has
sum `P[r+1] - P[l]`, so the windows summing to `S` are exactly the pairs of prefix indices `(i, j)`
with `i < j` and `P[j] - P[i] = S`. Sweeping `j` from left to right and asking "how many earlier
prefixes equal `P[j] - S`?" turns the count into a single hash-map pass. Two families of approach are
on the table before committing:

- **Brute force over all windows.** For each `l`, extend `r` and accumulate the running sum, testing
  against `S`. This is `O(n^2)` and obviously correct, but at `n = 2*10^5` it is far too slow.
- **Prefix-sum + hash map.** One left-to-right sweep, `O(n)` expected with an `unordered_map` keyed by
  prefix value. The open questions are the exact "have I counted the empty prefix?" bookkeeping and,
  critically, the integer widths: with `n` up to `2*10^5` the answer can reach about `2*10^10`, and
  the prefix total can reach about `2*10^14`.

## Evaluation settings

Judged on hidden tests covering: small mixed-sign arrays, `S = 0`, all-positive arrays (so `P` is
monotone and the magnitude of the prefix grows), targets `S` that no window achieves (answer `0`),
`n = 0`, a single element, and large adversarial cases — in particular `n = 2*10^5` with all-equal
values so that an enormous number of windows match and the count blows past the 32-bit range, and
all-`10^9` values with `S = 10^9` so the prefix total reaches `2*10^14`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long S;
    if (!(cin >> n >> S)) return 0;

    // TODO: count contiguous windows with sum exactly S using a prefix-sum + hash-map sweep.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
