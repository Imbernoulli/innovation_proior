# Counting fixed-balance windows in a warehouse ledger

## Research question

A warehouse keeps a single signed ledger of stock movements for one SKU over a shift: position `i`
records `a[i]`, the net change in units at minute `i` (a positive number is a delivery, a negative
number a withdrawal). An auditor wants to know how many **contiguous time windows** `[l, r]` had a
net movement of exactly `S` units, where `S` is a fixed target the auditor is reconciling against.
Formally, count the pairs `(l, r)` with `0 <= l <= r <= n-1` such that `a[l] + a[l+1] + ... + a[r] = S`.
Output that count.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `S` (`0 <= n <= 2*10^5`,
  `-2*10^14 <= S <= 2*10^14`). The second line has `n` integers `a[i]`
  (`-10^9 <= a[i] <= 10^9`), whitespace-separated. When `n = 0` the second line is empty or absent.
- Output (stdout): a single line with the number of contiguous windows whose sum equals `S`.
- Time limit: 1 second. Memory: 256 MB.

Example: for `S = 2` and `a = [3, -1, 1, 2, -2, 2]` the answer is `6`.

## Evaluation settings

Judged on hidden tests covering: small mixed-sign arrays, `S = 0`, all-positive arrays, targets `S`
that no window achieves (answer `0`), `n = 0`, a single element, and large adversarial cases at
`n = 2*10^5`, including arrays of all-equal values and arrays of large-magnitude values.

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

    // TODO: count contiguous windows [l, r] with a[l] + ... + a[r] == S.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
