# Minimum-cost relay chain with a multiplicative leg cost

## Research question

A signal must travel from an origin station `0` to a terminal station `n` along a line of stations
`0, 1, 2, ..., n`. The signal is forwarded in *legs*: a leg starts immediately after some station `j`
and ends by being received at a later station `i` (`j < i`). Receiving at station `i` finalizes a
cost `dp[i]`, and the cost of the whole chain ending at `i` decomposes as

```
dp[0] = 0
dp[i] = c[i] + min over 0 <= j < i of ( dp[j] + b[j] * a[i] )      for i = 1..n
```

Here `b[j]` is a per-station *transmit factor* (fixed by the station the leg starts after), `a[i]` is
a per-station *receive multiplier* (fixed by the station that receives the leg), and `c[i]` is an
additive receive cost. Any earlier station `j` may forward to `i`, so each `dp[i]` minimizes over all
predecessors. Report `dp[n]`, the minimum finalized cost at the terminal station.

This is the canonical setting for a transition of the form `dp[i] = min_j (dp[j] + b[j]·a[i])`: it
shows up inside cost-of-cutting, cost-of-building, and acquisition DPs, where the per-transition cost
is a product of a "left" quantity and a "right" quantity. Getting the one-dimensional version exactly
right — including negative factors and the interleaving of updates with queries — is what matters.

## Input / output contract

- Input (stdin), whitespace-separated, in this exact order:
  - `n`  (`0 <= n <= 2*10^5`).
  - `n` integers `a[1], a[2], ..., a[n]`  (`-10^6 <= a[i] <= 10^6`), the receive multipliers.
  - `n` integers `b[0], b[1], ..., b[n-1]`  (`-10^6 <= b[j] <= 10^6`), the transmit factors of the
    `n` possible *start* stations `0..n-1`.
  - `n` integers `c[1], c[2], ..., c[n]`  (`-10^6 <= c[i] <= 10^6`), the additive receive costs.
  - When `n = 0` there are no further numbers; the answer is `dp[0] = 0`.
- Output (stdout): a single line with `dp[n]`.
- Time limit: 1 second. Memory: 256 MB.

The maximum magnitude of `dp[n]` is about `2*10^5 * 10^6 * 10^6 = 2*10^17`, which exceeds 32-bit
range, so a 64-bit integer type is mandatory.

Example. For

```
4
3 1 4 2
5 -2 1 3
0 1 0 2
```

the answer is `10`. Working it out: `dp[1] = 0 + (0 + 5*3) = 15`; `dp[2] = 1 + min(0+5*1, 15-2*1) =
1 + 5 = 6`; `dp[3] = 0 + min(0+5*4, 15-2*4, 6+1*4) = 7`; `dp[4] = 2 + min(0+5*2, 15-2*2, 6+1*2,
7+3*2) = 2 + 8 = 10`. Note the chosen predecessor changes from one `i` to the next (`dp[3]` uses `j=1`,
`dp[4]` uses `j=2`): no single predecessor is best for every receive point.

## Background

The transition `dp[i] = c[i] + min_{j<i} (dp[j] + b[j]·a[i])` is a constrained selection over earlier
states. Two routes are on the table before committing.

- **Quadratic DP.** For every `i`, scan all `j < i` and take the minimum. This is `O(n^2)`, trivial to
  write, and obviously correct; at `n = 2*10^5` it performs `~2*10^10` operations and cannot finish in
  a second. It is the right *oracle* but the wrong *submission*.
- **Lower-envelope evaluation.** Each predecessor `j` defines a line `y = b[j]·X + dp[j]` in a variable
  `X`. Evaluating `dp[j] + b[j]·a[i]` is evaluating that line at `X = a[i]`. So
  `min_{j<i}(dp[j] + b[j]·a[i])` is the minimum of a set of lines at one query point — the *lower
  envelope* of the lines at `X = a[i]`. The open question is which envelope structure to use, given
  that lines are *added* (one per `j`) interleaved with *queries* (one per `i`), and the slopes `b[j]`
  and query points `a[i]` are not guaranteed monotone (values may be negative).

## Evaluation settings

Judged on hidden tests covering: all-positive arrays; arrays with negative `a`, `b`, and `c`; heavy
ties in `a[]` (so the distinct query coordinates collapse); `n = 0` and `n = 1`; configurations where
the optimal predecessor is far from `i-1`; and large `n = 2*10^5` with values near `10^6`, where the
running cost reaches `~2*10^17` (so a 32-bit accumulator is a silent wrong answer) and an `O(n^2)`
solution times out.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    if (n == 0) { cout << 0 << "\n"; return 0; }

    vector<long long> a(n + 1), b(n), c(n + 1);
    for (int i = 1; i <= n; i++) cin >> a[i];   // a[1..n]
    for (int j = 0; j < n; j++)  cin >> b[j];   // b[0..n-1]
    for (int i = 1; i <= n; i++) cin >> c[i];   // c[1..n]

    // TODO: compute dp[i] = c[i] + min_{0<=j<i}( dp[j] + b[j]*a[i] ), dp[0]=0; output dp[n].
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
