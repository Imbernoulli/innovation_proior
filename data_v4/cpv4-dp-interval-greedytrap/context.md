# Stringing beads onto glow-lines with minimum wasted slack

## Research question

A jeweller threads `n` beads, in a fixed left-to-right order, onto a display board made of equal-width
"glow-lines". Bead `i` has width `w[i] >= 1`. Each glow-line has the same usable width `W`. The beads
are cut into a sequence of **contiguous groups**: the first group is the prefix of beads on glow-line 1,
the next group the beads on glow-line 2, and so on; every bead lands on exactly one line and the order is
never reordered. Two beads that share a line need one unit of clearance between them, so a line that holds
beads `l..r` consumes

```
used(l, r) = (w[l] + w[l+1] + ... + w[r]) + (r - l)      # (r - l) = number of internal gaps
```

units of width, and this must satisfy `used(l, r) <= W` — a line may never overflow.

A line that is **not** the last line and leaves `slack = W - used` width unfilled is charged a penalty of
`slack^2` (squared, so the cost punishes one badly under-filled line far more than several slightly
under-filled ones). The **last** line is the bottom of the board and is charged **no** slack penalty
(trailing emptiness is free). Choose how to cut the bead sequence into lines so that the **total penalty**
— the sum of `slack^2` over all non-last lines — is **minimized**. Output that minimum.

Because every bead satisfies `w[i] <= W`, putting each bead on its own line is always a legal layout, so a
feasible answer always exists. If all beads fit on a single line the cost is `0`.

This is the line-breaking / text-justification cost in disguise: it is the cleanest setting in which the
"obvious" greedy (pack each line as full as you can, then start a new one) is provably suboptimal, while a
left-to-right interval/partition DP over contiguous bead groups is exact.

## Input / output contract

- Input (stdin): the first line holds two integers `n` and `W` (`1 <= n <= 5000`, `1 <= W <= 10^6`).
  The second line holds `n` integers `w[0..n-1]` (`1 <= w[i] <= W`), whitespace-separated.
- Output (stdout): a single line with the minimum total penalty (the sum of `slack^2` over every line
  except the last).
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 4`, `W = 6`, `w = [4, 1, 3, 3]` the answer is `5`. (Greedy fills line 1 with beads
`{0,1}` exactly to width `6`, then is forced into `{2}` then `{3}`, paying `(6-3)^2 = 9`. The optimum
instead puts `{0}` on line 1 paying `(6-4)^2 = 4`, `{1,2}` on line 2 paying `(6-5)^2 = 1`, and `{3}` on
the free last line, for `4 + 1 = 5`.)

## Background

The constraint that lines hold **contiguous** prefixes of the remaining beads, never overflow `W`, and pay
a **convex** (squared) slack penalty puts two very different strategies on the table:

- **Greedy first-fit.** Walk the beads once; keep adding the next bead to the current line while it still
  fits, and open a new line the instant it would overflow. This is `O(n)` and a dozen lines of code. The
  open question is whether maximally filling each line — which is locally the lowest-slack move — actually
  minimizes the *sum of squares* of slack across the whole board.
- **Interval / partition dynamic programming.** Let `dp[i]` be the minimum penalty to lay out the first
  `i` beads, where the layout's last line is the group `[j..i-1]`. Then `dp[i]` is the best over all legal
  last-group start points `j`, combining `dp[j-1]` with the slack cost of the group `[j..i-1]`. This is
  `O(n^2)` in the worst case (each line can scan back over the beads that still fit); the open question is
  the exact recurrence, the width/feasibility test, and the special handling of the last line.

## Evaluation settings

Judged on hidden tests covering: everything fitting on one line (`answer 0`), a single bead (`n = 1`,
`answer 0`), beads so wide each must occupy its own line, layouts where greedy first-fit is strictly worse
than the optimum (the core trap), `W` near `10^6` with many small beads so the running penalty exceeds a
32-bit integer, and the largest `n = 5000` so an `O(n^2)` partition scan and 64-bit accumulators are both
exercised.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    long long n, W;
    if (!(cin >> n >> W)) return 0;
    vector<long long> w(n);
    for (auto &x : w) cin >> x;

    // prefix sums of widths, so used(l, r) is O(1).
    vector<long long> pre(n + 1, 0);
    for (int i = 0; i < n; i++) pre[i + 1] = pre[i] + w[i];

    // TODO: partition the bead sequence into contiguous lines (each <= W),
    //       minimizing the sum of slack^2 over every line except the last.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
