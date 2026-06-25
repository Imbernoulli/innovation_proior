# Counting clean segmentations of a beacon strip

## Research question

A survey drone flies over a straight line of `n` beacons. Beacon `i` continuously broadcasts a fixed
30-bit identity code `a[i]` (`0 <= a[i] < 2^30`). To localize itself the drone slices the strip into
one or more **contiguous segments**; a segment is just a non-empty run of consecutive beacons. For a
segment it computes the **signature** = the bitwise XOR of all codes inside that segment:

```
sig(l, r) = a[l] XOR a[l+1] XOR ... XOR a[r].
```

A whole segmentation is **clean** when *every* segment's signature has an **even number of set bits**
(an even popcount). The drone wants to know how many clean segmentations of the full strip exist.

Report that count **modulo `1000000007`**. Two segmentations are different when their sets of cut
positions differ. With `n` beacons there are `n - 1` possible cut positions and `2^(n-1)` segmentations
in total, so brute enumeration is hopeless for large `n`; the task is to count the clean ones fast.

The predicate that drives everything is "even popcount of the segment XOR". It is a parity-of-bits
condition on an XOR of a range — not "the XOR equals zero", and not "the XOR is an even number". The
whole problem turns on translating that predicate into something a single left-to-right scan can
accumulate, without re-deriving the XOR of every segment from scratch.

## Input / output contract

- Input (stdin): the first token is `n` (`1 <= n <= 2*10^5`); then `n` integers `a[i]`
  (`0 <= a[i] < 2^30`), whitespace-separated. (An empty input stream, i.e. no tokens, is treated as
  "nothing to do" and prints nothing; `n = 0` denotes the empty strip and its answer is `1`, the empty
  segmentation.)
- Output (stdout): a single line with the number of clean segmentations modulo `1000000007`.
- Time limit: 1 second. Memory: 256 MB.

Example: for `a = [3, 1, 2]` the answer is `2`. The two clean segmentations are the single segment
`[3,1,2]` (signature `3 XOR 1 XOR 2 = 0`, popcount `0`, even) and the split `[3] | [1,2]` (signatures
`3` and `1 XOR 2 = 3`, each popcount `2`, even). The other two segmentations — `[3,1] | [2]` and
`[3] | [1] | [2]` — each contain a segment of odd popcount, so they are not clean.

## Background

Let `P[0..n]` be the prefix XOR with `P[0] = 0` and `P[i] = a[0] XOR ... XOR a[i-1]`. Then the
signature of the segment covering beacons `j .. i-1` (a last segment ending right before position `i`)
is `P[i] XOR P[j]`. Counting clean segmentations is naturally a one-dimensional dynamic program:

```
dp[i] = number of clean segmentations of the first i beacons,
dp[0] = 1,
dp[i] = sum over j in [0, i-1] of dp[j]   for which   popcount(P[i] XOR P[j]) is even.
```

The literal recurrence is `O(n^2)` (every `i` scans all earlier `j`). To reach `O(n)` you must replace
the per-`j` popcount test by a property of `j` alone, so the eligible `dp[j]` can be kept in a running
bucket. That hinges on a bit identity relating `popcount(P[i] XOR P[j])` to `popcount(P[i])` and
`popcount(P[j])` — an identity that is very tempting to state from memory and easy to state *wrongly*.

## Evaluation settings

Judged on hidden tests covering: tiny strips (`n = 1`, including a lone odd-popcount code whose answer
is `0`); the empty strip; codes restricted to `{0, 1}`; small-alphabet codes with many popcount
collisions; full 30-bit random codes; adversarial inputs where "even popcount" and "even value"
disagree on most beacons; and large `n = 2*10^5` so an `O(n^2)` scan times out and a 64-bit running sum
must be reduced modulo `1000000007` to avoid overflow.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;            // empty input -> nothing to print

    const long long MOD = 1000000007LL;

    // TODO: count, modulo MOD, the segmentations of a[0..n-1] in which every contiguous
    // segment has an XOR signature of even popcount. Use prefix XOR and a one-pass DP.
    long long answer = 1;

    cout << answer % MOD << "\n";
    return 0;
}
```
