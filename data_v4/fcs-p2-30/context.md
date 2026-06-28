# Counting decodings of a digit string (1->A .. 26->Z), modulo a prime

## Research question

A message of letters `A..Z` was encoded by replacing each letter with its position in the
alphabet: `A -> 1`, `B -> 2`, ..., `Z -> 26`. The separators between the codes were then dropped,
leaving a single run of digits `s`. Given `s`, count how many distinct original letter strings could
have produced it — that is, the number of ways to split `s` into consecutive groups where each group
is the decimal value of a code in `1..26`. Because the count can be astronomically large, report it
**modulo a prime `p`**.

A group of one digit is valid when it is `1..9`; a group of two digits is valid when its value lies
in `10..26`. A digit `0` can never stand alone (there is no code `0`) and can only appear as the
second digit of `10` or `20`. If `s` admits no valid split at all (for example it starts with `0`,
or contains a `0` not preceded by `1` or `2`), the answer is `0`.

This is the discrete "tiling / segmentation count" pattern: cut a sequence into legal pieces and
count the cuttings. The same shape appears in tokenization, run-length parsing, and restricted
composition counting, so getting the one- vs two-digit grouping and the zero corners exactly right is
the point.

## Input / output contract

- Input (stdin): two whitespace-separated tokens.
  - Token 1: a prime `p` with `2 <= p <= 2^31 - 1`.
  - Token 2: the digit string `s`, consisting only of characters `'0'..'9'`, with
    `1 <= |s| <= 10^5`.
- Output (stdout): a single line containing the number of valid decodings of `s`, taken modulo `p`.
- Time limit: 2 seconds. Memory: 256 MB.

Example: for `p = 1000000007` and `s = "226"` the answer is `3`. The three decodings are
`2 2 6 -> "BBF"`, `2 26 -> "BZ"`, and `22 6 -> "VF"`.

## Background

The constraint is local in an unusual way: whether a two-digit group is legal depends on a sliding
window of width two, and a single `0` can kill an entire prefix. Two routes are on the table before
committing to one.

- **Closed form via the Fibonacci/matrix angle.** When every adjacent pair of digits happens to form
  a valid two-digit code and no digit is `0`, each position independently offers "cut here or extend"
  and the count of splittings of a length-`m` block is the Fibonacci number `F(m+1)`. That invites a
  shortcut: detect maximal "fully flexible" blocks, look up or matrix-exponentiate a Fibonacci value
  per block modulo `p`, and multiply the blocks together. It is `O(log n)` per block and feels clever.
  The open question is whether real strings actually decompose into such clean independent blocks, or
  whether the `10..26` ceiling and the `0` rules couple positions in ways this factorization misses.

- **Linear segmentation DP.** Define `dp[i]` as the number of valid decodings of the prefix of length
  `i`, and extend by either a one-digit or a two-digit final group. This is `O(n)` with `O(1)` memory.
  The open question is only the exact recurrence and the base case, especially how zeros and the
  prefix boundary are handled.

## Evaluation settings

Judged on hidden tests covering: strings with no valid decoding (`0`, `06`, `100`, a stray `0`);
the two-digit boundary exactly at `26` and just past it (`27`, `30`); runs of `1`s and `2`s where the
count grows like Fibonacci; strings dense in `0` so the `10`/`20`-only rule dominates; small moduli
(`p = 2, 3, 5`) where many counts collapse to the same residue; and the maximum length `|s| = 10^5`
with `p` near `2^31` so the running count must be reduced every step.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    long long p;
    string s;
    if (!(cin >> p >> s)) return 0;
    int n = (int)s.size();

    // TODO: count the number of ways to split s into groups whose values lie in 1..26
    //       (one digit => 1..9, two digits => 10..26), reduced modulo p.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
