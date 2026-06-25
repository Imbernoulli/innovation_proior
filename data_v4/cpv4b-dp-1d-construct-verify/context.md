# Scoring a drum loop: the earliest-hit pattern with no long silence

## Research question

A drum machine plays a loop of exactly `n` beats. Each beat is either a **hit** `H` (the drum
sounds) or a **rest** `R` (silence). The loop must obey two rules:

- **No dead air.** The track may never contain `K` rests in a row. Equivalently, every maximal run
  of consecutive `R`s has length at most `K - 1`.
- **Hit budget.** The loop uses exactly `h` hits (so exactly `n - h` rests).

Among all loops meeting both rules, the producer wants the one that *kicks in earliest*: the
**lexicographically smallest** pattern under the ordering `H < R` (a hit is "smaller" than a rest, so
an earlier hit makes the whole string smaller). Output that pattern, or report that no legal loop
exists.

This is a *construction* problem in the one-dimensional DP family: the object you emit is a length-`n`
string, and its legality is a global property of all `n` of its characters. The trap that makes it
interesting is that a short, pretty pattern (a fixed period, or "place every hit as early as you
can") can satisfy the rules for a handful of small `n` purely by accident, then violate them — or
miss feasibility — at the scale the judge actually tests.

## Input / output contract

- Input (stdin): three integers `n K h` separated by whitespace, with
  `1 <= n <= 10^7`, `1 <= K <= n`, `0 <= h <= n`.
- Output (stdout):
  - If at least one legal loop exists, print the lexicographically smallest one (under `H < R`) as a
    single line of `n` characters from `{H, R}`, terminated by a newline.
  - If no legal loop exists, print a single line containing `-1`.
- Time limit: 2 seconds. Memory: 256 MB.

Example: for `n = 7`, `K = 3`, `h = 2` the answer is `RHRRHRR`. We have 2 hits and 5 rests, and no
run of 3 rests. Putting a hit at beat 0 would leave 6 beats with only 1 hit and 5 rests; those 5
rests cannot all be split by a single hit into runs of length `<= 2` (one hit makes at most two gaps,
holding `2 + 2 = 4 < 5` rests), so beat 0 must be a rest. The earliest legal hit lands at beat 1, and
the rest of the loop is forced.

## Background

Two facts have to be settled before a single character is written:

- **When does a legal loop exist at all?** With `h` hits there are `h + 1` gaps (before the first
  hit, between consecutive hits, after the last hit) into which the `n - h` rests must fall, each gap
  holding at most `K - 1` rests. So legality is a counting condition on `n`, `K`, `h`. It is very easy
  to get the `+1` wrong — to count the gaps *between* hits and forget the leading and trailing ones —
  and a feasibility test that is off by one passes every small case where the difference does not
  matter and then misfires on large inputs.
- **How to build the earliest-hit loop when it exists.** Greedily preferring `H` at each beat is
  correct *only if* placing that hit still leaves the remainder completable; a hit spent too early can
  leave too few hits to break up the trailing rests. The decision at each beat is therefore a
  suffix-feasibility query, and that query multiplies counts that reach `~10^14` — so the arithmetic
  must be 64-bit. A version that does the multiply in 32-bit reproduces the right answer on every
  small test and then silently overflows on the large ones.

## Evaluation settings

Judged on hidden tests covering: the trivial `n = 1`; `K = 1` (no rest is ever allowed, so the only
legal loop is all hits and any `h < n` is infeasible); `h = 0` and `h = n`; cases right on the
feasibility boundary `n - h = (h + 1)(K - 1)`; cases one beyond it (answer `-1`); and large
`n` up to `10^7` with `K` and `h` chosen so the gap-capacity product `(h + 1)(K - 1)` exceeds the
32-bit range. For each feasible case the checker confirms the output has length `n`, uses exactly `h`
hits, contains no run of `K` rests, and is the lexicographic minimum; for each infeasible case it
confirms the output is exactly `-1`. Emitting up to `10^7` characters within the limit requires a
single buffered write.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    long long n, K, h;
    if (!(cin >> n >> K >> h)) return 0;

    // TODO: decide feasibility, then either print the lexicographically smallest (H < R)
    //       length-n pattern with exactly h hits and no run of K rests, or print -1.

    return 0;
}
```
