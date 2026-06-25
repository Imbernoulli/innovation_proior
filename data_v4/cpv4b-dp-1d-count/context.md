# Counting distinct highlight reels from a broadcast

## Research question

A small radio station logs every jingle it plays during a show as a sequence of `n` tone ids
`t[0..n-1]` (the same jingle can be played many times, so ids repeat). A **highlight reel** is any
**non-empty** selection of logged jingles played back **in their original broadcast order** — formally,
a non-empty subsequence of the log. Two reels are considered the *same broadcast* when they produce
the **identical sequence of tone ids**; the producers do not care which physical log positions were
picked, only what the listener hears.

Count the number of **distinct** highlight reels (distinct non-empty tone-id sequences obtainable as a
subsequence of the log), modulo `1000000007`.

This is a one-dimensional counting DP. The subtlety is entirely in the word *distinct*: when a tone id
repeats, the naive "each new jingle doubles the count" recurrence counts some reels twice, and the
correction term has to reference exactly the right earlier prefix or it over- or under-subtracts.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 2*10^5`); then `n` integers `t[i]`
  (`1 <= t[i] <= 10^9`), whitespace-separated. If the input is empty (no tokens) treat it as `n = 0`.
- Output (stdout): a single line with the number of distinct non-empty highlight reels, taken modulo
  `1000000007`.
- Time limit: 1 second. Memory: 256 MB.

Example: for the log `t = [1, 2, 1]` the answer is `6`. The distinct reels are `1`, `2`, `(1,2)`,
`(2,1)`, `(1,1)`, and `(1,2,1)`. The two physical `1`s collapse to a single reel `1`, which is the
whole point of counting by content rather than by position.

## Background

The object being counted is "distinct subsequences of a sequence". Two routes are worth weighing
before committing:

- **Enumerate and deduplicate.** Generate every one of the `2^n` index subsets, render each to its
  tone-id sequence, drop the empty one, and put the rest in a hash set. Obviously correct and trivial
  to write, but `2^n` is hopeless past `n ~ 25`. It is only useful as an oracle on tiny inputs.
- **Linear counting DP.** Sweep left to right maintaining the number of distinct subsequences of the
  prefix seen so far. Each new tone *doubles* the count (every existing subsequence either ignores or
  appends the new tone), but when that tone has appeared before, the subsequences that already ended
  in an earlier copy of it get re-created identically, so a correction must be subtracted. The open
  questions are: what exactly is the correction term, which earlier prefix does it reference, how does
  the modulus interact with a subtraction that can go negative, and where does the empty subsequence
  enter and leave the bookkeeping.

## Evaluation settings

Judged on hidden tests covering: all-distinct logs (answer `2^n - 1 mod p`), all-identical logs
(answer `n`, since the only reels are the runs `t`, `tt`, `ttt`, ...), small alphabets with heavy
repetition (where double-counting is most likely), the empty log (`n = 0`, answer `0`), a single
jingle (`n = 1`, answer `1`), and large logs with `n = 2*10^5` and ids up to `10^9` (so ids do not fit
in a small fixed table and the running count must be reduced modulo `1000000007` throughout).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    const long long MOD = 1000000007LL;

    int n;
    if (!(cin >> n)) return 0;            // empty input -> n = 0
    vector<long long> t(n);
    for (auto &x : t) cin >> x;

    // TODO: count distinct non-empty subsequences of t (by content), modulo MOD.
    long long ans = 0;

    cout << ans << "\n";
    return 0;
}
```
