# Optimal-play stone game on a row (take from either end)

## Research question

A row of `n` stones carries integer values `a[0..n-1]` (values may be negative). Two players
alternate turns; the **first player moves first**. On each turn the player to move removes **one
stone from either the left end or the right end** of the remaining row and adds its value to their
own running total. Play continues until the row is empty (so the first player ends up with
`ceil(n/2)` stones and the second with `floor(n/2)`). **Both players play optimally to maximize
their own final total** (each player is purely self-interested, not trying to minimize the other).

Output the **first player's** final total under optimal play by both sides.

This is the canonical "stones / coins in a line" turn-based game. Its subtlety is that a player's
choice is constrained to the two ends, both players are optimizing simultaneously, and values can be
negative — so the row cannot simply be "left alone." Getting the recurrence exactly right (including
who profits from the leftover interval after a move) is the whole task.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 2000`); then `n` integers `a[i]`
  (`-10^9 <= a[i] <= 10^9`), whitespace-separated (newlines or spaces, format-agnostic).
- Output (stdout): a single line with one integer — the first player's optimal total.
- If `n = 0`, the row is empty and the first player's total is `0`.
- Time limit: 1 second. Memory: 256 MB.

Example: for `a = [1, 5, 233, 7]` the answer is `234`. The first player takes the left end `1`,
which makes taking the `233` unavoidable two moves later: optimal play runs `P1:1, P2:5, P1:233,
P2:7`, so the first player ends with `1 + 233 = 234`. Note that the *greedy* "take the larger end"
move (grabbing the `7`) does strictly worse here. See the samples below.

## Samples

```
Input:
4
1 5 233 7
Output:
234
```

```
Input:
4
1 1 3 2
Output:
4
```

```
Input:
1
-7
Output:
-7
```

The single-stone case forces the first player to take the only stone, even if it is negative — the
game runs until the row is empty, so passing is not allowed.

## Background

Two families of approach are on the table before committing to one:

- **Greedy "take the larger end."** On each turn the current player removes whichever end has the
  larger value. It is `O(n)` and a single loop. The open question is whether grabbing the locally
  larger end is actually optimal when the opponent then chooses optimally on what remains.
- **Interval dynamic programming.** For every contiguous subrow `a[i..j]`, compute the optimal
  outcome of the mover who faces it; combine subintervals to build up to the full row. This is
  `O(n^2)` time and `O(n^2)` memory. The open question is the exact recurrence — specifically how a
  move on `a[i..j]` hands a *fresh game* on the leftover interval to the opponent, and how that
  opponent's optimal take feeds back into the mover's total.

## Evaluation settings

Judged on hidden tests covering: all-positive rows, rows with negatives and zeros, the empty row
(`n = 0`), a single stone (`n = 1`, possibly negative), alternating large/small values designed to
mislead "take the larger end," and large `n = 2000` with values near `±10^9` (so totals can exceed
the 32-bit range — totals can reach `~2*10^12` in magnitude).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // TODO: compute the first player's optimal total in the take-from-either-end
    //       game, assuming both players maximize their own totals.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
