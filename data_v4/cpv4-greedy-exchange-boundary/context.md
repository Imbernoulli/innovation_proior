# Fewest snapshots to capture every pulse (half-open coverage window)

## Research question

A radio receiver logs `n` pulses on a one-dimensional timeline; pulse `i` arrives at integer time
`t[i]`. You capture the log by triggering *snapshots*. A snapshot triggered at an integer time `s`
records every pulse whose arrival time lies in the **half-open** window `[s, s + L)` — that is, every
pulse with `s <= t[i] < s + L`. The width `L` is fixed for the whole problem. Crucially, a pulse that
arrives at *exactly* `s + L` is **not** recorded by that snapshot: the right edge is open.

You may trigger snapshots at any integer times you like, as many as you like. Output the **minimum
number of snapshots** needed so that every one of the `n` pulses is recorded by at least one snapshot.

This is the interval point-cover problem in disguise — "stab all the pulses with the fewest fixed-width
windows" — but with a deliberately half-open window. Whether a pulse sitting on a window's right edge
counts as covered changes the answer, so the entire difficulty is getting the inclusive/exclusive
boundary exactly right.

## Input / output contract

- Input (stdin):
  - the first line has two integers `n` (`0 <= n <= 2*10^5`) and `L` (`1 <= L <= 10^9`);
  - then `n` integers `t[0..n-1]` (`0 <= t[i] <= 10^9`), the pulse arrival times, whitespace-separated
    (line breaks are not significant). The times are **not** guaranteed to be sorted or distinct.
- Output (stdout): a single line with the minimum number of snapshots.
- Time limit: 1 second. Memory: 256 MB.

Example: for `L = 3` and `t = [0, 3, 5, 6, 9]` the answer is `4`.

## Background

Each snapshot covers a contiguous block of the sorted timeline, so this is a covering problem on a
line. Two families of approach are on the table before committing to one:

- **Greedy exchange (leftmost-uncovered).** Sort the pulses. Repeatedly take the earliest pulse not
  yet recorded, place a snapshot whose window *starts* at that pulse's time, and mark every pulse the
  window reaches; repeat.
- **Set-cover / breadth-first search.** Treat each candidate window as a set of pulses it records and
  search for the fewest sets that cover everything. Exhaustive over integer window starts, this is
  exponential and only usable as an independent oracle on tiny inputs.

## Evaluation settings

Judged on hidden tests covering: `n = 0`; `n = 1`; many identical times; pulses spaced exactly `L`
apart; pulses spaced `L - 1` apart; `L = 1` with distinct integer times; and large instances with
`n = 2*10^5`, `t[i]` near `10^9`, and `L` near `10^9`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long L;
    if (!(cin >> n >> L)) return 0;
    vector<long long> t(n);
    for (auto &x : t) cin >> x;

    sort(t.begin(), t.end());

    // TODO: compute the minimum number of half-open windows [s, s+L) needed
    //       so that every pulse in t is covered by at least one window.
    long long snapshots = 0;

    cout << snapshots << "\n";
    return 0;
}
```
