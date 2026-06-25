# Firmware build farm: maximize how many modules ship on time

## Research question

A continuous-integration server compiles firmware **modules** one at a time on a single build core
(no parallelism, no preemption — once a compile starts it runs to completion). There are `n`
candidate modules. Module `i` takes `t[i]` seconds of core time to compile and has a hard release
**deadline** `d[i]` seconds from the start of the build window: if module `i` has not *finished*
compiling by time `d[i]`, it misses the release train and is dropped from this firmware image.

You get to choose **which** modules to compile and **in what order**. The clock starts at `0`; the
core is never idle while there is a chosen module left to run (idling can only hurt). A chosen module
"ships" if its completion time is `<= d[i]`. You want to choose a subset and an order so that the
**number of modules that ship on time is as large as possible**, and output that maximum count.

This is single-machine scheduling to maximize the number of on-time jobs. It looks like a textbook
greedy, but the *exact* objective here — maximize the **count** of on-time jobs, with **arbitrary
processing times** — is what decides which greedy is correct. A natural earliest-deadline-first
sweep that keeps a module whenever it still fits is *not* optimal, and the reason is structural, not
a coding slip.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 2*10^5`). Then follow `n` lines, each with two
  integers `t[i]` and `d[i]` (`1 <= t[i] <= 10^9`, `1 <= d[i] <= 10^9`), whitespace-separated:
  the compile time and the deadline of module `i`.
- Output (stdout): a single line with the maximum number of modules that can ship on time.
- Time limit: 1 second. Memory: 256 MB.

Example: for the four-line body `(t,d) = (3,3), (2,4), (2,4)` (`n = 3`) the answer is `2`. The
module with `(t,d) = (3,3)` is the only one that fits before time `3`, but keeping it forces a clock
of `3`, after which neither `(2,4)` fits; dropping it lets the two `(2,4)` modules both ship
(finishing at `2` and `4`), so the on-time count is `2`, not `1`.

## Background

Two facts frame the problem.

- **Feasibility of a chosen subset is decided by earliest-deadline-first (EDD).** If some ordering
  of a chosen subset lets every module ship, then sorting that subset by deadline and running it in
  that order also lets every module ship. So "is this subset schedulable?" reduces to "does the EDD
  order of the subset keep every cumulative finish time under its deadline?" The *selection*, not the
  ordering, is the hard part.
- **The objective is a count, and processing times differ.** This is the configuration where the
  obvious EDD sweep — walk modules in deadline order, keep each one if it still fits, otherwise skip
  it forever — is *not* optimal: a short-deadline but long module accepted early can block two later
  modules that together would have shipped. The fix is an exchange: when accepting a module would
  overflow the current deadline, you may *evict a previously accepted module* (the longest one) to
  make room, which never decreases the count and can shorten the clock. Whether that exchange step is
  present is the whole problem.

## Evaluation settings

Judged on hidden tests covering: instances where the plain EDD-keep-if-fits sweep undercounts (so the
exchange is forced), all modules trivially fitting, no module fitting (answer `0`), `n = 0`, `n = 1`,
ties in deadlines and in processing times, and large `n = 2*10^5` with `t[i], d[i]` near `10^9` (so
the running clock reaches `~2*10^14` and overflows 32-bit accumulators).

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
    vector<pair<long long,long long>> job(n);   // (deadline, processing time)
    for (int i = 0; i < n; i++) {
        long long t, d;
        cin >> t >> d;
        job[i] = {d, t};
    }

    // TODO: maximize the number of modules whose cumulative finish time stays
    //       within their deadline, choosing the subset and order.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
