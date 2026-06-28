# Serving every request from interval availabilities

## Research question

There are `T` discrete, indistinguishable time slots numbered `1..T`; each slot can serve **at most one**
request. There are `n` service requests. Request `i` is *available* only during a contiguous window of
slots `[l_i, r_i]` (with `1 <= l_i <= r_i <= T`) and, when served, occupies exactly one slot inside that
window.

The question is a pure feasibility one: **can every request be served simultaneously** — i.e. is there an
assignment of each request to a distinct slot, with each request landing inside its own availability
window? If yes, output `YES` together with one valid assignment; if no, output `NO`.

This is exactly a bipartite-matching feasibility test between requests and slots, restricted to the special
structure where every request's neighbourhood is a *contiguous interval* of slots. That interval structure
is what shows up in machine scheduling, channel/booking allocation, and "deadline + release-time" job
problems, so getting the one-shot feasibility decision exactly right — including the corners where the
answer hinges on a single overloaded window — is what matters.

## Input / output contract

- Input (stdin):
  - The first line contains two integers `n` and `T` (`0 <= n <= 2*10^5`, `1 <= T <= 2*10^5`).
  - Each of the next `n` lines contains two integers `l_i` and `r_i` (`1 <= l_i <= r_i <= T`), the
    availability window of request `i`, in input order.
- Output (stdout):
  - If every request can be served, print `YES` on the first line, then on the second line `n` integers:
    the slot assigned to request `i`, in input order, separated by single spaces. The slots must be
    pairwise distinct and each must satisfy `l_i <= slot_i <= r_i`. Any one valid assignment is accepted.
  - Otherwise print `NO`.
- Time limit: 1 second. Memory: 256 MB.

Example. For `n = 3`, `T = 3`, windows `[1,2], [1,1], [2,3]` the answer is `YES`; one valid assignment is
`2 1 3` (request 0 -> slot 2, request 1 -> slot 1, request 2 -> slot 3). If instead the windows are
`[1,1], [1,1], [2,3]`, the answer is `NO`: two requests can only use slot 1, so they collide.

## Background

The constraint "each slot serves at most one request, each request must fall inside its own window" is a
bipartite-matching feasibility question. Two families of approach are on the table before committing:

- **General bipartite matching.** Build the bipartite graph (request `i` adjacent to every slot in
  `[l_i, r_i]`) and run a maximum-matching algorithm; the answer is `YES` iff the matching saturates all
  `n` requests. This is the obviously-correct route, but the graph can have up to `n * T` edges, far too
  many to materialise at the stated scale.
- **Greedy assignment exploiting the interval structure.** Because each request's allowed slots form a
  contiguous interval, the requests can be ordered and served one at a time by a simple rule, without ever
  building the matching graph. The open questions are *which* order makes the greedy optimal, and how to
  find "the earliest still-free slot at or after `l_i`" fast enough.

## Evaluation settings

Judged on hidden tests covering: feasible and infeasible instances; instances where the answer turns on a
single overloaded window (Hall's condition tight by exactly one); `n = 0`; `n = 1`; `n > T` (always `NO`);
all windows equal; point windows (`l_i = r_i`); nested and overlapping windows; and large
`n = T = 2*10^5` for both the `YES` and `NO` cases. For `YES` instances the printed assignment is
re-validated (distinct slots, each inside its window), so an off-by-one in the assignment is caught even
when the verdict is right.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long T;
    if (!(cin >> n >> T)) return 0;

    vector<int> l(n), r(n);
    for (int i = 0; i < n; i++) cin >> l[i] >> r[i];

    // TODO: decide whether every request can be served (each in a distinct slot
    //       inside its window). If so, print YES and one valid assignment;
    //       otherwise print NO.

    return 0;
}
```
