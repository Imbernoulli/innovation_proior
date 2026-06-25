# Minimum gondola trips to evacuate a ridge

## Research question

A storm is closing in on an alpine ridge and `n` climbers must be brought down by a single cable
gondola. Climber `i` weighs `w[i]` kilograms. The gondola cabin is small: each trip it can carry
**at most two** climbers, and the **combined weight** of whoever rides must not exceed the cabin's
rated capacity `C` kilograms. Every climber's individual weight satisfies `w[i] <= C`, so anyone can
always ride alone; the only question is how often two can share a cabin.

A round trip (down and back up) is slow, so the rescue team wants the **minimum number of trips**
needed to get everyone down. Output that minimum.

This is a greedy-exchange problem: the right move is to sort the climbers and pair the lightest
still-waiting climber with the heaviest still-waiting one whenever they fit together. The trap is the
temptation to *skip the simulation* and read the answer off a closed-form lower bound — "it is just
the total weight divided by capacity," or "it is just half the head-count," or even the maximum of
those two bounds. Each of those is a genuine lower bound, and the combined one looks tight, but it is
**not** the answer, and asserting it without checking is a confidently-wrong shortcut.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `C`
  (`0 <= n <= 2*10^5`, `1 <= C <= 10^9`). The second line (present iff `n > 0`) has `n` integers
  `w[i]` (`1 <= w[i] <= C`), whitespace-separated.
- Output (stdout): a single line with the minimum number of gondola trips. When `n = 0` the answer
  is `0`.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 6`, `C = 5`, `w = [1, 1, 2, 5, 5, 5]` the answer is `5`. Each of the three
`5`-kg climbers must ride alone (any partner would push the cabin over `5` kg), the `1` and the `2`
can share one trip, and the remaining `1` rides alone — `3 + 1 + 1 = 5` trips. Note that the total
weight is `19`, so `ceil(19 / 5) = 4`, and the head-count bound is `ceil(6 / 2) = 3`; the true answer
`5` exceeds both, which is the whole point.

## Background

The constraint couples two scarce resources at once: each trip has a **slot** budget (at most two
climbers) and a **weight** budget (sum at most `C`). Two framings are on the table before committing:

- **Greedy-exchange, two-pointer.** Sort the weights. Repeatedly take the heaviest waiting climber;
  if the lightest waiting climber also fits alongside (their two weights sum to `<= C`), send the
  pair, otherwise send the heavy one alone. Each step removes the heaviest, so it terminates in
  `O(n)` after an `O(n log n)` sort. The open questions are *why* always pairing the current heaviest
  with the current lightest is optimal (an exchange argument), and getting the pointer bookkeeping
  exactly right when the two pointers meet on one remaining climber.
- **A closed-form bound.** Compute a lower bound such as `ceil(sum(w) / C)` (weight bound) or
  `ceil(n / 2)` (slot bound), or the maximum of the two, and report it directly. This is `O(n)` and
  one line. The open question — and the danger — is whether the bound is *tight*: a lower bound on the
  number of trips is not the same thing as the number of trips, and a "combined" bound that looks
  airtight can still under-count. This must be **checked numerically**, never asserted.

## Evaluation settings

Judged on hidden tests covering: `n = 0` and `n = 1`; everyone too heavy to pair (answer equals `n`);
everyone pairs perfectly (answer equals `ceil(n/2)`); a mix where neither bound is tight (like the
sample, where the answer beats both `ceil(sum/C)` and `ceil(n/2)`); odd and even `n`; many equal
weights; weights exactly at the capacity; and large `n = 2*10^5` with `C` near `10^9`, so the running
trip count and any weight sums must use 64-bit arithmetic.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long C;
    if (!(cin >> n >> C)) return 0;          // empty input -> no climbers -> 0 trips
    vector<long long> w(n);
    for (auto &x : w) cin >> x;

    // TODO: sort and greedily pair the lightest waiting climber with the heaviest
    // when they fit under capacity C; count the minimum number of trips.
    long long trips = 0;

    cout << trips << "\n";
    return 0;
}
```
