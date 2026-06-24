# Counting safe vote-counting orders (B never leads by more than m), modulo a prime

## Research question

A live election broadcast reveals the count one ballot at a time. Candidate A receives `a` ballots
in total and candidate B receives `b` ballots in total. The order in which the `a + b` ballots are
revealed is some interleaving of the A-ballots and the B-ballots; there are `C(a + b, a)` such orders
in all. The broadcaster calls an order **safe at margin `m`** (`m >= 0` an integer) if at every point
*after at least one ballot has been revealed*, candidate B is **never** more than `m` ballots ahead of
candidate A. Formally, if we read the order left to right and let `lead = (#A revealed so far) -
(#B revealed so far)`, then a safe order keeps `lead >= -m` after every single reveal.

For each of `q` independent scenarios you are given `a`, `b`, `m` and must output the number of safe
orders **modulo a prime `p`**. The interesting cases are `m` small relative to `b`, where the
constraint genuinely bites: most interleavings let B surge ahead early and are unsafe, and a tempting
"place each B-ballot in its earliest legal slot and multiply the free slots" greedy gets the count
wrong.

## Input / output contract

- Input (stdin): the first line holds two integers `q` and `p` (`1 <= q <= 2*10^5`; `p` is a prime
  with `2*10^6 < p < 2^31`, so `p` is strictly larger than any `a + b` below). Then `q` lines follow,
  each with three integers `a b m` (`0 <= a, b <= 10^6`, `a + b <= 2*10^6`, `0 <= m <= 2*10^6`).
- Output (stdout): `q` lines, the `i`-th being the number of safe orders for scenario `i`, taken
  modulo `p`.
- Time limit: 2 seconds. Memory: 256 MB.

Example: for `a = 3, b = 2, m = 0` the answer is `5`. The two B-ballots must never let B reach a lead,
i.e. every prefix has at least as many A's as B's; of the `C(5,3) = 10` interleavings exactly `5`
qualify (this is the Catalan-style ballot count).

## Background

The reveal order is a `+1 / -1` lattice path: an A-ballot is a `+1` step, a B-ballot is a `-1` step,
and "B never leads by more than `m`" is exactly "the path never drops below height `-m`". Two families
of approach are on the table before committing:

- **Local-multiply greedy.** Insert the B-ballots one at a time, each into the earliest position the
  margin still allows, and multiply the number of legal slots available to each. It is `O(b)` per
  query and feels like the staircase product that solves the *one-sided threshold* matching count. The
  open question is whether the per-B slot counts are really independent, or whether an early placement
  silently removes a slot a later B was counting on.
- **Reflection (the Andre / cycle-lemma identity).** Count *all* `C(a + b, a)` interleavings and
  subtract the unsafe ones via a single reflection across the forbidden barrier, which collapses the
  unsafe count to one binomial coefficient. This is `O(1)` per query after an `O(a + b)` precompute of
  factorials modulo `p`. The open question is the exact reflected index and the feasibility guard.

## Evaluation settings

Judged on hidden tests covering: `m` large enough to make the constraint vacuous (answer is the full
`C(a + b, a)`), tight `m = 0` ballot counts, infeasible scenarios with `a - b < -m` (answer `0`), the
empty scenario `a = b = 0` (one empty order, answer `1`), `b = 0` (one order, answer `1`), the
reflected index landing exactly on a boundary `b - m - 1 in {-1, 0, a+b, a+b+1}`, and large batches
`q = 2*10^5` with `a + b` up to `2*10^6` so that the factorial table and 64-bit modular products are
both exercised.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

ll MOD;

ll pw(ll base, ll e, ll mod) {
    base %= mod; if (base < 0) base += mod;
    ll r = 1 % mod;
    while (e > 0) {
        if (e & 1) r = (__int128)r * base % mod;
        base = (__int128)base * base % mod;
        e >>= 1;
    }
    return r;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int q;
    if (!(cin >> q >> MOD)) return 0;

    struct Query { ll a, b, m; };
    vector<Query> qs(q);
    ll maxn = 0;
    for (int i = 0; i < q; i++) {
        cin >> qs[i].a >> qs[i].b >> qs[i].m;
        maxn = max(maxn, qs[i].a + qs[i].b);
    }

    // TODO: precompute factorials mod p up to maxn, then answer each query by counting
    // all interleavings and subtracting the unsafe ones (reflection), guarding feasibility.

    for (int i = 0; i < q; i++) {
        ll answer = 0;
        cout << answer << "\n";
    }
    return 0;
}
```
