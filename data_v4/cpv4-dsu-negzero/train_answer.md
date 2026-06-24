**Problem.** There are `n` players, each starting alone in a guild with a signed balance `g[i]` (negative, zero, or positive). A stream of `q` events either merges two players' guilds (the merged balance is the sum) or queries: **among guilds with at least two members, what is the largest balance, reported only if it is strictly positive, else `0`** (and `0` if no two-member guild exists). Read `n`, `q`, the balances, then the events from stdin; print one line per query.

**Key idea — DSU with a per-component sum plus an exact multiset.** This is disjoint-set union: a `1 u v` event is a `union`, and each component (guild) stores its balance **sum** on its root, updated in `O(1)` per merge by adding the absorbed root's sum. The query asks for the maximum balance over components of size `>= 2`. Maintain a `multiset<long long>` of exactly those balances:

- On `union(ru, rv)` (`ru != rv`): if `sz[ru] >= 2` erase one copy of `bal[ru]`; if `sz[rv] >= 2` erase one copy of `bal[rv]`; attach smaller under larger; set `bal[survivor] += bal[absorbed]`; insert the new balance.
- On a query: if the multiset is empty answer `0`; else let `mx` be its maximum and answer `mx` if `mx > 0`, else `0`.

**Why the obvious shortcut is wrong.** Tracking a single running maximum updated at merge time fails because balances are signed, so component sums are **not monotone** under union. On `g = [10, 4, -20]`, merging {1,2} gives balance `14`; absorbing player 3's `-20` drops the only multi-member guild to `-6`. A running max keeps `14`, but no live guild has `14` — the true answer is `0`. The multiset is correct because it **removes** the absorbed guilds' old balances, so it only ever holds balances of currently-existing multi-member guilds.

**Correctness.** Invariant: after each event, `multi` contains the balance of every component whose current size is `>= 2`, with multiplicity. A union destroys the two old components (removed from `multi` iff they were size `>= 2`) and creates one new component of size `>= 2` (inserted). `find` returns the component, so `bal[root]` is the component's true sum. The query reads the exact maximum and applies the spec's strict-positive / empty rule. By induction over events, every answer is correct.

**Pitfalls.**
1. *Negative-driven staleness.* Component sums can decrease on merge, so any monotone "best so far" tracker is wrong; maintain the exact set and erase absorbed guilds.
2. *Erase one, not all.* `multi.erase(value)` removes **every** equal element. Use `multi.erase(multi.find(value))` so two distinct guilds sharing a balance are handled correctly (trace: `g=[3,0,3,0]`, two guilds of balance `3`).
3. *Base case / sign.* Report `0` when there is no multi-member guild and clamp with strict `mx > 0` (i.e. `max(0, mx)`); never print a negative or a zero "treasury."
4. *Overflow.* With `n` up to `2*10^5` and `|g[i]|` up to `10^9`, a balance reaches `~2*10^14`; use `long long`. An `int` is a silent wrong-answer on large tests.

**Edge cases (all `0`).** `n = 0` (empty world): no guilds, multiset stays empty. `n = 1` (lone player): no union possible, never size `>= 2`. All-negative merges: every balance negative, clamped to `0`. Balance exactly `0` after a merge: `mx > 0` is false, so `0`. Self-union / repeated union of already-merged players: `find(u) == find(v)`, skipped, no double counting.

**Complexity.** `O((n + q) log n)` time (DSU near-constant plus multiset ops), `O(n + k)` memory where `k` is the number of multi-member guilds (`<= n`). Comfortably within 1 s / 256 MB; at `n = q = 2*10^5` it runs in well under 0.1 s.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, q;
    if (!(cin >> n >> q)) return 0;          // n = 0 / empty input handled below

    vector<long long> bal(n + 1, 0);
    for (int i = 1; i <= n; i++) cin >> bal[i];

    // DSU with union by size; root carries the guild's total balance and member count.
    vector<int> par(n + 1), sz(n + 1, 1);
    for (int i = 1; i <= n; i++) par[i] = i;

    function<int(int)> find = [&](int x) {
        while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; }
        return x;
    };

    // Multiset of balances of guilds that currently have >= 2 members.
    // A merged balance can be SMALLER than its parts (negative members), so a
    // running max over merge-time balances is wrong; the multiset stays exact.
    multiset<long long> multi;

    string out;
    out.reserve(1 << 16);

    for (int e = 0; e < q; e++) {
        int type;
        cin >> type;
        if (type == 1) {
            int u, v;
            cin >> u >> v;
            int ru = find(u), rv = find(v);
            if (ru == rv) continue;          // already same guild: no change
            // A root with sz >= 2 is currently represented in the multiset.
            if (sz[ru] >= 2) multi.erase(multi.find(bal[ru]));
            if (sz[rv] >= 2) multi.erase(multi.find(bal[rv]));
            if (sz[ru] < sz[rv]) swap(ru, rv);
            par[rv] = ru;
            sz[ru] += sz[rv];
            bal[ru] += bal[rv];              // merged balance = sum (may be negative)
            multi.insert(bal[ru]);           // ru now has >= 2 members
        } else {
            // type == 2: max balance over multi-member guilds, but only if positive.
            long long ans = 0;
            if (!multi.empty()) {
                long long mx = *multi.rbegin();
                if (mx > 0) ans = mx;
            }
            out += to_string(ans);
            out += '\n';
        }
    }

    cout << out;
    return 0;
}
```
