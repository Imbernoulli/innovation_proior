I have `n` players, each starting alone with a signed gold balance `g[i]` that can be negative, zero, or positive. A stream of `q` events either merges two players' guilds (summing their balances) or asks a query. The query is specific in a way I must not gloss over: it wants the largest balance **among guilds with at least two members**, and it wants `0` unless that largest is **strictly positive** (or unless no two-member guild exists yet). So a one-player guild — even a wealthy one — never counts, and a merged guild that happens to total `0` or less reports `0`. Before any algorithm I fix the scale, because it decides the data types. `n, q <= 2*10^5` and `|g[i]| <= 10^9`. A single guild can swallow all `n` players, so its balance reaches `2*10^5 * 10^9 = 2*10^14`, far past the 32-bit ceiling of `~2.1*10^9`. Every balance and accumulator must be `long long`. The same magnitude argument runs on the negative side: a guild can reach `-2*10^14`, so my "is it positive?" test has to be a real signed comparison in 64-bit, not anything that could wrap. That is decision one and it is non-negotiable.

**Recognizing the shape: this is DSU.** Each guild is a connected component of players; "guild of `u`" is "the set reachable from `u` through alliance edges seen so far." A `1 u v` event is a `union`, and a `2` event reads an aggregate over components. That is textbook disjoint-set union. The two non-textbook pieces are (a) carrying the balance **sum** on each component, and (b) answering "max over multi-member components, but only if positive." I will get the DSU skeleton right first, then layer the aggregate, then layer the query, tracing at each layer because each layer has its own trap.

**The DSU skeleton.** I keep `par[]` (parent), `sz[]` (component size), and `bal[]` reused as the component balance stored on the root. `find` walks to the root; I will use path halving (`par[x] = par[par[x]]`) so `find` stays near-constant amortized without recursion depth worries on `2*10^5` elements. `union(u, v)`: find roots `ru, rv`; if equal, do nothing (the "already same guild" rule falls out for free); else attach the smaller tree under the larger (union by size) so trees stay shallow, then fold the absorbed root's size and balance into the surviving root. Storing the balance on the root and adding the absorbed root's balance is `O(1)` per merge and keeps the invariant "root holds the whole component's sum."

**Candidate approaches for the query — and the first one I want to break.** The merges are easy; the query is where I have to think. Two routes:

- *Single running maximum.* Keep one variable `best`, and every time a union produces a guild with `>= 2` members, do `best = max(best, newBalance)`. A `2` query answers `best > 0 ? best : 0`. This is `O(1)` per op and tempting. The danger smell: balances are signed, and `best` only ever moves up. If a guild that set `best` is later absorbed into a poorer guild, `best` keeps pointing at a value no live guild has. I refuse to trust this until I try to break it.
- *Exact multiset of live multi-member balances.* Keep a `multiset<long long>` holding the balance of every guild that currently has `>= 2` members. On a union, remove the old balances of whichever of the two roots already had `>= 2` members (they cease to exist as separate guilds), then insert the new merged balance. A `2` query reads `*multi.rbegin()` and applies the positivity rule. `O(log n)` per op, and crucially it stays *exact* under shrinking balances.

**Stress-testing the running-max idea on paper before committing.** Hand-waving "max only grows, so a running max is fine" is exactly how negatives bite. Let me attack it. Take three players with balances `g = [10, 4, -20]`. Events: `1 1 2`, then `2`, then `1 1 3`, then `2`.

- `1 1 2`: merge {1} and {2} -> guild {1,2} with balance `14`, size 2. Running-max sets `best = max(-inf, 14) = 14`.
- `2`: `best = 14 > 0`, answers `14`. Correct so far.
- `1 1 3`: merge {1,2} with {3}(balance `-20`) -> guild {1,2,3} with balance `14 + (-20) = -6`, size 3. Running-max does `best = max(14, -6) = 14` (or, if I only update on the new value, it simply leaves `best = 14`).
- `2`: running-max answers `best = 14 > 0` -> `14`.

But what guild actually has balance `14` now? **None.** The only multi-member guild is {1,2,3} with balance `-6`, which is not positive, so the true answer is `0`. The running-max is off by a mile — it reported a treasury (`14`) that was dismantled. This is the negative-balance trap made concrete: a merge can *lower* the best, and a max-only tracker cannot see a decrease. So the running-max approach is wrong, and I now see precisely why: monotone tracking is only valid when the tracked quantity is monotone, and signed sums under union are not. The multiset, which *removes* the stale `14` when {1,2} is absorbed and then holds only `-6`, gives `0`. The verification killed the cheap approach. I commit to the multiset.

**Deriving the multiset bookkeeping exactly.** Invariant I want: `multi` contains exactly the balance of each guild whose current size is `>= 2`, with multiplicity (two distinct guilds can share a balance value, so I must use a multiset and erase *one* occurrence, not all). On `union(u, v)` with roots `ru, rv`, `ru != rv`:

1. If `sz[ru] >= 2`, that guild is currently in `multi`; erase one copy of `bal[ru]`.
2. If `sz[rv] >= 2`, erase one copy of `bal[rv]`.
3. Decide survivor by size, attach, update `sz`, set `bal[survivor] += bal[absorbed]`.
4. Insert `bal[survivor]` — the new guild has size `>= 2` for sure (it has at least one member from each side... actually at least two total since both sides had `>= 1`), so it belongs in `multi`.

Step order matters: I must erase using the *old* `bal[ru]`, `bal[rv]` and *old* `sz` values, before I mutate them in step 3. If I attach first and then try to erase the old balances, the values are gone. So I do all erases up front.

One sharp point on the erase: `multi.erase(value)` erases **all** elements equal to `value`; that would be a bug if two live guilds share a balance and I only meant to remove one. I must use `multi.erase(multi.find(value))`, which removes a single occurrence. I will write it that way from the start and flag it in the trace.

**Designing the answer formula and the base case — where signs and emptiness bite.** A `2` query: if `multi` is empty (no two-member guild exists yet), the answer is `0`. Otherwise let `mx = *multi.rbegin()` (the largest live multi-member balance); answer `mx` if `mx > 0`, else `0`. The strictness matters: a guild totaling exactly `0` must report `0`, and `mx > 0` handles that. The empty-multiset case and the non-positive-max case both collapse to `0`, which is the "no treasury worth announcing" rule. This is the base case I have to get right, and it is exactly the corner the problem is built to test: all-negative worlds, the empty world, and the lone player.

**First implementation.** Here is my first cut of the event loop (DSU helpers elided, same as final):

```
multiset<long long> multi;
for (int e = 0; e < q; e++) {
    int type; cin >> type;
    if (type == 1) {
        int u, v; cin >> u >> v;
        int ru = find(u), rv = find(v);
        if (ru == rv) continue;
        if (sz[ru] < sz[rv]) swap(ru, rv);   // make ru the survivor
        par[rv] = ru;
        sz[ru] += sz[rv];
        bal[ru] += bal[rv];
        multi.insert(bal[ru]);               // new guild balance
    } else {
        long long ans = 0;
        if (!multi.empty()) ans = max(0LL, *multi.rbegin());
        cout << ans << "\n";
    }
}
```

**A trace that exposes the first real bug.** I deliberately removed nothing from `multi` above — let me see what that costs. Take `g = [10, 4, -20]`, events `1 1 2`, `2`, `1 1 3`, `2` (my earlier counterexample).

- `1 1 2`: `ru=1, rv=2`. `bal[1] = 10+4 = 14`. `multi.insert(14)` -> `multi = {14}`.
- `2`: `max(0, 14) = 14`. Good.
- `1 1 3`: `ru = find(1) = 1`, `rv = 3`. `sz[1]=2 >= sz[3]=1`, survivor `ru=1`. `bal[1] = 14 + (-20) = -6`. `multi.insert(-6)` -> `multi = {-6, 14}`. **I never removed the old `14` for guild {1,2}, even though that guild no longer exists separately.**
- `2`: `*multi.rbegin() = 14`, answer `max(0, 14) = 14`.

The answer is `14`, but the true answer is `0` (only live multi-member guild is {1,2,3} at `-6`). **Bug confirmed: I forgot to erase the absorbed/old balances on union.** This is the same staleness that killed the running-max, now reappearing because my multiset accumulated dead entries. The fix is the erase-before-mutate logic I derived: before attaching, if `sz[ru] >= 2` erase `bal[ru]`, if `sz[rv] >= 2` erase `bal[rv]`. Let me patch:

```
int ru = find(u), rv = find(v);
if (ru == rv) continue;
if (sz[ru] >= 2) multi.erase(multi.find(bal[ru]));
if (sz[rv] >= 2) multi.erase(multi.find(bal[rv]));
if (sz[ru] < sz[rv]) swap(ru, rv);
par[rv] = ru;
sz[ru] += sz[rv];
bal[ru] += bal[rv];
multi.insert(bal[ru]);
```

Re-trace `g = [10, 4, -20]`:

- `1 1 2`: `sz[1]=1`, `sz[2]=1`, neither `>= 2`, no erase. Merge -> `bal[1]=14`. Insert `14`. `multi={14}`.
- `2`: `14`. Good.
- `1 1 3`: `ru=1` with `sz[1]=2 >= 2` -> erase one `14`, `multi={}`. `rv=3`, `sz=1`, no erase. Merge -> `bal[1]=-6`. Insert `-6`. `multi={-6}`.
- `2`: `*rbegin() = -6`, `max(0, -6) = 0`. **Correct.** The stale `14` is gone, exactly because I erased the guild that got absorbed. The first bug is fixed and it failed for the reason I fixed.

**A second trace targeting the base-case / sign corner.** The problem's named trap is negatives, zeros, and the all-negative / empty corner. Let me hit those directly. First the all-negative merged case: `g = [-1, -2, -3]`, events `1 1 2`, `2`.

- `1 1 2`: merge -> `bal[1] = -1 + -2 = -3`, size 2. Insert `-3`. `multi = {-3}`.
- `2`: `*rbegin() = -3`. My formula `max(0LL, *multi.rbegin()) = max(0, -3) = 0`. **Correct** — a two-member but broke guild reports `0`, not `-3`.

Now imagine the base case written the *wrong* way, which is the classic failure: suppose I had written the query as `cout << *multi.rbegin()` when non-empty, forgetting the positivity clamp. Then this case prints `-3`, a negative "treasury," which is nonsense and a wrong answer. Or suppose I had initialized a single `best` variable to `0` and answered `best` directly — then I would also miss that the *only* multi-member guild is negative and would print a stale or default value. The `max(0LL, ...)` clamp (equivalently `mx > 0 ? mx : 0`) is what makes the all-negative world correct. I keep it.

Next the empty world: `n = 0`, `q = 2`, both events `2`.

- After `cin >> n >> q` with `n=0`, the balance loop runs zero times. `multi` is empty.
- Each `2`: `multi.empty()` is true, `ans = 0`. Prints `0`, `0`. **Correct** — there are no guilds at all.

And the lone player: `n = 1`, balance `g = [5]`, query `2`. No union can ever form (only one player), so `multi` stays empty and the query prints `0`. **Correct** — a single-member guild, however rich, is not eligible. A wrong base case that seeded `multi` with single-player balances, or that tracked the global max balance ignoring the size rule, would print `5` here. Mine prints `0`.

And the exactly-zero corner: `g = [0, 0]`, events `1 1 2`, `2`. Merge -> `bal = 0`, size 2, `multi = {0}`. Query: `max(0LL, 0) = 0`. **Correct**, and note this is correct *because* the rule and my clamp both treat `0` as not-a-treasury; a `>= 0` test would wrongly "succeed" but still print `0` here, yet the strict `> 0` is what the spec asks and is what I implemented via `max(0, mx)` (since `max(0, 0) = 0`).

**Sanity-checking the derivation on the documented sample.** Sample: `n = 5`, `q = 7`, balances `g = [10, -3, 4, -20, 7]`. Events: `1 1 3`, `2`, `1 1 2`, `2`, `1 1 4`, `2`, `2`. Trace the multiset and the answers.

- `1 1 3`: roots `1`(sz1), `3`(sz1), no erase. Merge -> `bal[1] = 10 + 4 = 14`, size 2. Insert `14`. `multi = {14}`.
- `2`: `max(0, 14) = 14`. -> **14**.
- `1 1 2`: `ru = find(1) = 1` (sz 2 -> erase one `14`, `multi = {}`), `rv = 2` (sz1, no erase). Survivor `1` (sz2 > sz1). `bal[1] = 14 + (-3) = 11`, size 3. Insert `11`. `multi = {11}`.
- `2`: `max(0, 11) = 11`. -> **11**.
- `1 1 4`: `ru = find(1) = 1` (sz 3 -> erase one `11`, `multi = {}`), `rv = 4` (sz1, no erase). Survivor `1`. `bal[1] = 11 + (-20) = -9`, size 4. Insert `-9`. `multi = {-9}`.
- `2`: `*rbegin() = -9`, `max(0, -9) = 0`. -> **0**. (Here is the whole story: absorbing the `-20` guild dropped the best from `11` to `-9`; a running-max tracker would have wrongly kept `11`.)
- `2`: still `multi = {-9}`, `max(0, -9) = 0`. -> **0**.

Answers `14, 11, 0, 0` — matches the documented sample. The derivation is internally consistent.

**Edge cases, deliberately enumerated.**
- `n = 0`: no balances, no possible union, every query prints `0`. Handled (empty multiset).
- `n = 1`: no union possible, every query prints `0`. Handled (multiset never gets a `>= 2` entry).
- `q = 0`: no events, no output. The loop runs zero times; program prints nothing. Fine.
- All-negative balances with merges: every merged balance is negative, `max(0, mx)` clamps to `0`. Handled.
- Balance exactly `0` after merge: `max(0, 0) = 0`. Handled, matches strict-positive rule.
- Self-union `1 u u` or repeated union of already-merged players: `find(u) == find(v)`, `continue`, no spurious multiset edits, no double-count. Handled.
- Overflow: balances are `long long`; max magnitude `~2*10^14` fits with three decimal orders of room. No accumulator other than `bal[]` grows, and it only ever holds a sum of inputs. Safe.
- Output volume: up to `2*10^5` query lines; I build them in a string buffer and flush once, and use `sync_with_stdio(false)` to keep I/O cheap. Reading is via `cin >>` so whitespace/newlines are irrelevant.

**A note on the multiset erase pitfall, re-confirmed.** I want to be sure `multi.erase(multi.find(bal[ru]))` removes exactly one element even when two live guilds share a balance. Construct it: `g = [3, 0, 3, 0]`, events `1 1 2` (-> guild {1,2} bal `3`), `1 3 4` (-> guild {3,4} bal `3`), now `multi = {3, 3}`. Then `1 1 3`: `ru = find(1)` has `bal 3`, sz2 -> `multi.find(3)` removes *one* `3`, `multi = {3}`; `rv = find(3)` has `bal 3`, sz2 -> removes the other `3`, `multi = {}`; merge -> `bal = 6`, insert -> `multi = {6}`. A query now gives `6`. If I had used `multi.erase(3)` (which erases *all* `3`s), the first erase would wipe both and the second `find` would point at a stale/`end` iterator — undefined behavior or a wrong count. The `find`-then-erase form is correct; I keep it.

**Final solution.** I disproved the cheap running-max with a traced negative-balance counterexample, derived the exact multiset bookkeeping, caught my first transcription bug (forgetting to erase absorbed guilds) by tracing that same counterexample, and pinned the base-case/sign behavior (`max(0, mx)`, strict positivity, empty/all-negative/lone-player) with direct traces. This is what I ship — one self-contained `O((n + q) log n)` file:

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

**Causal recap.** The cheap "single running maximum" looked right but a traced counterexample (`g = [10, 4, -20]`, merge {1,2} to balance `14`, then absorb the `-20` guild down to `-6`) showed that signed sums under union are not monotone, so a max-only tracker reports a treasury (`14`) that no live guild has — I moved to an exact multiset of current multi-member balances. My first multiset code reproduced the same staleness because I inserted the merged balance but never removed the absorbed guilds' old balances; the identical trace returning `14` instead of `0` pinpointed it, and erasing `bal[ru]`/`bal[rv]` (only when their old size was `>= 2`, and via `find`-then-`erase` to drop a single copy) before mutating fixes it. The base case is closed by answering `0` when the multiset is empty and clamping with `max(0, mx)` / strict `mx > 0`, which makes the empty world, the lone player, the all-negative merge, and the exactly-`0` guild all report `0`; and `long long` everywhere keeps the `~2*10^14` magnitudes and their signs intact.
