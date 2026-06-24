**Problem.** A robot walks the integer grid using only right (`+1` in `x`) and up (`+1` in `y`) steps.
For each of `q` queries you get a checkpoint `(cx, cy)` and a destination `(ex, ey)` with
`0 <= cx <= ex <= 10^6` and `0 <= cy <= ey <= 10^6`. Count the monotone walks from `(0,0)` that pass
through `(cx,cy)` and end at `(ex,ey)`, modulo a **prime `M`** given per test (guaranteed larger than
every factorial index that arises, and as large as about `2 * 10^9`). Print one reduced count per query.

**Key idea — factor the walk into two binomials.** A monotone walk from `(ax,ay)` to `(bx,by)` is an
arrangement of `R = bx-ax` rights among `R + U` steps (`U = by-ay` ups), so it is counted by
`C(R+U, R)`. A walk through a checkpoint splits into two independent legs (the checkpoint is visited
exactly once, since coordinates only increase), so

```
answer = C(cx+cy, cx) * C((ex-cx)+(ey-cy), ex-cx)   (mod M)
```

Compute the binomials with precomputed factorials mod `M`: `fact[i] = i! mod M` up to the largest leg
length `N` over all queries (`N <= 2*10^6`), one Fermat inverse `inv_fact[N] = fact[N]^(M-2) mod M`
(legal because `M` is prime and `M > N`, so no factorial is `0 mod M`), then fold down with
`inv_fact[i-1] = i * inv_fact[i] mod M`. Each binomial is `fact[n] * inv_fact[k] * inv_fact[n-k]`. Read
all queries first so you know `N` before sizing the tables.

**Pitfalls.**
1. *Overflow — the headline trap.* `M` can be near `2 * 10^9`, so a residue is near `2 * 10^9` and a
   product of two residues is near `4 * 10^18`. That overflows a 32-bit `int` (`~2.1 * 10^9`) but fits a
   signed 64-bit `long long` (`~9.2 * 10^18`). If the modular multiply is written as `(int)*(int) % M`,
   the product wraps **before** the `% M` and returns garbage — and it only shows up on large-`M` tests,
   passing every small case. Route `MOD`, `fact`, `inv_fact`, and **every** product through one
   64-bit `mul(a,b) = (a%MOD)*(b%MOD)%MOD`. (Concretely, `(M-3)*(M-5)` for `M=2000000011` is
   `4000000028000000048`; in 64-bit it reduces to `15`, in 32-bit it is nonsense.)
2. *Binomial range.* Guard `C(n,k)` to return `0` when `k < 0` or `k > n`, so a stray argument cannot
   index the inverse-factorial table out of bounds.

**Edge cases.** Checkpoint at the origin: `C(0,0)=1`, answer is the single second-leg binomial.
Checkpoint equal to destination: second leg is `C(0,0)=1`. Both endpoints at the origin: `N = 0`, the
fold loop runs zero times, answer `1`. Small primes (e.g. `M = 7`) reduce on small numbers and still
agree with a direct path-counting DP. `1 % MOD` in `power` is harmless since `M >= 2`.

**Complexity.** `O(N)` to build the factorial tables plus `O(log M)` for the single inverse, then
`O(1)` per query, with `N <= 2 * 10^6` and `q <= 10`. About `0.03 s` and under `20 MB` in practice —
well within a 1-second, 256-MB budget.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

typedef long long ll;

ll MOD;

ll mul(ll a, ll b) {
    return (a % MOD) * (b % MOD) % MOD;   // 64-bit product; int would overflow here
}

ll power(ll base, ll e) {
    ll r = 1 % MOD;
    base %= MOD;
    if (base < 0) base += MOD;
    while (e > 0) {
        if (e & 1) r = mul(r, base);
        base = mul(base, base);
        e >>= 1;
    }
    return r;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int q;
    if (!(cin >> q >> MOD)) return 0;

    // Maximum coordinate value across the whole input bounds factorial table size.
    // We pre-read all queries to know the max factorial index needed.
    // R + U for a leg can be up to (X + Y) <= 2*10^6.
    struct Query { ll cx, cy, ex, ey; };
    vector<Query> qs(q);
    long long maxN = 0;
    for (int i = 0; i < q; i++) {
        cin >> qs[i].cx >> qs[i].cy >> qs[i].ex >> qs[i].ey;
        maxN = max(maxN, qs[i].cx + qs[i].cy);
        maxN = max(maxN, (qs[i].ex - qs[i].cx) + (qs[i].ey - qs[i].cy));
    }

    // Factorials and inverse factorials mod MOD (MOD is prime, MOD > maxN).
    vector<ll> fact(maxN + 1), inv_fact(maxN + 1);
    fact[0] = 1 % MOD;
    for (ll i = 1; i <= maxN; i++) fact[i] = mul(fact[i - 1], i % MOD);
    inv_fact[maxN] = power(fact[maxN], MOD - 2);
    for (ll i = maxN; i >= 1; i--) inv_fact[i - 1] = mul(inv_fact[i], i % MOD);

    auto C = [&](ll n, ll k) -> ll {
        if (k < 0 || k > n) return 0;
        return mul(fact[n], mul(inv_fact[k], inv_fact[n - k]));
    };

    for (int i = 0; i < q; i++) {
        ll cx = qs[i].cx, cy = qs[i].cy, ex = qs[i].ex, ey = qs[i].ey;
        // Paths (0,0)->(cx,cy): choose which of the (cx+cy) steps go right (cx of them).
        ll leg1 = C(cx + cy, cx);
        // Paths (cx,cy)->(ex,ey): need ex-cx rights and ey-cy ups.
        ll dx = ex - cx, dy = ey - cy;
        ll leg2 = C(dx + dy, dx);
        cout << mul(leg1, leg2) << "\n";   // product also needs 64-bit before reduction
    }
    return 0;
}
```
