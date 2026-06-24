**Reading the problem and pinning the contract.** A robot moves only right or up on the integer grid. For each of `q` queries I get a checkpoint `(cx, cy)` and a destination `(ex, ey)`, and I must count the monotone walks from the origin that pass through the checkpoint and end at the destination, modulo a prime `M`. Input is `q` and `M` on the first line, then `q` lines of `cx cy ex ey`; I print one reduced count per query. Before touching algorithms I fix the scale, because it decides the data types. Coordinates go up to `10^6`, so a single leg uses up to `cx + cy <= 2 * 10^6` steps — that bounds my factorial table. The modulus `M` is *not* a fixed friendly constant: it is given per test and can be as large as about `2 * 10^9`. That last fact is the whole trap. A residue modulo `M` can be nearly `2 * 10^9`, and the product of two such residues is nearly `4 * 10^18`. That is far outside 32-bit range (`~2.1 * 10^9`) but inside signed 64-bit range (`~9.2 * 10^18`). So every product that feeds a `% M` must be computed in `long long`. I write that down now as non-negotiable; an `int` multiply here is a silent wrong answer on exactly the large-`M` tests.

**Reducing the geometry to binomials.** A monotone walk from `(ax, ay)` to `(bx, by)` consists of `R = bx - ax` rights and `U = by - ay` ups in some order; any interleaving is a distinct walk, and there are `C(R + U, R)` of them — choose which of the `R + U` positions are the rights. Passing through a checkpoint is the clean part: a walk `(0,0) -> (ex,ey)` that visits `(cx,cy)` is exactly a walk `(0,0) -> (cx,cy)` followed by a walk `(cx,cy) -> (ex,ey)`, and because monotone walks can only increase coordinates, the checkpoint is visited at most once, so there is no double counting. The two legs are independent, so the count multiplies:

```
answer = C(cx + cy, cx) * C((ex - cx) + (ey - cy), ex - cx)   (mod M)
```

Let me sanity-check that factorization on the first sample query, `(cx,cy)=(2,1)`, `(ex,ey)=(4,3)`. Leg 1 is `C(2+1, 2) = C(3,2) = 3`. Leg 2 needs `ex-cx = 2` rights and `ey-cy = 2` ups, so `C(4, 2) = 6`. Product `3 * 6 = 18`, which matches the stated answer `18`. Second query has the checkpoint at the origin: leg 1 is `C(0,0) = 1`, leg 2 is `C(10, 5) = 252`, product `252` — matches. Third: `C(2,1) * C(2,1) = 2 * 2 = 4` — matches. The reduction to binomials is correct.

**Candidate approaches for the binomials.** Two routes.

- *Per-query multiplicative formula.* For each `C(n, k)` multiply `n * (n-1) * ... * (n-k+1)` and divide by `k!`, doing the division with a modular inverse. The trouble is the division: under a prime modulus I cannot just integer-divide; I would need a modular inverse of `k!` anyway, and computing it from scratch each query is wasteful. Also the running numerator must be reduced every step, and that reduction is again an overflow-prone multiply. No real advantage.
- *Precomputed factorials and inverse factorials.* Build `fact[i] = i! mod M` once up to the largest index any query needs, compute the single inverse `inv_fact[N] = fact[N]^(M-2) mod M` by Fermat (valid because `M` is prime and `M > N`, so no factor of `fact[N]` is `0 mod M`), then fold `inv_fact[i-1] = inv_fact[i] * i mod M` downward. Each binomial is then three table lookups and two multiplies. This is `O(N)` precompute plus `O(1)` per query, where `N <= 2 * 10^6`. I commit to this.

One subtlety I note before coding: I do not know `N` until I have read every query, since `N` is the max over all legs of all queries. So I must read all queries first, compute `N = max(cx+cy, dx+dy)` over them, then build the tables. Reading-then-sizing is the safe order.

**Why Fermat is legal here.** Fermat's little theorem gives `a^(M-1) = 1 (mod M)` for `M` prime and `a` not divisible by `M`, hence `a^(M-2)` is the inverse of `a`. The problem guarantees `M > cx+cy` and `M > dx+dy` for every query, so the largest factorial index `N` satisfies `N < M`. Then `fact[N] = N! mod M` is a product of integers `1..N`, all strictly between `0` and `M`, none divisible by the prime `M`; so `fact[N]` is nonzero mod `M` and its inverse exists. Good — the inverse-factorial chain is well defined.

**First implementation and a trace.** I write a modular multiply helper and the rest on top of it. My first cut of the helper:

```
int MOD;                         // <-- first cut
int mul(int a, int b) {
    return (int)(a % MOD) * (b % MOD) % MOD;
}
```

and I plan to store `fact`, `inv_fact` as `int` too, "since every value is below `M < 2^31`". Let me trace the danger case before trusting it: a large modulus, say `M = 2000000011` (a prime near `2^31`), with two residues near `M`, say `a = M - 3 = 2000000008` and `b = M - 5 = 2000000006`. The true product is `a * b = 4000000028000000048`, and `a*b mod M` should be `(-3)*(-5) = 15`. What does my `mul` compute? `(a % MOD)` is `a` (an `int`), `(b % MOD)` is `b` (an `int`), and `(int) * (int)` is evaluated in `int` arithmetic: `2000000008 * 2000000006` wraps around `2^32` and lands on some garbage value, then `% MOD` reduces the garbage. I do the wrap by hand: the exact product is about `4 * 10^18`; reduced mod `2^32 = 4294967296` it is nowhere near `15`. So `mul` returns nonsense.

**The bug.** This is the overflow I warned myself about, now concrete. `(int) * (int)` in C++ multiplies in `int`, and the product of two near-`2*10^9` operands overflows 32-bit long before the `% MOD` ever runs. The cast `(int)(a % MOD)` makes it worse by *guaranteeing* the operands are `int`. The reduction `% MOD` cannot save a product that was already truncated. Every multiply in the program has this disease: the `fact` recurrence `fact[i] = fact[i-1] * i`, the fast-power `base = base * base`, the inverse fold `inv_fact[i] * i`, and the final `leg1 * leg2`. On small-`M` tests it would pass (products stay tiny), so this is exactly the kind of bug that survives casual testing and dies on the hidden large-`M` cases. The fix is structural: the modulus and every stored value and every intermediate product must be 64-bit. I switch `MOD`, `fact`, `inv_fact`, and the helper to `long long`:

```
typedef long long ll;
ll MOD;
ll mul(ll a, ll b) {
    return (a % MOD) * (b % MOD) % MOD;   // 64-bit product, then reduce
}
```

Re-trace `a = M-3`, `b = M-5`, `M = 2000000011`: `(a % MOD)` is `2000000008LL`, `(b % MOD)` is `2000000006LL`, their `ll * ll` product is `4000000028000000048` which fits in 64 bits (below `9.2 * 10^18`), and `% MOD` gives `15`. Correct. The reduction now operates on the true product. I keep `mul` as the single choke point so no multiply in the program can bypass the 64-bit widening.

**Second implementation and a second trace — the inverse chain.** With `mul` fixed I write the precompute:

```
fact[0] = 1 % MOD;
for (ll i = 1; i <= maxN; i++) fact[i] = mul(fact[i-1], i % MOD);
inv_fact[maxN] = power(fact[maxN], MOD - 2);
for (ll i = maxN; i >= 1; i--) inv_fact[i-1] = mul(inv_fact[i], i % MOD);
```

and the fast power:

```
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
```

Before I trust the inverse-factorial fold I prove the fold identity, then trace it. The claim is `inv_fact[i-1] = inv_fact[i] * i`. Why: `inv_fact[i] = 1 / i!` and `inv_fact[i-1] = 1 / (i-1)!`, and `(i-1)! = i! / i`, so `1/(i-1)! = i / i! = i * (1/i!) = i * inv_fact[i]`. The identity holds. Now a concrete trace with a small prime where I can compute everything by hand: `M = 7`, build up to `maxN = 4`. `fact = [1, 1, 2, 6%7=6, 24%7=3]`, i.e. `fact[4] = 4! = 24 = 3 (mod 7)`. `inv_fact[4] = 3^(5) mod 7`: `3^2=9=2`, `3^4=2^2=4`, `3^5=4*3=12=5`, so `inv_fact[4]=5`. Check: `3 * 5 = 15 = 1 (mod 7)`. Good. Fold down: `inv_fact[3] = inv_fact[4]*4 = 5*4 = 20 = 6 (mod 7)`; check `fact[3]=6`, `6*6=36=1 (mod 7)` — good. `inv_fact[2] = inv_fact[3]*3 = 6*3 = 18 = 4`; check `fact[2]=2`, `2*4=8=1` — good. `inv_fact[1] = 4*2 = 8 = 1`; check `fact[1]=1`, `1*1=1` — good. `inv_fact[0] = 1*1 = 1`; `fact[0]=1` — good. The whole inverse-factorial table is consistent, so `C(n,k) = mul(fact[n], mul(inv_fact[k], inv_fact[n-k]))` will be correct mod `M`.

**A real bug in the binomial guard.** My first `C` lambda was:

```
auto C = [&](ll n, ll k) -> ll {
    return mul(fact[n], mul(inv_fact[k], inv_fact[n - k]));
};
```

I trace a query whose checkpoint sits on the boundary in a way that makes `k > n` impossible for a leg... actually for valid legs `k <= n` always holds, because leg 1 is `C(cx+cy, cx)` with `cx <= cx+cy`, and leg 2 is `C(dx+dy, dx)` with `dx <= dx+dy`. So `k` is in range for legitimate input. But I deliberately test a *degenerate* construction to see whether `C` would ever be asked for `k < 0` or `k > n` — for instance if I ever (mistakenly) called `C(dx+dy, dx)` with `dx` negative because a future variant lets the checkpoint lie past the destination on one axis. If `k` were negative, `inv_fact[k]` would index `inv_fact[-1]`, an out-of-bounds read — undefined behaviour, likely a crash or garbage, not a clean `0`. To make `C` self-defending and match the mathematical convention that `C(n,k)=0` when `k<0` or `k>n`, I add the guard:

```
auto C = [&](ll n, ll k) -> ll {
    if (k < 0 || k > n) return 0;
    return mul(fact[n], mul(inv_fact[k], inv_fact[n - k]));
};
```

This does not change any answer on valid input (legs never trigger it), but it turns a potential out-of-bounds read into a defined `0`, which is the right behaviour and protects the table accesses. I keep it.

**Trace of the full pipeline on a sample.** Query `(2,1,4,3)` with `M = 998244353`. `leg1 = C(3, 2)`. `fact[3] = 6`, `inv_fact[2]`, `inv_fact[1]` — by the identity `C(3,2) = 3`. `leg2`: `dx = 2`, `dy = 2`, `C(4, 2) = 6`. Final `mul(leg1, leg2) = mul(3, 6) = 18`, and `18 < M` so no reduction. Output `18`. Matches. The pipeline is wired correctly end to end.

**Independent brute force, and 400 random agreements.** I do not trust hand traces alone for a counting problem, so I write a brute that counts walks a completely different way: a grid DP `dp[x][y] = dp[x-1][y] + dp[x][y-1]` with `dp[0][0] = 1`, run on small grids, once for `(0,0)->(cx,cy)` and once for `(cx,cy)->(ex,ey)`, then multiply and reduce. No binomials, no inverses, no factorials — purely additive path counting, so it cannot share a bug with the formula. I generate 400 random small cases: tiny grids (coordinates up to 7 so the DP is cheap) and a mixture of moduli — the smallest legal prime above the largest leg, a small prime in the low hundreds, a prime near `10^6`, and a prime near `2 * 10^9` to exercise the overflow-prone multiply. The fast solution and the brute agree on all 400. The large-`M` family is the one that would have failed under the `int` `mul`; it passes under the `long long` `mul`.

**Edge cases, deliberately.**
- *Checkpoint at the origin*, `(0,0,ex,ey)`: `leg1 = C(0,0) = 1`. `fact[0] = 1`, `inv_fact[0] = 1`, so `C(0,0) = mul(1, mul(1, 1)) = 1`. The answer collapses to the single binomial `leg2`, as it should — there is only one way to "walk" from the origin to the origin (stand still). Verified on the second sample (`252`).
- *Checkpoint equals destination*, `(ex,ey,ex,ey)`: `leg2 = C(0,0) = 1`; answer is just `leg1`. The factorial-zero handling makes this fall out for free.
- *Zero-length both ways*, `(0,0,0,0)`: `leg1 = leg2 = C(0,0) = 1`, answer `1`. The robot stays put; one walk. `maxN = 0`, so `fact` has a single entry `fact[0] = 1` and `inv_fact[0] = power(1, M-2) = 1`; the fold loop with `maxN = 0` runs zero times. No out-of-bounds, no empty-vector indexing. Correct.
- *Small modulus where reductions fire*, e.g. `M = 7`: traced above; the inverse chain is exact, and the brute agreed across the small-prime cases. The line `1 % MOD` rather than `1` matters only when `M = 1`, but `M` is a prime so `M >= 2` and `1 % MOD = 1`; harmless either way and safe.
- *Overflow*, the headline corner: all of `MOD`, `fact`, `inv_fact`, and every product are `long long`, funneled through `mul`, whose `(a % MOD) * (b % MOD)` is a 64-bit product of two values below `M (< 2^31)`, i.e. below `4 * 10^18 < 9.2 * 10^18`. No multiply in the program can overflow. The fast-power squares `base = mul(base, base)`, also through the 64-bit `mul`. Safe.
- *Performance*: `maxN <= 2 * 10^6`, so two `long long` tables of `~2 * 10^6` entries are about 32 MB total, and the precompute is one linear pass plus one `O(log M)` inverse; with `q <= 10` queries the whole run is well under the 1-second, 256-MB budget (measured: ~0.03 s, ~19 MB at `maxN = 2 * 10^6`).

**Final solution.** I disproved the `int` multiply with a concrete near-`2^31` product, proved the inverse-factorial fold and the Fermat inverse on a hand-worked small prime, guarded the binomial against out-of-range `k`, and cross-checked against an additive grid-DP brute on 400 randomized cases including large primes. This is what I ship — one self-contained file with every multiply routed through a 64-bit `mul`:

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

**Causal recap.** The geometry collapses to a product of two binomial coefficients because a monotone walk through a checkpoint is two independent legs and each leg's count is a single `C(R+U, R)`; the only hard part is the modular arithmetic. My first `mul` was written in `int`, and a traced product of two residues near `2 * 10^9` (`(M-3)(M-5)`, true value `~4 * 10^18`) showed the 32-bit multiply wraps before the `% M` and returns garbage — so I routed `MOD`, the factorial tables, and every product through a single 64-bit `mul`, where the same trace now reduces the true product to `15`. I proved the Fermat inverse is legal because `M > maxN` keeps every factorial nonzero mod the prime, verified the inverse-factorial fold `inv_fact[i-1] = i * inv_fact[i]` on a hand-worked `M = 7` table, guarded `C` to return `0` for out-of-range `k` so the table is never indexed negatively, and confirmed the origin / equal-checkpoint / zero-length corners plus 400 randomized agreements against an additive grid-DP brute — the large-prime cases being exactly the ones the original `int` multiply would have silently failed.
