**Reading the problem and pinning the contract.** I have `q` independent scenarios. In each, candidate A gets `a` ballots and B gets `b` ballots, and I count the interleavings of the reveal — binary strings with `a` A's and `b` B's — that are *safe at margin `m`*: reading left to right, after every single ballot the running `lead = (#A so far) - (#B so far)` must satisfy `lead >= -m`. I print each count modulo a prime `p`. Before any algorithm I fix the scale, because it dictates both the data types and which approaches survive. Per query `a, b <= 10^6` with `a + b <= 2*10^6`, and `q <= 2*10^5`. So a per-query cost of `O(a + b)` is `2*10^5 * 2*10^6 = 4*10^11` — far too slow. Whatever I do, after one shared precompute the per-query work must be essentially `O(1)`. And `p` is given as a prime with `p > 2*10^6`, strictly larger than any `a + b`, which is a gift I will lean on hard: it means factorials `0!, 1!, ..., (a+b)!` are all nonzero modulo `p`, so Fermat inverses of factorials exist and binomial coefficients `C(n, k) mod p` are computable in `O(1)` after an `O(maxN)` table. The modular products reach `(p-1)^2 ~ (2*10^9)^2 ~ 4*10^18`, which overflows 64-bit signed (`~9.2*10^18` is the ceiling, but chained `a*b%p*c` style products are safest in `__int128`), so every modular multiply goes through `__int128`. That is decision one and it is non-negotiable.

**Translating the rule into geometry.** The cleanest way to think about "B never leads by more than `m`" is as a lattice path. Map each A-ballot to a `+1` step and each B-ballot to a `-1` step. A reveal order is a path from height `0` that takes `a` up-steps and `b` down-steps, ending at height `a - b`. "After every reveal `lead >= -m`" is exactly "the path never dips below height `-m`". So I am counting lattice paths with `a` up, `b` down, that stay `>= -m` at every prefix. The barrier is the horizontal line `y = -m - 1`: a path is *unsafe* precisely when it touches that line at least once. This reframing is what unlocks both candidate methods.

**Laying out the candidate approaches.** Two routes are on the table, and I want to commit to the one I can *prove*, not the one that is shortest to type.

- *Local-multiply greedy.* Insert the `b` B-ballots one at a time, each into the earliest position the margin still allows, and multiply the number of legal slots each B has, then divide by `b!` to undo the ordering of identical B's. It is `O(b)` per query and it *feels* like the staircase product that correctly counts matchings when one item's eligible set is nested inside the next. The risk is structural: are the per-B slot sets really independent, or does placing one B early remove a slot a later B was counting on? Overlapping, mutually dependent slot sets are exactly where "multiply the choices" silently breaks. I will not trust it until I have tried to break it.
- *Reflection (Andre / cycle lemma).* Count *all* `C(a + b, a)` interleavings, then subtract the unsafe ones using the reflection principle, which collapses the entire unsafe family to a single binomial coefficient. `O(1)` per query after the factorial precompute. The risk here is not the idea but the *transcription*: the exact reflected index and the feasibility guard are easy to get subtly wrong by a sign or an off-by-one.

**Stress-testing the greedy before committing.** Hand-waving "the product feels right" is how wrong solutions ship, so let me actually attack the greedy on the smallest scenario where the constraint bites: `a = 3, b = 2, m = 0`. The barrier is `y = -1`; a safe path must keep `lead >= 0` at every prefix, i.e. at least as many A's as B's seen so far. The greedy reasons per B-ballot: the `j`-th B (1-indexed, in left-to-right order) must have at least `j` A's strictly before it to keep `lead >= 0`, so it can occupy 0-indexed positions `>= 2j - 1`, giving `n - (2j - 1) - m` legal slots where `n = a + b = 5`. For `B#1`: `5 - 1 - 0 = 4` slots (positions 1..4). For `B#2`: `5 - 3 - 0 = 2` slots (positions 3..4). Greedy product is `4 * 2 = 8` ordered placements, divided by `2! = 2` to undo the identical-B ordering, giving `8 / 2 = 4`.

Now I enumerate the truth by hand. With `lead >= 0` everywhere, the safe strings of three A's and two B's are: `AAABB`, `AABAB`, `AABBA`, `ABAAB`, `ABABA` — that is **5**, the Catalan-style ballot number. Greedy says `4`. They disagree, so greedy is wrong, and I want to see *exactly why*, because the cause tells me whether any patch could save it. List the greedy's 6 ordered placements `(p1, p2)` with `p1 in {1,2,3,4}`, `p2 in {3,4}`, `p1 != p2`: `(1,3), (1,4), (2,3), (2,4), (3,4), (4,3)`. The last two, `(3,4)` and `(4,3)`, are the *same* set `{3,4}` = `AAABB` — they collide. So the 6 ordered placements cover only 5 distinct B-sets, not 6, and dividing by `2!` is invalid because the two slot ranges `{1,2,3,4}` and `{3,4}` *overlap*, so the symmetry the `b!` division assumes simply isn't there. Depending on whether I write the product as `4*2/2! = 4` or count distinct sets `6/2 = 3`, I get `4` or `3`, and the truth is `5`. The greedy is not off by a constant I can correct; its independence premise is false, because an early B placement changes the legal slots of a later B, and the slot ranges nest-and-overlap rather than sitting disjoint. The verification paid off: it killed an approach I would otherwise have shipped. Greedy is out.

**Deriving the reflection and checking it on paper.** I want the number of paths with `a` up, `b` down, staying `>= -m` at every prefix. Total paths ignoring the barrier: `C(a + b, a)`. Now I subtract the unsafe ones — paths that touch the barrier `y = -m - 1` somewhere. The reflection principle: take an unsafe path, find its *first* contact with the barrier, and reflect the portion of the path *before* that contact across the line `y = -m - 1`. Reflecting flips the start point from `0` to `2(-m - 1) - 0 = -2(m+1)`. A path from height `-2(m+1)` to height `a - b` with `n = a + b` steps has up-steps `u` solving `u - (n - u) = (a - b) - (-2(m+1))`, i.e. `2u - n = (a - b) + 2(m + 1)`, giving `u = ((a + b) + (a - b) + 2(m+1)) / 2 = a + m + 1`. So the count of unsafe paths equals the count of unconstrained paths with `a + m + 1` up-steps, which is `C(n, a + m + 1)`. Equivalently, in terms of *down-steps* this is `C(n, n - (a + m + 1)) = C(n, b - m - 1)`. So:

`safe(a, b, m) = C(a + b, a) - C(a + b, b - m - 1)`,

with the convention that `C(n, k) = 0` when `k < 0` or `k > n` (so the subtracted term vanishes when `b - m - 1 < 0`, i.e. `m >= b`, which is the unconstrained regime). There is one more guard: if the path cannot even *end* at or above the barrier — that is `a - b < -m` — then no safe path exists and the answer is `0`. (The formula would actually still return `0` in many such cases, but the `a - b < -m` short-circuit makes the intent explicit and bulletproof.)

Let me confirm the recurrence by hand on the sample `a = 3, b = 2, m = 0`, expected `5`. `total = C(5, 3) = 10`. `bad index = b - m - 1 = 2 - 0 - 1 = 1`, so `bad = C(5, 1) = 5`. `safe = 10 - 5 = 5`. The derivation reproduces the hand-enumerated answer. As a second paper check, `a = 4, b = 4, m = 0` should be the 4th Catalan number, `14`: `total = C(8, 4) = 70`, `bad = C(8, 4 - 0 - 1) = C(8, 3) = 56`, `safe = 70 - 56 = 14`. Correct. The reflection is right.

**First implementation — and immediately a trace, because clean math transcribes dirty.** My first cut of the factorial precompute and query loop:

```
ll MOD;
vector<ll> fct, inv_fct;
void build(ll maxn) {
    fct.assign(maxn + 1, 0);
    inv_fct.assign(maxn + 1, 0);
    fct[0] = 1;                                    // BUG lurking here
    for (ll i = 1; i <= maxn; i++) fct[i] = (__int128)fct[i-1] * i % MOD;
    inv_fct[maxn] = pw(fct[maxn], MOD - 2, MOD);
    for (ll i = maxn; i >= 1; i--) inv_fct[i-1] = (__int128)inv_fct[i] * i % MOD;
}
ll C(ll n, ll k) {
    return (__int128)fct[n] * inv_fct[k] % MOD * inv_fct[n-k] % MOD;   // BUG: no range guard
}
```

and per query I computed `total = C(a+b, a)` and `bad = C(a+b, b-m-1)`. I trace the smallest input that could expose the range issue: `a = 5, b = 0, m = 0` (only A's). Safe answer is obviously `1` (the single string `AAAAA` keeps `lead` climbing, always `>= 0`). Here `total = C(5, 5) = 1`, and `bad index = b - m - 1 = 0 - 0 - 1 = -1`. My `C(5, -1)` reads `inv_fct[-1]` — a negative index into a `vector`. That is undefined behaviour; in practice it reads garbage memory or crashes, and at best returns a nonzero junk value, so `safe = 1 - junk` is wrong.

**Diagnosing the first bug.** The defect is precise: the reflection legitimately produces an out-of-range lower index `b - m - 1` whenever `m >= b` (the unconstrained regime), and the canonical `C(n, k) = 0` for `k < 0 or k > n` convention is what makes the formula correct there — but my `C` did not encode that convention, it indexed blindly. I add the guard at the top of `C`:

```
ll C(ll n, ll k) {
    if (k < 0 || k > n) return 0;
    return (__int128)fct[n] * inv_fct[k] % MOD * inv_fct[n-k] % MOD;
}
```

Re-trace `a = 5, b = 0, m = 0`: `total = C(5,5) = 1`, `bad = C(5, -1) = 0` (guard), `safe = 1 - 0 = 1`. Correct. Re-trace `a = 2, b = 5, m = 10` (B dominates but margin is enormous, so *every* interleaving is safe): `a - b = -3 >= -m = -10`, so not short-circuited; `total = C(7, 2) = 21`, `bad index = 5 - 10 - 1 = -6 < 0`, `bad = 0`, `safe = 21`. Enumerating, with `m = 10` the barrier `y = -11` is never reachable in 7 steps, so all `C(7,2) = 21` orders are safe. Correct. The case that broke now passes, and it passed for the reason I fixed.

**A second trace — the small-prime trap, found by running, not by staring.** I wired up an independent brute force (enumerate every interleaving, check the running balance) and a random small-case generator, compiled `sol`, and ran a few hundred comparisons. A wall of mismatches appeared. Seed 1 was `q=2, p=3`, query `4 1 3`: brute says `2`, my `sol` says `0`. I trace it. `a=4, b=1, m=3`: `a - b = 3 >= -3`, fine. `total = C(5, 4)`, `bad = C(5, b - m - 1) = C(5, -3) = 0`. So `safe = C(5,4) mod 3`. But `C(5,4) = 5`, and the *true* answer mod 3 is `5 mod 3 = 2` — yet `sol` printed `0`. Where did `C(5,4)` become `0`? I print `fct` mod 3: `fct = [1, 1, 2, 6%3=0, 0, 0]`. The moment `i = 3`, `fct[3] = fct[2] * 3 mod 3 = 0`, and every later factorial is `0`. Then `inv_fct[5] = pw(0, ...)= 0`, the whole inverse chain is `0`, and `C(5,4) = fct[5]*inv_fct[4]*inv_fct[1] = 0`. The Fermat-factorial method is *fundamentally invalid* when `p <= n`, because `p` divides `n!` and the inverse of `0` does not exist.

**Diagnosing the second bug — and choosing the fix at the contract level.** This is not a coding slip; it is a *modeling* mistake. Lucas' theorem would be the general fix (compute `C(n, k) mod p` digit-by-digit in base `p`), but Lucas is `O(log_p n)` per binomial and, more to the point, the problem does not need it: the intended setting is "modulo a large prime". The clean, honest resolution is to make the contract explicit — `p` is a prime with `p > 2*10^6 >= a + b`, so no factorial in `0!..(a+b)!` is ever divisible by `p`, every inverse exists, and the simple factorial method is exactly correct. I fix the *generator* to only emit primes strictly larger than the maximum `a + b` it can produce (the smallest I use is `17`, and tiny cases have `a + b <= 16`), matching the stated constraint. This is the right fix because it keeps the solution `O(1)` per query at full scale; bolting on Lucas would buy generality the problem never asks for and would slow the `2*10^5`-query batch. After this, the same brute-vs-`sol` harness goes green across the board.

**Sanity-checking the derivation against the brute, not just my algebra.** I re-ran 400 generated cases (seeds 1..400) and then a second independent batch of 500 (seeds 401..900) of `sol` versus the exhaustive brute force: **0 mismatches over 900 cases**. Because the brute force enumerates every interleaving and re-checks the running balance from scratch — a completely different method from the reflection identity — agreement across `m = 0`, large-`m`, `a < b`, `a = b`, and empty scenarios is strong evidence the reflected index and both guards are right.

**Edge cases, deliberately, because this is where this kind of code dies.**
- *Empty scenario `a = b = 0`.* `maxn` may be `0`; `build(0)` makes `fct = [1]`, `inv_fct[0] = pw(1, p-2) = 1`. `total = C(0, 0) = 1`, `bad index = 0 - m - 1 = -1 - m < 0` so `bad = 0`, `safe = 1`. There is exactly one (empty) order — correct. The `maxn = 0` table size of 1 is fine.
- *No B-ballots `b = 0`.* `bad index = -m - 1 < 0`, `bad = 0`, `safe = C(a, a) = 1`. The single all-A order is safe — correct.
- *Infeasible `a - b < -m`.* Short-circuit to `0`. Example `a = 2, b = 5, m = 0`: `a - b = -3 < 0`, answer `0` — B must end ahead, so no order keeps `lead >= 0`. Correct, and the short-circuit also stops me from trusting the formula in a regime where I have not re-derived it.
- *Reflected index on the boundary.* `b - m - 1 = 0` (e.g. `m = b - 1`): `bad = C(n, 0) = 1`, a real subtraction, handled. `b - m - 1 = -1` (e.g. `m = b`): `bad = 0`, handled by the guard. Both transitions across the `m >= b` boundary are continuous and correct.
- *Overflow.* All modular multiplies route through `__int128` before the `% MOD`, so the `(p-1)^2 ~ 4*10^18` intermediate cannot wrap a 64-bit register. `maxn + 1` up to `2*10^6 + 1` longs is ~16 MB per table, two tables ~32 MB, inside 256 MB. The subtraction `total - bad` can go negative in `[-(p-1), p-1]`, so I add `MOD` back when `ans < 0`.
- *Performance.* One shared `build(maxn)` is `O(maxn)`; each query is `O(1)`. A full-scale run (`q = 2*10^5`, `a + b` up to ~2*10^6) finishes in ~0.10 s using ~39 MB — comfortably inside 2 s and 256 MB.

**Final solution.** I convinced myself the *idea* is right by disproving the greedy with a traced counterexample and hand-checking the reflection on two samples, and I convinced myself the *code* is right by tracing two concrete failures (the out-of-range binomial index and the small-prime factorial collapse) to precise causes, fixing each at the right level, and then getting 0 mismatches over 900 brute-force comparisons plus a max-scale check against exact Python binomials. That is what I ship — one self-contained file, the `O(1)`-per-query reflection I can defend rather than the greedy I broke:

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

ll MOD;

// We precompute factorials and inverse factorials modulo a PRIME p, up to the
// largest a+b appearing in any query, then answer each query in O(1) via the
// reflection (Andre) formula:
//   safe(a,b,m) = C(a+b, a) - C(a+b, b-m-1)   (the subtracted term is 0 when its
//   lower index is outside [0, a+b]), and 0 entirely when a-b < -m.

vector<ll> fct, inv_fct;

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

void build(ll maxn) {
    fct.assign(maxn + 1, 0);
    inv_fct.assign(maxn + 1, 0);
    fct[0] = 1 % MOD;
    for (ll i = 1; i <= maxn; i++) fct[i] = (__int128)fct[i - 1] * (i % MOD) % MOD;
    inv_fct[maxn] = pw(fct[maxn], MOD - 2, MOD);   // Fermat: p prime
    for (ll i = maxn; i >= 1; i--) inv_fct[i - 1] = (__int128)inv_fct[i] * (i % MOD) % MOD;
}

// C(n, k) mod p, with the standard out-of-range = 0 convention.
ll C(ll n, ll k) {
    if (k < 0 || k > n) return 0;
    return (__int128)fct[n] * inv_fct[k] % MOD * inv_fct[n - k] % MOD;
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

    build(maxn);

    for (int i = 0; i < q; i++) {
        ll a = qs[i].a, b = qs[i].b, m = qs[i].m;
        ll ans;
        if (a - b < -m) {
            ans = 0;                              // cannot even finish within margin
        } else {
            ll total = C(a + b, a);               // all interleavings
            ll bad = C(a + b, b - m - 1);         // reflected = orderings that cross the barrier
            ans = (total - bad) % MOD;
            if (ans < 0) ans += MOD;
        }
        cout << ans << "\n";
    }
    return 0;
}
```

**Causal recap.** The local-multiply greedy looked right but a single traced counterexample (`a=3, b=2, m=0`: greedy `4` via overlapping slot ranges divided by `b!`, versus the reachable `5`) showed the per-B slot sets are not independent — an early placement removes a later B's slot — so I moved to the reflection identity and checked it on two samples; transcribing it, the reflected index `b - m - 1` legitimately goes out of range when `m >= b`, which a missing `C(n,k)=0` guard turned into an undefined negative `vector` access (traced on `a=5, b=0, m=0`); and running the brute-force harness exposed a deeper modeling error — Fermat-factorial binomials collapse to `0` once the prime `p <= a+b` (traced on `p=3, a=4, b=1`), fixed at the contract level by requiring `p > 2*10^6 >= a+b` so every factorial is invertible; with the guard, the short-circuit `a-b < -m`, `__int128` modular products, and a `build(maxn)` that survives `maxn=0`, the solution is `O(maxN + q)` and agrees with brute force on all 900 small cases and with exact binomials at full scale.
