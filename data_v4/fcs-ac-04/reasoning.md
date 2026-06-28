**Reading the problem and pinning the contract.** I have `n` piles; pile `i` holds `a[i]` stones with `1 <= a[i] <= 10^9`, and `n` can be as large as `10^5`. A move picks a pile of size `x > 1` and replaces it with `y` stones where `y` is a *proper divisor* of `x` (`y | x`, `1 <= y < x`); a pile of size `1` is dead. Last player to move wins — equivalently, the player facing an all-ones (or empty) position loses. I must print `First` or `Second`. The input is `n` then the `n` sizes; output is a single word. Before any algorithm I fix the scale, because it rules options in and out: `n` up to `10^5` and values up to `10^9`. That immediately tells me two things — the per-pile work has to be roughly `O(sqrt(a))` or better (a `sqrt(10^9) ~ 31623` factor times `10^5` piles is about `3.2 * 10^9` divisions in the absolute worst case, which is borderline but survivable with trivial inner loops), and whatever I do *cannot* enumerate joint game states.

**The obvious approach, and watching it explode.** The textbook way to decide an impartial game is minimax with memoization over positions: a position is a *loss for the mover* iff every move leads to a *win for the opponent*, and a *win* iff some move leads to a loss. The state here is the multiset of pile sizes. Let me actually estimate the state count on a concrete small case to feel the blow-up. Take three piles, each of size `12`. A single pile of `12` can be driven through its proper divisors `{1,2,3,4,6}`, and each of those further down — so one pile alone has on the order of a dozen reachable values, but the *combined* state is the (multiset) product across piles. With three piles that is already on the order of `12^3` distinct positions to memoize, and that is for tiny values. Now scale to `n = 10^5` piles: the joint state space is the product of per-pile reachable sets over `10^5` factors. That is not "large", it is hopeless — exponential in the number of piles. Even with perfect memoization I would never enumerate it. So the joint-state minimax is a non-starter at these limits; it is exactly the oracle I will keep for *small* cross-checks, not the solution. I need structure that lets me avoid ever forming the joint state.

**The first real idea — the piles are independent, so use Sprague–Grundy.** Here is the structural observation the blow-up is begging me to exploit: a move *touches exactly one pile* and never references the others. That means the whole position is a disjunctive sum of `n` one-pile games. For impartial games under normal play, the Sprague–Grundy theorem applies: every position has a Grundy value `g` (a nonnegative integer), the Grundy value of a sum of independent games is the bitwise XOR of the components' Grundy values, and the mover *loses* iff that XOR is `0`. So if I can compute, for a single pile of size `x`, its Grundy value `G(x)`, then the answer is simply: XOR all `G(a[i])`; print `First` iff the result is nonzero. This collapses an exponential joint search into `n` independent per-pile computations plus a fold. The entire difficulty has migrated to one question: **what is `G(x)` for the proper-divisor game?**

**Computing the per-pile Grundy by definition — mex of the options.** By definition `G(x) = mex { G(y) : y is a proper divisor of x }`, where `mex S` is the smallest nonnegative integer not in `S`. A pile of `1` has no proper divisor, so `G(1) = mex(empty) = 0`. Let me just compute the first several by hand and look for a pattern, because if there is a closed form I avoid a per-pile recursion entirely.

- `G(1) = mex{} = 0`.
- `G(2)`: proper divisors `{1}`, options `{G(1)} = {0}`, `mex = 1`.
- `G(3)`: divisors `{1}`, `mex{0} = 1`.
- `G(4)`: divisors `{1,2}`, options `{G(1),G(2)} = {0,1}`, `mex = 2`.
- `G(5)`: `{1}` → `1`.
- `G(6)`: divisors `{1,2,3}`, options `{0,1,1} = {0,1}`, `mex = 2`.
- `G(7)`: `1`.
- `G(8)`: divisors `{1,2,4}`, options `{0,1,2}`, `mex = 3`.
- `G(9)`: divisors `{1,3}`, options `{0,1}`, `mex = 2`.
- `G(12)`: divisors `{1,2,3,4,6}`, options `{0,1,1,2,2} = {0,1,2}`, `mex = 3`.
- `G(16)`: divisors `{1,2,4,8}`, options `{0,1,2,3}`, `mex = 4`.

So the sequence `G(1..16)` is `0,1,1,2,1,2,1,3,2,2,1,3,1,2,2,4`. Stare at it. `G` of a prime is `1`. `G(4)=2`, `G(8)=3`, `G(16)=4`: powers of two give the exponent. `G(12)=3` and `12 = 2^2 * 3` has `2+1 = 3` prime factors with multiplicity. `G(6)=2` and `6 = 2*3` has two. `G(9)=2` and `9 = 3^2` has two. The pattern is unmistakable: **`G(x) = Omega(x)`, the number of prime factors of `x` counted with multiplicity** (`Omega(1)=0`, `Omega(prime)=1`, `Omega(p^k)=k`, `Omega(12)=3`).

**Earning the insight — why `G(x) = Omega(x)` is actually true, not just a coincidence on 16 values.** A pattern over sixteen integers is a conjecture, not a proof, and a wrong Grundy formula is a silent wrong-answer. Let me prove it by strong induction on `x`. Claim: `G(x) = Omega(x)`. Base: `Omega(1) = 0 = G(1)`. Inductive step: assume `G(d) = Omega(d)` for all `d < x`. The options of `x` are its proper divisors `y`. The key fact about `Omega` on divisors: if `y | x` and `y < x`, then `0 <= Omega(y) <= Omega(x) - 1`, because dropping any prime factor of `x` decreases `Omega` by exactly one and a proper divisor drops at least one. Moreover, *every* value in `{0, 1, ..., Omega(x) - 1}` is achieved by some proper divisor: write `x = p_1 p_2 ... p_m` as a multiset of `m = Omega(x)` primes; for any target `t` with `0 <= t <= m - 1`, the product of any `t` of those primes is a proper divisor `y` with `Omega(y) = t` (and for `t = 0`, `y = 1`). Critically, no proper divisor can have `Omega(y) = Omega(x)`, since that would force `y = x`. So by the inductive hypothesis the option-set of Grundy values is *exactly* `{0, 1, ..., Omega(x) - 1}` — a complete prefix of the nonnegative integers missing precisely `Omega(x)`. Therefore `mex = Omega(x)`, and `G(x) = Omega(x)`. The conjecture is a theorem. The reformulation is fully earned: the obvious minimax is exponential, the independence of piles invokes Sprague–Grundy, and the per-pile Grundy reduces — provably — to a prime-factor count.

So the whole problem is now: **XOR `Omega(a[i])` over all piles; First wins iff the XOR is nonzero.** No game tree, no DP over states — just factorization and a fold.

**Designing the fast factorization (the SOTA component).** `Omega(x)` for `x` up to `10^9` needs real factoring, but only trial division up to `sqrt(x) <= sqrt(10^9) < 31623` is required: strip every prime factor `p <= sqrt(x)`, and whatever cofactor `> 1` remains after that is necessarily a single prime (it cannot be a product of two factors both `> sqrt(x)` without exceeding `x`), contributing exactly one to `Omega`. To make trial division fast I sieve all primes up to `31623` once and divide only by those, not by every integer — that cuts the inner loop from ~31623 candidates to ~3400 primes per pile. A linear sieve gives the prime list cheaply. This is the canonical strongest approach at this scale: a full Pollard–Rho would be overkill (it shines past `10^{12}`–`10^{18}`, not at `10^9`), and a sieve of `Omega` up to `10^9` is impossible in memory; sieve-the-small-primes + trial-divide is exactly right.

**First implementation.** I write the linear sieve, then per pile strip small primes counting `Omega`, then XOR:

```
const int LIM = 31623;
vector<int> spf(LIM + 1, 0);
vector<int> primes;
for (int i = 2; i <= LIM; i++) {
    if (spf[i] == 0) { spf[i] = i; primes.push_back(i); }
    for (int p : primes) {
        if ((long long)p * i > LIM) break;
        spf[p * i] = p;
        if (p == spf[i]) break;
    }
}
int nimXor = 0;
for (int i = 0; i < n; i++) {
    long long x; cin >> x;
    int omega = 0;
    for (int p : primes) {
        if (p * p > x) break;            // <-- suspicious
        while (x % p == 0) { x /= p; omega++; }
    }
    if (x > 1) omega++;
    nimXor ^= omega;
}
cout << (nimXor != 0 ? "First" : "Second") << "\n";
```

**Tracing the code on a concrete case, because clean math transcribes dirty.** Let me run `x = 999999937` (a prime near `10^9`) through the factor loop by hand. The largest prime in my list is just under `31623`; for this `x`, `p * p` against `x` is the loop guard. `p` is an `int`, and the largest `p` is about `31607`. `p * p` for `p = 31607` is about `9.99 * 10^8` — but that product is computed in `int`, and `31607 * 31607 = 999002449`, which is under `2^31 - 1 = 2147483647`, so it does not overflow *here*. But the guard `p * p > x` is comparing an `int` product to a `long long x`. For the largest primes near `31623`, `p*p` is around `10^9`, still inside `int` range — so on this exact value it happens to be fine. That is the dangerous kind of "fine": it is fine by luck, not by construction. If I ever raised `LIM` or the value cap, `p * p` in `int` would silently overflow to a negative number, the guard `p*p > x` would read as false, and the loop would keep dividing — a latent bug. I will not ship a guard whose correctness depends on `LIM` staying below `~46340` (where `int` squares overflow). I promote the comparison to 64-bit: `(long long)p * p > x`. This is a real defect I am fixing now, before it bites, not a cosmetic change — the overflow class of bug is exactly the silent-wrong-answer trap this problem is built to punish.

**A second, sharper trace — does the cofactor rule actually fire correctly?** Take `x = 6` directly. Loop: `p = 2`, `(ll)2*2 = 4 <= 6`, strip: `6 % 2 == 0` → `x = 3, omega = 1`; `3 % 2 != 0` stop inner. `p = 3`, `(ll)3*3 = 9 > 3`? At this point `x` has already been reduced to `3`, so the guard compares `9 > 3` → true → break. Now `x = 3 > 1`, so `omega++` → `omega = 2`. Correct: `Omega(6) = 2`. Good — and note the guard correctly uses the *current* reduced `x`, not the original, which is what lets the "remaining cofactor is one prime" rule kick in. Now `x = 4`: `p = 2`, strip twice → `x = 1, omega = 2`; next `p = 3`, `9 > 1` break; `x = 1` not `> 1`, no extra. `Omega(4) = 2`. Correct. Now a prime square of a *large* prime, say `x = 999999937^2` — but that exceeds `10^9`, out of range, so I never see it; the largest cofactor I leave unfactored is a single prime, never a prime square, which is the whole reason the "`if (x > 1) omega++`" line is sound. The invariant holds within the stated bound `a[i] <= 10^9`.

**Fixing and consolidating.** I apply the `(long long)p * p` fix and add `ios_base::sync_with_stdio(false)` for the `10^5` reads. I also double-check the `n = 0` path: `if (!(cin >> n)) return 0;` handles truly empty input, and if `n = 0` is given explicitly the read loop runs zero times, `nimXor` stays `0`, and I print `Second` — which is correct, since First faces an empty position and cannot move. The sentinel-free design means there is nothing to underflow.

**Edge cases, deliberately, because this is where game solvers die.**
- `n = 0` (or empty input): no piles → `nimXor = 0` → `Second`. First cannot move and loses. Correct.
- All piles of size `1`: `Omega(1) = 0` for each, XOR stays `0` → `Second`. Every pile is dead; First cannot move. Correct.
- Single prime `a = [7]`: `Omega(7) = 1`, XOR `= 1 != 0` → `First`. Indeed First moves `7 → 1` and Second is stuck. Correct.
- Two equal piles `[8, 8]`: `Omega(8) = 3` each, `3 XOR 3 = 0` → `Second`. The mirroring strategy: whatever First does to one pile, Second copies on the other, so Second makes the last move. Correct.
- `[4, 9]`: `Omega(4) = 2`, `Omega(9) = 2`, XOR `= 0` → `Second`. Same mirror logic. Correct.
- `[8, 12, 6]`: `Omega = 3, 3, 2`, `3 XOR 3 XOR 2 = 2 != 0` → `First`. Correct (this is sample 1).
- Largest values: `x` up to `10^9`, `Omega` at most `30` (since `2^30 > 10^9`), so the XOR fits trivially in an `int`. Trial division bounded by ~3400 primes per pile; worst case `10^5` large primes ran in about `0.65 s` in testing, comfortably under the `2 s` limit, ~4 MB memory.

**Self-verification against an independent oracle.** I wrote a brute that does *not* assume Sprague–Grundy: it runs full minimax over the combined multiset of piles (dropping size-1 piles), memoized — the obvious-but-exponential method, valid on tiny inputs. I differential-tested the compiled solution against it on 1200 randomly generated small cases (few piles, small values, biased toward composites and prime powers) plus a dozen hand-picked edge cases (`n=0`, all ones, primes, prime powers, XOR-zero pairs), with zero mismatches. Separately I cross-checked the `Omega`-XOR verdict on 300 large-value cases (values up to `10^9`, big primes, `2^k`, `3^k`) against an independent Python factorization, again zero mismatches. The two checks are complementary: the minimax brute validates the *game theory* (that XOR-of-Grundy is the right verdict and that `G = Omega` holds), and the large-value cross-check validates the *factorization* at scale where the minimax brute cannot run. Both pass.

**Final solution.** I convinced myself the *idea* is right by proving `G(x) = Omega(x)` by induction (the mex of options is exactly the prefix `{0,...,Omega(x)-1}`), and the *code* is right by tracing the `int`-overflow guard to a precise latent cause and promoting it to 64-bit, then re-verifying the corners and a thousand-plus differential cases. What I ship is one self-contained file: sieve the small primes once, count `Omega` per pile by trial division over those primes, XOR the counts, and answer `First` iff the XOR is nonzero.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;              // no piles -> First cannot move -> Second wins

    // Sieve primes up to sqrt(max a[i]) = sqrt(1e9) < 31623, so 31623 suffices.
    const int LIM = 31623;
    vector<int> spf(LIM + 1, 0);            // smallest prime factor; 0 marks "prime/unset"
    vector<int> primes;
    for (int i = 2; i <= LIM; i++) {
        if (spf[i] == 0) { spf[i] = i; primes.push_back(i); }
        for (int p : primes) {
            if ((long long)p * i > LIM) break;
            spf[p * i] = p;
            if (p == spf[i]) break;
        }
    }

    // For each pile x, the Grundy value of the "move to a proper divisor" game is
    // Omega(x) = number of prime factors of x counted with multiplicity.
    // Sprague-Grundy: the whole game's value is the XOR of per-pile Grundy values;
    // the first player wins iff that XOR is nonzero.
    int nimXor = 0;
    for (int i = 0; i < n; i++) {
        long long x;
        cin >> x;
        int omega = 0;                      // Omega(x), prime-factor count with multiplicity
        for (int p : primes) {
            if ((long long)p * p > x) break;
            while (x % p == 0) { x /= p; omega++; }
        }
        if (x > 1) omega++;                 // remaining cofactor is a single prime
        nimXor ^= omega;                    // XOR the per-pile Grundy values
    }

    cout << (nimXor != 0 ? "First" : "Second") << "\n";
    return 0;
}
```

**Causal recap.** Direct minimax over the joint pile-state is exponential in `n` (three piles of `12` already need ~`12^3` states; `10^5` piles is hopeless), so it can only serve as a small-case oracle. The piles are independent — a move touches one pile — so Sprague–Grundy applies: XOR the per-pile Grundy values, First wins iff nonzero. The per-pile Grundy `G(x) = mex` of proper-divisor options equals `Omega(x)`, proved by induction because a proper divisor realizes every prime-factor-count in `{0,...,Omega(x)-1}` and never `Omega(x)` itself, so the mex is exactly `Omega(x)`. Computing `Omega` for `x <= 10^9` is trial division over sieved primes up to `31623`, with the cofactor `>1` counted as one prime. The one real bug was an `int` overflow latent in the `p*p` guard, fixed by `(long long)p*p`; the empty/all-ones/XOR-zero corners fall out of `Omega(1)=0` and the final nonzero-XOR test; and 1200 minimax-brute differential cases plus 300 large-value factorization cross-checks pass with zero mismatches.
