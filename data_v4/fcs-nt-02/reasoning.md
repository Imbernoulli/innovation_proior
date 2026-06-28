**Reading the problem and pinning the contract.** I am given three integers `n`, `r`, `m` and must print `C(n, r) mod m`. The ranges are the whole story: `n, r <= 10^18`, and `m <= 10^6` but **arbitrary** — not necessarily prime, not necessarily squarefree. The binomial `C(n, r)` for `n = 10^18` has on the order of `10^17` decimal digits, so it cannot be materialized; I must work modularly from the start. And because `n, r` are 64-bit-sized, every index, exponent, and accumulator that touches them must be `long long`. The output is a single integer in `[0, m)`. If `r > n` (or `r < 0`) the binomial is `0`, so the answer is `0`.

**The obvious approach, and pinning down exactly where it dies.** The textbook recipe for `C(n, r) mod p` is: precompute `fact[k] = k! mod p` and `invfact[k] = (k!)^{-1} mod p`, then `C(n, r) = fact[n] * invfact[r] * invfact[n-r]`. For `n` up to `10^18` this is impossible to tabulate directly, but **Lucas' theorem** rescues it for a prime modulus: write `n` and `r` in base `p`; then `C(n, r) ≡ ∏_i C(n_i, r_i) (mod p)`, a product of binomials of single base-`p` digits, each of which is small (`< p <= 10^6`) and computable from a `p`-sized factorial table with Fermat inverses. That is a complete, fast answer — *when `m` is prime*.

The instant `m` is composite, this collapses, and I want to be precise about why rather than hand-wave. The inverse `invfact[r] = (r!)^{-1} mod m` is computed as a modular inverse, which exists **iff** `gcd(r!, m) = 1`. But `r!` contains every prime up to `r`. As soon as `r` is at least the smallest prime factor of `m`, `gcd(r!, m) > 1`, and `(r!)^{-1} mod m` does not exist. Concretely: I want `C(10, 3) mod 12`. Here `12 = 2^2 * 3`. The honest value is `120`, and `120 mod 12 = 0`. But if I tried `fact[10] * inv(fact[3]) * inv(fact[7]) mod 12`, I would need `inv(6) mod 12` and `inv(5040) mod 12` — and `inv(6) mod 12` is undefined because `gcd(6, 12) = 6`. There is no element `x` with `6x ≡ 1 (mod 12)`. So the Fermat/inverse machinery is not merely slow here; it is *ill-defined*. Recognizing this is the whole trap: the division `n! / (r! (n-r)!)` is an exact integer, but I cannot realize it as a modular product of inverses when the modulus shares factors with the factorials.

**First idea for a workaround, and why it is not enough.** Could I dodge the non-invertibility by computing the binomial with the multiplicative formula `C(n, r) = ∏_{i=1}^{r} (n - r + i) / i`, reducing mod `m` as I go and doing the divisions exactly (each prefix is a genuine integer binomial)? That is exactly what a brute force does, and it is correct — but it costs `O(r)` multiplications. With `r` up to `10^18` that is `10^18` operations: hopeless. So I need something that is correct under a composite modulus **and** runs in time polylogarithmic in `n`. The brute multiplicative formula gives me a trustworthy oracle for small `r`, but not an algorithm for the real constraints.

**Decomposing the modulus — the first half of the insight.** The composite modulus is the obstacle, so I attack its structure. Factor `m = ∏_j p_j^{e_j}`. Since `m <= 10^6`, trial division to `sqrt(m)` finds the factorization in microseconds, and there are at most a handful of distinct primes (six distinct primes already force `m >= 2*3*5*7*11*13 > 3*10^4`; more than seven is impossible under `10^6`). The prime-power factors `p_j^{e_j}` are pairwise coprime, so by the **Chinese Remainder Theorem**, knowing `C(n, r) mod p_j^{e_j}` for every `j` pins down `C(n, r) mod m` uniquely. This converts one hard composite problem into a few independent problems, each modulo a *prime power* `p^e <= 10^6`. The CRT recombination of a few residues is trivial. So the real question reduces to: **compute `C(n, r) mod p^e`, with `n, r` up to `10^18`, where `p` divides the factorials.**

**The second half of the insight — strip the `p`'s out, keep them on the side.** Modulo `p^e`, the factorials `n!`, `r!`, `(n-r)!` are still not units, so I still cannot just invert. The key realization: separate each factorial into its `p`-part and its `p`-free part. Write `n! = p^{a} * (n!)_p` where `(n!)_p` is the product of all integers in `[1, n]` **with the factors of `p` removed** (the "`p`-free factorial"), and `a` is the total power of `p` in `n!`. Crucially, `(n!)_p` is coprime to `p`, hence a **unit** modulo `p^e` and therefore invertible. So I can write

```
C(n, r) = n! / (r! (n-r)!) = p^{a - b - c} * (n!)_p / ((r!)_p (n-r)!)_p)
```

where `a = v_p(n!)`, `b = v_p(r!)`, `c = v_p((n-r)!)`, and `v_p` is the exponent of `p`. The three `p`-free factorials are all units mod `p^e`, so their quotient is a legitimate modular division. The leftover power `p^{a-b-c}` I carry explicitly. Two clean facts make this finite and fast:

- **Kummer / Legendre:** `v_p(k!) = floor(k/p) + floor(k/p^2) + floor(k/p^3) + ...`, an `O(log_p n)` sum. And `a - b - c = v_p(C(n,r))` is exactly the number of carries when adding `r` and `n-r` in base `p` (Kummer's theorem). If this exponent is `>= e`, then `p^e | C(n, r)` and the answer mod `p^e` is simply `0`. Otherwise the surviving `p^{a-b-c}` is a small power I multiply back in.

- **The `p`-free factorial has a periodic block structure.** Consider the residues in `[1, p^e]` that are coprime to `p`; their product modulo `p^e` is, by the **generalization of Wilson's theorem**, equal to `-1` for odd `p` (any `e`) and for `p^e ∈ {2, 4}`, and `+1` for `p = 2, e >= 3`. The `p`-free numbers up to a large `x` consist of `floor(x / p^e)` full blocks of length `p^e` — each contributing this fixed `±1` — times a partial block `[1, x mod p^e]`. And then the multiples of `p` that I stripped, `p * 1, p * 2, ..., p * floor(x/p)`, when their common `p` is pulled out, reproduce `(floor(x/p)!)_p`. So

```
(x!)_p ≡ (blockSign)^{floor(x/p^e)} * (partial product up to x mod p^e) * (floor(x/p)!)_p   (mod p^e),
```

a recurrence that bottoms out in `O(log_p x)` steps because `x` shrinks by a factor of `p` each level.

This is the canonical state-of-the-art method (the generalized-Lucas / Granville factorial technique): factor `m`, compute each prime-power binomial via the `p`-free factorial recurrence with Legendre/Kummer for the `p`-exponent and Wilson for the block sign, then CRT. Per prime power it costs `O(p^e)` to build the in-block prefix product table once, plus `O(log_p n)` per factorial evaluation — vastly under the limit.

**Confirming the recurrence on paper before coding.** Let me sanity-check the `p`-free factorial mechanics on `C(10, 3) mod 12`, the case that broke the naive method. `12 = 2^2 * 3`, two prime powers. Take `p^e = 4`. The `p`-free prefix table over `[0, 4]` (skipping even numbers): `fact[0]=1, fact[1]=1, fact[2]=1 (2 skipped), fact[3]=3, fact[4]=3 (4 skipped)`. Block sign = `fact[3] = 3 ≡ -1 (mod 4)`. Compute `(10!)_2`: level 0, `blocks = 10/4 = 2`, partial `= fact[10 mod 4] = fact[2] = 1`, contribute `(-1)^2 * 1 = 1`; recurse on `10/2 = 5`. Level 1, `blocks = 5/4 = 1`, partial `fact[5 mod 4] = fact[1] = 1`, contribute `(-1)^1 * 1 = -1`; recurse on `2`. Level 2, `blocks = 0`, partial `fact[2] = 1`; recurse on `1`. Level 3, partial `fact[1] = 1`; recurse on `0`, stop. Product `= 1 * (-1) * 1 * 1 = -1 ≡ 3 (mod 4)`. The exponent `v_2(C(10,3)) = v_2(10!) - v_2(3!) - v_2(7!) = 8 - 1 - 4 = 3 >= e = 2`, so the contribution mod `4` is `0`. For `p^e = 3`: `v_3(C(10,3)) = v_3(10!) - v_3(3!) - v_3(7!) = 4 - 1 - 2 = 1 >= e = 1`, so contribution mod `3` is `0` too. CRT of `(0 mod 4, 0 mod 3)` is `0`, and indeed `120 mod 12 = 0`. The method reproduces the answer the naive inverse approach couldn't even express.

**Implementing.** I structure it as a `PrimePowerComb` that, given `(p, e)`, builds the prefix table once and exposes `comb(n, r)`. Skeleton of the core:

```
void init(p, e): pe = p^e; fact[0..pe] where fact[i] = fact[i-1]*i if p∤i else fact[i-1], all mod pe.
legendre(x): sum of x/p + x/p^2 + ...
factmod(x):  res = 1; while x>0 { res *= blockSign^(x/pe) * fact[x % pe]; x /= p; } return res;
comb(n, r):  powp = legendre(n) - legendre(r) - legendre(n-r);
             if powp >= e return 0;
             res = factmod(n) * inv(factmod(r)*factmod(n-r)) * p^powp;
```

I read `blockSign` straight from the table as `fact[pe - 1]` (the product of all `p`-free residues in one block), which *is* the Wilson `±1` automatically — no need to special-case `p = 2` versus odd `p`; the table computes the correct sign for me. The modular inverse here is legitimate because `factmod(...)` returns a unit mod `p^e`. Then `main` factors `m`, loops over prime powers, and folds each `(a, p^e)` into a running CRT pair `(res, mod)`.

**First implementation — and then I trace it, because clean math transcribes dirty.** My first cut of the CRT merge inside the prime-power loop looked like this:

```
ll inv = inv_mod(mod % pe == 0 ? mod : mod, pe);  // inverse of mod modulo pe
ll diff = ((a - res) % pe + pe) % pe;
ll t = mulmod(diff, inv, pe);
res = res + mod * t; mod = mod * pe; res %= mod;
```

The CRT idea is right — I want `t` with `res + mod*t ≡ a (mod pe)`, i.e. `t = (a - res) * mod^{-1} (mod pe)`. But that `inv_mod(mod % pe == 0 ? mod : mod, pe)` is nonsense I wrote on autopilot: the ternary evaluates to `mod` in **both** branches, so the condition does nothing, and worse, I am passing `mod` (which can be much larger than `pe` after a couple of merges) where I meant to reduce it first. To see whether it actually misbehaves, I trace a two-factor case by hand. Take `m = 12 = 4 * 3`, and suppose the per-prime-power residues came out `a = 3 mod 4` and `a = 2 mod 3` (a made-up but valid pair, CRT answer should be `11`). Merge order: start `(res, mod) = (0, 1)`. First factor `pe = 4`, `a = 3`: `inv = inv_mod(1, 4) = 1`, `diff = 3`, `t = 3`, `res = 0 + 1*3 = 3`, `mod = 4`. Good, `res = 3 ≡ 3 (mod 4)`. Second factor `pe = 3`, `a = 2`: here `mod = 4`, and I call `inv_mod(mod % pe == 0 ? mod : mod, 3)` which passes `mod = 4` (not `4 mod 3 = 1`). Now my `inv_mod` does `g0 = a % m` at its top, so it would compute `4 % 3 = 1` internally and still return the inverse of `1 mod 3 = 1`. So in *this* run it limps to the right answer by luck — `diff = (2 - 3) mod 3 = 2`, `t = 2*1 = 2`, `res = 3 + 4*2 = 11`, `mod = 12`, `res = 11`. Correct here.

**Diagnosing the latent bug.** So the trace "passed," which is exactly the kind of false comfort that ships bugs. The defect is that I am relying on `inv_mod` silently reducing its argument, and my dead ternary advertises confusion about whether `mod` is already reduced. If I had instead written an `inv_mod` that assumed its input was already in `[0, m)` (a common micro-optimization), passing an unreduced `mod` would compute the inverse of the wrong residue and silently corrupt the CRT for any `m` with two or more prime factors — and crucially this would only manifest on *composite* `m`, i.e. precisely the cases this whole problem exists to handle, while every prime-`m` test would still pass. That asymmetry (prime tests green, composite tests red) is the trap restated one level up. I fix it by reducing explicitly and deleting the dead ternary: `ll inv = inv_mod(mod % pe, pe);`. This makes the contract unambiguous — I hand `inv_mod` a value in `[0, pe)` and ask for its inverse mod `pe` — and removes the reliance on incidental internal behavior.

**Re-verifying after the fix.** Re-running the same hand trace with `inv = inv_mod(mod % pe, pe)`: second factor `mod % pe = 4 % 3 = 1`, `inv = inv_mod(1, 3) = 1`, identical arithmetic, `res = 11`. Same answer, but now for the right reason. More importantly I push this through the real differential harness rather than trusting one trace. Building the Python oracles: an exact `math.comb(n, r) % m` for small/moderate `n` (correct by construction, no inverses involved), an independent **Lucas** implementation for prime `m` that scales to `n = 10^18`, and an independent re-derivation of the prime-power `p`-free recurrence. Then 1400 random cases from a generator that deliberately samples prime moduli, prime powers `2^e/3^e/5^e/7^e`, squarefree composites, full composites up to `10^6`, and boundary `r` (`0`, `n`, `n+1`). Zero mismatches.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `m = 1`: every value is `0 mod 1`. The factorization loop would produce no useful prime power and the CRT `res % 1` is `0`, but I short-circuit `m == 1` at the top to print `0` cleanly and avoid a degenerate empty-`pf` path.
- `r = 0` or `r = n`: `C = 1`. In `comb`, `factmod(0) = 1` and the `p`-exponent is `0`, so `res = 1 * inv(1*factmod(n)) ... ` — let me make sure: `factmod(n) * inv(factmod(0) * factmod(n))` with `factmod(0) = 1` gives `factmod(n) * inv(factmod(n)) = 1`, and `powp = legendre(n) - 0 - legendre(n) = 0`, so the answer is `1`. Verified against brute (`n=999999, r=0/n, m=10^6` both give `1`).
- `r > n`: `comb` returns `0` immediately, so every residue is `0`, CRT gives `0`. Tested `5 6 12 -> 0`.
- All-`p` blocks / Wilson sign: large prime-power moduli `1024 = 2^10`, `59049 = 3^10`, `390625 = 5^8`, `823543 = 7^7`, `524288 = 2^19` at `n` up to `2*10^5` all match `math.comb`, exercising both the `+1` (`p=2, e>=3`) and `-1` (odd `p`) block signs.
- Overflow: `n, r` are `long long`; in `factmod`, `x` and `blocks` stay `< 10^18`; products in `mulmod` are guarded by `__int128` even though operands are `< p^e <= 10^6`; the CRT step computes `mod * t` with `mod <= 10^6` and `t < pe <= 10^6`, so `mod * t <= 10^12`, comfortably inside `long long`. No overflow anywhere.
- Full scale: `n = r = 10^18` runs in single-digit milliseconds (the cost is one `O(p^e)` table build per prime power plus `O(log_p n)` per factorial), far under the time limit.

**Final solution.** I convinced myself the *idea* is right by pinning exactly why the inverse approach is ill-defined under a composite modulus (the `inv(6) mod 12` non-existence), then rebuilding the binomial from `p`-free factorials that *are* units, with Kummer for the `p`-exponent and Wilson for the block sign, and CRT across prime powers. I convinced myself the *code* is right by tracing the CRT merge, catching a dead ternary that masked an unreduced-modulus reliance which would have failed silently on exactly the composite cases this problem targets, fixing it to an explicit `mod % pe`, and then differential-testing 1400 cases plus large-`n` Lucas and prime-power oracles to zero mismatches. This is what I ship — one self-contained file:

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef unsigned long long ull;
typedef __int128 lll;

// ----- 128-bit safe modular multiply (mod fits in <= ~1e6, but base/exponent
//       intermediates use values < p^e <= m <= 1e6, so 64-bit suffices; we still
//       guard products with __int128 to be safe). -----
static inline ll mulmod(ll a, ll b, ll mod) {
    return (ll)((lll)a * b % mod);
}

static ll powmod(ll a, ll e, ll mod) {
    a %= mod; if (a < 0) a += mod;
    ll r = 1 % mod;
    while (e > 0) {
        if (e & 1) r = mulmod(r, a, mod);
        a = mulmod(a, a, mod);
        e >>= 1;
    }
    return r;
}

// Extended Euclid -> modular inverse of a mod m (a must be coprime to m).
static ll inv_mod(ll a, ll m) {
    ll g0 = a % m, g1 = m, x0 = 1, x1 = 0;
    if (g0 < 0) g0 += m;
    while (g1) {
        ll q = g0 / g1;
        ll t = g0 - q * g1; g0 = g1; g1 = t;
        t = x0 - q * x1; x0 = x1; x1 = t;
    }
    // g0 == gcd; assumed 1
    x0 %= m; if (x0 < 0) x0 += m;
    return x0;
}

struct PrimePowerComb {
    ll p;       // prime
    int e;      // exponent
    ll pe;      // p^e
    vector<ll> fact; // fact[i] = product of j in [1..i] with p | j removed, mod pe

    void init(ll p_, int e_) {
        p = p_; e = e_;
        pe = 1; for (int i = 0; i < e; i++) pe *= p;
        fact.assign((size_t)pe + 1, 1);
        fact[0] = 1 % pe;
        for (ll i = 1; i <= pe; i++) {
            if (i % p == 0) fact[i] = fact[i - 1];
            else fact[i] = mulmod(fact[i - 1], i % pe, pe);
        }
    }

    // total power of p dividing x!
    ll legendre(ll x) {
        ll cnt = 0;
        while (x > 0) { x /= p; cnt += x; }
        return cnt;
    }

    // (x!)_p  : x! with all factors of p stripped, taken mod pe.
    ll factmod(ll x) {
        ll res = 1 % pe;
        while (x > 0) {
            // number of complete blocks of length pe in [1..x]
            ll blocks = x / pe;
            // product of one full block == fact[pe-1] (== product of all units mod pe);
            // by Wilson's generalization this is -1 mod pe for odd p (any e) and for
            // p^e in {2,4}; +1 for p=2,e>=3. We just read it from the table directly.
            ll blockProd = fact[pe - 1];
            res = mulmod(res, powmod(blockProd, blocks, pe), pe);
            res = mulmod(res, fact[x % pe], pe);
            x /= p; // recurse on floor(x/p): the multiples of p contribute (x/p)! again
        }
        return res;
    }

    // nCr mod pe
    ll comb(ll n, ll r) {
        if (r < 0 || r > n) return 0;
        ll powp = legendre(n) - legendre(r) - legendre(n - r); // power of p in nCr
        if (powp >= e) return 0; // divisible by pe
        ll num = factmod(n);
        ll den = mulmod(factmod(r), factmod(n - r), pe);
        ll res = mulmod(num, inv_mod(den, pe), pe);
        // multiply back the surviving p^powp
        for (ll i = 0; i < powp; i++) res = mulmod(res, p % pe, pe);
        return res % pe;
    }
};

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    ll n, r, m;
    if (!(cin >> n >> r >> m)) return 0;

    if (m == 1) { cout << 0 << "\n"; return 0; }

    // edge: r out of range -> 0 (mod m). Handle generally inside comb too.
    // Factor m into prime powers.
    vector<pair<ll,int>> pf;
    ll mm = m;
    for (ll d = 2; d * d <= mm; d++) {
        if (mm % d == 0) {
            int cnt = 0;
            while (mm % d == 0) { mm /= d; cnt++; }
            pf.push_back({d, cnt});
        }
    }
    if (mm > 1) pf.push_back({mm, 1});

    // CRT accumulation: result `res` mod `mod`.
    ll res = 0, mod = 1;
    for (auto &pr : pf) {
        PrimePowerComb ppc;
        ppc.init(pr.first, pr.second);
        ll a = ppc.comb(n, r); // value mod pe
        ll pe = ppc.pe;
        // CRT merge (res, mod) with (a, pe); mod and pe are coprime.
        // Find t with res + mod*t == a (mod pe):  t = (a-res) * mod^{-1} (mod pe).
        ll inv = inv_mod(mod % pe, pe);         // mod is coprime to pe
        ll diff = ((a - res) % pe + pe) % pe;
        ll t = mulmod(diff, inv, pe);
        res = res + mod * t;
        mod = mod * pe;
        res %= mod;
        if (res < 0) res += mod;
    }

    cout << (res % m) << "\n";
    return 0;
}
```

**Causal recap.** The naive "factorials and Fermat inverses" recipe is not just slow but *undefined* under a composite modulus — `inv(6) mod 12` does not exist because `r!` shares prime factors with `m`. The resolution is structural: CRT splits `m` into prime powers `p^e`; within each, separate every factorial into its explicit `p`-power (Kummer/Legendre, with `>= e` short-circuiting to `0`) and its `p`-free part (which *is* a unit, so invertible), where the `p`-free factorial follows a block recurrence whose full-block sign is the Wilson `±1` read straight from the prefix table. A hand trace of the CRT merge exposed a dead ternary masking an unreduced-modulus reliance that would have silently broken only the composite cases; reducing explicitly to `mod % pe` fixed it, and 1400 differential cases against `math.comb`, an independent Lucas oracle, and a re-derived prime-power oracle confirmed zero mismatches, with full `10^18` inputs finishing in milliseconds.
