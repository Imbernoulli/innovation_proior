Two numbers set the shape of this problem before I write anything: `n, r` up to `10^18`, and `m` up to `10^6` but arbitrary. The first says `C(n, r)` has on the order of `10^17` digits — it can never be materialized, so I work modularly from the start, and every index and exponent that touches `n` or `r` is `long long`. The second is the real trap: `m` is not promised prime, and not even squarefree. The output is a single integer in `[0, m)`, and `r > n` (or `r < 0`) means `C = 0`.

The textbook recipe for `C(n, r) mod p` is `fact[n] * invfact[r] * invfact[n-r]` with `fact`/`invfact` precomputed and inverses from Fermat. For `n = 10^18` the tables are impossible, but **Lucas' theorem** rescues a prime modulus: write `n, r` in base `p`, then `C(n, r) ≡ ∏_i C(n_i, r_i) (mod p)`, a product over single base-`p` digits, each small and computed from a `p`-sized factorial table. Complete and fast — *when `m` is prime*.

The moment `m` is composite this doesn't slow down, it stops being defined. `invfact[r]` is a modular inverse of `r!`, which exists iff `gcd(r!, m) = 1`. But `r!` contains every prime up to `r`; as soon as `r` reaches the smallest prime factor of `m`, `gcd(r!, m) > 1` and the inverse simply does not exist. Concretely, `C(10, 3) mod 12` with `12 = 2^2·3`: the true value is `120`, so `120 mod 12 = 0`. The inverse formula would need `inv(6) mod 12`, and no `x` satisfies `6x ≡ 1 (mod 12)` since `gcd(6, 12) = 6`. The exact integer `n!/(r!(n-r)!)` is fine; realizing it as a modular product of inverses is what breaks when the modulus shares factors with the factorials.

One correct workaround exists and is useless at scale: `C(n, r) = ∏_{i=1}^{r} (n-r+i)/i`, reducing mod `m` and dividing exactly each step (every prefix is a genuine integer). That is what a brute force does, and it gives me a trustworthy oracle for small `r` — but at `O(r)` multiplications with `r` up to `10^18` it is hopeless. I need something correct under a composite modulus *and* polylogarithmic in `n`.

So I attack the modulus's structure. Factor `m = ∏_j p_j^{e_j}`; since `m <= 10^6`, trial division to `sqrt(m)` finds this in microseconds, and there are at most a handful of distinct primes (`2·3·5·7·11·13 > 3·10^4` already, so at most seven). The prime powers are pairwise coprime, so by the **Chinese Remainder Theorem**, `C(n, r) mod p_j^{e_j}` for each `j` pins down `C(n, r) mod m`. The recombination of a few residues is trivial. The problem collapses to: **compute `C(n, r) mod p^e`, `n, r` up to `10^18`, where `p` still divides the factorials.**

Modulo `p^e` the factorials are still not units, so I separate each into its `p`-part and its `p`-free part. Write `k! = p^{v_p(k!)} · (k!)_p`, where `(k!)_p` is the product over `[1, k]` of the integers *not* divisible by `p`. That `p`-free part is coprime to `p`, hence a **unit** mod `p^e`, hence invertible — the ill-defined division becomes legitimate:

```
C(n, r) ≡ p^{v_p(C)} · (n!)_p / ((r!)_p (n-r)!)_p)   (mod p^e).
```

Two facts make this finite and fast. **Legendre/Kummer** gives the exponent: `v_p(k!) = floor(k/p) + floor(k/p^2) + ...`, an `O(log_p k)` sum, and `v_p(C) = v_p(n!) - v_p(r!) - v_p((n-r)!)`. If `v_p(C) >= e` then `p^e | C(n, r)` and the residue is just `0`. Otherwise a small surviving `p^{v_p(C)}` gets multiplied back. And the `p`-free factorial has a **periodic block structure**: the product of all `p`-coprime residues in one block `[1, p^e]` is, by Wilson's generalization, `-1 mod p^e` for odd `p` (any `e`) and for `p^e ∈ {2, 4}`, and `+1` for `p = 2, e >= 3`. The `p`-free numbers up to `x` are `floor(x/p^e)` full blocks (each contributing that fixed sign) times a partial block, and the stripped multiples of `p` reproduce `(floor(x/p)!)_p`:

```
(x!)_p ≡ blockSign^{floor(x/p^e)} · (partial up to x mod p^e) · (floor(x/p)!)_p   (mod p^e),
```

bottoming out in `O(log_p x)` steps as `x` shrinks by `p` each level. This is the generalized-Lucas / Granville factorial technique: factor `m`, per prime power compute the binomial via the `p`-free recurrence with Legendre for the exponent and Wilson for the block sign, then CRT. Per prime power costs `O(p^e)` to build the in-block prefix product once, plus `O(log_p n)` per factorial.

The trap case `C(10, 3) mod 12` now resolves without ever touching an inverse of a non-unit. For `p^e = 4`: `v_2(C) = v_2(10!) - v_2(3!) - v_2(7!) = 8 - 1 - 4 = 3 >= 2`, so the residue mod `4` is `0`. For `p^e = 3`: `v_3(C) = 4 - 1 - 2 = 1 >= 1`, so the residue mod `3` is `0`. CRT of `(0, 0)` is `0`, matching `120 mod 12`. Both factors short-circuit on the exponent — exactly the divisibility the naive formula couldn't even express.

I structure the core as a `PrimePowerComb(p, e)` that builds the size-`p^e` prefix table once and exposes `comb(n, r)`:

```
init(p, e): pe = p^e; fact[i] = fact[i-1]*i if p∤i else fact[i-1], mod pe.
legendre(x): sum of x/p + x/p^2 + ...
factmod(x):  res = 1; while x>0 { res *= blockSign^(x/pe) * fact[x % pe]; x /= p; }
comb(n, r):  powp = legendre(n) - legendre(r) - legendre(n-r);
             if powp >= e return 0;
             res = factmod(n) * inv(factmod(r)*factmod(n-r)) * p^powp;
```

I read `blockSign` straight off the table as `fact[pe-1]` — the product of one full block *is* the Wilson `±1`, so the table computes the correct sign and no `p = 2` special-case is needed. The inverse here is legitimate because every `factmod(...)` is a unit mod `p^e`.

The one place the clean math can go wrong in code is the CRT merge. Folding `(a mod pe)` into a running `(res mod M)`, I need `t` with `res + M·t ≡ a (mod pe)`, i.e. `t = (a - res)·M^{-1} (mod pe)`. The inverse has to be of `M mod pe`, not `M`: after the first merge `M` exceeds `pe`, and inverting the unreduced value returns the wrong residue. This only bites composite `m` — with a single prime factor the merge is trivial and every prime-`m` case passes regardless — so a missing reduction would silently corrupt exactly the cases this problem exists for. Hence `inv_mod(M % pe, pe)`.

For confidence beyond hand-checking I run a differential harness: an exact `math.comb(n, r) % m` oracle (no inverses, correct by construction) for `n` up to a few thousand, and an independent **Lucas** implementation for prime `m` that scales to `n = 10^18`. Roughly a thousand cases from a generator targeting the structural regimes — prime moduli, prime powers `2^e/3^e/5^e/7^e`, squarefree composites, general composites up to `10^6`, and boundary `r` (`0`, `n`, `n+1`) — zero mismatches.

The edges I make sure of:
- `m = 1`: short-circuit to `0` at the top, avoiding a degenerate empty-factorization path.
- `r = 0` or `r = n`: `factmod(0) = 1` and `powp = 0`, so `comb` returns `factmod(n)·inv(factmod(n)) = 1`.
- `r > n`: `comb` returns `0` immediately, every residue `0`, CRT `0`.
- Wilson sign: large prime-power moduli `1024 = 2^10`, `524288 = 2^19` (both `+1` blocks) and `59049 = 3^10`, `390625 = 5^8`, `823543 = 7^7` (`-1` blocks) match `math.comb` at `n` up to `2·10^5`, exercising both signs.
- Overflow: `x` and `blocks` in `factmod` stay `< 10^18`; `mulmod` products are guarded by `__int128`; the CRT step multiplies `M <= 10^6` by `t < pe <= 10^6`, so `M·t <= 10^12`, inside `long long`.
- Full scale: `n = r = 10^18` finishes in single-digit milliseconds — one `O(p^e)` table build per prime power plus `O(log_p n)` per factorial.

The full self-contained C++ program is the answer: factor `m`, run `PrimePowerComb` per prime power, fold the residues through the reduced-modulus CRT merge, print `res % m`.
