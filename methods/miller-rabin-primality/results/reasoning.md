OK, let me think this through from scratch. I have an odd integer `n` — could be a thousand bits — and I need to decide: prime or composite? Fast, and I need to trust the answer. The obvious thing, trial division up to `sqrt(n)`, is dead on arrival: `sqrt(n)` for a 1024-bit number is `2^512` operations. I need something polynomial in `log n`. And I want to do it *without* factoring `n`, because if I could factor I wouldn't be asking this question in the first place.

So what do I actually have that's cheap? Modular exponentiation. `a^m mod n` costs `O(log m)` modular multiplications by square-and-multiply — scan the bits of `m`, square the running value each step, multiply in `a` on a set bit. That's polynomial. So any test I build has to be expressible as "compute a few modular powers and look at the residues."

The one identity that connects "prime" to a modular power is Fermat's little theorem: if `n` is prime and `gcd(a,n)=1`, then `a^(n-1) ≡ 1 (mod n)`. Great — that's a *necessary* condition I can check cheaply. So flip it: pick some `a`, compute `a^(n-1) mod n`. If it isn't `1`, then `n` is definitely composite, and `a` proved it — call `a` a witness. If it is `1`, then `n` *might* be prime; `a` is a nonwitness.

Now the question that decides whether this is any good: when `n` is composite, how many bases `a` actually expose it? If most do, a handful of random tries nails it. Let me look at the structure. Fix `n` and consider the nonwitnesses, `{a : a^(n-1) ≡ 1 (mod n)}`. This set is closed under multiplication and inverses mod `n` — it's a *subgroup* of the units `(Z/nZ)*`. And a subgroup is either the whole group or at most half of it. So if the nonwitness subgroup is a *proper* subgroup, at least half of all coprime bases are witnesses, and `k` random rounds give error at most `2^(-k)`. That would be wonderful.

So the whole thing hinges on: is the nonwitness subgroup always proper for composite `n`? Let me try to break it. I want a composite `n` where `a^(n-1) ≡ 1` for *all* coprime `a`. Take `n = 561 = 3·11·17`. By CRT, `a^(n-1) ≡ 1 (mod n)` iff it holds mod `3`, mod `11`, mod `17`. Now `n - 1 = 560`. Mod `3`: the units have order dividing `2`, and `2 | 560`, so `a^560 ≡ 1`. Mod `11`: order divides `10`, `10 | 560`, so `a^560 ≡ 1`. Mod `17`: order divides `16`, `16 | 560`, so `a^560 ≡ 1`. So for *every* `a` coprime to `561`, `a^560 ≡ 1 (mod 561)`. The nonwitness subgroup is the entire unit group. The Fermat test fails completely on `561`.

That stings. And it's not a one-off — these are the Carmichael numbers, composite `n` with `a^(n-1) ≡ 1` for all coprime `a`, and there are infinitely many of them. So Fermat's test has a whole infinite family of composites it cannot reliably catch. The contrapositive of FLT is true but toothless here: `a^(n-1) ≡ 1` for every base I'd ever randomly pick, so I learn nothing. I've walled myself in. `a^(n-1) ≡ 1` is simply too weak a fingerprint of primality — it's satisfied by these structured composites too.

Let me stare at *why* `561` slips through. The trouble is that `a^(n-1) ≡ 1` is a single end-state; once I've squared my way up to the top of the exponent and landed on `1`, there's no more information. I've thrown away the *path*. Maybe the path to `1` carries a fingerprint that primes have and composites don't.

What does a prime "know" that `561` doesn't? Here's a property of primes that has nothing to do with Fermat directly: modulo a prime `n`, the equation `x^2 ≡ 1` has *only* the solutions `x ≡ ±1`. Why — `x^2 - 1 = (x-1)(x+1) ≡ 0 (mod n)`, and a prime that divides a product divides one of the factors, so `n | x-1` or `n | x+1`, i.e. `x ≡ 1` or `x ≡ -1`. Clean. Now what about a composite with two distinct prime factors? Say `n = 15 = 3·5`. By CRT a residue mod `15` is a pair (mod 3, mod 5). The square roots of `1` are the pairs `(±1, ±1)`: that's `(1,1) = 1`, `(-1,-1) = 14`, `(1,-1) = 4`, `(-1,1) = 11`. Four of them, not two. The two "extra" ones, `4` and `11`, satisfy `x^2 ≡ 1` but `x != ±1`.

That's the leverage. If I ever find an `x` with `x^2 ≡ 1 (mod n)` but `x !≡ ±1 (mod n)` — a *nontrivial* square root of `1` — then `n` is provably composite, because a prime can't have one. And crucially this works *even on Carmichael numbers*: a Carmichael number is composite with several distinct prime factors, so CRT hands me extra square roots of `1` for free. The Fermat test never looked for these; it only ever checked the topmost value.

So I want to manufacture square roots of `1` and inspect them. Where do they live? Right on the path up to `a^(n-1)`. Since `n` is odd, `n - 1` is even — write `n - 1 = 2^e · k` with `k` odd and `e ≥ 1`. Then `a^(n-1) = a^(2^e k) = (((a^k)^2)^2 \cdots)^2`, squared `e` times starting from `a^k`. So consider the sequence

`a^k, a^(2k), a^(4k), …, a^(2^(e-1)k), a^(2^e k) = a^(n-1)`,

where each term is the square of the previous one. Suppose `n` is prime. Then the last term is `1` by FLT. Now walk *backwards*: the second-to-last term squares to `1`, so it's a square root of `1`, so — primes only — it's `±1`. If it's `+1`, walk back again: the term before it also squares to `1`, so it's `±1` too. Keep going. Either every term is `1` all the way down to `a^k`, or at some point I hit the first term that *isn't* `1`, and that term squared to `1` so it must be `-1`. There's no third option for a prime.

Let me also see this as a factorization, to be sure I have all the cases. `x^(2^e k) - 1` factors as

`(x^k - 1)(x^k + 1)(x^(2k) + 1)(x^(4k) + 1) \cdots (x^(2^(e-1)k) + 1)`

— each squaring step splits off one `(\cdot + 1)` factor. For prime `n`, `a^(n-1) - 1 ≡ 0`, so the prime divides this product, hence divides one factor:

`a^k ≡ 1 (mod n)`  or  `a^(2^i k) ≡ -1 (mod n)` for some `i ∈ {0, …, e-1}`.   (★)

Same conclusion, two ways. So every base `a` satisfies (★) when `n` is prime. Therefore if I find an `a` for which (★) *fails* — meaning `a^k !≡ 1` and none of `a^k, a^(2k), …, a^(2^(e-1)k)` is `-1` — then `n` is composite. And look at *why* it's composite: if (★) fails, the sequence does reach `1` at the top (or it would already have failed the Fermat condition `a^(n-1)≡1`... wait, let me be careful, I'll come back to that), and the first time it becomes `1` it was preceded by a value `≠ ±1` that squared to `1`. A nontrivial square root of `1`. That's exactly the certificate I wanted, and it fell out of the same modular-exponentiation chain the Fermat test was already computing. No extra cost.

Let me pin down the witness definition cleanly. Write `n - 1 = 2^e k`, `k` odd. For a base `a`:

- `a` is a **nonwitness** if `a^k ≡ 1 (mod n)`, or `a^(2^i k) ≡ -1 (mod n)` for some `i ∈ {0,…,e-1}`.
- `a` is a **witness** if neither holds: `a^k !≡ 1` and `a^(2^i k) !≡ -1` for all `i ∈ {0,…,e-1}`.

A witness proves `n` composite. Equivalently, look at the sequence `(a^k, a^(2k), …, a^(2^(e-1)k))` mod `n`: `a` is a nonwitness exactly when the sequence starts with `1`, or contains `-1` somewhere; a witness when it does neither. (Note the `i=0` case of the second condition is `a^k ≡ -1`, so I could merge: nonwitness iff `a^k ≡ ±1`, or `a^(2^i k) ≡ -1` for some `i ≥ 1`.) Note `1` and `n-1` are always nonwitnesses — `1^k = 1`, and `(n-1)^k ≡ (-1)^k ≡ -1` since `k` is odd, hitting the `i=0` case — so I should pick random bases from `2` to `n-2` and not waste draws on those.

Now does this actually defeat `561`? Let me check base `a=2`. `561 - 1 = 560 = 2^4 · 35`, so `e=4`, `k=35`. Compute `2^35 mod 561`. I'll trust the chain: `2^35 ≡ 263`. Is that `±1`? No. Square: `2^70 ≡ 263^2 = 69169 ≡ 166 (mod 561)`. Not `-1`. Square: `2^140 ≡ 166^2 = 27556 ≡ 67`. Not `-1`. Square: `2^280 ≡ 67^2 = 4489 ≡ 1 (mod 561)`. So the sequence is `(263, 166, 67, 1)` — it reaches `1`, but the value just before, `67`, is a square root of `1` that isn't `±1`. Witness found. `561` is exposed as composite by `a=2`, even though `2^560 ≡ 1` made the Fermat test shrug. The path carried the proof.

Good. Now the only thing that matters for *reliability*: when `n` is composite, what fraction of bases are witnesses? With Fermat I got at least half — *unless* `n` was Carmichael, where I got essentially none. I need to show the new test has *no* such hole: a guaranteed large fraction of witnesses for **every** composite `n`, Carmichael included.

Mirror the Fermat argument: show the nonwitnesses sit inside a *proper* subgroup of `(Z/nZ)*`. Then witnesses are at least half, with no exceptions. The catch from before: are the nonwitnesses even a subgroup? Let me check `n = 65 = 5·13`. Its nonwitnesses turn out to be `{1, 8, 18, 47, 57, 64}`. Take `8·18 = 144 ≡ 14 (mod 65)`. Is `14` a nonwitness? `65 - 1 = 64 = 2^6 · 1`, so `k=1`; the sequence from `14` is `(14, 14^2,…)`; `14^2 = 196 ≡ 1 (mod 65)`, so the sequence is `(14, 1, 1, …)` — starts with `14 ≠ 1`, no `-1`. That's a *witness*. So the product of two nonwitnesses is a witness: the nonwitnesses are **not closed under multiplication**, not a subgroup. The clean Fermat-style argument doesn't transfer directly. Wall.

Why did closure fail? `8` and `18` each have `-1` appearing in the *same* position of their sequences (here position `i=1`: `8^2 ≡ 64 ≡ -1`, `18^2 ≡ 64 ≡ -1`), and when I multiply, those `-1`s cancel to `+1` too early, while the earlier terms `8·18 = 14` don't multiply to `±1`. So the obstruction is specific: two nonwitnesses whose sequences first hit `-1` at the *same* exponent. I need a subgroup that *contains* all the nonwitnesses but is engineered to be closed despite this.

Here's the construction. Let `i0` be the **largest** index in `{0,…,e-1}` such that *some* integer `a0` has `a0^(2^i0 k) ≡ -1 (mod n)`. Such an `i0` exists, because `a0 = -1` gives `(-1)^(2^0 k) = (-1)^k = -1` (k odd), so at least `i=0` works. Now define

`G_n = { a ∈ (Z/nZ)* : a^(2^i0 k) ≡ ±1 (mod n) }`.

This *is* a subgroup — `{±1}` is closed under multiplication, and `a -> a^(2^i0 k)` is a homomorphism, so its preimage of `{±1}` is a subgroup. Cleanly closed, no path-cancellation issues, because I've collapsed every sequence to a single fixed exponent `2^i0 k`.

Does `G_n` contain every nonwitness? Take a nonwitness `a`. Either `a^k ≡ 1`, in which case raising to the `2^i0` power gives `a^(2^i0 k) ≡ 1 ∈ {±1}`, so `a ∈ G_n`. Or `a^(2^i k) ≡ -1` for some `i ∈ {0,…,e-1}`. By the maximality of `i0`, this `i ≤ i0`. If `i = i0`, then `a^(2^i0 k) ≡ -1 ∈ {±1}`, done. If `i < i0`, square `a^(2^i k) ≡ -1` repeatedly `(i0 - i)` times: `(-1)^2 = 1`, so `a^(2^i0 k) ≡ 1 ∈ {±1}`. Either way `a ∈ G_n`. So `G_n ⊇` all nonwitnesses.

Now I just need `G_n` to be a **proper** subgroup of the units. Suppose `n` is not a prime power, so write `n = p^α n'` with `p ∤ n'` and `n' > 1` (both factors `> 1` and odd). By CRT choose `a` with

`a ≡ a0 (mod p^α)`,  `a ≡ 1 (mod n')`.

Then `a` is a unit (coprime to both), and look at `a^(2^i0 k)`: mod `p^α` it equals `a0^(2^i0 k) ≡ -1` (so `a^(2^i0 k) !≡ 1 (mod n)`, since it's `-1 ≠ 1` mod `p^α`); mod `n'` it equals `1^(2^i0 k) = 1` (so `a^(2^i0 k) !≡ -1 (mod n)`, since it's `1 ≠ -1` mod `n'`). So `a^(2^i0 k) ≡ neither +1 nor -1 mod n`: `a ∉ G_n`. A unit outside `G_n` exists, so `G_n` is proper. (And the prime-power case `n = p^α`, `α ≥ 2`, is even easier — there the nonwitnesses turn out to be exactly `{a : a^(p-1) ≡ 1 (mod p^α)}`, a subgroup of order `p-1`, while the units number `ϕ(p^α) = p^(α-1)(p-1) > p-1`; proper because `1 + p^(α-1)` is a unit of order `p` not in it.) So for every odd composite `n`: nonwitnesses lie in a proper subgroup, hence are **at most half** the units, hence the witnesses are **more than half** of `{2,…,n-2}`. No Carmichael escape — the construction never needed `n` to be non-Carmichael. The hole is gone.

Let me make sure the counting from "half the *units*" to "more than half of `{2,…,n-2}`" is honest. Nonwitnesses are all coprime to `n` (if `a^k ≡ 1` or `a^(2^i k) ≡ -1`, then some power of `a` is `±1`, a unit, so `a` is a unit). So all the non-coprime residues are automatically witnesses, on top of at least half the units being witnesses. With `W` = number of witnesses in `{1,…,n-1}`, I get `W/(n-1) ≥ 1/2`, and removing `1` and `n-1` (both nonwitnesses) only helps: `W/(n-3) ≥ W/(n-1) ≥ 1/2`. Solid.

A `1/2` error-reduction per round is already infinitely better than Fermat-on-Carmichael. But I recall the test is usually quoted with `≤ 4^(-k)`, i.e. nonwitnesses below `1/4`. Let me see if `1/2` is loose. The lever for `1/4` is to show the proper subgroup `G_n` actually has **index at least 4** in the units, i.e. `ϕ(n)/|G_n| ≥ 4`, whenever `n` is not a prime power. (Prime powers I'll handle separately.)

First, a fact I'll reuse: every `a ∈ G_n` satisfies `a^(n-1) ≡ 1 (mod n)`. Because `i0 ≤ e-1`, the exponent `2^(i0+1) k` divides `2^e k = n-1`; and `a ∈ G_n` means `a^(2^i0 k) ≡ ±1`, so squaring once gives `a^(2^(i0+1) k) ≡ 1`, and raising to the remaining power gives `a^(n-1) ≡ 1`. So `G_n` sits inside the Fermat-nonwitness group `F_n = {a : a^(n-1) ≡ 1}`.

Split on Carmichael-ness. **If `n` is not Carmichael** (and has ≥ 2 distinct prime factors): then `F_n` is a *proper* subgroup of the units — that's exactly what "not Carmichael" means, some coprime `a` has `a^(n-1) !≡ 1`. So `units ⊋ F_n ⊋ G_n` with both containments strict (the second strict because the CRT-built `a` above lies in `F_n` — it satisfies `a^(2^(i0+1)k) ≡ 1` mod both `p^α` and `n'`, hence `a^(n-1) ≡ 1` — but not in `G_n`). Two strict steps in a chain of subgroups, each at least doubling the index: `ϕ(n)/|G_n| = (ϕ(n)/|F_n|)·(|F_n|/|G_n|) ≥ 2·2 = 4`. So nonwitnesses `≤ |G_n| ≤ ϕ(n)/4 < (n-1)/4`.

**If `n` is Carmichael**, then `F_n` is the *whole* unit group, so that chain only gives one strict step and I'd be stuck at `1/2` — exactly the hard case. But a Carmichael number has at least **three** distinct prime factors. That's the resource. Write `n = p1^α1 ⋯ pr^αr`, `r ≥ 3`, and define a finer object using all the factors:

`H_n = { a : a^(2^i0 k) ≡ ±1 (mod p_l^α_l) for each l = 1,…,r }`.

So `units ⊇ H_n ⊇ G_n`: membership in `G_n` forces `a^(2^i0 k) ≡ ±1` mod `n`, i.e. the *same* sign `+1` everywhere or the same sign `-1` everywhere across the factors; `H_n` allows each factor its own independent `±1`. Consider the homomorphism

`f : H_n -> ∏_{l=1}^r {±1 mod p_l^α_l}`,  `f(a) = (a^(2^i0 k) mod p_l^α_l)_l`.

Its target has order `2^r`. I claim `f` is *surjective*. By symmetry it's enough to hit `(-1, 1, 1, …, 1)`. The defining `a0` has `a0^(2^i0 k) ≡ -1 (mod n)`, hence `≡ -1` mod each `p_l^α_l`. By CRT pick `a ≡ a0 (mod p1^α1)` and `a ≡ 1 (mod p_l^α_l)` for `l ≥ 2`; then `a^(2^i0 k) ≡ -1` in the first slot and `≡ 1` in the rest. Each standard-basis sign-flip is reachable, so all `2^r` sign patterns are, `f` is onto. Meanwhile `f(G_n) = {(1,1,…,1), (-1,-1,…,-1)}` has order `2`. With `K_n = ker f`: `|H_n|/|K_n| = 2^r` and `|G_n|/|K_n| = 2`, so `|H_n|/|G_n| = 2^(r-1) ≥ 2^2 = 4`. Therefore `ϕ(n)/|G_n| ≥ |H_n|/|G_n| ≥ 4`. Nonwitnesses below `1/4` again — and this is precisely the Carmichael case, the one Fermat couldn't touch, now pinned at `≤ 1/4`.

The prime-power case `n = p^α`, `α ≥ 2`: nonwitnesses number exactly `p - 1` (the solutions of `a^(p-1) ≡ 1 mod p^α`), out of `p^α - 1` residues in range, so the proportion is `(p-1)/(p^α - 1) = 1/(1 + p + ⋯ + p^(α-1)) ≤ 1/(1+p) ≤ 1/(1+3) = 1/4`, since `p ≥ 3` and `α ≥ 2`. Equality only at `p=3, α=2`, i.e. `n = 9`. So in every case, for odd composite `n`, the nonwitnesses are at most `1/4` of `{2,…,n-2}` — and the bound is essentially sharp (it limits to `3/4` witnesses on `n = p(2p-1)` with `p, 2p-1` both prime, `p ≡ 3 mod 4`).

Now I can state the reliability cleanly. Run `t` independent rounds with fresh random `a`. If `n` is prime, every round passes (every `a` is a nonwitness) and I never wrongly reject. If `n` is composite, each round independently catches it with probability `> 3/4`, so the chance all `t` rounds miss — a false "probably prime" — is `< (1/4)^t = 4^(-t)`. Take `t = 20` and the error is below `10^(-12)`; `t = 40` and it's cryptographically negligible. And the whole test is just `t` modular exponentiations plus `O(e) = O(log n)` squarings each, so `O(t · log n)` modular multiplications — polynomial, exactly what I needed, and now reliable on the Carmichael family that sank the Fermat test.

One more thing worth nailing down because it's a *deterministic* payoff. The randomness was only to find a witness when one is guaranteed to be common. If I'm willing to fix a base list, I can ask: for `n` below some bound, does a small *fixed* set of bases always contain a witness for every composite `n` in range? It does, by exhaustive search over the composites: the smallest odd composite with no witness among `{2, 3}` is `1373653`; the smallest with none among `{2, 3, 5, 7}` is `3215031751`. Pushing this out, the bases `{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}` — the primes up to 37 — contain a witness for *every* composite `n < 2^64`. So for 64-bit integers the test becomes fully **deterministic**: try exactly those twelve bases; if none is a witness, `n` is prime, no probability involved. (And under GRH one can even bound, for arbitrary `n`, that a witness exists among bases up to `2(log n)^2`, giving a deterministic polynomial-time test in general — but that leans on an unproven hypothesis, whereas the `4^(-t)` randomized guarantee and the `< 2^64` deterministic base set are unconditional.)

Let me write the code, mirroring exactly the structure I derived: strip small factors; decompose `n-1 = 2^e k`; a per-base routine that computes `a^k` and squares up to `e` times watching for the nonwitness conditions; and a driver that either samples `t` random bases (randomized, error `4^(-t)`) or sweeps the fixed list (deterministic below `2^64`).

```python
import random

def modpow(base, exp, mod):
    # square-and-multiply: base^exp mod mod in O(log exp) modular mults
    result = 1
    base %= mod
    while exp > 0:
        if exp & 1:
            result = (result * base) % mod
        base = (base * base) % mod
        exp >>= 1
    return result

def _decompose(n):
    # n-1 = 2^e * k with k odd; this exposes the square-root chain
    k = n - 1
    e = 0
    while k % 2 == 0:
        k //= 2
        e += 1
    return e, k

def is_composite_witness(n, a, e, k):
    # Returns True if a is a Miller-Rabin WITNESS (proves n composite).
    x = modpow(a, k, n)            # first term of the chain a^k
    if x == 1 or x == n - 1:
        return False              # sequence starts at 1, or hits -1 at i=0: nonwitness
    for _ in range(e - 1):        # square up to e-1 more times: a^(2k), a^(4k), ...
        x = (x * x) % n
        if x == n - 1:
            return False          # found a^(2^i k) == -1: nonwitness
        if x == 1:
            return True           # squared to 1 from x != +-1: nontrivial sqrt of 1 -> composite
    return True                   # never reached -1 and a^(n-1) != 1: composite

def is_probably_prime(n, t=40):
    # randomized: error < 4^(-t) when n is composite, never wrong when n is prime
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n == p:
            return True
        if n % p == 0:
            return n == p
    e, k = _decompose(n)
    for _ in range(t):
        a = random.randrange(2, n - 1)   # skip 1 and n-1, always nonwitnesses
        if is_composite_witness(n, a, e, k):
            return False                  # certainly composite
    return True                           # probably prime

def is_prime_deterministic_64(n):
    # exact for all n < 2^64: this fixed base set contains a witness for every composite in range
    if n < 2:
        return False
    e, k = _decompose(n) if n % 2 == 1 else (None, None)
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n == a:
            return True
        if n % a == 0:
            return False
        if is_composite_witness(n, a, e, k):
            return False
    return True
```

The causal chain, end to end: Fermat's `a^(n-1) ≡ 1` is a fast necessary condition for primality, but it collapses the whole exponentiation to one end-value and so the Carmichael numbers — composite yet satisfying it for every coprime base — slip through with no error bound. The fix is to stop discarding the path: factor `n-1 = 2^e k` and watch the chain `a^k, a^(2k), …, a^(2^e k)` climb to `a^(n-1)`. A prime forces this chain either to start at `1` or to pass through `-1` before reaching `1`, because modulo a prime the only square roots of `1` are `±1`; a base for which it does neither has exposed a *nontrivial* square root of `1` and thereby proven `n` composite — Carmichael numbers included, since their several prime factors guarantee such square roots exist. Quantifying it, the nonwitnesses for any odd composite `n` sit inside a proper subgroup of index at least `4` (via the maximal-`i0` group `G_n`, refined by the Fermat group `F_n` off Carmichael numbers and by the per-factor sign homomorphism on Carmichael numbers with their ≥ 3 prime factors), so more than `3/4` of bases are witnesses; `t` random rounds drive the one-sided error below `4^(-t)`, and a fixed twelve-base list makes it exactly deterministic below `2^64`.
