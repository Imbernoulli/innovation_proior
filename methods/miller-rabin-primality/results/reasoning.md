OK, let me think this through from scratch. I have an odd integer `n` — could be a thousand bits — and I need to decide: prime or composite? Fast, and I need to *trust* the answer. There already is a fast test on the table: Miller's. Decompose `n-1` and check a structured congruence for small bases `a`; if `n` is prime they all pass, and the claim is that if `n` is composite one of those small bases must expose it, so checking them all decides primality deterministically in polynomial time. Beautiful — except the claim that a small base must expose every composite is conditional. It comes from the extended Riemann hypothesis. And that gnaws at me. The procedure's guarantee is leveraged on a conjecture nobody has proven; without that conjecture, the fixed small-base search has no unconditional proof behind it. I don't want my answer to be a hostage to an open problem in analytic number theory. I want a test whose guarantee I can prove *today*, with no unproven hypothesis underneath it.

So the goal sharpens: keep the speed of Miller's test — polynomial in `log n`, no factoring of `n`, because if I could factor I wouldn't be asking — but remove the dependence on the Riemann hypothesis. The thing the hypothesis was buying me was a *deterministic* guarantee from a *short, fixed* list of bases: "these specific small `a` are certain to contain an exposer." That's exactly the part that isn't proven. So drop the demand that the bases be a fixed short list, and drop the demand for a zero-error deterministic verdict. I've spent years on probabilistic automata — machines that flip a coin to choose their next transition, that don't promise certainty on a single run but, run enough times, are wrong with probability I can drive as low as I like. That trade is the escape here. If, instead of needing a *specific* short list of bases to provably work, I could prove that for any composite `n` a *large fraction* of all bases expose it, then I don't need the Riemann hypothesis at all: I pick bases *at random* and let independent sampling do the work. The price is a controllable, vanishingly small probability of error instead of certainty; the payoff is a guarantee that stands on proven ground. So: randomize over the bases, and to make that work I need the right notion of "exposer" and a proof that exposers are dense for every composite `n`.

So what do I actually have that's cheap? Modular exponentiation. `a^m mod n` costs `O(log m)` modular multiplications by square-and-multiply — scan the bits of `m`, square the running value each step, multiply in `a` on a set bit. That's polynomial. So any test I build has to be expressible as "compute a few modular powers and look at the residues," and to make the randomization argument work I'll need: a condition on a base `a` that *proves* `n` composite when it fails, and a proof that for every composite `n` the failing bases are a guaranteed-large fraction of all bases — so a handful of random draws finds one.

The one identity that connects "prime" to a modular power is Fermat's little theorem: if `n` is prime and `gcd(a,n)=1`, then `a^(n-1) ≡ 1 (mod n)`. Great — that's a *necessary* condition I can check cheaply, and it's exactly the shape of exposer I want. So flip it: pick some `a`, compute `a^(n-1) mod n`. If it isn't `1`, then `n` is definitely composite, and `a` proved it — call `a` a witness. If it is `1`, then `n` *might* be prime; `a` is a nonwitness.

Now the question that decides the whole randomization plan: when `n` is composite, how many bases `a` actually expose it? If most do, a handful of random tries nails it — that's the density I need to prove, and if I can prove it I never have to invoke the Riemann hypothesis. Let me look at the structure. Fix `n` and consider the nonwitnesses, `{a : a^(n-1) ≡ 1 (mod n)}`. This set is closed under multiplication and inverses mod `n` — it's a *subgroup* of the units `(Z/nZ)*`. And a subgroup is either the whole group or at most half of it. So if the nonwitness subgroup is a *proper* subgroup, at least half of all coprime bases are witnesses, and `r` random rounds give error at most `2^(-r)`. That would be wonderful.

So the whole thing hinges on: is the nonwitness subgroup always proper for composite `n`? Let me try to break it. I want a composite `n` where `a^(n-1) ≡ 1` for *all* coprime `a`. Take `n = 561 = 3·11·17`. By CRT, `a^(n-1) ≡ 1 (mod n)` iff it holds mod `3`, mod `11`, mod `17`. Now `n - 1 = 560`. Mod `3`: the units have order dividing `2`, and `2 | 560`, so `a^560 ≡ 1`. Mod `11`: order divides `10`, `10 | 560`, so `a^560 ≡ 1`. Mod `17`: order divides `16`, `16 | 560`, so `a^560 ≡ 1`. So for *every* `a` coprime to `561`, `a^560 ≡ 1 (mod 561)`. The nonwitness subgroup is the entire unit group. The Fermat test fails completely on `561`.

That stings. And it's not a one-off — these are the Carmichael numbers, composite `n` with `a^(n-1) ≡ 1` for all coprime `a`; `561` is just the first visible warning sign. So Fermat's test has a structured class of composites it cannot reliably catch. The contrapositive of FLT is true but toothless here: `a^(n-1) ≡ 1` for every coprime base I would deliberately sample, so I learn nothing. I've walled myself in. `a^(n-1) ≡ 1` is simply too weak a fingerprint of primality — it's satisfied by these structured composites too.

Let me stare at *why* `561` slips through. The trouble is that `a^(n-1) ≡ 1` is a single end-state; once I've squared my way up to the top of the exponent and landed on `1`, there's no more information. I've thrown away the *path*. Maybe the path to `1` carries a fingerprint that primes have and composites don't.

What does a prime "know" that `561` doesn't? Here's a property of primes that has nothing to do with Fermat directly: modulo a prime `n`, the equation `x^2 ≡ 1` has *only* the solutions `x ≡ ±1`. Why — `x^2 - 1 = (x-1)(x+1) ≡ 0 (mod n)`, and a prime that divides a product divides one of the factors, so `n | x-1` or `n | x+1`, i.e. `x ≡ 1` or `x ≡ -1`. Clean. Now what about a composite with two distinct prime factors? Say `n = 15 = 3·5`. By CRT a residue mod `15` is a pair (mod 3, mod 5). The square roots of `1` are the pairs `(±1, ±1)`: that's `(1,1) = 1`, `(-1,-1) = 14`, `(1,-1) = 4`, `(-1,1) = 11`. Four of them, not two. The two "extra" ones, `4` and `11`, satisfy `x^2 ≡ 1` but `x != ±1`.

That's the leverage. If I ever find an `x` with `x^2 ≡ 1 (mod n)` but `x !≡ ±1 (mod n)` — a *nontrivial* square root of `1` — then `n` is provably composite, because a prime can't have one. And crucially this works *even on Carmichael numbers*: a Carmichael number is composite with several distinct prime factors, so CRT hands me extra square roots of `1` for free. The Fermat test never looked for these; it only ever checked the topmost value.

So I want to manufacture square roots of `1` and inspect them. Where do they live? Right on the path up to `a^(n-1)`. Since `n` is odd, `n - 1` is even — write `n - 1 = 2^s · d` with `d` odd and `s ≥ 1`. Then `a^(n-1) = a^(2^s d) = (((a^d)^2)^2 \cdots)^2`, squared `s` times starting from `a^d`. So consider the sequence

`a^d, a^(2d), a^(4d), …, a^(2^(s-1)d), a^(2^s d) = a^(n-1)`,

where each term is the square of the previous one. Suppose `n` is prime. Then the last term is `1` by FLT. Now walk *backwards*: the second-to-last term squares to `1`, so it's a square root of `1`, so — primes only — it's `±1`. If it's `+1`, walk back again: the term before it also squares to `1`, so it's `±1` too. Keep going. Either every term is `1` all the way down to `a^d`, or at some point I hit the first term that *isn't* `1`, and that term squared to `1` so it must be `-1`. There's no third option for a prime.

Let me also see this as a factorization, to be sure I have all the cases. `x^(2^s d) - 1` factors as

`(x^d - 1)(x^d + 1)(x^(2d) + 1)(x^(4d) + 1) \cdots (x^(2^(s-1)d) + 1)`

— each squaring step splits off one `(\cdot + 1)` factor. For prime `n`, `a^(n-1) - 1 ≡ 0`, so the prime divides this product, hence divides one factor:

`a^d ≡ 1 (mod n)`  or  `a^(2^i d) ≡ -1 (mod n)` for some `i ∈ {0, …, s-1}`.   (★)

Same conclusion, two ways. So every base `a` satisfies (★) when `n` is prime. Therefore if I find an `a` for which (★) *fails* — meaning `a^d !≡ 1` and none of `a^d, a^(2d), …, a^(2^(s-1)d)` is `-1` — then `n` is composite. There are two ways that failure can look. If the top value `a^(n-1)` is not `1`, then I already have the old Fermat certificate. If the top value is `1`, then the sequence first becomes `1` at some squaring step, and the value just before that is neither `1` nor `-1`, because the sequence never started at `1` and never hit `-1`. That predecessor is a nontrivial square root of `1`. So the path gives me a stronger certificate than Fermat when Fermat's end-value is silent, and it costs only the squarings I was already doing.

Let me pin down the witness definition cleanly. Write `n - 1 = 2^s d`, `d` odd. I will call `a` a nonwitness if `a^d ≡ 1 (mod n)` or if `a^(2^i d) ≡ -1 (mod n)` for some `i ∈ {0,…,s-1}`. I will call `a` a witness if neither condition holds: `a^d !≡ 1` and `a^(2^i d) !≡ -1` for every `i ∈ {0,…,s-1}`. A witness proves `n` composite. Equivalently, look at the sequence `(a^d, a^(2d), …, a^(2^(s-1)d))` mod `n`: `a` is a nonwitness exactly when the sequence starts with `1`, or contains `-1` somewhere; a witness when it does neither. The `i=0` case of the second condition is `a^d ≡ -1`, so I could also say: nonwitness iff `a^d ≡ ±1`, or `a^(2^i d) ≡ -1` for some `i ≥ 1`. The bases `1` and `n-1` are always nonwitnesses — `1^d = 1`, and `(n-1)^d ≡ (-1)^d ≡ -1` since `d` is odd, hitting the `i=0` case — so I should pick random bases from `2` to `n-2` and not waste draws on those.

Now does this actually defeat `561`? Let me check base `a=2`. `561 - 1 = 560 = 2^4 · 35`, so `s=4`, `d=35`. Square-and-multiply gives `2^10 ≡ 463`, `2^20 ≡ 67`, `2^30 ≡ 166`, and therefore `2^35 ≡ 263 (mod 561)`. Is that `±1`? No. Square: `2^70 ≡ 263^2 = 69169 ≡ 166 (mod 561)`. Not `-1`. Square: `2^140 ≡ 166^2 = 27556 ≡ 67`. Not `-1`. Square: `2^280 ≡ 67^2 = 4489 ≡ 1 (mod 561)`. So the sequence is `(263, 166, 67, 1)` — it reaches `1`, but the value just before, `67`, is a square root of `1` that isn't `±1`. Witness found. `561` is exposed as composite by `a=2`, even though `2^560 ≡ 1` made the Fermat test shrug. The path carried the proof.

Good. Now the only thing that matters for *reliability*: when `n` is composite, what fraction of bases are witnesses? With Fermat I got at least half — *unless* `n` was Carmichael, where I got essentially none. I need to show the new test has *no* such hole: a guaranteed large fraction of witnesses for **every** composite `n`, Carmichael included.

Mirror the Fermat argument: show the nonwitnesses sit inside a *proper* subgroup of `(Z/nZ)*`. Then witnesses are at least half, with no exceptions. The catch from before: are the nonwitnesses even a subgroup? Let me check `n = 65 = 5·13`. Its nonwitnesses turn out to be `{1, 8, 18, 47, 57, 64}`. Take `8·18 = 144 ≡ 14 (mod 65)`. Is `14` a nonwitness? `65 - 1 = 64 = 2^6 · 1`, so `d=1`; the sequence from `14` is `(14, 14^2,…)`; `14^2 = 196 ≡ 1 (mod 65)`, so the sequence is `(14, 1, 1, …)` — starts with `14 ≠ 1`, no `-1`. That's a *witness*. So the product of two nonwitnesses is a witness: the nonwitnesses are not closed under multiplication, not a subgroup. The clean Fermat-style argument doesn't transfer directly; I need a container that is larger than the nonwitness set but still closed.

Why did closure fail? `8` and `18` each have `-1` appearing in the *same* position of their sequences (here position `i=1`: `8^2 ≡ 64 ≡ -1`, `18^2 ≡ 64 ≡ -1`), and when I multiply, those `-1`s cancel to `+1` too early, while the earlier terms `8·18 = 14` don't multiply to `±1`. So the obstruction is specific: two nonwitnesses whose sequences first hit `-1` at the *same* exponent. I need a subgroup that *contains* all the nonwitnesses but is engineered to be closed despite this.

Let `i0` be the largest index in `{0,…,s-1}` such that some unit `a0` has `a0^(2^i0 d) ≡ -1 (mod n)`. Such an `i0` exists, because `a0 = -1` gives `(-1)^(2^0 d) = (-1)^d = -1` (`d` odd), so at least `i=0` works. Now define

`G_n = { a ∈ (Z/nZ)* : a^(2^i0 d) ≡ ±1 (mod n) }`.

This *is* a subgroup — `{±1}` is closed under multiplication, and `a -> a^(2^i0 d)` is a homomorphism, so its preimage of `{±1}` is a subgroup. Cleanly closed, no path-cancellation issues, because I've collapsed every sequence to a single fixed exponent `2^i0 d`.

Does `G_n` contain every nonwitness? Take a nonwitness `a`. Either `a^d ≡ 1`, in which case raising to the `2^i0` power gives `a^(2^i0 d) ≡ 1 ∈ {±1}`, so `a ∈ G_n`. Or `a^(2^i d) ≡ -1` for some `i ∈ {0,…,s-1}`. By the maximality of `i0`, this `i ≤ i0`. If `i = i0`, then `a^(2^i0 d) ≡ -1 ∈ {±1}`, done. If `i < i0`, square `a^(2^i d) ≡ -1` repeatedly `(i0 - i)` times: `(-1)^2 = 1`, so `a^(2^i0 d) ≡ 1 ∈ {±1}`. Either way `a ∈ G_n`. So `G_n ⊇` all nonwitnesses.

Now I just need `G_n` to be a **proper** subgroup of the units. Suppose `n` is not a prime power, so write `n = p^α n'` with `p ∤ n'` and `n' > 1` (both factors `> 1` and odd). By CRT choose `a` with

`a ≡ a0 (mod p^α)`,  `a ≡ 1 (mod n')`.

Then `a` is a unit (coprime to both), and look at `a^(2^i0 d)`: mod `p^α` it equals `a0^(2^i0 d) ≡ -1` (so `a^(2^i0 d) !≡ 1 (mod n)`, since it is `-1 ≠ 1` mod `p^α`); mod `n'` it equals `1^(2^i0 d) = 1` (so `a^(2^i0 d) !≡ -1 (mod n)`, since it is `1 ≠ -1` mod `n'`). So `a^(2^i0 d)` is neither `+1` nor `-1` mod `n`: `a ∉ G_n`. A unit outside `G_n` exists, so `G_n` is proper. For the prime-power case `n = p^α`, `α ≥ 2`, the CRT escape is unavailable, so I use the cyclic structure of the units modulo an odd prime power. If `a` is a nonwitness, then `a^(n-1) ≡ 1`, hence `a` is a unit; Euler also gives `a^ϕ(n) ≡ 1`, and `gcd(p^α - 1, p^(α-1)(p-1)) = p - 1`, so the order of `a` divides `p - 1`. Thus nonwitnesses lie in the subgroup `{a : a^(p-1) ≡ 1 (mod p^α)}`, whose size is at most `p - 1`, while the whole unit group has order `ϕ(p^α) = p^(α-1)(p-1) > p - 1`; for instance `1 + p^(α-1)` is a unit of order `p`, so it is outside that subgroup. So for every odd composite `n`, the nonwitnesses lie in a proper subgroup of the units. No Carmichael escape — the construction never needed `n` to be non-Carmichael. The hole is gone.

Let me make sure the counting from "half the *units*" to "more than half of `{2,…,n-2}`" is honest. Nonwitnesses are all coprime to `n` (if `a^d ≡ 1` or `a^(2^i d) ≡ -1`, then some power of `a` is `±1`, a unit, so `a` is a unit). So all the non-coprime residues are automatically witnesses, on top of at least half the units being witnesses. Since `n` is composite there is at least one nonunit in `{1,…,n-1}`, while `1` and `n-1` are the two units I know are nonwitnesses. With `W` = number of witnesses in `{1,…,n-1}`, I get `W/(n-1) > 1/2`; removing the two guaranteed nonwitnesses `1` and `n-1` only raises the ratio, so `W/(n-3) > W/(n-1) > 1/2`.

A `1/2` error-reduction per round already removes the Fermat-on-Carmichael failure, but the subgroup proof has more room in it. To get an error of about `4^(-t)`, I need nonwitnesses to occupy at most one quarter of the full residue range, and strictly less than one quarter after I remove the always-useless bases `1` and `n-1`. The lever is to show the proper subgroup `G_n` actually has index at least `4` in the units, i.e. `ϕ(n)/|G_n| ≥ 4`, whenever `n` is not a prime power. Prime powers need their own count.

First, a fact I'll reuse: every `a ∈ G_n` satisfies `a^(n-1) ≡ 1 (mod n)`. Because `i0 ≤ s-1`, the exponent `2^(i0+1) d` divides `2^s d = n-1`; and `a ∈ G_n` means `a^(2^i0 d) ≡ ±1`, so squaring once gives `a^(2^(i0+1) d) ≡ 1`, and raising to the remaining power gives `a^(n-1) ≡ 1`. So `G_n` sits inside the Fermat-nonwitness group `F_n = {a : a^(n-1) ≡ 1}`.

If `n` is not Carmichael and has at least two distinct prime factors, then `F_n` is a proper subgroup of the units — that is exactly what "not Carmichael" means, some coprime `a` has `a^(n-1) !≡ 1`. So `units ⊋ F_n ⊋ G_n` with both containments strict. The second containment is strict because the CRT-built `a` above lies in `F_n`: it satisfies `a^(2^(i0+1)d) ≡ 1` mod both `p^α` and `n'`, hence `a^(n-1) ≡ 1`, but it is not in `G_n`. Two strict steps in a chain of subgroups each at least double the index, so `ϕ(n)/|G_n| = (ϕ(n)/|F_n|)·(|F_n|/|G_n|) ≥ 2·2 = 4`. The nonwitnesses are inside `G_n`, so their count is at most `|G_n| ≤ ϕ(n)/4 < (n-1)/4`.

If `n` is Carmichael, then `F_n` is the whole unit group, so that chain only gives one strict step and I would be stuck at `1/2`. But a Carmichael number is squarefree and has at least three distinct prime factors. That's the resource. Write `n = p1 ⋯ pr`, `r ≥ 3`, and define a finer object using all the factors:

`H_n = { a : a^(2^i0 d) ≡ ±1 (mod p_l) for each l = 1,…,r }`.

This is a group for the same preimage reason as `G_n`: take the fixed-power homomorphism to the product of the unit groups modulo the `p_l`, and pull back the subgroup `∏_l {±1}`. So `units ⊇ H_n ⊇ G_n`: membership in `G_n` forces `a^(2^i0 d) ≡ ±1` mod `n`, i.e. the *same* sign `+1` everywhere or the same sign `-1` everywhere across the factors; `H_n` allows each factor its own independent `±1`. Consider the homomorphism

`f : H_n -> ∏_{l=1}^r {±1 mod p_l}`,  `f(a) = (a^(2^i0 d) mod p_l)_l`.

Its target has order `2^r`. I claim `f` is surjective. By symmetry it is enough to hit `(-1, 1, 1, …, 1)`. The defining `a0` has `a0^(2^i0 d) ≡ -1 (mod n)`, hence `≡ -1` mod each `p_l`. By CRT pick `a ≡ a0 (mod p1)` and `a ≡ 1 (mod p_l)` for `l ≥ 2`; then `a^(2^i0 d) ≡ -1` in the first slot and `≡ 1` in the rest. Because `H_n` is a group and `f` is a homomorphism, multiplying these one-coordinate sign flips reaches every sign pattern, so `f` is onto. Meanwhile `f(G_n) = {(1,1,…,1), (-1,-1,…,-1)}` has order `2`, and the kernel `K_n` of `f` lies inside `G_n` because the all-`+1` sign pattern is a global `+1` modulo `n`. Thus `|H_n|/|K_n| = 2^r` and `|G_n|/|K_n| = 2`, so `|H_n|/|G_n| = 2^(r-1) ≥ 2^2 = 4`. Therefore `ϕ(n)/|G_n| ≥ |H_n|/|G_n| ≥ 4`, and the nonwitness count is at most `ϕ(n)/4 < (n-1)/4`. The Carmichael case, the one Fermat could not touch, is now pinned below one quarter.

For the prime-power case `n = p^α`, `α ≥ 2`, I should not wave my hands. If `a` is a nonwitness, then `a^(n-1) ≡ 1`; since `a` is a unit, Euler also gives `a^ϕ(n) ≡ 1`. Here `n-1 = p^α - 1`, `ϕ(n)=p^(α-1)(p-1)`, and `gcd(p^α-1, p^(α-1)(p-1)) = p-1`, so the order of `a` divides `p-1`, hence `a^(p-1) ≡ 1`. Conversely, if `a^(p-1) ≡ 1`, write `p-1 = 2^f ℓ` with `ℓ` odd. Since `p-1 | p^α-1 = 2^s d`, I have `f ≤ s` and `ℓ | d`; because both `d` and `ℓ` are odd, `d/ℓ` is odd. The element `a^ℓ` has order a power of `2`; if that order is `1`, then `a^d = (a^ℓ)^(d/ℓ) ≡ 1`. If the order is `2^j` with `j ≥ 1`, then `j ≤ f ≤ s`, so the index `j-1` is allowed, and `(a^ℓ)^(2^(j-1))` is the unique element of order `2` in the cyclic subgroup it generates. Modulo an odd prime power the only square roots of `1` are `±1`, so that element is not `1` and must be `-1`; raising by the odd factor `d/ℓ` preserves `-1`, giving `a^(2^(j-1)d) ≡ -1`. Thus the nonwitnesses are exactly the solutions of `a^(p-1) ≡ 1 (mod p^α)`. The unit group modulo an odd prime power is cyclic of order `p^(α-1)(p-1)`, so that equation has exactly `p-1` solutions. Their proportion in `{1,…,p^α-1}` is `(p-1)/(p^α - 1) = 1/(1 + p + ⋯ + p^(α-1)) ≤ 1/(1+p) ≤ 1/4`, with equality only at `p=3, α=2`, i.e. `n=9`. When I remove `1` and `n-1`, which are both nonwitnesses, the sampled range `{2,…,n-2}` has a strictly smaller nonwitness proportion. So in every odd composite case, more than `3/4` of the sampled bases are witnesses.

Now I can state the reliability cleanly. Run `t` independent rounds with fresh random `a`. If `n` is prime, every round passes and I never wrongly reject. If `n` is composite, each sampled round from `{2,…,n-2}` catches it with probability `> 3/4`, so the chance all `t` rounds miss — a false "probably prime" — is `< (1/4)^t = 4^(-t)`. Take `t = 20` and the error is already below `10^(-12)`; more rounds drive it down geometrically. And the whole test is just `t` modular exponentiations plus `O(s) = O(log n)` squarings each, so `O(t · log n)` modular multiplications at this level of accounting — polynomial, exactly what I needed, and now reliable on the Carmichael family that sank the Fermat test.

Let me write the code, mirroring exactly the structure I derived: strip small factors; prepare the candidate by decomposing `n-1 = 2^s d`; run a per-base routine that computes `a^d` and squares up the chain watching for the nonwitness conditions; and sample `t` independent bases, with error below `4^(-t)`.

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

def small_factor_prefilter(n):
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n == p:
            return True
        if n % p == 0:
            return False
    return None

def prepare_candidate(n):
    # n-1 = 2^s * d with d odd; this exposes the square-root chain.
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    return s, d

def single_base_test(n, a, state):
    # Returns False if a proves n composite, True if this base passes.
    s, d = state
    x = modpow(a, d, n)            # first term of the chain a^d
    if x == 1 or x == n - 1:
        return True               # sequence starts at 1, or hits -1 at i=0
    for _ in range(s - 1):        # square up to s-1 more times: a^(2d), a^(4d), ...
        x = (x * x) % n
        if x == n - 1:
            return True           # found a^(2^i d) == -1
        if x == 1:
            return False          # nontrivial square root of 1 -> composite
    return False                  # never reached -1; either Fermat failed or the path exposed composite

def is_probably_prime(n, trials=40):
    # randomized: for trials > 0, error < 4^(-trials) when n is composite, never rejects a prime
    if n < 2:
        return False
    pre = small_factor_prefilter(n)
    if pre is not None:
        return pre
    state = prepare_candidate(n)
    for _ in range(trials):
        a = random.randrange(2, n - 1)   # skip 1 and n-1, always nonwitnesses
        if not single_base_test(n, a, state):
            return False                  # certainly composite
    return True                           # probably prime
```
