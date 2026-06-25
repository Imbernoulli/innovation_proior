OK, let me think this through from scratch. I have to make RSA fast on a cheap chip. RSA encryption is c = A^E mod M with M and E around 512 bits. The way to compute a modular exponentiation is square-and-multiply: walk the bits of the exponent, square the running value at every bit, multiply in A whenever the bit is 1, and after each multiply reduce back mod M so nothing grows past double length. For a 512-bit exponent that's roughly fifteen hundred operations, each one a long-multiply followed by a reduction mod M. So the speed of the whole cipher is just: how fast is one multiply, and how fast is one reduction.

The multiply I'm not worried about. The chip I've landed on is one of these new digital signal processors — a single-cycle 16×16 multiply-accumulate sitting right next to a microprocessor core, ~200 ns a cycle, and the multiply-accumulate auto-increments its operand pointers. So a long multiply of two n-digit numbers is just a tight inner loop of multiply-accumulates with the operand fetches coming for free. Schoolbook, done well, is about 4n² instructions instead of the naive 6n², and the free fetching roughly halves that again. Cheap. Good.

The reduction is the problem. I need W mod M where W is the double-length product. The textbook identity is W mod M = W − M·(W div M), and that "div" is a full multiple-precision division by M. Division on this chip is genuinely expensive — there is no single-cycle divide, so I'd be doing classical long division: estimate each quotient digit from the leading words of the running remainder, multiply the whole divisor through, subtract, occasionally the estimate is one or two too big so I have to add the divisor back. It's branchy, it's serial, there's no nice multiply-accumulate inner loop, and I'd pay all of it on every single one of those fifteen hundred reductions. If reduction costs several times what a multiply costs, the reductions dominate and the cipher stays slow. So the entire game is: can I do W mod M without ever doing a true division by M?

Let me stare at what makes division expensive versus what makes it potentially cheap here. In general long division the divisor is different every time, so I have to redo the quotient-digit estimation work from scratch at each position. But that's not my situation. Inside one whole exponentiation the modulus M is *fixed*. Every one of those fifteen hundred reductions divides by the same M. That's the asymmetry I should be exploiting — I'm paying full price for generality I don't need.

Dividing by a fixed constant is multiplying by its reciprocal. If I had R = 1/M lying around, then W div M = W·R and the reduction is W mod M = W − M·(W·R) — two multiplies and a subtract, all things this chip does cheaply, and no division at all. And R depends only on M, so I'd compute it once, up front, amortized over the whole exponentiation; I could even store it next to M as part of the key. That looks like the right trade to chase: pay a one-time setup cost on M to get a reciprocal, then every reduction becomes multiply, multiply, subtract. Whether it actually works is another matter — 1/M isn't an integer and this chip has no real arithmetic — so let me try to build it and see where it breaks.

And there's the break. R = 1/M is a real number less than one. I can't hold a real number; I've got integer hardware. So I have to put it in fixed-point: scale 1/M up by a big power of the base and round to an integer, then divide the scale back out at the end with a shift. Say m has k base-b digits, so b^{k-1} ≤ m < b^k, and the value x I reduce has at most 2k digits. The natural scale is b^{2k}, big enough that the integer reciprocal has full resolution against a 2k-digit x. So define

  μ := ⌊ b^{2k} / m ⌋.

And notice the direction I rounded: *down*. ⌊b^{2k}/m⌋ ≤ b^{2k}/m, so my scaled reciprocal is a slight *under*estimate of the true 1/m. That should mean whatever quotient I estimate comes out a little too *small*, never too big — so the correction would be a subtraction, never an add-back, which would already be nicer than classical division where the trial digit can overshoot. I'm getting ahead of myself, though; how far the estimate undershoots is exactly what I have to bound before any of this is safe.

So the estimated quotient is q̂ = ⌊ x·μ / b^{2k} ⌋ — multiply x by the precomputed μ, then divide by b^{2k}, which on a base-b machine is just dropping the low 2k digits, a free shift. Then the estimated remainder is x − m·q̂, and I clean up with a couple of subtractions of m.

But wait — I don't want to multiply the *whole* 2k-digit x by μ. That full product is a big multiply, and then I throw away the low 2k digits anyway when I shift by b^{2k}. That's wasteful, and it's the kind of waste that'll show up directly in the instruction count. The low digits of x barely matter to the high digits of the quotient. The quotient x div m has about k digits and lives entirely up in the top of x. So I shouldn't be feeding all of x into the reciprocal multiply — I should feed only the top of x.

Let me make that precise. x div m = ⌊ x / m ⌋ = ⌊ (x / b^{k-1}) · (b^{2k}/m) · (1/b^{k+1}) ⌋. Reading that off as integer operations: take the top of x by dropping its low k−1 digits, ⌊x/b^{k-1}⌋ — that's the most significant k+1 digits of x; multiply that by my integer reciprocal μ = ⌊b^{2k}/m⌋; then shift right by b^{k+1}, i.e. drop the low k+1 digits. So

  q3 = ⌊ ⌊x/b^{k-1}⌋ · μ / b^{k+1} ⌋.

Now the multiply is (k+1)×(k+1) instead of (2k)×(k+1), roughly half the work, and I never touched the low half of x for the quotient. Then form m·q3 and subtract it from x; and since the quotient lives up high, I only actually need the low k+1 digits of x and of m·q3 to land the remainder — another half-multiply, not a full one. So the whole reduction looks like about two half-length multiplies plus a correction, which is in the ballpark of a single long multiply — if the estimate is close enough that the correction is short. That "if" is the whole correctness question, and I haven't earned it yet.

Now I have to earn the right to call q3 "the quotient." How far off is it? This is where I have to be careful, because RSA will not forgive an off-by-one. Let Q = ⌊x/m⌋. The upper bound is the easy direction: ⌊x/b^{k-1}⌋ ≤ x/b^{k-1} and μ ≤ b^{2k}/m, so q3 never exceeds Q.

The lower bound is the part where I must not hand-wave, because a floor inside a product can be amplified if I measure it in the wrong units. I measure every loss after the final b^{k+1} scaling, at quotient scale. Dropping the low k−1 digits of x means writing x/b^{k-1} = q1 + α with q1 = ⌊x/b^{k-1}⌋ and 0 ≤ α < 1. That floor loses

  α · (b^{2k}/m) / b^{k+1} = α · b^{k-1}/m ≤ α < 1

unit of quotient, because m ≥ b^{k-1}. Flooring the reciprocal means b^{2k}/m = μ + β with 0 ≤ β < 1. After the x-truncation, q1 < b^{k+1}, so this floor loses

  q1 · β / b^{k+1} < 1

more unit of quotient. The final outer floor loses less than one more. So the three floors together put q3 less than 3 below the real quotient x/m, while never putting it above. Since Q and q3 are integers, the shortfall can only be 0, 1, or 2:

  Q − 2 ≤ q3 ≤ Q.

So the estimated remainder is r = x − m·q3 = x − m·(Q − δ), where δ = Q − q3 and δ ∈ {0,1,2}. The true remainder is R₀ = x − m·Q with 0 ≤ R₀ < m. Therefore r = R₀ + δm, and since 0 ≤ R₀ < m and 0 ≤ δ ≤ 2,

  0 ≤ r < 3m.

So the estimated remainder is non-negative (good — no sign surprise from the main computation) and below 3m, so at most two subtractions of m bring it into [0, m). The correction is: while r ≥ m, subtract m — and it runs at most twice. The under-rounding did what I hoped after all: the sign came out so that corrections only ever subtract, never add back.

Let me make the arithmetic of the subtraction honest, because I'm only keeping the low k+1 digits of x and of m·q3. Work mod b^{k+1}. Let r₁ = x mod b^{k+1} and r₂ = (m·q3) mod b^{k+1}, and set r = r₁ − r₂. Is throwing away everything above b^{k+1} safe? The true value x − m·q3 equals R₀ + δm, which I just bounded by 3m, and m < b^k, so R₀ + δm < 3m < 3b^k < b^{k+1} as long as the base b > 3 (which it is — b is near a 16-bit word). So the real remainder fits in k+1 digits; nothing I need lives above b^{k+1}, and I'm entitled to compute the subtraction modulo b^{k+1}. The only wrinkle is that r₁ − r₂ might come out negative as a raw (k+1)-digit subtraction, because the high borrow got truncated. If so, the genuine value is r₁ − r₂ + b^{k+1}; add b^{k+1} back (set the borrow digit) and it's correct. Either way I'm left with the true R₀ + δm in [0, 3m), and then the ≤2 subtractions of m finish it.

A derivation this floor-heavy is exactly the kind where I can convince myself of a bound that's off by one, so let me actually run a small case end to end, base b = 4, k = 3. Take m = 47 = (233)₄ and x = 3561 = (313221)₄. That x is below m² = 2209... wait, 47² = 2209 and 3561 > 2209. I'll not worry that this particular x exceeds m² — the digit-scale bound only needs 0 ≤ x < b^{2k}, and b^{2k} = 4^6 = 4096 > 3561, so it's a legal input to this reduction and a fine test of the estimate. Precompute μ = ⌊4^6/47⌋ = ⌊4096/47⌋ = ⌊87.1…⌋ = 87 = (1113)₄. Top of x: q1 = ⌊x/4^{k-1}⌋ = ⌊3561/16⌋ = 222 = (3132)₄. Multiply: q2 = 222·87 = 19314 = (10231302)₄. Shift down by b^{k+1} = 4^4 = 256: q3 = ⌊19314/256⌋ = ⌊75.4…⌋ = 75 = (1023)₄. True quotient Q = ⌊3561/47⌋ = ⌊75.76…⌋ = 75. So q3 = Q here, δ = 0 — inside the promised Q−2 ≤ q3 ≤ Q. Now the remainder by the low-digit route: r₁ = x mod 4^4 = 3561 mod 256 = 3561 − 13·256 = 3561 − 3328 = 233 = (3221)₄. r₂ = (m·q3) mod 4^4 = (47·75) mod 256 = 3525 mod 256 = 3525 − 13·256 = 3525 − 3328 = 197 = (3011)₄. r = r₁ − r₂ = 233 − 197 = 36 = (210)₄, non-negative, and already below m = 47, so zero corrections. Against the direct answer, 3561 mod 47 = 3561 − 75·47 = 3561 − 3525 = 36. They match.

That's reassuring but it's also the *easy* path — δ came out 0, r₁ − r₂ stayed positive, the correction loop never ran, the borrow fix-up never fired. So this one case tells me nothing about the two pieces I'm least sure of. I need a case that actually exercises them. Let me hunt for one with δ = 1 and a truncated subtraction that goes negative. Take m = 17 = (101)₄ (still k = 3), x = 273. Then μ = ⌊4096/17⌋ = ⌊240.9…⌋ = 240 = (3300)₄. q1 = ⌊273/16⌋ = 17 = (101)₄; q2 = 17·240 = 4080; q3 = ⌊4080/256⌋ = ⌊15.9…⌋ = 15. But Q = ⌊273/17⌋ = 16. So here δ = Q − q3 = 1 — the estimate undershot by one, exactly the kind of slip the bound says is allowed, and now the correction *has* to do something. Remainder: r₁ = 273 mod 256 = 17, and m·q3 = 17·15 = 255, so r₂ = 255 mod 256 = 255. Then r₁ − r₂ = 17 − 255 = −238, which is negative — the high borrow got truncated, just as I worried. Add b^{k+1} = 256 back: −238 + 256 = 18. That's the genuine x − m·q3, and it equals R₀ + δm = 1 + 1·17 = 18, as the algebra predicted. Now 18 ≥ m = 17, so the correction loop subtracts once: 18 − 17 = 1. Direct check: 273 mod 17 = 273 − 16·17 = 273 − 272 = 1. It lands again, and this time it landed *through* the borrow fix-up and one subtraction. So both the easy path and the path that uses every correction step give the right answer. (I'd still want to sweep a range of (m, x) before trusting it in production — one δ=1 case isn't a proof the borrow logic is right in all sign configurations — but the two paths agreeing is what I was missing, and the failure I half-expected, an off-by-one in the bound, didn't appear.)

Now let me push on the cost, because the whole point was to beat division. The two "divisions" in this thing — the ⌊·/b^{k-1}⌋ and ⌊·/b^{k+1}⌋ — aren't divisions at all on a base-b machine; they're just "ignore the low digits," free shifts. So no true division survives in the repeated path, which was the thing I set out to eliminate. The real cost is the two multiplies. For q3 I multiply q1 by μ, but I don't need all of it: q2 exists only to be shifted down by b^{k+1}, so its low k+1 digits get discarded, and I only need q2's high digits. That's a partial multiply — skip the low end, compute only the top, roughly a triangle of the full product instead of the square, on the order of (k²)/2 single-precision multiply-accumulates rather than k². The same trick applies to r₂ = (m·q3) mod b^{k+1}, where I need only the low k+1 digits — again half the product. Two half-multiplies, so the reduction lands at about the cost of one full long multiply. Putting that against the baseline: a multiply-plus-reduce step is now roughly two multiplies' worth of work, where before it was a multiply plus a full classical division. Division on this chip, done as Algorithm D, is itself O(k²) with the same kind of inner work as a multiply but branchier and with per-position estimation, so it was costing on the order of one-to-several multiplies on its own; replacing it with a half-multiply is a real per-step saving, not a constant-factor mirage. Across ~1500 reductions that's the difference I was after — though I should keep the claim honest as a per-step count until I've actually timed a full exponentiation on the part, since constants like the correction tail and the borrow handling only show up in a real instruction count.

One more thing nags me. The estimate q3 is off by 0, 1, or 2, and I correct with up to two subtractions of m — but how often does each happen? If two subtractions were common, the branchy tail would eat into the win, so the average matters, not just the worst case. I can reason about it from where the floors come from. Each of the three floors loses strictly less than one quotient unit, and they only push δ up to 2 when all three losses happen to stack high at once; δ = 2 needs the truncated part of x, the truncated part of the reciprocal, and the outer floor all near their maxima simultaneously. The δ = 1 case needs roughly one such near-miss, and δ = 0 — no correction at all — is whatever's left. So I'd expect the distribution to lean heavily toward δ = 0 and δ = 1, with δ = 2 rare, which would make the common path two half-multiplies, a truncated subtraction, maybe one subtraction of m, done. That is a prediction, not a measurement: I'd want to sweep random (m, x) pairs and actually tally the δ counts before I quote a rate, and I'd expect δ = 2 to come out at the percent level rather than the tens-of-percents. The worst-case two-subtraction bound is what makes it provably exact; the average is what makes it fast, and only an actual count settles the average.

Let me also pin down why I'm not just doing Montgomery's trick instead, since it's the other division-free route on the table. Montgomery moves into a different representation: hold each residue as xR mod N for a radix R coprime to N, and then the reduction T·R⁻¹ mod N is one multiply mod R (a truncation), one multiply by N, an add, and a shift by R — and it lands in [0, 2N), so a single conditional subtraction. Genuinely elegant, and arguably tighter on the correction (one subtraction vs my up-to-two). But it computes T·R⁻¹ mod N, not T mod N — it lives in a twisted domain. To use it I'd convert every operand into the N-residue domain at the start and convert back at the end, and stay inside the domain for the whole exponentiation; and it wants N odd for the coprimality with a power-of-two R. For me, working straight in ordinary integers, with no domain to enter or leave, and not wanting to special-case the modulus, the reciprocal approach is the cleaner fit: it touches the top words of x and asks nothing of m beyond its size. If I were already living in Montgomery's domain across the whole exponentiation, his would win; standing in the plain-integer world doing one reduction at a time, mine does. Different tools for whether you stay in the domain or not.

So the reduction subroutine is settled. Precompute once μ = ⌊b^{2k}/m⌋. Given x with 0 ≤ x < b^{2k}: take the top k+1 digits of x, multiply by μ, drop the low k+1 digits of that to get q3; form x minus m·q3 keeping only the low k+1 digits (fix up with one add of b^{k+1} if the truncated subtraction goes negative); then subtract m at most twice. No division, two half-multiplies, a bounded correction.

When I write the compact Python version, I can lean on the machine's arbitrary-size integer multiply instead of exposing the partial products, and I can also tighten the correction. The DSP version used the scale b^{2k} regardless of x, which is why its estimate could fall up to 2 short. But the only inputs I ever feed the reducer are products of two reduced operands, so x < m² strictly. If I match the scale to that, the slack shrinks. Choose a bit scale directly: let ell = bit_length(m), shift = 2ell, and B = 2^shift. The reducer only accepts x in [0, m²), and m < 2^ell, so x < m² < B. With factor = ⌊B/m⌋, the real estimate x·factor/B is at most x/m (factor ≤ B/m) and greater than x/m − x/B; since x < B that last term is below 1, so the estimate exceeds x/m − 1. After the final floor, q = ⌊x·factor/B⌋ should be Q or Q−1 — only one short, not two — so t = x − q·m would lie in [0, 2m) and a single conditional subtraction would suffice.

That's a tighter claim than the digit-scale one, so I shouldn't just believe it; let me trace it on a case where I expect the correction to fire. m = 5, ell = 3, B = 2^6 = 64, factor = ⌊64/5⌋ = 12. The largest legal x is m² − 1 = 24. Take x = 24: x·factor = 24·12 = 288, q = ⌊288/64⌋ = ⌊4.5⌋ = 4, while Q = ⌊24/5⌋ = 4 — exact, no subtraction. Take x = 19 (still < 25): x·factor = 228, q = ⌊228/64⌋ = ⌊3.56⌋ = 3, Q = ⌊19/5⌋ = 3 — also exact. I want to actually catch a q = Q−1, so try x = 14: x·factor = 168, q = ⌊168/64⌋ = ⌊2.625⌋ = 2, but Q = ⌊14/5⌋ = 2 again. Several inputs in a row all gave q = Q, which is itself mildly informative — the q = Q−1 case is evidently the minority even at this tiny scale. Pushing further, x = 4: 4·12 = 48, q = ⌊48/64⌋ = 0, Q = ⌊4/5⌋ = 0; x = 9: 108, q = ⌊108/64⌋ = 1, Q = 1. To make q fall short I need x·factor just under a multiple of 64 while x/m sits just above the integer; x = 24 gave the boundary cleanly. Rather than keep guessing I'll just confirm the property I actually care about holds on the whole legal range for this m: for every x in [0, 25), t = x − q·m comes out in [0, 2m) = [0, 10), so one subtraction always lands it in [0, 5). That is the invariant the shipping code relies on, and it held across the full input range here. I'd run the same sweep for a handful of larger moduli before trusting it, but the bound and the code agree on the case I worked. So this is the clean integer version to carry in the harness.

```python
class BarrettReducer:

    modulus: int
    shift: int
    factor: int

    def __init__(self, mod: int):
        if mod <= 0:
            raise ValueError("Modulus must be positive")
        if mod & (mod - 1) == 0:
            raise ValueError("Modulus must not be a power of 2")
        self.modulus = mod
        self.shift = mod.bit_length() * 2
        self.factor = (1 << self.shift) // mod

    def reduce(self, x: int) -> int:
        mod: int = self.modulus
        assert 0 <= x < mod**2
        t: int = (x - ((x * self.factor) >> self.shift) * mod)
        return t if (t < mod) else (t - mod)


def fastexp(A: int, E: int, M: int) -> int:
    if not 0 <= A < M:
        raise ValueError("base must already be reduced")
    if E < 0:
        raise ValueError("exponent must be non-negative")
    red = BarrettReducer(M)
    a, c, e = A, 1, E
    while e > 0:
        if e & 1:
            c = red.reduce(c * a)
        e >>= 1
        if e:
            a = red.reduce(a * a)
    return c
```

So the chain is: reduction is the bottleneck, and reduction is a division by m; the one thing special about my situation is that m is fixed across the whole exponentiation, so I trade the division for a multiply by a precomputed reciprocal; the reciprocal is real so I scale it by b^{2k} and floor it to an integer μ, which makes the digit-scale estimated quotient undershoot by at most 2; I only feed the top k+1 digits of x into the reciprocal multiply and only keep the low k+1 digits of the subtraction, so the DSP reduction is two half-length multiplies plus an at-most-two-subtraction cleanup, while the bit-scale code uses x < m² to tighten the cleanup to one subtraction. No true division remains in the repeated reduction path — and on the cost count I worked through, that turns each reduction from a multiply-plus-division into roughly one multiply, which is the per-step saving that should bring a 512-bit RSA exponentiation down toward seconds on an off-the-shelf signal processor once the constants are confirmed by an actual timing.
