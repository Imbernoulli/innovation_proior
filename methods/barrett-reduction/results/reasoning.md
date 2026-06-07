OK, let me think this through from scratch. I have to make RSA fast on a cheap chip. RSA encryption is c = A^E mod M with M and E around 512 bits. The way to compute a modular exponentiation is square-and-multiply: walk the bits of the exponent, square the running value at every bit, multiply in A whenever the bit is 1, and after each multiply reduce back mod M so nothing grows past double length. For a 512-bit exponent that's roughly fifteen hundred operations, each one a long-multiply followed by a reduction mod M. So the speed of the whole cipher is just: how fast is one multiply, and how fast is one reduction.

The multiply I'm not worried about. The chip I've landed on is one of these new digital signal processors — a single-cycle 16×16 multiply-accumulate sitting right next to a microprocessor core, ~200 ns a cycle, and the multiply-accumulate auto-increments its operand pointers. So a long multiply of two n-digit numbers is just a tight inner loop of multiply-accumulates with the operand fetches coming for free. Schoolbook, done well, is about 4n² instructions instead of the naive 6n², and the free fetching roughly halves that again. Cheap. Good.

The reduction is the problem. I need W mod M where W is the double-length product. The textbook identity is W mod M = W − M·(W div M), and that "div" is a full multiple-precision division by M. Division on this chip is genuinely expensive — there is no single-cycle divide, so I'd be doing classical long division: estimate each quotient digit from the leading words of the running remainder, multiply the whole divisor through, subtract, occasionally the estimate is one or two too big so I have to add the divisor back. It's branchy, it's serial, there's no nice multiply-accumulate inner loop, and I'd pay all of it on every single one of those fifteen hundred reductions. If reduction costs several times what a multiply costs, the reductions dominate and the cipher stays slow. So the entire game is: can I do W mod M without ever doing a true division by M?

Let me stare at what makes division expensive versus what makes it potentially cheap here. In general long division the divisor is different every time, so I have to redo the quotient-digit estimation work from scratch at each position. But that's not my situation. Inside one whole exponentiation the modulus M is *fixed*. Every one of those fifteen hundred reductions divides by the same M. That's the asymmetry I should be exploiting — I'm paying full price for generality I don't need.

Dividing by a fixed constant is multiplying by its reciprocal. If I had R = 1/M lying around, then W div M = W·R and the reduction is W mod M = W − M·(W·R) — two multiplies and a subtract, all things this chip does cheaply, and no division at all. And R depends only on M, so I compute it once, up front, amortized over the whole exponentiation; I could even store it next to M as part of the key. So the shape of the answer is clear: precompute a reciprocal, turn every reduction into multiply, multiply, subtract.

Now the wall. R = 1/M is a real number less than one. I can't hold a real number; I've got integer hardware. So I have to put it in fixed-point: scale 1/M up by a big power of the base and round to an integer, then divide the scale back out at the end with a shift. Say m has k base-b digits, so b^{k-1} ≤ m < b^k, and the value x I reduce has at most 2k digits. The natural scale is b^{2k}, big enough that the integer reciprocal has full resolution against a 2k-digit x. So define

  μ := ⌊ b^{2k} / m ⌋.

And notice: I rounded *down*. ⌊b^{2k}/m⌋ ≤ b^{2k}/m, so my scaled reciprocal is a slight *under*estimate of the true 1/m. Hold onto that — it means whatever quotient I estimate will come out a little too *small*, never too big. The correction will be a subtraction, never an add-back. That's already nicer than classical division, where the trial digit can overshoot.

So the estimated quotient is q̂ = ⌊ x·μ / b^{2k} ⌋ — multiply x by the precomputed μ, then divide by b^{2k}, which on a base-b machine is just dropping the low 2k digits, a free shift. Then the estimated remainder is x − m·q̂, and I clean up with a couple of subtractions of m.

But wait — I don't want to multiply the *whole* 2k-digit x by μ. That full product is a big multiply, and then I throw away the low 2k digits anyway when I shift by b^{2k}. That's wasteful, and it's the kind of waste that'll show up directly in the instruction count. The low digits of x barely matter to the high digits of the quotient. The quotient x div m has about k digits and lives entirely up in the top of x. So I shouldn't be feeding all of x into the reciprocal multiply — I should feed only the top of x.

Let me make that precise. x div m = ⌊ x / m ⌋ = ⌊ (x / b^{k-1}) · (b^{2k}/m) · (1/b^{k+1}) ⌋. Reading that off as integer operations: take the top of x by dropping its low k−1 digits, ⌊x/b^{k-1}⌋ — that's the most significant k+1 digits of x; multiply that by my integer reciprocal μ = ⌊b^{2k}/m⌋; then shift right by b^{k+1}, i.e. drop the low k+1 digits. So

  q3 = ⌊ ⌊x/b^{k-1}⌋ · μ / b^{k+1} ⌋.

Now the multiply is (k+1)×(k+1) instead of (2k)×(k+1), roughly half the work, and I never touched the low half of x for the quotient. Then form m·q3 and subtract it from x; and since the quotient lives up high, I only actually need the low k+1 digits of x and of m·q3 to land the remainder — another half-multiply, not a full one. So the whole reduction is about two half-length multiplies plus the correction, which is in the ballpark of a single long multiply. That's the win: reduction stops costing several multiplies and costs about one.

Now I have to earn the right to call q3 "the quotient." How far off is it? This is where I have to be careful, because RSA will not forgive an off-by-one. Let Q = ⌊x/m⌋. The upper bound is the easy direction: ⌊x/b^{k-1}⌋ ≤ x/b^{k-1} and μ ≤ b^{2k}/m, so q3 never exceeds Q.

The lower bound is the part where I must not hand-wave, because a floor inside a product can be amplified if I measure it in the wrong units. I measure every loss after the final b^{k+1} scaling, at quotient scale. Dropping the low k−1 digits of x means writing x/b^{k-1} = q1 + α with q1 = ⌊x/b^{k-1}⌋ and 0 ≤ α < 1. That floor loses

  α · (b^{2k}/m) / b^{k+1} = α · b^{k-1}/m ≤ α < 1

unit of quotient, because m ≥ b^{k-1}. Flooring the reciprocal means b^{2k}/m = μ + β with 0 ≤ β < 1. After the x-truncation, q1 < b^{k+1}, so this floor loses

  q1 · β / b^{k+1} < 1

more unit of quotient. The final outer floor loses less than one more. So the three floors together put q3 less than 3 below the real quotient x/m, while never putting it above. Since Q and q3 are integers, the shortfall can only be 0, 1, or 2:

  Q − 2 ≤ q3 ≤ Q.

So the estimated remainder is r = x − m·q3 = x − m·(Q − δ), where δ = Q − q3 and δ ∈ {0,1,2}. The true remainder is R₀ = x − m·Q with 0 ≤ R₀ < m. Therefore r = R₀ + δm, and since 0 ≤ R₀ < m and 0 ≤ δ ≤ 2,

  0 ≤ r < 3m.

That's it — the estimated remainder is non-negative (good, no sign surprise from the main computation) and below 3m, so at most two subtractions of m bring it into [0, m). The correction is: while r ≥ m, subtract m — and it runs at most twice. The under-rounding paid off exactly as predicted: corrections only ever subtract.

Let me make the arithmetic of the subtraction honest, because I'm only keeping the low k+1 digits of x and of m·q3. Work mod b^{k+1}. Let r₁ = x mod b^{k+1} and r₂ = (m·q3) mod b^{k+1}, and set r = r₁ − r₂. Is throwing away everything above b^{k+1} safe? The true value x − m·q3 equals R₀ + δm, which I just bounded by 3m, and m < b^k, so R₀ + δm < 3m < 3b^k < b^{k+1} as long as the base b > 3 (which it is — b is near a 16-bit word). So the real remainder fits in k+1 digits; nothing I need lives above b^{k+1}, and I'm entitled to compute the subtraction modulo b^{k+1}. The only wrinkle is that r₁ − r₂ might come out negative as a raw (k+1)-digit subtraction, because the high borrow got truncated. If so, the genuine value is r₁ − r₂ + b^{k+1}; add b^{k+1} back (set the borrow digit) and it's correct. Either way I'm left with the true R₀ + δm in [0, 3m), and then the ≤2 subtractions of m finish it.

Let me actually walk a small case to make sure I haven't fooled myself, base b = 4, k = 3. Take m = 47 = (233)₄ and x = 3561 = (313221)₄, which is below m² = 2209... wait, 47² = 2209 and 3561 > 2209. Let me not worry that this particular x exceeds m² — the digit-scale bound only needs 0 ≤ x < b^{2k}, and b^{2k} = 4^6 = 4096 > 3561, so it's a legal input to this reduction and a fine test of the estimate. Precompute μ = ⌊4^6/47⌋ = ⌊4096/47⌋ = ⌊87.1…⌋ = 87 = (1113)₄. Top of x: q1 = ⌊x/4^{k-1}⌋ = ⌊3561/16⌋ = 222 = (3132)₄. Multiply: q2 = 222·87 = 19314 = (10231302)₄. Shift down by b^{k+1} = 4^4 = 256: q3 = ⌊19314/256⌋ = ⌊75.4…⌋ = 75 = (1023)₄. True quotient Q = ⌊3561/47⌋ = ⌊75.76…⌋ = 75. So q3 = Q exactly here, δ = 0 — within the promised Q−2 ≤ q3 ≤ Q. Now the remainder by the low-digit route: r₁ = x mod 4^4 = 3561 mod 256 = 3561 − 13·256 = 3561 − 3328 = 233 = (3221)₄. r₂ = (m·q3) mod 4^4 = (47·75) mod 256 = 3525 mod 256 = 3525 − 13·256 = 3525 − 3328 = 197 = (3011)₄. r = r₁ − r₂ = 233 − 197 = 36 = (210)₄, already non-negative and already below m = 47, zero corrections. And 3561 mod 47 = 3561 − 75·47 = 3561 − 3525 = 36. It lands. The machinery is sound.

Now let me push on the cost, because the whole point was to beat division. The divisions in this thing — the ⌊·/b^{k-1}⌋ and ⌊·/b^{k+1}⌋ — aren't divisions at all on a base-b machine, they're just "ignore the low digits," free shifts. So there is no true division anywhere; I kept my promise. The real cost is the two multiplies. For q3 I multiply q1 by μ, but I don't need all of it. q2 is only there to be shifted down by b^{k+1}, so its low k+1 digits get discarded; I only need q2's high digits. So that's a partial multiply — skip the low end, compute only the top, roughly a triangle of the full product instead of the square, on the order of (k²)/2 single-precision multiply-accumulates rather than k². Same trick on r₂ = (m·q3) mod b^{k+1}: there I need only the low k+1 digits, so again only half the product. Two half-multiplies. So the reduction costs about what one full long multiply costs — and a multiply-plus-reduce step is now roughly two multiplies' worth of work instead of a multiply plus an expensive division. That's the factor that turns minutes into seconds.

One more thing nags me. The estimate q3 is off by 0, 1, or 2, and I correct with up to two subtractions of m — but how often does each happen? If two subtractions were common the branchy tail would eat into the win. The error calculation says the initial value should already be below m for most x and m, and exceeding 2m should be rare, so the common path is: two half-multiplies, a truncated subtraction, done. The correction loop is there for exactness but almost never fully exercised. Exactly the cost profile I want: cheap on average, and provably correct in the worst case.

Let me also pin down why I'm not just doing Montgomery's trick instead, since it's the other division-free route on the table. Montgomery moves into a different representation: hold each residue as xR mod N for a radix R coprime to N, and then the reduction T·R⁻¹ mod N is one multiply mod R (a truncation), one multiply by N, an add, and a shift by R — and it lands in [0, 2N), so a single conditional subtraction. Genuinely elegant, and arguably tighter on the correction (one subtraction vs my up-to-two). But it computes T·R⁻¹ mod N, not T mod N — it lives in a twisted domain. To use it I'd convert every operand into the N-residue domain at the start and convert back at the end, and stay inside the domain for the whole exponentiation; and it wants N odd for the coprimality with a power-of-two R. For me, working straight in ordinary integers, with no domain to enter or leave, and not wanting to special-case the modulus, the reciprocal approach is the cleaner fit: it touches the top words of x and asks nothing of m beyond its size. If I were already living in Montgomery's domain across the whole exponentiation, his would win; standing in the plain-integer world doing one reduction at a time, mine does. Different tools for whether you stay in the domain or not.

So the reduction subroutine is settled. Precompute once μ = ⌊b^{2k}/m⌋. Given x with 0 ≤ x < b^{2k}: take the top k+1 digits of x, multiply by μ, drop the low k+1 digits of that to get q3; form x minus m·q3 keeping only the low k+1 digits (fix up with one add of b^{k+1} if the truncated subtraction goes negative); then subtract m at most twice. No division, two half-multiplies, a bounded correction.

When I write the compact Python version, I can use the machine's arbitrary-size integer multiply instead of exposing the partial products. Then I choose a bit scale directly: let ell = bit_length(m), shift = 2ell, and B = 2^shift. The reducer only accepts x in [0, m²), and m < 2^ell, so x < m² < B. With factor = ⌊B/m⌋, the real estimate x·factor/B is at most x/m and greater than x/m − x/B, hence greater than x/m − 1. After the final floor, q = ⌊x·factor/B⌋ is Q or Q−1. Then t = x − q·m is in [0, 2m), and one conditional subtraction is enough. Let me write that version because it is the clean integer code I actually want to carry in the harness.

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

So the chain is: reduction is the bottleneck, and reduction is a division by m; the one thing special about my situation is that m is fixed across the whole exponentiation, so I trade the division for a multiply by a precomputed reciprocal; the reciprocal is real so I scale it by b^{2k} and floor it to an integer μ, which makes the digit-scale estimated quotient undershoot by at most 2; I only feed the top k+1 digits of x into the reciprocal multiply and only keep the low k+1 digits of the subtraction, so the DSP reduction is two half-length multiplies plus an at-most-two-subtraction cleanup, while the bit-scale code uses x < m² to tighten the cleanup to one subtraction. No true division remains in the repeated reduction path, and that is what makes a 512-bit RSA exponentiation finish in seconds on an off-the-shelf signal processor.
