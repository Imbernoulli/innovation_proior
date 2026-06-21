The operation that hurts is reduction, not multiplication. With a fixed odd modulus $N$, I repeatedly need products $a b \bmod N$, and forming the double-width product $T = ab$ from $k$-bit inputs is regular, predictable work. The pain is turning $T$ back into a residue. The textbook route computes $q = \lfloor T / N \rfloor$ and returns $T - qN$, which means dividing by an arbitrary multi-precision integer: estimating quotient digits, correcting them, and propagating carries. That trial-quotient machinery is irregular in software and awkward in hardware, and it sits inside every single product of a modular exponentiation or a factoring loop, so it dominates the cost. The alternatives each dodge the problem only partially. Binary exponentiation controls how many products there are but leaves each one as an ordinary division-heavy reduction. Special-form moduli near a power of two permit fast high-bit folding, but they require $N$ to have a convenient shape that general public-key moduli do not. Residue-number and redundant representations cut carry propagation but add representation machinery and are not a drop-in for repeated products modulo an arbitrary odd $N$. What I want is a reduction that works for any odd modulus and replaces division by $N$ with shifts, masks, and multiply-adds, accepting some one-time per-modulus setup because cryptographic chains amortize it over many products.

The method is Montgomery multiplication. The crucial observation is that division by a radix is cheap in a way division by $N$ is not: if I pick $R = 2^k$, then $T \bmod R$ is just the low $k$ bits of $T$ and exact division by $R$ is a right shift. The difficulty is that $R$ has nothing to do with the residue class modulo $N$ — but because $N$ is odd, $\gcd(N, R) = 1$, so $R$ is invertible modulo $N$ and $N$ is invertible modulo $R$. I exploit this by adding a carefully chosen multiple of $N$ to $T$ to force it divisible by $R$ without changing its residue. Given $T$ with $0 \le T < RN$, I want an integer congruent to $T R^{-1} \bmod N$, since multiplying by $R^{-1}$ in the ring is exactly what dividing by $R$ in the integers will accomplish once the value is made divisible by $R$. Adding $mN$ leaves the residue mod $N$ untouched, so I solve for the $m$ that kills the low bits: I require $T + mN \equiv 0 \pmod R$, i.e. $mN \equiv -T \pmod R$, and since $N$ is invertible mod $R$ this gives $m \equiv -T N^{-1} \pmod R$. Only the low bits of $T$ matter here, so I precompute $N' = -N^{-1} \bmod R$ once per modulus and set $m = (T \bmod R)\, N' \bmod R$. The reduction step REDC is therefore
$$m = (T \bmod R)\, N' \bmod R, \qquad t = \frac{T + mN}{R}, \qquad t \leftarrow t - N \ \text{if}\ t \ge N.$$
That the quotient is right follows from multiplying its definition by $R$: $tR = T + mN$, and reducing modulo $N$ the $mN$ term vanishes, so $tR \equiv T \pmod N$, hence $t \equiv T R^{-1} \pmod N$ because $R$ is invertible mod $N$. The range is controlled too: the choice gives $0 \le m < R$, and with $0 \le T < RN$ we get $0 \le T + mN < 2RN$, so after the shift $0 \le t < 2N$. There is at most one excess copy of $N$, so a single conditional subtraction — cheaper than any division — lands $t$ in $[0, N-1]$ while preserving the congruence. The only operations are a mask, a multiply by the precomputed $N'$, an add, a right shift, and one compare-subtract; division by $N$ is gone.

A single reduction alone is not enough, because applying REDC to $ab$ returns $ab R^{-1}$, which is off by a stray factor of $R^{-1}$. The fix is to change the representation so that the factor REDC removes is exactly the one multiplication introduces. I store a residue $x$ in the Montgomery domain as $\bar{x} = xR \bmod N$. Then the product of two stored values is $\bar{a}\bar{b} = (aR)(bR) = abR^2 \bmod N$, and one REDC removes a single factor of $R$ to give $abR \bmod N$, which is precisely $\overline{ab}$, the stored representative of the product. The representation is closed under multiplication: a whole chain of squares and multiplies stays encoded with one REDC per product and never touches a division by $N$. The bounds also line up for free — encoded values live in $[0, N-1]$, so their product is below $N^2$, and since I choose $R > N$ this is below $RN$, exactly the REDC input condition, so the same inequality serves the proof and the implementation. The remaining worry is the boundary: computing $\bar{x} = xR \bmod N$ directly is itself a modular reduction, the thing I am trying to avoid. But with $N$ fixed I precompute $R^2 \bmod N$ once, and then $\mathrm{REDC}\!\big((x \bmod N)(R^2 \bmod N)\big) = xR^2 R^{-1} = xR \bmod N$ encodes with one REDC, while $\mathrm{REDC}(\bar{x}) = xR R^{-1} = x \bmod N$ decodes with one REDC. For a lone product this is overhead, but for exponentiation I convert the base once, keep the accumulator encoded through every square and multiply, and decode once at the end, so the conversion is paid only at the two edges of a long chain. I also note that for a multi-word radix $R = b^n$ the same congruence can be enforced one word at a time: with $\mu = -N_0^{-1} \bmod b$ for the low word $N_0$, at step $i$ choosing $q_i = T_i \mu \bmod b$ makes word $i$ of $T + q_i N b^i$ equal to $T_i(1 + \mu N_0) \equiv 0 \pmod b$, so after $n$ steps the low $n$ words are zero and the final shift is just dropping them — letting a multi-precision implementation interleave the reduction with the multiply-add instead of building a separate divider. A quick numerical check on $N = 17$, $R = 32$ confirms it: $N' = 15$ since $17 \cdot 15 = 255 \equiv -1 \pmod{32}$; encoding $7 \mapsto 3$ and $11 \mapsto 12$, their encoded product $T = 36$ gives $m = 4 \cdot 15 \bmod 32 = 28$ and $t = (36 + 28\cdot 17)/32 = 16$, which equals $9$ encoded ($9 \cdot 32 \bmod 17 = 16$), and decoding $16$ returns $9 = 7 \cdot 11 \bmod 17$.

```python
from random import Random


def inverse_mod(a, m):
    """Return a^{-1} mod m for coprime positive integers a and m."""
    if m <= 0:
        raise ValueError("modulus must be positive")
    a %= m
    old_r, r = a, m
    old_s, s = 1, 0
    while r:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
    if old_r != 1:
        raise ValueError("inverse does not exist")
    return old_s % m


class FixedModulusArithmetic:
    def __init__(self, modulus, word_bits=None):
        if modulus <= 1 or modulus % 2 == 0:
            raise ValueError("the modulus must be an odd integer greater than 1")
        self.N = modulus
        self.word_bits = word_bits

        if word_bits is None:
            self.k = modulus.bit_length()
        else:
            if word_bits <= 0:
                raise ValueError("word_bits must be positive")
            self.k = ((modulus.bit_length() + word_bits - 1) // word_bits) * word_bits

        self.R = 1 << self.k
        self.mask = self.R - 1
        self.N_prime = (-inverse_mod(self.N, self.R)) & self.mask
        self.R2 = (self.R * self.R) % self.N

    def reduce_product(self, T):
        """REDC: return T * R^{-1} mod N for 0 <= T < R*N."""
        if T < 0 or T >= self.R * self.N:
            raise ValueError("REDC input must satisfy 0 <= T < R*N")
        m = ((T & self.mask) * self.N_prime) & self.mask
        t = (T + m * self.N) >> self.k
        if t >= self.N:
            t -= self.N
        return t

    def redc(self, T):
        """Explicit REDC alias for the product reduction step."""
        return self.reduce_product(T)

    def encode(self, x):
        """Return xR mod N."""
        return self.reduce_product((x % self.N) * self.R2)

    def decode(self, x_encoded):
        """Return the standard residue represented by x_encoded."""
        return self.reduce_product(x_encoded % self.N)

    def mul_encoded(self, a_encoded, b_encoded):
        """Multiply two Montgomery-domain representatives."""
        a_encoded %= self.N
        b_encoded %= self.N
        return self.reduce_product(a_encoded * b_encoded)

    def modmul(self, a, b):
        a_encoded = self.encode(a)
        b_encoded = self.encode(b)
        return self.decode(self.mul_encoded(a_encoded, b_encoded))

    def modpow(self, base, exponent):
        if exponent < 0:
            raise ValueError("negative exponents are not supported")
        result = self.encode(1)
        x = self.encode(base)
        while exponent:
            if exponent & 1:
                result = self.mul_encoded(result, x)
            x = self.mul_encoded(x, x)
            exponent >>= 1
        return self.decode(result)


def verify():
    cases = [
        (3, 0, 2, 0),
        (17, 7, 11, 13),
        (187, 42, 55, 19),
        (104729, 12345, 67890, 1000),
        (1_000_000_007, 123456789, 987654321, 13579),
        ((1 << 64) - 59, (1 << 63) - 1, (1 << 62) + 1, 257),
    ]

    rng = Random(20260607)
    for bits in (8, 16, 32, 80, 127):
        for _ in range(20):
            N = rng.getrandbits(bits) | 1 | (1 << (bits - 1))
            a = rng.randrange(0, 3 * N)
            b = rng.randrange(0, 3 * N)
            exponent = rng.randrange(0, 5000)
            cases.append((N, a, b, exponent))

    checks = 0
    for N, a, b, exponent in cases:
        for word_bits in (None, 8, 16, 32):
            engine = FixedModulusArithmetic(N, word_bits=word_bits)
            checks += 1

            expected_mul = (pow(a, 1, N) * pow(b, 1, N)) % N
            assert engine.decode(engine.encode(a)) == a % N
            assert engine.decode(engine.encode(b)) == b % N
            assert engine.modmul(a, b) == expected_mul
            assert engine.modpow(a, exponent) == pow(a, exponent, N)
            assert engine.modpow(a, 0) == pow(a, 0, N)

            R_inverse = inverse_mod(engine.R, N)
            for T in (0, 1, min(engine.R * N - 1, a * b), engine.R * N - 1):
                assert engine.redc(T) == (T * R_inverse) % N

    print(f"all tests passed: {checks} configurations over {len(cases)} modmul/modpow cases")


if __name__ == "__main__":
    verify()
```
