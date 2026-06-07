# Montgomery Multiplication

## Summary

Montgomery multiplication computes modular products without division by the
modulus. For an odd modulus `N`, choose a power of two `R = 2^k` with `R > N`.
Represent a residue `x` by its Montgomery-domain value

```text
x_bar = xR mod N.
```

Multiplying two encoded values gives an extra factor of `R`:

```text
(aR)(bR) = abR^2 mod N.
```

The reduction step removes one factor of `R` by computing `T R^{-1} mod N`
using only a low-bit mask, a multiplication by a precomputed inverse, an
addition, a right shift, and one conditional subtraction. Conversions into and
out of the Montgomery domain are paid only at the boundary of a long chain such
as modular exponentiation.

## Algorithm

For fixed odd `N > 1`:

```text
choose R = 2^k with R > N
N_prime = -N^{-1} mod R
R2 = R^2 mod N
```

`N_prime` exists because `gcd(N, R) = 1`.

The reduction operation is:

```text
REDC(T):
    require 0 <= T < RN
    m = (T mod R) * N_prime mod R
    t = (T + mN) / R
    if t >= N:
        t = t - N
    return t
```

Because `R` is a power of two, `T mod R` is a mask of the low `k` bits and
division by `R` is a right shift.

Montgomery-domain multiplication is:

```text
encode(x) = REDC((x mod N) * R2) = xR mod N
mul_encoded(a_bar, b_bar) = REDC(a_bar * b_bar)
decode(x_bar) = REDC(x_bar)
modmul(a, b) = decode(mul_encoded(encode(a), encode(b)))
```

Inside an exponentiation loop, keep the accumulator and base encoded. Then each
square or multiply is one `mul_encoded`, and the final answer is decoded once.

## Correctness

Let `N_prime = -N^{-1} mod R`, so

```text
N * N_prime = -1 mod R.
```

For `0 <= T < RN`, REDC computes

```text
m = (T mod R) * N_prime mod R.
```

Modulo `R`, this means `m` is congruent to `T N_prime`, so

```text
T + mN = T + T N_prime N = T + T(-1) = 0 mod R.
```

Therefore `T + mN` is exactly divisible by `R`, and `t = (T + mN) / R` is an
integer.

Now reduce the defining equation modulo `N`:

```text
tR = T + mN = T mod N.
```

Since `R` is invertible modulo `N`,

```text
t = T R^{-1} mod N.
```

The output is almost canonical before the final subtraction. Since `0 <= m < R`
and `0 <= T < RN`,

```text
0 <= T + mN < RN + RN = 2RN,
```

so

```text
0 <= t < 2N.
```

One conditional subtraction maps `t` into `[0, N - 1]` without changing its
residue modulo `N`.

For encoded inputs `a_bar = aR mod N` and `b_bar = bR mod N`, the product
`T = a_bar b_bar` satisfies `T < N^2 < RN`. REDC gives

```text
REDC(a_bar b_bar) = a_bar b_bar R^{-1}
                  = (aR)(bR)R^{-1}
                  = abR mod N,
```

which is the encoded product. This is why a chain of products stays in the
Montgomery domain until the final decode.

## Runnable Code

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

## Cost

The one-time setup computes `N_prime` and `R2`. Each encoded multiplication then
uses one ordinary product, one REDC step, and no division by `N`. The conversion
cost is one REDC per input and one REDC at the end, so modular exponentiation
amortizes the boundary work over all of its squarings and multiplies.
