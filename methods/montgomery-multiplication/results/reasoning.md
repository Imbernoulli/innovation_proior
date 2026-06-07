Let me start from the operation that hurts. I have a fixed odd modulus `N`, and
I need to multiply residues modulo `N` again and again. The raw product is not
the difficult part: if `a` and `b` are `k`-bit numbers, producing the
double-width product `T = ab` is regular work. The difficult part is turning
`T` back into a residue. The standard reduction asks for `q = floor(T / N)`,
then returns `T - qN`, so it asks for division by an arbitrary multi-precision
integer. That quotient estimate is exactly the operation I want to stop doing.

The cheap division is division by a radix. If I choose `R = 2^k`, then
`T mod R` is just the low `k` bits and exact division by `R` is a right shift.
That does not immediately reduce modulo `N`, because `R` has nothing to do with
the residue class modulo `N`. But if `N` is odd, `R` and `N` are coprime. So
`R` has an inverse modulo `N`, and `N` has an inverse modulo `R`. That is a
useful opening: maybe I can add a multiple of `N` to the product, leave the
residue modulo `N` unchanged, and force the result to become divisible by `R`.

Suppose I have an integer `T` with `0 <= T < RN`. I want something congruent to
`T R^{-1}` modulo `N`, because dividing by `R` in the integers is cheap and
multiplying by `R^{-1}` is what that division means in the residue ring. The
problem is that `T / R` is not usually an integer. I need to add `mN`, because
adding a multiple of `N` does not change the residue modulo `N`, and I want

```text
T + mN = 0 mod R.
```

This congruence determines `m` modulo `R`:

```text
mN = -T mod R.
```

Since `N` is invertible modulo `R`, this becomes

```text
m = -T N^{-1} mod R.
```

The low bits of `T` are all that matter modulo `R`, so I can precompute

```text
N_prime = -N^{-1} mod R
```

and then compute

```text
m = (T mod R) * N_prime mod R.
```

Now the sum `T + mN` is exactly divisible by `R`. I can set

```text
t = (T + mN) / R.
```

This is an integer division by a power of two, so it is just a shift. I still
need to check that this quotient is the residue I intended. Multiplying the
definition of `t` by `R` gives

```text
tR = T + mN.
```

Modulo `N`, the `mN` term vanishes, so

```text
tR = T mod N.
```

Because `R` is invertible modulo `N`,

```text
t = T R^{-1} mod N.
```

The quotient is not necessarily already in `[0, N - 1]`, but it is very close.
The choice of `m` gives `0 <= m < R`, and I am requiring `0 <= T < RN`, so

```text
0 <= T + mN < RN + RN = 2RN.
```

After dividing by `R`,

```text
0 <= t < 2N.
```

So there is only one possible excess copy of `N`. If `t >= N`, subtract `N`;
otherwise keep `t`. That gives a canonical representative while preserving the
congruence `t = T R^{-1} mod N`. This reduction is exactly the division-free
piece I need:

```text
REDC(T):
    m = (T mod R) * N_prime mod R
    t = (T + mN) / R
    if t >= N:
        t = t - N
    return t
```

Now I need the representation that makes this useful for multiplication, not
just for one isolated reduction. If I store `x` as itself, then applying this
reduction to `ab` returns `abR^{-1}`, which is off by a factor of `R^{-1}`. That
looks wrong. I need the stored representation to have one spare factor of `R`
so that losing one factor during reduction leaves the product in the same
representation.

So I store the residue `x` as

```text
x_bar = xR mod N.
```

Now multiply two stored representatives:

```text
a_bar b_bar = (aR)(bR) = abR^2 mod N.
```

Apply the reduction that multiplies by `R^{-1}`:

```text
REDC(a_bar b_bar) = abR mod N.
```

That is the stored representative of `ab`. The reduction has stopped being just
a way to make a product small; it is the operation that removes exactly the
extra factor introduced by multiplying two encoded values.

The input bound also lines up without extra work. Encoded representatives are
kept in `[0, N - 1]`, so their product `T` is less than `N^2`. Since I choose
`R > N`, I get `N^2 < RN`, exactly the REDC input condition. The proof and the
implementation are now using the same bound.

There is still the boundary problem: how do I get into this representation
without doing the expensive reduction I am trying to avoid? Directly computing
`xR mod N` is a modular reduction. But once the modulus is fixed, I can
precompute `R^2 mod N`. Then

```text
REDC((x mod N)(R^2 mod N)) = xR^2 R^{-1} = xR mod N.
```

That is one reduction per input. To leave the representation,

```text
REDC(x_bar) = xR R^{-1} = x mod N.
```

For a single product, this adds conversion overhead. For exponentiation, the
overhead is amortized. I convert the base once, keep the accumulator encoded,
perform every square and multiply as one encoded multiplication, and decode once
at the end. A long chain pays for the representation at its boundaries, while
the loop avoids division by `N` throughout.

Let me check the arithmetic on a small case before trusting the symbols. Take
`N = 17` and `R = 32`. Then `N_prime` must satisfy

```text
17 * N_prime = -1 mod 32.
```

`17 * 15 = 255 = -1 mod 32`, so `N_prime = 15`. Encode `7` and `11`:

```text
7_bar = 7 * 32 mod 17 = 3
11_bar = 11 * 32 mod 17 = 12
```

The encoded product uses `T = 3 * 12 = 36`. REDC gives

```text
m = (36 mod 32) * 15 mod 32 = 4 * 15 mod 32 = 28
t = (36 + 28 * 17) / 32 = 512 / 32 = 16
```

`16 < 17`, so the encoded product is `16`. The ordinary product is
`7 * 11 = 77 = 9 mod 17`, and `9` encoded is

```text
9 * 32 mod 17 = 16.
```

The representation has stayed closed under multiplication. Decoding `16` should
return `9`:

```text
m = 16 * 15 mod 32 = 16
t = (16 + 16 * 17) / 32 = 288 / 32 = 9.
```

That checks the boundary conversion too.

For multi-precision arithmetic I do not actually want to form every operation
as a single enormous "mod `R`" computation if `R = b^n` is an `n`-word radix.
The same congruence can be enforced one word at a time. Let `b` be the machine
word base and let `N_0` be the low word of `N`. Since `N` is odd when `b` is a
power of two, `N_0` is invertible modulo `b`. Precompute

```text
mu = -N_0^{-1} mod b.
```

At step `i`, assume the lower `i` words have already been zeroed. Look at the
current word `T_i`. Choose

```text
q_i = T_i * mu mod b.
```

Adding `q_i N b^i` changes word `i` by `q_i N_0`. Modulo `b`, that word becomes

```text
T_i + q_i N_0
  = T_i + T_i mu N_0
  = T_i(1 + mu N_0)
  = 0 mod b.
```

Carries move upward, but the lower words stay zero. After `n` such steps, the
low `n` words are zero, so the whole adjusted value is divisible by `b^n = R`.
The final shift is just dropping those zero words. This is the same reduction,
organized so a multi-word implementation can interleave multiply-add work with
the reduction instead of building a separate divider.

The code I want now has fixed modulus setup, encode, decode, product reduction,
encoded multiplication, ordinary modular multiplication, and exponentiation that
stays encoded internally.

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

The pieces now fit: arbitrary division by `N` is expensive; a power
of two gives cheap low bits and cheap exact division; adding `mN` preserves the
residue modulo `N` while choosing `m` with `-N^{-1} mod R` forces divisibility
by `R`; the quotient is `T R^{-1} mod N` and is below `2N`; storing `x` as
`xR mod N` makes that loss of one `R` factor exactly what multiplication needs;
conversion costs are paid at the edges and amortized over the products inside.
