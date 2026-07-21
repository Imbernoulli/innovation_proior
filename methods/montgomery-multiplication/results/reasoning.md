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

The implementation follows this derivation directly, with a fixed-modulus
object holding the precomputed constants and one shared REDC step underneath
everything. `__init__` fixes an odd `N`, takes `k` as `N`'s bit length (rounded
up to a whole number of machine words when a `word_bits` size is given, so
`R = 2^k` lands on a realistic word boundary), and computes the two constants
the derivation needed once per modulus: `N_prime = -N^{-1} mod R` by extended
Euclid, and `R2 = R^2 mod N` for boundary conversions. `reduce_product` is
REDC itself — mask for `m`, add `mN`, shift by `k`, one conditional
subtraction — guarded by the same `0 <= T < RN` domain check the bound above
requires. `encode` and `decode` are each a single call to `reduce_product`,
using `R2` so that entering the domain never needs a real division by `N`.
`mul_encoded` multiplies two domain values and reduces once, and `modpow` is
ordinary square-and-multiply with the accumulator and base kept encoded for
every step, decoded only once at the end — the amortization argument made
concrete. A test harness exercises this across bit-widths from 8 to 127 bits
and several `word_bits` settings, checking `modmul` and `modpow` against
Python's built-in `pow` for `a`, `b` drawn from up to `3N` (so `encode` is
also tested on inputs outside the canonical range), and separately calling
`redc` directly on `T` at the edges of its domain — `0`, `1`, and `RN - 1`,
the largest input the `0 <= T < RN` requirement allows — against the raw
`T * R^{-1} mod N` definition, so every constant and every step of the
derivation above has something in the code and the test run checking it.
