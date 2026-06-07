# Barrett reduction

## Problem

Compute r = x mod m repeatedly for a single fixed modulus m, on hardware where multiplication is
cheap but division is expensive — the inner loop of RSA modular exponentiation, where every
square-and-multiply step reduces a double-length product modulo the same m. The goal is to do the
reduction with only multiplies, shifts, and subtractions, never a true multiple-precision division
by m, while staying exact.

## Key idea

Dividing by a fixed m is multiplying by 1/m. Precompute, once per modulus, a scaled integer
reciprocal

    μ = ⌊ b^{2k} / m ⌋,

where m has k base-b digits (b near the machine word size) and the input x has at most 2k digits
(0 ≤ x < b^{2k}; in particular x < m² for a product of two reduced operands). Because μ is rounded
*down*, it underestimates 1/m, so the estimated quotient is never too large — only too small — and
the correction is always a subtraction, never an add-back.

Estimate the quotient Q = ⌊x/m⌋ using only the top digits of x and a shift:

    q3 = ⌊ ⌊x / b^{k-1}⌋ · μ / b^{k+1} ⌋.

The two divisions are by powers of b, i.e. free truncations/shifts. The estimate cannot exceed Q
because every approximation rounds downward. For the lower bound, write x/b^{k-1} = q1+α and
b^{2k}/m = μ+β with 0 ≤ α,β < 1, then count the three losses at quotient scale: dropping the low
k−1 digits loses α·b^{k-1}/m ≤ α < 1; replacing b^{2k}/m by μ loses q1·β/b^{k+1} < 1 because
q1 < b^{k+1}; and the outer floor loses less than 1. Therefore Q − q3 < 3, so the
integer estimate satisfies

    Q − 2 ≤ q3 ≤ Q,

hence the estimated remainder x − m·q3 = R + δm with 0 ≤ R < m and δ ∈ {0,1,2}, giving 0 ≤ r < 3m.
Since 3m < b^{k+1} (using m < b^k and b > 3), the subtraction needs only the low k+1 digits, and at
most two subtractions of m land r in [0, m).

## Algorithm

Precomputation (once per m): μ = ⌊b^{2k}/m⌋.

Reduction of x (0 ≤ x < b^{2k}):
1. q1 = ⌊x / b^{k-1}⌋,  q2 = q1·μ,  q3 = ⌊q2 / b^{k+1}⌋.
2. r1 = x mod b^{k+1},  r2 = (q3·m) mod b^{k+1},  r = r1 − r2.
3. If r < 0: r = r + b^{k+1}.
4. While r ≥ m: r = r − m.                                     (at most twice)
5. Return r.

Cost: no true division. q2 only needs its high digits (it is shifted down by b^{k+1}) and r2 only
needs its low k+1 digits, so steps 1 and 2 are *partial* multiplies — each about half of a full
long multiply. The whole reduction costs roughly one long multiply; computing q3 takes at most
(k²+5k+2)/2 single-precision multiplications. The correction loop almost always runs zero times.

## Code

A compact bit-scale implementation uses shift = 2·bitlength(m) and accepts only x in [0, m²). With
B = 2^shift, x < m² < B, and the real estimate x·⌊B/m⌋/B is at most x/m and greater than x/m−1.
After the final floor, q satisfies Q−1 ≤ q ≤ Q. The temporary t = x−q·m is therefore below 2m, and
one conditional subtraction suffices.

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
    """A**E mod M by square-and-multiply, reducing after each multiply."""
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

Trade-off versus Montgomery reduction: Montgomery computes x·R⁻¹ mod N with a single conditional
subtraction but requires moving operands into an N-residue domain (and N coprime to the radix, i.e.
odd for a power-of-two radix); it wins when you stay in that domain across a whole exponentiation.
Barrett stays in the ordinary integer domain, places no oddness/coprimality requirement on m, and
inspects the high words of x — the better fit for an isolated reduction or a plain-integer pipeline,
at the cost of a slightly looser (≤2-subtraction) tail.
