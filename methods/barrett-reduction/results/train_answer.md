The task is to compute r = x mod m over and over inside RSA modular exponentiation, where m is fixed for the whole exponentiation but x is the double-length product of two reduced numbers. Schoolbook long division works, but it is genuinely expensive on a cheap digital signal processor: every quotient digit must be estimated from the leading words of the running remainder, the divisor must be multiplied back and subtracted, and the estimate sometimes overshoots and requires an add-back correction. Worse, classical division treats each reduction as a fresh division by a general divisor, even though the modulus never changes during the exponentiation. Montgomery reduction avoids division, but it computes x·R^{-1} mod m in a special residue domain rather than the ordinary remainder, so it only pays off if every operand is converted into that domain once and left there for the whole exponentiation. What is needed is a way to stay in ordinary integers, impose no oddness or coprimality condition on m, and reduce a double-length product using only the operations the DSP does well: multiplies, shifts, and subtractions.

The answer is Barrett reduction. It exploits the fact that m is fixed by precomputing an integer approximation to its reciprocal, then using that approximation in place of division for every subsequent reduction. Let m have k base-b digits, where b is the machine word base, and suppose each input satisfies 0 <= x < b^{2k}, which is exactly the precondition after multiplying two already-reduced numbers. Precompute mu = floor(b^{2k} / m). Because this is rounded down, it is a slight underestimate of the true reciprocal 1/m, so any quotient estimated from it can only be too small, never too large. The estimated quotient is q3 = floor( floor(x / b^{k-1}) * mu / b^{k+1} ). The two divisions by powers of b are not real divisions: on a base-b machine they are just truncations, so they are free. Only the top k+1 digits of x are fed into the reciprocal multiplication, and only the low k+1 digits of the subtraction x - m*q3 are needed. A careful error bound shows that q3 is at most two below the true quotient floor(x/m), so the estimated remainder lies in [0, 3m) and at most two subtractions of m finish the reduction. In the bit-scale version, where x is known to be below m^2, the underestimate is at most one, so a single conditional subtraction suffices.

This reorganizes the reduction so that its cost is about two partial multiplications plus a tiny correction loop, which is comparable to one full long multiplication. Since the reciprocal is computed once and reused for every one of the roughly thousand reductions in a 512-bit exponentiation, the expensive per-reduction division is eliminated entirely, and RSA becomes fast enough to run on an off-the-shelf DSP.

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
        assert 0 <= x < mod ** 2
        t: int = x - ((x * self.factor) >> self.shift) * mod
        return t if t < mod else t - mod


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
