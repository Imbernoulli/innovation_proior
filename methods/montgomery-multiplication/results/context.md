## Research question

Large-integer modular arithmetic needs an efficient way to compute `(a * b) mod N`
for a fixed odd modulus `N`. The ordinary route multiplies first, producing a
double-width product `T`, and then reduces `T` by division against `N`.
Repeated modular exponentiation and factoring routines perform many such
reductions under the same `N`, so a method that pays some one-time setup per
modulus and handles arbitrary odd moduli is of interest.

## Background

Residue classes modulo `N` can be represented by any complete residue system.
The standard representative `x` in `[0, N - 1]` is convenient for comparison and
addition, but after multiplying two representatives the product still requires
reduction by `N`.

Division by a power of two is different. If `R = 2^k`, then `T mod R` is just the
low `k` bits of `T`, and exact division by `R` is a right shift. Choosing a radix
`R` coprime to `N` with `R > N` makes division by `R` a simple shift operation.

The relevant algebraic primitive is available. Since `N` is odd and `R` is a power
of two, `gcd(N, R) = 1`, so extended Euclid gives inverses modulo both `N` and
`R`.

Repeated modular multiplication is the setting that makes a representation
change worthwhile. RSA exponentiation, Pollard-style factoring methods, and
probabilistic algebraic tests all do long chains of products modulo one fixed
modulus. Paying to enter and leave a nonstandard representation can be amortized
over those chains if every internal multiplication is cheaper.

## Baselines

Classical long division reduces `T` by computing `q = floor(T / N)` and returning
`T - qN`. It works for every modulus and leaves numbers in the standard
representative range. Multi-precision division repeatedly estimates quotient
digits, corrects them, and propagates carries.

Binary modular exponentiation reduces an exponentiation to a chain of squares
and selected multiplies. That is the right high-level structure for RSA-style
workloads, with each product computed as an ordinary modular multiplication.

Special-form moduli, such as numbers close to powers of two, allow reductions by
folding high bits back into low bits. Those reductions can be extremely fast for
the specific moduli they support.

Residue-number and redundant-representation approaches can reduce carry
propagation or decompose arithmetic across channels.

## Evaluation settings

The natural yardsticks are computations that perform many modular products under
one modulus:

- modular exponentiation for public-key encryption, signatures, and key exchange
- Pollard `p - 1` and Pollard rho style factoring loops
- probabilistic polynomial identity checks over modular rings or fields
- software multi-precision arithmetic with radix `b^n`
- hardware datapaths where shifts, additions, and multiply-add steps are much
  simpler than a divider

The relevant measurements are per-product word operations, number of full
divisions by `N`, carry propagation, temporary storage width, conversion cost at
the start and end of a chain, and whether the arithmetic keeps results in a
canonical interval after each product.

## Code framework

```python
class FixedModulusArithmetic:
    def __init__(self, modulus, word_bits=None):
        if modulus <= 1 or modulus % 2 == 0:
            raise ValueError("the modulus must be an odd integer greater than 1")
        self.N = modulus
        self.word_bits = word_bits

    def encode(self, x):
        # TODO: choose the working representative for x modulo self.N.
        pass

    def decode(self, x_encoded):
        # TODO: map a working representative back to the standard residue.
        pass

    def reduce_product(self, T):
        # TODO: reduce a product-shaped intermediate without trial division by self.N.
        pass

    def mul_encoded(self, a_encoded, b_encoded):
        # TODO: multiply two working representatives and reduce the product.
        pass

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
```
