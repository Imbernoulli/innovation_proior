# Blum-Micali Pseudorandom Bit Generator

Let `B = {B_i : D_i -> {0,1}}` be an accessible, input-hard predicate family. For each public index `i`, let `f_i : D_i -> D_i` be an efficiently computable permutation, and assume

```text
h_i(x) = B_i(f_i(x))
```

is efficiently computable. For seed `(i, x)` and output length `m`, compute the orbit

```text
x_0 = x
x_1 = f_i(x_0)
...
x_m = f_i(x_(m-1))
```

and output the bits in the original right-to-left order:

```text
B_i(x_m), B_i(x_(m-1)), ..., B_i(x_1)
```

Each bit is computable as `B_i(x_j) = h_i(x_(j-1))`. Modern hard-core-bit presentations often assume `B` itself is efficient and use the forward loop

```text
S <- seed
repeat m times:
    output B(S)
    S <- f(S)
```

with the proof applied to the reversed sequence.

Security: if an efficient circuit predicts a next output bit from the previous bits with advantage `epsilon`, then one constructs an efficient circuit for the hard predicate. On a challenge state `y`, compute the visible prefix

```text
B_i(f_i^k(y)), B_i(f_i^(k-1)(y)), ..., B_i(f_i(y))
```

feed it to the stream predictor, and use the predictor's answer as the guess for `B_i(y)`. The prefix is computable from `y`, and the permutation property makes the shifted orbit distributed like a real generator prefix. This contradicts input-hardness of `B`, so the stream is next-bit unpredictable for polynomially many output bits.

Concrete discrete-log version: choose an odd prime `p` and generator `g` of `Z_p^*`, with parameters avoiding smooth `p - 1`. Represent `D` by the integers `1, ..., p - 1` and use

```text
f_(p,g)(x) = g^x mod p
```

Define `B_(p,g)(x) = 1` if `x` is the principal square root of `x^2 mod p`, and `0` otherwise. If `x = g^z mod p` with `1 <= z <= p - 1`, then

```text
B_(p,g)(x) = 1  for 1 <= z <= (p - 1)/2
B_(p,g)(x) = 0  for (p + 1)/2 <= z <= p - 1
```

There is no equality case at `p/2`. The generator can compute `B_(p,g)(g^x mod p)` by comparing the known exponent `x` with `p/2`; an adversary given only the group element would have to solve the principal-square-root predicate. A predictor for that predicate with advantage over `1/2` can be amplified into a discrete-log algorithm, so the construction is secure under the stated discrete-log hardness assumption.
