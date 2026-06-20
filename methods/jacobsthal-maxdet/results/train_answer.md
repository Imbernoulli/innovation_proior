We want the `29 × 29` sign matrix with the largest possible determinant, and the first thing to
settle is what "largest possible" can even mean, because everything turns on `29 mod 4`. If a
`±1` matrix had orthogonal rows, its Gram matrix would be `29·I`, its determinant would hit the
Hadamard ceiling `29^{29/2}`, and the problem would be solved. But the rows have odd length, and
the inner product of two `±1` vectors of odd length is a sum of an odd number of `±1`'s, an odd
integer that can never be zero. So no two rows can be orthogonal, `HHᵀ` can never be `29·I`, and
the Hadamard bound is provably unreachable at this order. That fixes the character of the problem:
the best matrices are only *almost* orthogonal, and the whole question is how close one can get.
"Almost orthogonal" has a precise meaning here — since the off-diagonal entries of the Gram matrix
are odd, the smallest they can be in magnitude is `±1`, so the cleanest reachable target is a Gram
matrix with `29` on the diagonal and `−1` everywhere off it.

The method is the Jacobsthal construction `R = Q + I`. For the prime `q = 29` it builds the
quadratic-residue matrix `Q` with `Q_{ij} = χ(i − j)`, where `χ` is the Legendre symbol — `+1` on
nonzero quadratic residues, `−1` on non-residues, `0` at zero. Because `29 ≡ 1 (mod 4)`, the value
`−1` is itself a quadratic residue, which forces `χ(−a) = χ(a)` and makes `Q` symmetric; and the
multiplicative structure of the residues gives the exact identity `QQᵀ = qI − J`, which is
precisely the almost-orthogonal Gram structure parity allows — every pair of rows overlapping by
exactly `−1`. The only obstacle is that `Q` has zeros on its diagonal, so it is not yet a sign
matrix. Filling the diagonal with `+1`, `R = Q + I`, repairs this: every entry is now `±1`, and
the Gram matrix becomes `RRᵀ = (q+1)I − J + 2Q`, still highly regular, with an integer determinant
that factors as a small multiplier of `2^28 · 7^12`.

The construction has no parameter to tune, which is exactly what one expects from a rigid
symmetric design. The single apparent degree of freedom — filling the diagonal with `+1` versus
`−1`, giving `R = Q + I` or `R = Q − I` — is illusory: the two are determinant-twins,
`|det(Q + I)| = |det(Q − I)|`, because the residue spectrum is symmetric under the sign flip. So
this is offered not as a strong answer but as the right floor: principled, guaranteed legal, and
fully determined. Measured exactly with Bareiss elimination, it yields multiplier `49.00` and score
`0.14330` — in fact `|det| = 2^28 · 7^14`, the clean closed-form value of the symmetric design,
coinciding exactly with the published task baseline. That floor is honest about its own ceiling:
the symmetry that makes the matrix elegant is the same symmetry that pins it to one configuration,
and the maximal-determinant records at this order (multiplier `320`, Barba bound `369.94`) come
precisely from *breaking* that symmetry under search. The entire distance from `49` to the record
is what any later, searched construction has to buy.

```python
import numpy as np

def construct():
    q = 29
    qr = set((i * i) % q for i in range(1, q))           # nonzero quadratic residues mod 29
    chi = lambda a: 0 if a % q == 0 else (1 if a % q in qr else -1)   # Legendre symbol
    Q = np.array([[chi(i - j) for j in range(q)] for i in range(q)], dtype=int)
    # Q is symmetric (29 ≡ 1 mod 4) with Q Qᵀ = 29 I − J; fill the zero diagonal with +1.
    return Q + np.eye(q, dtype=int)
```
