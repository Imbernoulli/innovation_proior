# Riemann-Roch

Let `X` be a nonsingular projective curve of genus `g`, let `D` be a divisor on `X`, and define

`L(D) = { f in k(X) : div(f) + D >= 0 } union {0}`

with `l(D) = dim L(D)`. Let `K_X` be a canonical divisor. The theorem is

`l(D) - l(K_X - D) = deg D + 1 - g`.

The term `deg D + 1 - g` is the expected count: local pole freedom plus constants, corrected by the genus-sized global period conditions. The term `l(K_X - D)` is the exact special correction: it measures the independent dual differentials that make some of those global conditions dependent.

For a divisor of `m` allowed simple poles, this is Roch's corrected count in modern form: the generic `m - g + 1` becomes `m - g + 1 + q`, where `q` is the dimension of the dual vanishing space. When `deg D > 2g - 2`, the correction vanishes, so `l(D) = deg D + 1 - g`.
