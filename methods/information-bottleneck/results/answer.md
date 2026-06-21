# Information Bottleneck Method

Given a joint distribution `p(x,y)`, choose a stochastic compressed representation `T` of `X` with the Markov chain

```text
Y -> X -> T.
```

For `beta >= 0`, solve the tradeoff

```text
min_{p(t|x)} I(X;T) - beta I(T;Y),
```

using one fixed logarithm base throughout. The base only changes the information units and rescales `beta`; it does not change the fixed-point equations.

The stationary equations are

```text
p(t|x) = p(t) exp(-beta D_KL[p(y|x) || p(y|t)]) / Z(x,beta)

p(t) = sum_x p(x) p(t|x)

p(y|t) = sum_x p(y|x) p(t|x) p(x) / p(t)

Z(x,beta) = sum_t p(t) exp(-beta D_KL[p(y|x) || p(y|t)]).
```

The induced distortion is therefore

```text
d_IB(x,t) = D_KL[p(y|x) || p(y|t)].
```

Equivalently, the alternating minimization uses the free-energy form

```text
F = I(X;T) + beta E_{p(x,t)} D_KL[p(y|x) || p(y|t)].
```

This differs from the Lagrangian above only by the constant `beta I(X;Y)`, since under `Y -> X -> T`,

```text
E_{p(x,t)} D_KL[p(y|x) || p(y|t)] = I(X;Y) - I(T;Y).
```

Finite-alphabet algorithm:

1. Choose `|T|`, initialize a row-stochastic encoder `p(t|x)`, and fix `beta`.
2. Set `p(t) = sum_x p(x)p(t|x)`.
3. Set `p(y|t) = sum_x p(y|x)p(t|x)p(x) / p(t)`.
4. Update `p(t|x)` with the exponential KL rule.
5. Repeat steps 2-4 until the encoder stabilizes.
6. Vary `beta` to trace the curve `(I(X;T), I(T;Y))`.

Boundary cases matter. At `beta = 0`, the update makes `T` independent of `X`; with a minimal active code this collapses all inputs to one codeword. As `beta` increases, assignments increasingly favor codewords whose predictive distributions over `Y` match `p(y|x)`. Terms with `p(y|x)=0` contribute zero to the KL, while any `p(y|t)=0` under a positive `p(y|x)` makes that assignment impossible.

The executable artifact in `code/information_bottleneck.py` implements exactly this finite-distribution protocol: it validates the joint table, computes `p(y|t)` by the posterior average above, performs the assignment update in log space, and returns `I(X;T)`, `I(T;Y)`, expected KL distortion, and free energy.
