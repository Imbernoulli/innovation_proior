# Rademacher Complexity

## Definition

Bartlett and Mendelson use the absolute `2/n` convention. For a class `F` of real-valued functions and a sample `X_1,...,X_n`,

$$
\widehat R_n(F)=E_\sigma\left[\sup_{f\in F}\left|\frac{2}{n}\sum_{i=1}^n\sigma_i f(X_i)\right|\mid X_1,\ldots,X_n\right],
\qquad
R_n(F)=E\,\widehat R_n(F),
$$

where the `sigma_i` are independent fair signs. This measures how well the class can correlate with pure random noise on the realized sample.

## Risk Bound

For a bounded loss `L : Y x A -> [0,1]`, a dominating cost `phi : Y x A -> [0,1]`, and

$$
\widetilde\phi\circ F=\{(x,y)\mapsto \phi(y,f(x))-\phi(y,0):f\in F\},
$$

with probability at least `1-delta`, every `f in F` satisfies

$$
E\,L(Y,f(X))
\le
\widehat E_n\,\phi(Y,f(X))
 + R_n(\widetilde\phi\circ F)
 + \sqrt{\frac{8\ln(2/\delta)}{n}}.
$$

For binary classification with `F subset {+1,-1}^X`, this specializes to

$$
P(Y\ne f(X))
\le
\widehat P_n(Y\ne f(X))
 + \frac{R_n(F)}2
 + \sqrt{\frac{\ln(1/\delta)}{2n}}.
$$

For a real-valued class with finite pointwise envelope, and a margin cost `phi` that dominates `1(alpha <= 0)` and is `L`-Lipschitz,

$$
P(Yf(X)\le0)
\le
\widehat E_n\,\phi(Yf(X))
 + 2L\,R_n(F)
 + \sqrt{\frac{\ln(2/\delta)}{2n}}.
$$

## Sample Estimation

For classes mapping to `[-1,1]`, the sample quantity concentrates around its expectation:

$$
P\left(\left|R_n(F)-\widehat R_n(F)\right|\ge \epsilon\right)
\le 2\exp\left(-\frac{\epsilon^2 n}{8}\right).
$$

The same theorem controls a single Rademacher draw, so the penalty can be estimated by repeated sign draws and optimization of the signed correlation over `F`. There is no separate canonical software implementation for this theorem artifact; the faithful object is the definition and bounds above.

## Structural Rules

The penalty is useful because it composes:

- `F subset H` implies `R_n(F) <= R_n(H)`.
- `R_n(F)=R_n(conv F)=R_n(absconv F)`.
- `R_n(cF)=|c|R_n(F)`.
- If `phi(0)=0` and `phi` is `L`-Lipschitz, then `R_n(phi o F) <= 2L R_n(F)`.
- For uniformly bounded `h`, `R_n(F+h) <= R_n(F)+||h||_infty/sqrt(n)`.
- If `L_{F,h,q}={|f-h|^q:f in F}`, `1 <= q < infinity`, `h` is uniformly bounded, and `||f-h||_infty <= 1` for all `f`, then

$$
R_n(L_{F,h,q})\le 2q\left(R_n(F)+\frac{||h||_\infty}{\sqrt n}\right).
$$

- `R_n(sum_j F_j) <= sum_j R_n(F_j)`.

For kernel functions `f(x)=<w,Phi(x)>` with `||w|| <= B`,

$$
\widehat R_n(F)\le \frac{2B}{n}\left(\sum_{i=1}^n k(X_i,X_i)\right)^{1/2}.
$$

Thus the method's core artifact is a data-dependent generalization penalty: a function class is complex exactly to the extent that it can fit independent random signs on the actual sample.
