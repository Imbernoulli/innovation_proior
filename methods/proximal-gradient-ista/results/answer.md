# Proximal Gradient, ISTA, and FISTA

## Composite Setup

Solve

`min_x F(x)=f(x)+g(x)`,

where `f` is convex differentiable with `L(f)`-Lipschitz gradient and `g` is closed proper convex, possibly nonsmooth, with a tractable proximal map.

At an anchor `y`, majorize only the smooth part:

`Q_L(x,y)=f(y)+<x-y,grad f(y)>+(L/2)||x-y||_2^2+g(x)`.

For `L >= L(f)`, `Q_L(.,y)` is an upper model of `F`. Its minimizer is

`p_L(y)=argmin_x Q_L(x,y)=prox_{(1/L)g}(y-grad f(y)/L)`,

where `prox_{t g}(v)=argmin_x g(x)+(1/(2t))||x-v||_2^2`.

This is the essential split: a gradient step on the smooth term followed by the exact proximal step for the nonsmooth term.

## ISTA

The basic proximal-gradient iteration is

`x_k=p_L(x_{k-1})`.

For sparse least squares,

`f(x)=||Ax-b||_2^2`, `g(x)=lambda ||x||_1`,

`grad f(x)=2A^T(Ax-b)`, and `L(f)=2 lambda_max(A^T A)`.

The `l1` proximal map is coordinatewise soft-thresholding:

`T_tau(v)_i=sign(v_i) max(|v_i|-tau,0)`.

Thus ISTA becomes

`x_{k+1}=T_{lambda/L}(x_k - (1/L) 2A^T(Ax_k-b))`.

It satisfies

`F(x_k)-F(x*) <= Lmax||x_0-x*||_2^2/(2k)`,

where `Lmax` is any uniform upper bound on the curvatures used. In the fixed Lipschitz-constant setting `Lmax=L(f)`.

## FISTA

FISTA uses the same proximal point `p_L`, but evaluates it at an extrapolated anchor:

`y_1=x_0`, `t_1=1`,

`x_k=p_L(y_k)`,

`t_{k+1}=(1+sqrt(1+4t_k^2))/2`,

`y_{k+1}=x_k+((t_k-1)/t_{k+1})(x_k-x_{k-1})`.

The recurrence enforces `t_k^2=t_{k+1}^2-t_{k+1}`, which is what makes the Beck-Teboulle potential telescope. Since `t_k >= (k+1)/2`,

`F(x_k)-F(x*) <= 2 Lmax||x_0-x*||_2^2/(k+1)^2`.

FISTA is not necessarily monotone in `F(x_k)`; the decreasing object in the proof is a weighted potential.

## Backtracking

When `L(f)` is unknown, start from the previous accepted curvature `L_{k-1}` and choose the smallest integer `i_k >= 0` such that `bar L=eta^{i_k}L_{k-1}` satisfies

`F(p_{bar L}(y)) <= Q_{bar L}(p_{bar L}(y),y)`.

Then set `L_k=bar L` and use `p_{L_k}(y)`, with `y=x_{k-1}` for ISTA and `y=y_k` for FISTA. This condition is guaranteed once the tested curvature is at least `L(f)`. If `L_0 <= L(f)`, the accepted values satisfy `L_k <= eta L(f)`, so `Lmax=eta L(f)`; if `L_0` is already larger, use `Lmax=max(L_0, eta L(f))`.

## Code Artifact

The local executable artifact is `code/ista_fista_lasso.py`. It matches the canonical forward-backward convention `gamma=1/L`: form `y=x-gamma grad f(x)`, call the proximal map with parameter `gamma`, and for acceleration use the Nesterov coefficient `(t-1)/t_next`. It implements:

- `soft_threshold(v, threshold)`
- `ista(A, b, lam, steps=...)`
- `fista(A, b, lam, steps=...)`

using the convention `f(x)=||Ax-b||_2^2`, so the gradient is `2 A.T @ (A @ x - b)` and the shrinkage threshold is `lambda / L`.
