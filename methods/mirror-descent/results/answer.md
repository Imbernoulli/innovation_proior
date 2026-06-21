# Mirror Descent

Original dual-space form: choose a regular function `V:E* -> R`, keep the main iterate `g_k` in the dual space, form the primal image

`x_k = V'(g_k)`,

and update the dual variable by a first-order direction,

`g_{k+1} = g_k - p_k xi_k`.

The primal iterate is the image or feasible projection induced by the same geometry. In the Hilbert/Euclidean case `V(g)=(1/2)||g||_2^2`, so `V'(g)=g` and the method reduces to ordinary gradient descent.

Modern Bregman form: choose a norm `||.||`, dual norm `||.||_*`, and a differentiable potential `psi` that is `sigma`-strongly convex with respect to `||.||`. Define

`B_psi(x,y) = psi(x) - psi(y) - <grad psi(y), x-y>`.

Given `g_k in partial f(x_k)`, update by

`x_{k+1} = argmin_{x in X} { t_k <g_k,x> + B_psi(x,x_k) }`.

Equivalently,

`0 in t_k g_k + grad psi(x_{k+1}) - grad psi(x_k) + N_X(x_{k+1})`.

In the unconstrained interior,

`grad psi(x_{k+1}) = grad psi(x_k) - t_k g_k`,

so the subgradient step happens in dual coordinates and the mirror map returns the point to the primal feasible geometry.

The basic guarantee is

`t_k(f(x_k)-f(x*)) <= B_psi(x*,x_k)-B_psi(x*,x_{k+1}) + t_k^2 ||g_k||_*^2/(2 sigma)`.

Summing gives

`min_{1<=s<=k} f(x_s)-f(x*) <= (B_psi(x*,x_1) + (1/(2 sigma)) sum_{s=1}^k t_s^2 ||g_s||_*^2) / sum_{s=1}^k t_s`.

If `||g_s||_* <= L` and the horizon `k` is known, the constant step

`t_s = sqrt(2 sigma B_psi(x*,x_1)) / (L sqrt(k))`

yields

`min_{1<=s<=k} f(x_s)-f(x*) <= L sqrt(2 B_psi(x*,x_1)/sigma) / sqrt(k)`.

Special cases:

- Euclidean: `psi(x)=(1/2)||x||_2^2` gives ordinary projected subgradient descent.
- Simplex entropy: `psi(x)=sum_i x_i log x_i` gives

`x_{k+1,i}=x_{k,i} exp(-t_k g_{k,i}) / sum_j x_{k,j} exp(-t_k g_{k,j})`.

For the simplex with `l1` geometry, negative entropy is 1-strongly convex, the dual norm is `l_infinity`, and the uniform start has radius at most `log n`. The rate scales as

`sqrt(2 log n) ||g||_infinity / sqrt(k)`,

matching the original geometry-adapted improvement over the Euclidean `sqrt(n)` dependence.
