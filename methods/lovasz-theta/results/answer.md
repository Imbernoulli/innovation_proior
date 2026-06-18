# Lovasz Theta

For a confusability graph `G`,

```text
Theta(G) = lim_{k -> infinity} alpha(G^k)^(1/k),
```

where `G^k` is the kth strong power. In the capacity convention, choose unit vectors `u_v` with
`u_v^T u_w = 0` whenever `v` and `w` are non-adjacent in `G`, and choose a unit handle `c`. Define

```text
theta(G) = min_{U,c} max_v 1 / (c^T u_v)^2.
```

Then every independent set is a mutually orthogonal vector family, so `alpha(G) <= theta(G)`. Tensor products give
`theta(G * H) <= theta(G) theta(H)`, and the dual semidefinite formulation gives the reverse inequality, hence

```text
theta(G * H) = theta(G) theta(H),
```

and therefore

```text
Theta(G) <= theta(G).
```

The matching SDP form, used by the reference CSDP/Sage implementation, is

```text
theta(G) = max <J, X>
subject to Tr(X) = 1,
           X_ij = 0 for every edge ij in G,
           X is positive semidefinite.
```

For `C5`, the length-2 code `(0,0), (1,2), (2,4), (3,1), (4,3)` gives `Theta(C5) >= sqrt(5)`. The umbrella
representation has handle projection `5^(-1/4)`, so `theta(C5) <= sqrt(5)`; the matching bounds give

```text
Theta(C5) = theta(C5) = sqrt(5).
```

Equivalently, for a graph `H` in the coloring convention,

```text
omega(H) <= theta(complement(H)) <= chi(H).
```

Thus the method converts a hard asymptotic independence problem into a multiplicative semidefinite/geometric
upper bound, while also connecting independence and coloring through the same convex relaxation.
