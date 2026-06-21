# Hamilton-Perelman Ricci Flow

The method converts 3-manifold topology into controlled geometric evolution.

Start with a closed oriented 3-manifold `M` and any smooth Riemannian metric `g(0)`. Run Ricci flow

```text
partial_t g_ij = -2 R_ij.
```

The flow is allowed to fail smoothly only after the singular regions have been made standard enough to cut.

The control mechanism is:

1. Use Perelman's monotone functionals. For

```text
F(g,f) = integral_M (R + |grad f|^2)e^{-f} dV
```

and

```text
W(g,f,tau) = integral_M [tau(R + |grad f|^2) + f - n]
              (4 pi tau)^(-n/2)e^{-f} dV,
```

with `integral_M (4 pi tau)^(-n/2)e^{-f} dV = 1` and `tau_t = -1`,

```text
dW/dt = 2 tau integral_M |Ric + Hess f - (1/(2 tau))g|^2
        (4 pi tau)^(-n/2)e^{-f} dV >= 0.
```

The minimized `mu(g,tau)` is therefore monotone along the coupled flow, and equality is the gradient shrinking soliton case.

2. Use reduced geometry. Backward `L`-length

```text
L(gamma) = integral sqrt(tau) (R(gamma(tau)) + |gamma'(tau)|^2) d tau
```

defines reduced distance `l = L/(2 sqrt(tau))`, and the reduced volume

```text
Vtilde(tau) = integral_M tau^(-n/2) exp(-l(q,tau)) dq
```

is nonincreasing as the backward parameter `tau` increases. Together with the entropy formula, this gives no local collapsing: on a finite time interval, at a scale where `|Rm| <= r^{-2}`, metric balls have volume at least `kappa r^n`.

3. Blow up high-curvature points. Noncollapsing plus Hamilton compactness gives ancient `kappa`-solution limits. In dimension three, Hamilton-Ivey pinching makes high-curvature limits nonnegatively curved. The canonical-neighborhood theorem then forces sufficiently high-curvature regions to be modeled by strong necks, caps, or closed positive-curvature pieces.

4. Perform delta-cutoff surgery. At a singular time, keep the bounded-curvature region. In each high-curvature horn, find a sufficiently round neck of small cutoff radius `h`, cut along its middle `S^2`, remove the horn end, and glue in an almost standard cap. Choose `h` much smaller than the canonical-neighborhood scale `r`; the cap is chosen to preserve the pinching condition, and the small-cutoff induction restores the canonical-neighborhood and noncollapsing estimates after surgery.

5. Continue the flow with surgery. The process is controlled on finite time intervals. Components close to round spherical quotients are declared extinct in the surgery construction, and positive-scalar-curvature components become extinct in finite time.

Consequences:

- If the prime decomposition has no aspherical factors, the Ricci flow with surgery becomes extinct in finite time for every initial metric. In particular, a closed simply connected 3-manifold is diffeomorphic to `S^3`.
- In the long-time case, the manifold decomposes into thick regions converging to finite-volume hyperbolic pieces and thin regions that collapse with lower curvature bound and are graph manifolds.
- Combining the finite-extinction and long-time alternatives gives the geometrization decomposition: finite-extinction summands such as spherical space forms and `S^2 x S^1` factors, hyperbolic thick pieces, and graph-manifold or Seifert-fibered thin pieces separated by the sphere and torus decompositions.

The core artifact is therefore:

```text
arbitrary smooth metric
  -> Ricci flow
  -> entropy/reduced-volume noncollapsing
  -> ancient kappa-solution blow-up models
  -> canonical neck/cap neighborhoods
  -> delta-cutoff surgery
  -> extinction or thick-thin geometric decomposition
```

Hamilton supplied the flow program and the surgery vision. Perelman supplied the monotone quantities, noncollapsing, canonical-neighborhood analysis, and surgery implementation that make the singularities usable rather than fatal.
