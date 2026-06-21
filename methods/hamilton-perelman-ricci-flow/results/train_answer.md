Closed three-manifolds can look locally trivial while hiding a complicated global topology built from prime connected sums, incompressible tori, finite-volume hyperbolic pieces, and graph-manifold pieces. The natural classification strategy is to find the sphere and torus splittings that reduce the manifold to geometric building blocks, but those splitting surfaces are not visible in an arbitrary presentation. Older geometric approaches either assumed the decomposition was known or tried to smooth the metric without a plan for the singularities that inevitably form, so they stalled as soon as curvature blew up.

The missing ingredient is a way to let the metric itself reveal where to cut. Running Ricci flow is the obvious candidate: it is an intrinsic geometric heat equation that tends to round positively curved regions and can in principle simplify topology. Yet in a general three-manifold the curvature can become unbounded in finite time at isolated regions, and without quantitative control those singularities destroy the evolution rather than expose structure. The breakthrough is to make singularity formation trustworthy enough to be used as a topological tool.

The method is Hamilton-Perelman Ricci flow. It begins with an arbitrary smooth Riemannian metric on a closed 3-manifold and evolves it by the equation ∂g/∂t = -2 Ric. To prevent the flow from collapsing, Perelman's W-entropy and reduced volume supply monotone quantities that give a uniform lower bound on the volume of balls at curvature-bounded scales. That noncollapsing lets us rescale around high-curvature points and obtain ancient κ-solutions as blow-up limits. In dimension three, the Hamilton-Ivey pinching estimate forces these limits to have nonnegative curvature, and the canonical-neighborhood theorem then classifies sufficiently high-curvature regions as strong necks, caps, or closed positive-curvature components.

Once the singular regions are known to be standard, delta-cutoff surgery becomes a controlled operation. At a singular time we keep the bounded-curvature part of the manifold, locate round necks of a small cutoff scale h inside the high-curvature horns, cut through their middle two-spheres, discard the horn tips, and glue in almost-standard caps. The cutoff scale h is chosen much smaller than the canonical-neighborhood scale r, and the caps are chosen so that the pinching condition and the noncollapsing estimates survive after the cut. The flow with surgery then continues, and the entire inductive procedure repeats on the modified manifold.

The topological consequences follow from the controlled evolution. Components with positive scalar curvature become extinct in finite time. If the prime decomposition has no aspherical factors, the flow with surgery becomes extinct entirely, and in the simply connected case the manifold is the 3-sphere. If the flow persists, the long-time geometry has a thick-thin decomposition: thick regions, after scaling by t^{-1}, converge to finite-volume hyperbolic pieces, while sufficiently collapsed thin regions with local lower curvature bounds are graph manifolds. Together these alternatives give the full geometrization decomposition, recovering spherical space forms, S^2 × S^1 factors, hyperbolic pieces, and graph-manifold pieces separated by the sphere and torus decompositions.

```python
import numpy as np

def round_sphere_ricci_flow(r0, t):
    """
    Exact Ricci-flow radius of a round 3-sphere.
    For g(t) = r(t)^2 g_{S^3}, Ric = (2/r^2) g, so dr/dt = -1/r
    and r(t)^2 = r0^2 - 2t. The flow becomes extinct at t = r0^2 / 2.
    """
    return np.sqrt(np.maximum(r0**2 - 2.0 * t, 0.0))

def sphere_geometry(r):
    """Scalar curvature R = 6/r^2 and volume V = 2π^2 r^3 of a round 3-sphere."""
    R = 6.0 / r**2
    V = 2.0 * np.pi**2 * r**3
    return R, V

def w_entropy_round_sphere(r, tau):
    """
    Perelman's W-entropy for the round 3-sphere with the constant weight f
    normalized so that (4πτ)^{-3/2} e^{-f} V = 1. The round sphere is a
    gradient shrinking soliton, so this W is constant in τ along the flow.
    """
    n = 3
    R, V = sphere_geometry(r)
    f = np.log(V / (4.0 * np.pi * tau)**(n / 2.0))  # |grad f| = 0
    return tau * R + f - n

r0 = 2.0
T = r0**2 / 2.0  # extinction time
print(f"Extinction time T = {T:.4f}")

for t in np.linspace(0.0, 0.95 * T, 10):
    r = round_sphere_ricci_flow(r0, t)
    tau = T - t
    R, V = sphere_geometry(r)
    W = w_entropy_round_sphere(r, tau)
    print(f"t={t:.4f}, r={r:.4f}, R={R:.4f}, V={V:.4f}, W={W:.6f}")
```
