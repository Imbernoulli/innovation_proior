# Deterministic Chaos and Sensitive Dependence — the Lorenz System

## The problem it solves

Can a finite system of deterministic ordinary differential equations, with constant forcing, sustain
nonperiodic motion — and if so, does a small uncertainty in the initial state stay small, or can it
grow until prediction fails? The answer reframes predictability: determinism does not guarantee
long-range predictability. A perfectly deterministic, low-dimensional system can amplify imperceptible
differences in initial conditions until two trajectories that started indistinguishably close evolve
into completely different states. This is **sensitive dependence on initial conditions**, the defining
feature of deterministic chaos, and it places an intrinsic limit on weather forecasting that no amount
of better instrumentation (short of perfect measurement) can overcome.

## The key idea

Truncate the equations of thermal convection — a fluid layer heated from below — to their three most
essential modes: one for the intensity of the convective motion, one for the temperature contrast
between rising and sinking fluid, and one for the distortion of the vertical temperature profile. The
resulting three-equation system is **forced** (steady heating), **dissipative** (every phase-space
volume contracts), and **bounded** (no trajectory escapes), yet for the standard parameters has **no
stable fixed point** anywhere. Numerical integration shows the remaining sustained motion is
nonperiodic except for exceptional unstable periodic sequences; the instability theorem for bounded
nonperiodic flow then gives sensitive dependence. The trajectories pile onto a bounded, zero-volume,
infinitely-layered attracting set that they orbit forever without repeating.

## The system

With state `(X, Y, Z)`, dimensionless time `τ`, and a prime denoting `d/dτ`:
```
X' = σ(Y − X)
Y' = r X − Y − X Z
Z' = X Y − b Z
```
- `X` ∝ intensity of the convective overturning.
- `Y` ∝ temperature difference between the ascending and descending currents.
- `Z` ∝ distortion of the vertical temperature profile from linear.
- `σ = ν/κ` is the Prandtl number; `r = Ra/Rc` is the Rayleigh number relative to its critical value;
  `b = 4/(1+a²)` is a geometric constant. The nonlinearity is the two advection terms `XZ` and `XY`.

**Standard parameters:** `σ = 10`, `b = 8/3` (from `a² = ½`, the most easily convecting aspect ratio),
and `r = 28`.

## Why the motion is nonperiodic and sensitive

**Fixed points.** `(0,0,0)` is the rest state (no convection). For `r > 1` two convecting states appear:
```
C, C' = (±√(b(r−1)), ±√(b(r−1)), r−1)   →   (±6√2, ±6√2, 27) ≈ (±8.485, ±8.485, 27)  at r = 28.
```

**No stable fixed point.** For `r > 1` the origin is a saddle (the convection instability). Linearizing about
`C` (or `C'`) gives the characteristic equation
```
λ³ + (σ+b+1)λ² + (r+σ)b λ + 2σb(r−1) = 0,
```
whose complex roots cross the imaginary axis — so the steady convecting states lose stability — at
```
r = σ(σ+b+3)/(σ−b−1) = 470/19 ≈ 24.74   (for σ=10, b=8/3).
```
At `r = 28 > 24.74`, **all three fixed points are unstable.**

**Bounded.** The right Lyapunov quantity is shifted by the forcing level:
```
W = ½[X² + Y² + (Z−r−σ)²].
```
Along the flow,
```
W' = −σX² − Y² − bZ² + b(r+σ)Z
   = −σX² − Y² − b[Z−(r+σ)/2]² + b(r+σ)²/4,
```
so `W' < 0` outside a fixed ellipsoid and every trajectory is trapped in a bounded region forever.

**Dissipative.** The flow's divergence is constant and negative:
```
∂X'/∂X + ∂Y'/∂Y + ∂Z'/∂Z = −(σ+b+1),
```
so any phase volume contracts as `V(t) = V(0) e^{−(σ+b+1)t}` (= `e^{−(41/3)t}` for these constants).

**Nonperiodic attracting set and sensitivity.** Trapped in a bounded box, with every volume crushed to
zero and no stable point to land on, trajectories accumulate on a zero-volume set rather than escaping
or resting. Under uniqueness, continuity, and boundedness, a stable central trajectory must be
quasi-periodic; therefore a nonperiodic central trajectory is unstable, and noncentral nonperiodic
trajectories are not uniformly stable. For prediction, nearby nonperiodic trajectories are effectively
unstable. The simultaneous volume contraction (folding sheets together) and trajectory divergence
(prying them apart) produce an infinitely-layered attracting set. Successive maxima of `Z` fall on a
tent-like return curve; the ideal rescaled model is `M_{n+1} = 2M_n` (`M_n<½`) / `2−2M_n` (`M_n>½`),
and the observed curve has slope magnitude greater than 1 along its branches. Periodic maxima
sequences are therefore exceptional and unstable, while almost every observed sequence is nonperiodic.

## Numerical illustration

A small integration (RK4, `Δt = 0.01`, `σ=10, b=8/3, r=28`) of two starts differing by `10⁻⁶` in one
coordinate — a smaller version of the rounded restart from `0.506127` to `0.506` — confirms the
separation grows from `~10⁻⁶` to the scale of the attractor itself within a few dozen time units, while
the orbit stays bounded and winds around the two convecting states `(±6√2, ±6√2, 27)`.

```python
import math

sigma, b, r = 10.0, 8.0/3.0, 28.0

def deriv(s):
    x, y, z = s
    return (sigma*(y - x), r*x - y - x*z, x*y - b*z)

def rk4_step(s, dt):
    k1 = deriv(s)
    k2 = deriv(tuple(s[i] + 0.5*dt*k1[i] for i in range(3)))
    k3 = deriv(tuple(s[i] + 0.5*dt*k2[i] for i in range(3)))
    k4 = deriv(tuple(s[i] + dt*k3[i] for i in range(3)))
    return tuple(s[i] + (dt/6.0)*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) for i in range(3))

def run(s0, dt=0.01, n=4000):
    s = s0; out = [s]
    for _ in range(n):
        s = rk4_step(s, dt); out.append(s)
    return out

def fixed_points():
    c = math.sqrt(b*(r - 1.0))
    return ((0.0, 0.0, 0.0), (c, c, r - 1.0), (-c, -c, r - 1.0))

def divergence():
    return -(sigma + b + 1.0)

def z_maxima(traj, start=0):
    return [traj[i][2] for i in range(max(1, start), len(traj)-1)
            if traj[i-1][2] < traj[i][2] > traj[i+1][2]]

def diagnostics(traj, other=None, dt=0.01):
    xs = [p[0] for p in traj]; zs = [p[2] for p in traj]
    print("fixed points:", fixed_points())
    print("divergence=%.6g, volume factor over one time unit=%.3e" %
          (divergence(), math.exp(divergence())))
    print("X in [%.1f, %.1f],  Z in [%.1f, %.1f]" %
          (min(xs), max(xs), min(zs), max(zs)))
    maxes = z_maxima(traj, start=1000)
    print("post-transient Z-maximum return pairs:",
          [(round(maxes[i], 3), round(maxes[i+1], 3))
           for i in range(min(5, len(maxes)-1))])
    if other is not None:
        for k in (0, 1000, 2000, 3000, 4000):
            d = sum((traj[k][i]-other[k][i])**2 for i in range(3))**0.5
            print("t=%4.1f  separation=%.3e" % (k*dt, d))

a = run((1.0, 1.0, 1.0))
bb = run((1.0, 1.0, 1.000001))
diagnostics(a, bb)
```

## What it establishes

A three-variable deterministic system, derived from real convection, exhibits bounded nonperiodic
motion with exponential sensitivity to initial conditions. Applied to the atmosphere — which shares the
ingredients of nonlinear advection, constant uneven forcing, and dissipation — it implies that
long-range weather prediction is impossible in principle unless the present state is known exactly:
two states differing by as little as the immediate influence of a single butterfly may, given enough
time, evolve into two states differing as much as the presence of a tornado.
