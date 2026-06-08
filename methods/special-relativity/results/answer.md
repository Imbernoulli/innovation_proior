# Special Relativity — the kinematics of moving bodies

## The problem

Maxwell's electrodynamics fixes a definite speed of light `c` but is silent about the frame it refers to; the standard "ether rest frame" answer makes `c` frame-dependent and predicts an ether wind that no experiment (notably Michelson–Morley) ever detects. Meanwhile the moving-body electrodynamics of the day produces asymmetries — the magnet/conductor case tells two different stories for one observable — that are artifacts of an assumed absolute rest. The task: a single kinematics in which the laws (Maxwell included) are identical in every inertial frame, `c` is the same in every inertial frame, and no frame is privileged.

## The key idea

Hold two postulates exactly and follow them honestly:

- **(P1) Principle of relativity** — the laws of physics take the same form in all inertial frames.
- **(P2) Constancy of light** — light propagates in vacuum at speed `c`, independent of the motion of the source.

They appear to contradict each other only because of an unstated third assumption — **absolute simultaneity** (`t' = t`, one universal "now"). Define simultaneity *operationally* instead: two clocks at A and B are synchronized when a light signal takes equally long each way,
```
t_B − t_A = t'_A − t_B.
```
Light-synchronization makes simultaneity frame-relative, the apparent contradiction dissolves, and the transformation between frames is then forced.

## Definition of simultaneity (operational time)

A clock dates only events at its own location. Distant simultaneity has no meaning until a synchronization rule is fixed; choose the symmetric light rule above. It is equivalent to dating a remote event at the midpoint of light emission and reflected return, and it builds in `2·AB/(t'_A − t_A) = c`.

## Relativity of simultaneity

A rod of length `r_AB` (measured in frame K) moves at `v`; its end-clocks, synchronized by the light rule in the rod's frame, are timed in K using P2:
```
forward:  t_B − t_A = r_AB/(c − v),     back:  t'_A − t_B = r_AB/(c + v).
```
These differ, so clocks synchronous in one inertial frame are not synchronous in another. **Simultaneity is relative.**

## The Lorentz transformation (from the two postulates)

For K → k with relative velocity `v` along the shared x-axis, axes coinciding at `t = τ = 0`:

1. **Linearity** from homogeneity of space and time.
2. **Define `τ`** by the synchronization rule applied in k. The midpoint condition `½(τ₀+τ₂)=τ₁`, with K-frame light-times `x'/(c−v)` and `x'/(c+v)` (where `x' = x − vt`), gives in the infinitesimal limit
   `∂τ/∂x' + (v/(c²−v²)) ∂τ/∂t = 0`, and `∂τ/∂y = ∂τ/∂z = 0`, hence `τ = a(t − (v/(c²−v²))x')`.
3. **Impose P2 in k** (`ξ = cτ` for a ray, with `x' = (c−v)t`): yields `ξ`, `η`, `ζ` in terms of `a`.
4. **Fix the scale** `φ(v)`: reciprocity (composing `v` then `−v` is the identity) gives `φ(v)φ(−v)=1`; transverse isotropy gives `φ(v)=φ(−v)`; together `φ(v)=1`.

Result, with `β = 1/√(1 − v²/c²)`:
```
τ = β(t − v x/c²),
ξ = β(x − v t),
η = y,
ζ = z.
```

**Compatibility of the postulates.** Under this transformation `x² + y² + z² − c²t²` is invariant, so a spherical light pulse `x²+y²+z²=c²t²` becomes `ξ²+η²+ζ²=c²τ²` — still a sphere of speed `c` in k. The two postulates are consistent.

## Consequences (read off the transformation)

- **Length contraction.** A rest-frame sphere `ξ²+η²+ζ²=R²`, taken at one K-instant, is the ellipsoid `x²/(1−v²/c²)+y²+z²=R²`: the longitudinal dimension is shortened by `√(1−v²/c²)`; transverse dimensions unchanged.
- **Time dilation.** A clock at k's origin (`x=vt`) reads `τ = t√(1−v²/c²)` — it runs slow; lag per second `≈ ½v²/c²`. At constant speed around a closed loop, the carried clock returns behind a stationary one by `≈ ½(v²/c²)t`.
- **Velocity addition.** A velocity `w` in k composes with the frame velocity `v` as
  `V = (v + w)/(1 + vw/c²)` (collinear); general case `V = √((v²+w²+2vw cos a) − (vw sin a/c²)²)/(1 + vw cos a/c²)`. Two sub-`c` velocities give `V < c`; `c` composed with any `w` gives `c`. The transformations form a group.

## Final artifact (a numerical realization and self-checks)

```python
import numpy as np

C = 1.0  # units with the speed of light equal to 1

def gamma(v):
    return 1.0 / np.sqrt(1.0 - (v / C) ** 2)            # beta = 1/sqrt(1 - v^2/c^2)

def synchronize_event_time(t_emit, t_return):
    return 0.5 * (t_emit + t_return)                    # light-signal midpoint rule

def lorentz_transform(x, t, v):
    b = gamma(v)
    return b * (x - v * t), b * (t - v * x / C ** 2)    # xi = b(x - v t),  tau = b(t - v x/c^2)

def length_contraction(rest_length, v):
    return rest_length * np.sqrt(1.0 - (v / C) ** 2)     # longitudinal only

def moving_clock_time(t, v):
    return t * np.sqrt(1.0 - (v / C) ** 2)               # clock at the moving origin runs slow

def add_velocities(w, v):
    return (v + w) / (1.0 + v * w / C ** 2)              # collinear composition; replaces v + w

if __name__ == "__main__":
    v = 0.6
    for (x, t) in [(0.6, 1.0), (-0.3, 0.5), (0.9, 0.95)]:
        xp, tp = lorentz_transform(x, t, v)
        assert abs((x**2 - (C*t)**2) - (xp**2 - (C*tp)**2)) < 1e-12   # interval invariant
    assert abs(add_velocities(C, 0.3) - C) < 1e-12                    # light stays light
    assert add_velocities(0.9, 0.9) < C                              # sub-c stays sub-c
    assert abs(length_contraction(1.0, v) - moving_clock_time(1.0, v)) < 1e-12
    print("Lorentz transform consistent with both postulates.")
```

This is **special relativity**: two postulates, an operational redefinition of simultaneity, and the Lorentz transformation with its consequences — contraction, dilation, and the velocity-addition law — following as forced kinematic conclusions, with no ether and no privileged frame.
