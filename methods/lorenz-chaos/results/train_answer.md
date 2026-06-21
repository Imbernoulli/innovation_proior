The question I set out to settle was a practical one about forecasting, but it turned into something far sharper. Classical determinism, inherited from celestial mechanics, carries a tacit promise: approximate knowledge of the present buys approximate knowledge of the future, smoothly and without limit, so that refining the observations a little improves the forecast a little. Astronomy seemed to vouch for it — nudge a comet's initial position slightly and its predicted position shifts slightly. Operational meteorology of the day was built on the same faith, in its linear-statistical form: regress tomorrow's weather on today's, fit to decades of maps, extrapolate. My instinct was that integrating the fluid equations forward dynamically would do better, and I wanted a fair test — build a small model atmosphere, generate a long artificial weather record, and see whether the statistical method could predict it. To make the test punishing I drove the model toward nonperiodic output, since a scheme premised on recurring patterns should fail hardest where the record never repeats. It did wander, and the statistical method did badly, but that was not what stopped me. Restarting a run from a printed line of numbers — three decimals where the machine internally carried six — produced a two-month "forecast" with nothing in common with the original. It was not a hardware fault: the divergence did not break suddenly, it grew smoothly, the gap roughly doubling every few simulated days from a seed of about one part in a thousand. That is exponential amplification of a tiny error, a property of the dynamics, not of round-off accumulation, which is additive and slow. And if the system amplifies my typo, it amplifies any difference between the true atmosphere and the best measurement of it. The existing catalogue of long-term behaviors — steady point, periodic loop, quasi-periodic torus — has no slot for bounded, dissipative, deterministic motion that never repeats; linear stability theory speaks only at the onset of convection and goes silent in the finite-amplitude regime where the interesting motion lives; and blaming the irregularity on external noisy forcing fails, because constant-forced laboratory convection is irregular too. What was needed was the smallest honest system that exhibits the amplification, small enough to reason about rather than merely watch misbehave.

I propose the three-equation convection system now called the Lorenz system. It is obtained by truncating two-dimensional Rayleigh–Bénard convection — a fluid layer of depth $H$ heated from below with fixed top-to-bottom temperature difference $\Delta T$, stress-free boundaries — down to its three most essential modes. Saltzman's numerical experience was the warrant: in his spectral integrations almost all modes died away and only a few survived, varying irregularly. So I keep exactly one mode for the motion and the two lowest modes for the temperature field, writing the stream function and the temperature departure as $\psi \propto X\,\sin(\pi a H^{-1}x)\sin(\pi H^{-1}z)$ and $\theta \propto Y\,\cos(\pi a H^{-1}x)\sin(\pi H^{-1}z) - Z\,\sin(2\pi H^{-1}z)$, with $X,Y,Z$ functions of time alone and the spatial shapes frozen. Substituting into the convection equations, performing the $x$- and $z$-integrals that orthogonality makes clean, and discarding every term that would feed a spatial mode I did not keep, then introducing dimensionless time $\tau = \pi^2 H^{-2}(1+a^2)\kappa\,t$, collapses the partial differential equations to

$$X' = \sigma(Y - X), \qquad Y' = rX - Y - XZ, \qquad Z' = XY - bZ.$$

Here $X$ measures the intensity of the convective overturning, $Y$ the temperature difference between the rising and sinking fluid (equal signs of $X$ and $Y$ meaning warm fluid rises and cold sinks, the natural sense), and $Z$ the distortion of the vertical temperature profile away from linear. The Prandtl number is $\sigma = \nu/\kappa$, the relative Rayleigh number $r = \mathrm{Ra}/\mathrm{Rc}$ is how hard I heat relative to the convection threshold, and $b = 4/(1+a^2)$ is geometric. The entire nonlinearity has collapsed to the two quadratic advection terms $XZ$ and $XY$ — the surviving fingerprints of the fluid carrying its own heat and vorticity around. These quadratic terms are exactly what makes the system worth studying: a linear dissipative system under constant forcing relaxes provably to a steady response, so nonlinearity is the only way constant forcing can yield variable motion. I fix the constants at $\sigma = 10$ (a representative Prandtl number), $a^2 = \tfrac12$ (the aspect ratio that convects most easily, so $b = 8/3$), and treat $r$ as the knob.

What makes this system produce sustained nonperiodic motion, rather than settling, is the simultaneous truth of three facts that leave no other possibility. First, no stable resting place. The origin is the rest state; for $r > 1$ a pair of steady convecting states appears at $X=Y=\pm\sqrt{b(r-1)},\,Z=r-1$ — at $r=28$ these sit at $(\pm 6\sqrt2,\pm 6\sqrt2,27)\approx(\pm 8.485,\pm 8.485,27)$ — one for each roll direction, matching Rayleigh's onset at $r=1$. The origin is a saddle (the convection instability itself). Linearizing about a convecting state gives the characteristic equation

$$\lambda^3 + (\sigma+b+1)\lambda^2 + (r+\sigma)b\,\lambda + 2\sigma b(r-1) = 0,$$

whose complex roots cross the imaginary axis — so steady convection itself loses stability — at $r = \sigma(\sigma+b+3)/(\sigma-b-1) = 470/19 \approx 24.74$ for these constants. At $r = 28 > 24.74$ all three fixed points are unstable; every one repels nearby states in at least one direction. Second, despite this, no trajectory can run away. The naive distance $\tfrac12(X^2+Y^2+Z^2)$ fails because the $rX$ forcing leaves an unsigned cross term, so I shift the vertical-temperature variable by the forcing level and use $W = \tfrac12[X^2 + Y^2 + (Z-r-\sigma)^2]$. Along the flow the quadratic advective contributions cancel exactly,

$$W' = -\sigma X^2 - Y^2 - bZ^2 + b(r+\sigma)Z = -\sigma X^2 - Y^2 - b\!\left[Z - \tfrac{r+\sigma}{2}\right]^2 + \tfrac{b(r+\sigma)^2}{4},$$

so $W' < 0$ outside a fixed ellipsoid: every trajectory is driven inward and trapped in a bounded region forever. The shift is the load-bearing trick — it is what makes the advective terms cancel and leaves a manifestly negative-definite remainder. Third, the flow is strongly dissipative. The divergence of the velocity field is constant and negative,

$$\frac{\partial X'}{\partial X} + \frac{\partial Y'}{\partial Y} + \frac{\partial Z'}{\partial Z} = -(\sigma+b+1),$$

so any phase-space volume contracts as $V(t) = V(0)\,e^{-(\sigma+b+1)t}$, here $e^{-(41/3)t}$ — every blob is crushed toward zero volume, uniformly, forgetting the size of where it began.

Hold the three together. Confined to a bounded box, every volume crushed to zero, and no stable point to land on, the trajectories must accumulate on a set of zero volume that is neither a point nor a loop. The decisive step from there to unpredictability is a general theorem about bounded flows: under uniqueness and continuity, if a central trajectory were stable, two late near-recurrences to the same limit point would force its future to nearly repeat after a large time shift — that is quasi-periodicity. Contrapositively, a genuinely nonperiodic central trajectory cannot be stable, and noncentral nonperiodic trajectories are not uniformly stable; for prediction, when one cannot measure precisely enough to tell a central path from a nearby transient, nonperiodic motion is effectively unstable. So nonperiodic implies unstable implies sensitive dependence — exactly the logical shape of the rounded-restart accident. The remaining question, whether the sustained motion truly is nonperiodic, I settle numerically: integrating at $r=28$, the trajectory spirals outward around one convecting state, is flung across to the other when it gets far enough, spirals there, is flung back, and the number of loops on each side before switching never settles. Two surfaces that appear to merge cannot actually merge (uniqueness forbids it; the volume law presses them exponentially close), so what looks like one sheet is two, then four, then eight — an infinitely-layered attracting set of zero volume, the folding from volume contraction and the prying-apart from neighbor divergence reconciled in its infinite intricacy. Quantitatively, the successive maxima of $Z$ fall almost on a single tent-shaped return curve, the next maximum nearly determined by the current; its idealized rescaled model is $M_{n+1} = 2M_n$ for $M_n < \tfrac12$ and $M_{n+1} = 2 - 2M_n$ for $M_n > \tfrac12$, and the measured slope has magnitude greater than one along every branch. Because the slope exceeds one everywhere, periodic maxima sequences — like rational starting values — are exceptional and unstable, while almost every sequence, like the irrationals, is nonperiodic. That is the smoking gun: the system is nonperiodic by overwhelming measure, not by accident. Carried back to the atmosphere, which shares the very ingredients of nonlinear advection, constant uneven forcing, and dissipation, this means long-range prediction is impossible in principle unless the present state is known exactly: two states differing by as little as the immediate influence of a single butterfly may, given enough time, evolve into two states differing as much as the presence of a tornado.

A small integration of exactly these three equations makes the divergence concrete — two starts differing by only $10^{-6}$ in one coordinate, a smaller version of the rounded restart from $0.506127$ to $0.506$, confirming the separation grows to the scale of the attractor within a few dozen time units while the orbit stays bounded and winds around the two convecting states.

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
