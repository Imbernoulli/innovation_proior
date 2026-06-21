I would call this method Huygens' wavefront construction for fastest paths, or simply the Huygens-Fermat boundary-crossing method. The problem is a classic lifeguard rescue: a straight shoreline separates sand, where the lifeguard runs at speed v1 = 5 m/s, from water, where she swims at speed v2 = 2 m/s. She starts at point A on the beach, 30 m from the shoreline, and must reach a swimmer at point B in the water, 40 m from the shoreline on the opposite side and 100 m down the beach from A's nearest shoreline point. The goal is to choose the entry point P on the shoreline that minimizes total travel time.

The first thing I notice is that this is exactly the same mathematical problem as light refracting at the interface between two media. Light minimizes travel time through Fermat's principle, and the path it takes obeys Snell's law. But I want to understand *why* the optimal path obeys a sine law, not just derive it by setting a derivative to zero. The calculus approach works: if the crossing point is at horizontal coordinate x, then total time is T(x) = sqrt(x^2 + 30^2)/5 + sqrt((100-x)^2 + 40^2)/2, and T'(x) = 0 gives x / (5 sqrt(x^2 + 900)) = (100-x) / (2 sqrt((100-x)^2 + 1600)). This already looks like a sine law, because x/sqrt(x^2 + 30^2) is the sine of the angle the running segment makes with the normal to the shoreline, and (100-x)/sqrt((100-x)^2 + 40^2) is the corresponding sine on the water side. But the calculus is hiding the deeper reason why the distances 30, 40, and 100 drop out of the local condition at the boundary.

To see that reason, I use Huygens' principle. Imagine the lifeguard's "reachable set" as an expanding wavefront. At time t, the wavefront is the set of all points she could have reached by time t if she had traveled optimally from A in every possible direction. In a uniform medium, this wavefront is just a circle expanding at the local speed. Now here is the key idea: every point on the current wavefront acts as a secondary source, emitting a small wavelet of radius (local speed) times dt. The wavefront at time t + dt is the envelope of all these little wavelets. Because each wavelet is tangent to the envelope along its radius, the envelope advances everywhere perpendicular to itself at exactly the local speed. The optimal path, the ray, is the normal to the wavefront.

This wavefront picture becomes powerful when the wavefront reaches the shoreline. On the sand side, it arrives at speed v1 = 5 m/s. As soon as a point on the shoreline is reached, it becomes a source for waves entering the water, but those waves expand at the slower speed v2 = 2 m/s. The crucial observation is that the intersection point of the wavefront with the shoreline moves along the shoreline at some trace speed u, and this trace speed must be the same whether computed from the sand side or the water side, because it is the motion of one geometric point.

Let theta1 be the angle between the ray in region 1 and the normal to the boundary, and theta2 the same angle in region 2. In a small time dt, the wavefront advances v1 dt perpendicular to itself, while its intersection with the shoreline slides u dt along the shoreline. The perpendicular advance equals the along-shore slide times sin(theta1), so v1 dt = (u dt) sin(theta1), which gives u = v1 / sin(theta1). On the water side, the same geometry gives u = v2 / sin(theta2). Equating the two expressions for the single trace speed yields v1 / sin(theta1) = v2 / sin(theta2), or equivalently sin(theta1)/v1 = sin(theta2)/v2. This is Snell's law, and it fell out of the continuity of the wavefront across the boundary rather than from differentiating a travel-time function.

The law is purely local: it relates only the speeds and the angles at the crossing point. The global positions of A and B determine *which* crossing point can realize these angles, but the matching condition itself knows nothing about the distances 30, 40, or 100. Because v2 is smaller than v1, we must have sin(theta2) smaller than sin(theta1), so theta2 < theta1: the path bends toward the normal in the slower medium. That matches the intuition that one wants to minimize time spent in the slow region.

To apply this to the lifeguard, I place the foot of A at the origin and the shoreline along the x-axis, so A = (0, 30) and B = (100, -40). If the entry point is P = (x, 0), then sin(theta1) = x / sqrt(x^2 + 30^2) and sin(theta2) = (100-x) / sqrt((100-x)^2 + 40^2). Snell's law sin(theta1)/sin(theta2) = v1/v2 = 5/2 becomes x / sqrt(x^2 + 900) = (5/2) * (100-x) / sqrt((100-x)^2 + 1600). The left side increases with x and the right side decreases, so the solution is unique. Numerically, x is approximately 83.74 m. At this point, sin(theta1) is about 0.9414 and sin(theta2) is about 0.3766, and their ratio is 2.5, confirming the refraction law. The corresponding angles are theta1 ≈ 70.3 degrees and theta2 ≈ 22.1 degrees. The least total time is T_min = sqrt(83.74^2 + 30^2)/5 + sqrt(16.26^2 + 40^2)/2 ≈ 17.79 + 21.59 ≈ 39.4 seconds. For comparison, the straight-line dash from A to B crosses the shoreline at about x = 42.9 m and takes roughly 45.3 seconds, so the refracted path saves nearly six seconds by trading a longer run on fast sand for a shorter swim in slow water.

The following Python script verifies the computation numerically: it solves the one-dimensional stationarity equation for the crossing point, checks the resulting Snell ratio, and reports the minimum time alongside the straight-line time for comparison.

```python
import numpy as np
from scipy.optimize import brentq

v1 = 5.0          # speed on sand (m/s)
v2 = 2.0          # speed in water (m/s)
yA = 30.0         # perpendicular distance of A from shoreline (m)
yB = 40.0         # perpendicular distance of B from shoreline (m)
L = 100.0         # along-shore separation of the feet of A and B (m)

A = np.array([0.0, yA])
B = np.array([L, -yB])

def total_time(x):
    P = np.array([x, 0.0])
    return np.linalg.norm(P - A) / v1 + np.linalg.norm(B - P) / v2

def derivative_time(x):
    return x / (v1 * np.sqrt(x**2 + yA**2)) - (L - x) / (v2 * np.sqrt((L - x)**2 + yB**2))

x_opt = brentq(derivative_time, 0.0, L)
t_min = total_time(x_opt)

x_straight = L * yA / (yA + yB)  # straight-line crossing point
t_straight = total_time(x_straight)

s1 = x_opt / np.sqrt(x_opt**2 + yA**2)
s2 = (L - x_opt) / np.sqrt((L - x_opt)**2 + yB**2)

print(f"Optimal shoreline entry point: x = {x_opt:.4f} m")
print(f"Angles: theta1 = {np.degrees(np.arcsin(s1)):.2f} deg, theta2 = {np.degrees(np.arcsin(s2)):.2f} deg")
print(f"Snell ratio sin(theta1)/sin(theta2) = {s1/s2:.4f} (target {v1/v2:.4f})")
print(f"Minimum travel time: {t_min:.4f} s")
print(f"Straight-line travel time: {t_straight:.4f} s")
```

In summary, Huygens' wavefront construction turns a boundary-crossing optimization into a continuity statement about how a wavefront's footprint slides along the interface. The resulting Snell relation determines the optimal entry point, and for the given numbers the lifeguard should enter the water about 83.7 m down the shore from A's foot, reaching the swimmer in roughly 39.4 seconds.
