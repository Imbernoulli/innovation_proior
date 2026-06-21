I am asked for the largest possible relative speed of two satellites that orbit Earth in the same plane on ellipses of eccentricities $e_1$ and $e_2$, with angle $\alpha$ between their major axes and areal rates $L_1$ and $L_2$. The period ratio is irrational, so the pair of orbital phases never repeats; over time it explores the phase torus densely. Because relative speed is a continuous function of the two phases, the supremum along the actual motion equals the maximum over the full product of the two phase circles. That lets me treat the two phases as independent.

The key is to stop parametrizing by position and instead ask what curve the tip of a single satellite’s velocity vector traces. For a Kepler orbit this curve is especially simple. Place Earth’s mass $M$ at the origin and let the satellite have position $\vec r$, velocity $\vec v$, and conserved angular momentum $\vec J=m\,\vec r\times\vec v=J\hat z$. Newton’s law is $\dot{\vec v}=-(GM/r^2)\,\hat r$. Using $\dot{\hat r}=\dot\phi\,\hat\phi$ and $\hat r\times\hat z=-\hat\phi$, the time derivative of $\vec v\times\vec J$ becomes

$$\frac{d}{dt}(\vec v\times\vec J)=-\frac{GMJ}{r^2}(\hat r\times\hat z)=\frac{GMJ}{r^2}\hat\phi=GMm\,\dot\phi\,\hat\phi=GMm\,\dot{\hat r}.$$

Hence the vector

$$\vec\varepsilon=\frac{\vec v\times\vec J}{GMm}-\hat r$$

is constant. This is the eccentricity, or Laplace–Runge–Lenz, vector. It has length $e$ and points along the major axis toward perigee. Its conservation is special to the inverse-square force: the $r^2$ hidden inside angular momentum cancels the $1/r^2$ in the acceleration.

The problem gives the areal rate $L=\tfrac12 r^2\dot\phi$, so $J=2mL$. Crossing the identity $\vec v\times\vec J=GMm(\vec\varepsilon+\hat r)$ with $\hat z$ and using $\hat z\times(\vec v\times\hat z)=\vec v$ gives

$$\vec v=\frac{GM}{2L}\,\hat z\times(\vec\varepsilon+\hat r)=\frac{GM}{2L}(\vec\chi+\hat\phi),$$

where $\vec\chi=\hat z\times\vec\varepsilon$ is a fixed vector of length $e$ perpendicular to the major axis, and $\hat\phi$ is the unit azimuthal vector that rotates once per orbit. Therefore the velocity vector is a fixed offset plus a vector of constant length $GM/(2L)$ that swings through a full turn. The tip of $\vec v$ traces a circle: center $\vec C=(GM/2L)\,\vec\chi$, radius $\rho=GM/(2L)$. This is the Kepler velocity hodograph.

The canonical name I would give this approach is the Laplace–Runge–Lenz hodograph method for Kepler orbits.

For the two satellites I now have two circles in velocity space. Satellite $i$ lives on a circle of radius $\rho_i=GM/(2L_i)$ centered at $\vec C_i=(GM/2L_i)\,\vec\chi_i$, where $|\vec\chi_i|=e_i$ and the angle between $\vec\chi_1$ and $\vec\chi_2$ is $\alpha$. The relative velocity $\Delta\vec v=\vec v_2-\vec v_1$ is a vector from a point on circle 1 to a point on circle 2. The largest such distance is the separation of the centers plus the two radii, achieved when the two surface points lie on the line of centers on the far sides. Thus

$$|\Delta\vec v|_{\max}=|\vec C_2-\vec C_1|+\rho_1+\rho_2.$$

By the law of cosines,

$$|\vec C_2-\vec C_1|=\frac{GM}{2}\sqrt{\frac{e_1^2}{L_1^2}+\frac{e_2^2}{L_2^2}-2\,\frac{e_1e_2}{L_1L_2}\cos\alpha},$$

assuming both satellites orbit in the same sense; counter-rotation would reflect one hodograph and change the sign of $\cos\alpha$. Adding the radii and clearing denominators gives the compact result

$$|\Delta\vec v|_{\max}=\frac{GM}{2L_1L_2}\left[\sqrt{e_1^2L_2^2+e_2^2L_1^2-2\,e_1e_2L_1L_2\cos\alpha}+L_1+L_2\right].$$

For the special case $L_1=L_2=L$ and $\alpha=90^\circ$, where $\cos\alpha=0$, this simplifies to

$$\Delta v_{\max}=\frac{GM}{2L}\left(\sqrt{e_1^2+e_2^2}+2\right).$$

The term $\sqrt{e_1^2+e_2^2}$ is the scaled separation of the two hodograph centers, while the $2$ comes from adding the two equal radii. Even for circular orbits with $e_1=e_2=0$ the maximal relative speed is $GM/L$, twice the circular orbital speed, obtained when the two velocity vectors point opposite each other.

The following Python script illustrates the construction numerically. It parametrizes each Kepler orbit by true anomaly, computes the velocity from the vis-viva relation and the local tangent direction, confirms that the points lie on the predicted hodograph circle, and compares a dense scan over the two anomalies with the analytic formula.

```python
import numpy as np

def hodograph_speed_max(GM, e1, e2, L1, L2, alpha):
    """Analytic maximal relative speed for the two-satellite problem."""
    centers = np.sqrt(e1**2 * L2**2 + e2**2 * L1**2
                      - 2 * e1 * e2 * L1 * L2 * np.cos(alpha))
    radii = L1 + L2
    return (GM / (2 * L1 * L2)) * (centers + radii)

def velocity(true_anomaly, GM, L, e, major_axis_angle=0.0):
    """Velocity of a satellite on a Kepler ellipse with given areal rate L."""
    theta = true_anomaly
    # Specific angular momentum h = 2L, and h^2 = GM * p, so p = 4 L^2 / GM.
    p = 4.0 * L**2 / GM
    r = p / (1.0 + e * np.cos(theta))
    # vis-viva speed
    a = p / (1.0 - e**2)
    v = np.sqrt(GM * (2.0 / r - 1.0 / a))
    # local direction: tangent to ellipse in orbital plane
    # radial/transverse decomposition using h = 2L
    vr = (GM / (2.0 * L)) * e * np.sin(theta)
    vt = (GM / (2.0 * L)) * (1.0 + e * np.cos(theta))
    # rotate by major-axis orientation
    phi = theta + major_axis_angle
    # radial and transverse unit vectors in the fixed lab frame
    rhat = np.array([np.cos(phi), np.sin(phi)])
    phihat = np.array([-np.sin(phi), np.cos(phi)])
    return vr * rhat + vt * phihat

if __name__ == "__main__":
    GM = 1.0
    e1, e2 = 0.5, 0.3
    L1, L2 = 1.2, 0.9
    alpha = np.deg2rad(60.0)

    thetas = np.linspace(0, 2 * np.pi, 2000)
    v1 = np.array([velocity(t, GM, L1, e1, 0.0) for t in thetas])
    v2 = np.array([velocity(t, GM, L2, e2, alpha) for t in thetas])

    # Check that each hodograph is a circle with the predicted center and radius.
    rho1, rho2 = GM / (2 * L1), GM / (2 * L2)
    chi1 = np.array([0.0, e1])
    chi2 = e2 * np.array([-np.sin(alpha), np.cos(alpha)])
    C1 = (GM / (2 * L1)) * chi1
    C2 = (GM / (2 * L2)) * chi2
    print("radius 1 error:", np.max(np.abs(np.linalg.norm(v1 - C1, axis=1) - rho1)))
    print("radius 2 error:", np.max(np.abs(np.linalg.norm(v2 - C2, axis=1) - rho2)))

    # Dense scan over independent anomalies.
    dv = v2[:, None, :] - v1[None, :, :]
    max_numerical = np.max(np.linalg.norm(dv, axis=2))
    max_analytic = hodograph_speed_max(GM, e1, e2, L1, L2, alpha)
    print("max numerical:", max_numerical)
    print("max analytic: ", max_analytic)
    print("relative error:", abs(max_numerical - max_analytic) / max_analytic)

    # Special case L1=L2=L, alpha=90 degrees.
    L = 1.0
    a90 = np.deg2rad(90.0)
    v1s = np.array([velocity(t, GM, L, e1, 0.0) for t in thetas])
    v2s = np.array([velocity(t, GM, L, e2, a90) for t in thetas])
    dvs = v2s[:, None, :] - v1s[None, :, :]
    max_numerical_90 = np.max(np.linalg.norm(dvs, axis=2))
    max_analytic_90 = (GM / (2 * L)) * (np.sqrt(e1**2 + e2**2) + 2.0)
    print("special case relative error:",
          abs(max_numerical_90 - max_analytic_90) / max_analytic_90)
```

So the answer is the boxed formula above, with the simplified expression $\Delta v_{\max}=\tfrac{GM}{2L}\big(\sqrt{e_1^2+e_2^2}+2\big)$ when the areal rates are equal and the major axes are perpendicular.
