I set the speed of light to one and work with the Minkowski metric signature $(+,-,-,-)$. The rocket starts at rest with rest mass $m$. During the first stage it burns photons in some fixed direction with an unknown thrust history and an unknown amount of fuel, ending that stage with some rest mass $m_1$ and moving along the burn line. Then, in the rocket's instantaneous rest frame after the first stage, the thrust direction is turned by an angle $\alpha$, and a second burn brings the rocket to a final rest mass $m/4$ and a final speed $4/5$ in the original rest frame. The question is the smallest onboard turn angle $\alpha$ that can realize this final state.

The first thing I need is a clean way to describe a single straight photon burn. Suppose a body of rest mass $M$ is initially at rest and emits photons all in one direction, ending with rest mass $\kappa M$. The total exhaust four-momentum is lightlike, and because all emitted photons are proportional to the same lightlike vector their sum is also lightlike regardless of how the thrust varies in time. Conservation of four-momentum gives $P_\gamma = P_i - P_f$, and squaring this gives $0 = M^2 - 2 M E + \kappa^2 M^2$, so the final energy is $E = M(1+\kappa^2)/2$. The mass shell $E^2 - p^2 = (\kappa M)^2$ then gives $p = M(1-\kappa^2)/2$. Thus a straight photon burn is completely summarized by the initial and final rest masses; no thrust profile details survive.

I introduce the dimensionless intermediate mass ratio $\eta$ by $m_1 = \eta m$, and the fixed final ratio $f = 1/4$. The core idea is to evaluate the invariant dot product $P_i \cdot P_f$ in two different frames and equate the results. In the original rest frame the initial four-momentum is $P_i = (m,0,0,0)$, and the final four-momentum has energy $\gamma_{\rm fin} f m$ with $\gamma_{\rm fin} = 1/\sqrt{1-(4/5)^2} = 5/3$, so $P_i \cdot P_f = \gamma_{\rm fin} f m^2$.

In the post-first-burn rest frame the second burn starts from rest mass $\eta m$ and ends at rest mass $f m$. Using the single-burn formulas with $M = \eta m$ and $\kappa = f/\eta$, the final energy and momentum magnitude after the second burn are $E_f = (m/2)(\eta + f^2/\eta)$ and $|\vec p_f| = (m/2)(\eta - f^2/\eta)$. The spatial momentum makes angle $\alpha$ with the first-burn direction. The original rocket four-momentum in this same frame is obtained by reversing the first burn: $E_i = (m/2)(\eta + 1/\eta)$ and $\vec p_i$ points backward along the first-burn direction with magnitude $(m/2)(1/\eta - \eta)$. Taking the dot product and simplifying gives
\[
P_i \cdot P_f = \frac{m^2}{4}\left[\left(1+\eta^2+f^2+\frac{f^2}{\eta^2}\right) + \left(1-\eta^2+f^2-\frac{f^2}{\eta^2}\right)\cos\alpha\right].
\]

Equating the two evaluations of the same invariant and solving for $\cos\alpha$ yields
\[
\cos\alpha = 1 + \frac{4\gamma_{\rm fin} f - 2 f^2 - 2}{1 - \eta^2 + f^2 - f^2/\eta^2}.
\]
For the given final state the numerator of the fraction equals $-11/24$. Inside the physical range $f < \eta < 1$ the denominator is positive, so minimizing the angle is the same as maximizing this denominator. The denominator can be rewritten as $(1-\eta^2)(1-f^2/\eta^2)$, and maximizing it is equivalent to minimizing $\eta^2 + f^2/\eta^2$. By the AM-GM inequality this minimum occurs at $\eta^2 = f$, so $\eta = \sqrt{f} = 1/2$. This means the optimal two-burn sequence reduces rest mass by equal ratios: $m \to m/2 \to m/4$.

Substituting $\eta = 1/2$, or equivalently using the optimized formula with the denominator equal to $(1-f)^2$, gives
\[
\cos\alpha_{\min} = 1 + \frac{-11/24}{(3/4)^2} = 1 - \frac{22}{27} = \frac{5}{27}.
\]
Therefore the smallest possible onboard turn is
\[
\alpha_{\min} = \arccos\left(\frac{5}{27}\right) \approx 79.3^\circ.
\]

The canonical name for this approach is relativistic photon-rocket turn-angle optimization via the Minkowski invariant dot product. The hidden thrust history disappears because collinear photon exhaust has a lightlike total four-momentum; the frame mismatch disappears because the same invariant $P_i \cdot P_f$ encodes the final energy in one frame and the onboard angle in another. All that remains is the mass split between the two burns, and the equal-ratio split is forced by the AM-GM inequality.

```python
import math

f = 0.25
gamma_fin = 5.0 / 3.0

def cos_alpha(eta):
    num = 4 * gamma_fin * f - (1 + eta**2 + f**2 + f**2 / eta**2)
    den = 1 - eta**2 + f**2 - f**2 / eta**2
    return num / den

eta_opt = math.sqrt(f)
cos_opt = 5.0 / 27.0
alpha_opt = math.degrees(math.acos(cos_opt))

best = -1.0
best_eta = None
for k in range(1, 10000):
    eta = f + (1.0 - f) * k / 10000.0
    c = cos_alpha(eta)
    if c > best:
        best = c
        best_eta = eta

print(f"Analytic eta_opt = {eta_opt:.6f}")
print(f"Analytic cos(alpha_min) = {cos_opt:.10f}")
print(f"Analytic alpha_min = {alpha_opt:.4f} deg")
print(f"Numerical best cos = {best:.10f} at eta = {best_eta:.6f}")
print(f"Numerical best alpha = {math.degrees(math.acos(best)):.4f} deg")
```
