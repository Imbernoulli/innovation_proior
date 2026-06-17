Two satellites orbit Earth (mass $M$) in a common plane on ellipses of eccentricities $e_1,e_2$, with areal rates (the rate at which the focus-to-satellite segment sweeps area) $L_1,L_2$, and angle $\alpha$ between their major axes. The orbital-period ratio is irrational, so the two-phase motion is dense in the phase torus; the exact equality phases need not be hit, but the supremum along the actual motion is the same as the maximum over independent phase choices. The maximal relative velocity in this precise sense is obtained by noting that the **velocity hodograph of a Kepler orbit is a circle**, which reduces the problem to the geometry of two circles.

**The eccentricity (Laplace–Runge–Lenz) vector.** With $M$ at the origin, a body of mass $m$ obeys $\dot{\vec v}=-\dfrac{GM}{r^2}\hat r$, and $\vec J=m\,\vec r\times\vec v=J\hat z$ is conserved with $J=mr^2\dot\phi$. Using $\dot{\hat r}=\dot\phi\,\hat\phi$,

$$\frac{d}{dt}\big(\vec v\times\vec J\big)=-\frac{GMJ}{r^2}(\hat r\times\hat z)=GMm\,\dot\phi\,\hat\phi=GMm\,\dot{\hat r},$$

so

$$\vec\varepsilon=\frac{\vec v\times\vec J}{GMm}-\hat r$$

is conserved. It has magnitude $|\vec\varepsilon|=e$ and points along the major axis (toward perigee). The conservation is special to the inverse-square law: the $r^2$ from $\vec J$ cancels the $1/r^2$ from the force.

**The hodograph is a circle.** The problem's areal rate is $L=\tfrac12 r^2\dot\phi$, hence $J=2mL$. Solving the eccentricity-vector identity for $\vec v$ (cross with $\hat z$, use $\hat z\times(\vec v\times\hat z)=\vec v$ for planar $\vec v$):

$$\vec v=\frac{GM}{2L}\,\hat z\times(\vec\varepsilon+\hat r)=\frac{GM}{2L}\big(\vec\chi+\hat\phi\big),\qquad \vec\chi\equiv\hat z\times\vec\varepsilon .$$

Here $\vec\chi$ is fixed, of length $e$, perpendicular to the major axis, while $\hat\phi$ is a unit vector that sweeps a full turn over one orbit. Thus the tip of $\vec v$ traces a **circle**:

$$\text{center}\ \vec C=\frac{GM}{2L}\vec\chi\ \ (\,|\vec C|=\tfrac{GMe}{2L}\,),\qquad \text{radius}\ \rho=\frac{GM}{2L}.$$

**Maximal relative velocity = maximal distance between two circles.** Each $\vec v_i$ lies on a circle $(\vec C_i,\rho_i)$ with $\rho_i=\tfrac{GM}{2L_i}$ and $\vec C_i=\tfrac{GM}{2L_i}\vec\chi_i$. The largest distance between a point of circle 1 and a point of circle 2 is the centers' separation plus both radii, with the two surface points diametrically opposite along the line of centers. The irrational period ratio makes the actual pair of phases dense, so this geometric maximum is the supremum of the relative speed:

$$|\Delta\vec v|_{\max}=|\vec C_2-\vec C_1|+\rho_1+\rho_2 .$$

The centers' separation, with $|\vec\chi_i|=e_i$ and angle $\alpha$ between $\vec\chi_1,\vec\chi_2$ (equal to the angle between the major axes), is

$$|\vec C_2-\vec C_1|=\frac{GM}{2}\sqrt{\frac{e_1^2}{L_1^2}+\frac{e_2^2}{L_2^2}-2\,\frac{e_1e_2}{L_1L_2}\cos\alpha}.$$

**Result.**

$$\boxed{\,|\Delta\vec v|_{\max}=\frac{GM}{2L_1L_2}\left[\sqrt{e_1^2L_2^2+e_2^2L_1^2-2\,e_1e_2L_1L_2\cos\alpha}+L_1+L_2\right]\,}$$

For $L_1=L_2=L$ and $\alpha=90^\circ$ ($\cos\alpha=0$) this simplifies to

$$\boxed{\,\Delta v_{\max}=\frac{GM}{2L}\Big(\sqrt{e_1^2+e_2^2}+2\Big)\,.}$$

(The sign of $\cos\alpha$ corresponds to both satellites orbiting in the same sense, as the statement implies; if they counter-rotate, one hodograph circle is reflected and the sign before $\cos\alpha$ becomes $+$.)
