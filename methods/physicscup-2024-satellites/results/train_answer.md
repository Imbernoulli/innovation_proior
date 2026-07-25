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

That closed form is the deliverable: for two satellites orbiting Earth (mass $M$, gravitational constant $G$) in the same sense, on ellipses of eccentricities $e_1,e_2$ and areal rates $L_1,L_2$ whose major axes are separated by angle $\alpha$,

$$\boxed{|\Delta\vec v|_{\max}=\frac{GM}{2L_1L_2}\left[\sqrt{e_1^2L_2^2+e_2^2L_1^2-2\,e_1e_2L_1L_2\cos\alpha}+L_1+L_2\right]},$$

attained (in the closure of the motion, guaranteed by the irrational period ratio) when the two velocity-hodograph points sit diametrically opposite one another along the line joining the two circle centers; if the satellites counter-rotate, one hodograph is mirrored and the sign in front of $\cos\alpha$ flips. For the requested special case $L_1=L_2=L$, $\alpha=90^\circ$, where the two hodograph circles have equal radius and perpendicular center offsets, this reduces to

$$\boxed{\Delta v_{\max}=\frac{GM}{2L}\left(\sqrt{e_1^2+e_2^2}+2\right)}.$$
