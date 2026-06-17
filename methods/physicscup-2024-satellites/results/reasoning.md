Two satellites, two ellipses sharing a focus at Earth's centre, tilted by an angle $\alpha$ between their major axes. I want the largest possible value of $|\vec v_2-\vec v_1|$ as each satellite runs around its own orbit. The relative velocity at a given instant is fixed once I know where each body sits on its ellipse, so the natural maximum is over two phase angles, one per orbit. The reassurance that I am allowed to use that two-phase maximum is buried in one innocent-looking sentence: the ratio of the periods is irrational. That means the pair of phases never locks into a fixed relationship; over time the two bodies visit the phase torus densely. A particular pair of phases may only be approached, not hit exactly, but relative speed is continuous, so the supremum along the actual motion is the same as the maximum over the full two-phase torus. Good; the problem really is "maximize over the closure of both phases independently," with no resonance lowering the least upper bound.

If I attack this head-on I have to write each velocity as a function of true anomaly. On an ellipse $r=p/(1+e\cos\theta)$ the speed has that ugly $\sqrt{1+2e\cos\theta+e^2}$ shape, the direction of $\vec v$ swings between perigee and apogee, and then I have to subtract two such vectors living on two differently-oriented ellipses and maximize the magnitude over both anomalies. Two coupled transcendental extremization conditions. That is a lot of grinding, and it would hide whatever structure is actually doing the work. Let me not parametrize by position at all and instead ask: as a single body sweeps its orbit, what shape does its *velocity vector* trace out? If that velocity locus is something simple, the whole "maximize a vector difference" question collapses into geometry of that shape.

So fix one satellite and watch the tip of $\vec v$, anchored at a common origin, as the body goes around once. The speed is largest at perigee and smallest at apogee, and the direction rotates through a full turn over one orbit. My instinct says ellipse, because everything about a Kepler orbit feels elliptical. But I should not trust that; I should compute it. The clean way to find the velocity locus is to find a conserved vector that ties $\vec v$ down. Energy and angular momentum are the usual conserved scalars, but a scalar only constrains $\vec v$ to a curve implicitly; what I want is a *vector* identity that expresses $\vec v$ directly in terms of fixed quantities plus the one thing that genuinely rotates as the body moves. There is exactly such a vector in the inverse-square problem, and the reason it exists is special to the $1/r^2$ force — it is why Kepler orbits close.

Let me reconstruct it. Put the heavy mass $M$ at the origin, a satellite of mass $m$ at $\vec r=r\hat r$, working in the orbital plane with polar unit vectors $\hat r,\hat\phi$ and the out-of-plane $\hat z$. Newton's law is $\dot{\vec v}=-\dfrac{GM}{r^2}\hat r$. The angular momentum $\vec J=m\,\vec r\times\vec v=J\hat z$ is conserved, and in polar form $J=mr^2\dot\phi$. I also need the basis-vector kinematics: $\dot{\hat r}=\dot\phi\,\hat\phi$ and $\dot{\hat\phi}=-\dot\phi\,\hat r$.

Now I look for a vector built from $\vec v$ and $\hat r$ whose time derivative cancels. Consider $\vec v\times\vec J$. Its derivative is $\dot{\vec v}\times\vec J$ because $\vec J$ is constant, and

$$\dot{\vec v}\times\vec J=-\frac{GM}{r^2}\,\hat r\times(J\hat z)=-\frac{GMJ}{r^2}\,(\hat r\times\hat z).$$

In a right-handed $(\hat r,\hat\phi,\hat z)$ frame, $\hat r\times\hat z=-\hat\phi$, so this is $+\dfrac{GMJ}{r^2}\hat\phi$. Substitute $J=mr^2\dot\phi$ and the $r^2$ cancels:

$$\frac{d}{dt}(\vec v\times\vec J)=GM\,m\,\dot\phi\,\hat\phi.$$

That is begging to be matched against $\dot{\hat r}=\dot\phi\,\hat\phi$. Indeed $GMm\,\dot\phi\,\hat\phi=GMm\,\dot{\hat r}$, so

$$\frac{d}{dt}\Big(\vec v\times\vec J-GMm\,\hat r\Big)=0.$$

A conserved vector falls out. Divide by $GMm$ to make it dimensionless and define

$$\vec\varepsilon=\frac{\vec v\times\vec J}{GMm}-\hat r.$$

This is constant along the entire orbit. Its constancy is exactly the inverse-square cancellation I need here: $\dot{\vec v}\propto 1/r^2$ meets the $r^2$ inside angular momentum, and the radial dependence disappears. That cancellation is what gives me a fixed eccentricity vector for the Kepler problem instead of another scalar constraint.

What is this constant vector geometrically? At perigee, $\vec r$ and $\vec v$ are perpendicular and the body is moving fastest. There $\hat r$ points from the focus toward the near apse and $\vec v=v_p\hat\phi$, so $\vec J=mr_pv_p\hat z$ and $\vec v\times\vec J=mr_pv_p^2\hat r$. The vis-viva relation at $r_p=a(1-e)$ gives $v_p^2=GM(2/r_p-1/a)=GM(1+e)/r_p$, hence $\vec v\times\vec J/(GMm)=(1+e)\hat r$ and $\vec\varepsilon=e\hat r$ at perigee. Since $\vec\varepsilon$ is conserved, it is always a fixed vector of length $e$ along the major axis toward perigee. This is the eccentricity vector, the normalized Laplace–Runge–Lenz vector.

Now I have what I wanted: a vector identity pinning $\vec v$. Rearrange $\vec\varepsilon=\dfrac{\vec v\times\vec J}{GMm}-\hat r$ into

$$\vec v\times\vec J=GMm\,(\vec\varepsilon+\hat r).$$

The left side has $\vec v$ crossed with a fixed vector $\vec J=J\hat z$, so I can invert it. Cross both sides with $\hat z$ on the left, or just recall that for a planar $\vec v$, $\vec v\times(J\hat z)=J(\vec v\times\hat z)$ and $\hat z\times(\vec v\times\hat z)=\vec v$. Crossing $\hat z$ into the equation,

$$J\,\hat z\times(\vec v\times\hat z)=GMm\,\hat z\times(\vec\varepsilon+\hat r)\quad\Rightarrow\quad J\,\vec v=GMm\,\hat z\times(\vec\varepsilon+\hat r),$$

so

$$\vec v=\frac{GMm}{J}\,\hat z\times(\vec\varepsilon+\hat r).$$

I should write $J$ in the language the problem actually gives me. The areal rate — the rate the focus-to-satellite segment sweeps area — is $L=\tfrac12 r^2\dot\phi$, because the little triangle swept in $dt$ has area $\tfrac12 r^2\dot\phi\,dt$. And $J=mr^2\dot\phi$. So $J=2mL$, and $L$ is a constant of the orbit exactly because $J$ is. The mass cancels cleanly:

$$\vec v=\frac{GM}{2L}\,\hat z\times(\vec\varepsilon+\hat r)=\frac{GM}{2L}\big(\,\hat z\times\vec\varepsilon+\hat z\times\hat r\,\big)=\frac{GM}{2L}\big(\vec\chi+\hat\phi\big),$$

where I named $\vec\chi\equiv\hat z\times\vec\varepsilon$ and used $\hat z\times\hat r=\hat\phi$. Look at what each piece does. The vector $\vec\chi$ is $\vec\varepsilon$ rotated ninety degrees in the plane, so it is a *fixed* vector of length $e$, perpendicular to the major axis. The vector $\hat\phi$ is a *unit* vector that swings all the way around as the body orbits. So

$$\vec v=\underbrace{\frac{GM}{2L}\vec\chi}_{\text{fixed center}}+\underbrace{\frac{GM}{2L}\hat\phi}_{\text{rotating, length }GM/2L}.$$

There it is — and it is not an ellipse. The velocity vector is a constant offset plus a vector of *constant length* $GM/2L$ that rotates through a full turn. The tip of $\vec v$ traces a perfect circle. Its center sits at $\dfrac{GM}{2L}\vec\chi$, a point at distance $\dfrac{GMe}{2L}$ from the origin, and its radius is $\dfrac{GM}{2L}$. The velocity hodograph of a Kepler orbit is a circle. I half-expected an ellipse and the algebra corrected me: it is cleaner than that.

Two things about the orientation matter for later. The center direction $\vec\chi$ is perpendicular to the major axis (it is $\vec\varepsilon$ turned by $90^\circ$). And the origin — the tip of zero velocity — lies *inside* this circle for a bound orbit, since the center is at distance $\tfrac{GMe}{2L}$ and the radius is $\tfrac{GM}{2L}$, and $e<1$. Fine. The essential output: each satellite's velocity lives on its own circle, fully determined by $(e,L,\text{axis direction})$.

So now both velocities are points on circles. Satellite 1: a circle of radius $\rho_1=\dfrac{GM}{2L_1}$ centered at $\vec C_1=\dfrac{GM}{2L_1}\vec\chi_1$. Satellite 2: radius $\rho_2=\dfrac{GM}{2L_2}$ centered at $\vec C_2=\dfrac{GM}{2L_2}\vec\chi_2$. The relative velocity $\Delta\vec v=\vec v_2-\vec v_1$ is the vector from a point on circle 1 to a point on circle 2, and I want the largest value over the closure of all allowed phase pairs. The irrational-period clause is what makes that closure the whole product of the two hodograph circles. This is now pure plane geometry: the largest distance between a point on one circle and a point on another.

Picture two circles. Take any point $\vec C_1$ at one center and $\vec C_2$ at the other; the segment between two points, one on each circle, is at most the distance between the centers plus the two radii — and that bound is achieved by walking each point radially *outward along the center-to-center line, away from the other circle*. The two surface points then sit on the line through both centers, on the far sides. So

$$|\Delta\vec v|_{\max}=|\vec C_2-\vec C_1|+\rho_1+\rho_2.$$

I just have to compute the center separation. Writing it out,

$$\vec C_2-\vec C_1=\frac{GM}{2}\left(\frac{\vec\chi_2}{L_2}-\frac{\vec\chi_1}{L_1}\right),$$

and $\vec\chi_1,\vec\chi_2$ both have length $e_1,e_2$ respectively, with the angle between them equal to the angle between the major axes, namely $\alpha$ — because each $\vec\chi$ is its $\vec\varepsilon$ rotated by the same $90^\circ$, so the relative angle is unchanged. By the law of cosines on $\vec\chi_2/L_2-\vec\chi_1/L_1$,

$$|\vec C_2-\vec C_1|=\frac{GM}{2}\sqrt{\frac{e_1^2}{L_1^2}+\frac{e_2^2}{L_2^2}-2\,\frac{e_1e_2}{L_1L_2}\cos\alpha}.$$

Let me pause on the sign of that cosine, because there is a real physical choice hiding in it. I implicitly took both satellites circulating the same way, so that both $\hat\phi$'s are oriented by the same $\hat z$ and the angle between the two $\vec\chi$'s is just $\alpha$. Almost all satellites do orbit the same sense, and the statement implies it. If instead they counter-rotate, one body's $\hat z$ flips, its hodograph circle is reflected, and the effective angle between the center vectors becomes $\pi-\alpha$, which turns the $-\cos\alpha$ into $+\cos\alpha$. I will carry the same-direction case, $-\cos\alpha$, and note the flip. Adding the two radii $\rho_1+\rho_2=\dfrac{GM}{2}\left(\dfrac{1}{L_1}+\dfrac{1}{L_2}\right)$,

$$|\Delta\vec v|_{\max}=\frac{GM}{2}\left[\sqrt{\frac{e_1^2}{L_1^2}+\frac{e_2^2}{L_2^2}-2\,\frac{e_1e_2}{L_1L_2}\cos\alpha}+\frac{1}{L_1}+\frac{1}{L_2}\right].$$

Before trusting this I should make sure the configuration that achieves the two-circle bound is physically relevant, not just geometrically optimal. The extremal $\Delta\vec v$ wants $\vec v_2$ at the far point of circle 2 and $\vec v_1$ at the far point of circle 1, both along the line of centers and pointing away from each other. In velocity terms, $\vec v=\dfrac{GM}{2L}(\vec\chi+\hat\phi)$ reaches its extreme along the center line when $\hat\phi$ aligns with $\pm$ the center-line direction, and each $\hat\phi$ sweeps every direction over its own orbit. With an irrational period ratio, the actual time evolution is dense in the pair of phases, so it can come arbitrarily close to that pair. If the initial phases happen to place the dense orbit through the extremal pair, the value is attained; otherwise this is the exact supremum. That is the mathematically precise content of the maximal value.

It is cleaner to clear the denominators. Multiply inside by $L_1L_2$:

$$|\Delta\vec v|_{\max}=\frac{GM}{2L_1L_2}\left[\sqrt{e_1^2L_2^2+e_2^2L_1^2-2\,e_1e_2L_1L_2\cos\alpha}+L_1+L_2\right].$$

Now the requested special case. Set $L_1=L_2=L$ and $\alpha=90^\circ$, so $\cos\alpha=0$. The square root becomes $\sqrt{e_1^2L^2+e_2^2L^2}=L\sqrt{e_1^2+e_2^2}$, and the bracket is $L\sqrt{e_1^2+e_2^2}+2L$. The prefactor is $\dfrac{GM}{2L^2}$, so

$$\Delta v_{\max}=\frac{GM}{2L^2}\cdot L\left(\sqrt{e_1^2+e_2^2}+2\right)=\frac{GM}{2L}\left(\sqrt{e_1^2+e_2^2}+2\right).$$

That is a strikingly compact result, and it carries the whole story in it. The "$2$" is the sum of the two equal radii $\tfrac{GM}{2L}+\tfrac{GM}{2L}$ in units of $\tfrac{GM}{2L}$, the contribution of the rotating unit vectors. Even when $e=0$ and the hodograph centers coincide, the two-phase geometry still allows antiparallel velocities with relative speed twice the orbital speed. The $\sqrt{e_1^2+e_2^2}$ is the separation of the two hodograph centers, the part that the eccentricities and the right-angle between the axes contribute. Everything that looked like it needed grinding over true anomalies was all the geometry of two circles, and the only physics input was that the inverse-square law makes the eccentricity vector constant, which is what bent the velocity locus into a circle in the first place.

To recap the causal chain: the $1/r^2$ force makes $\vec\varepsilon=\vec v\times\vec J/(GMm)-\hat r$ a conserved vector of length $e$ along the major axis; solving that identity for $\vec v$ gives $\vec v=\frac{GM}{2L}(\vec\chi+\hat\phi)$, a fixed offset plus a constant-length rotating vector, so each velocity hodograph is a circle of radius $\frac{GM}{2L}$ centered at distance $\frac{GMe}{2L}$; the maximal relative speed in the phase closure is then the largest distance between two such circles, centers' separation plus both radii, with the irrational period ratio guaranteeing that the actual motion approaches the optimal phase pair arbitrarily closely; and that distance is the boxed formula, collapsing to $\frac{GM}{2L}(\sqrt{e_1^2+e_2^2}+2)$ when the areal rates match and the axes are perpendicular.
