## Fastest path across a boundary, via Huygens' wavefront construction

**Problem.** A point travels at speed $v_1$ in region 1 and speed $v_2$ in region 2, the two regions meeting along a straight boundary. It must go from $A$ in region 1 to $B$ in region 2 in minimum time, by a straight segment in each region meeting at a crossing point $P$ on the boundary. For the lifeguard instance: $v_1=5\ \mathrm{m/s}$ (sand), $v_2=2\ \mathrm{m/s}$ (water); $A$ is $30\ \mathrm{m}$ from the boundary, $B$ is $40\ \mathrm{m}$ from it on the far side, and the feet of $A$ and $B$ are $100\ \mathrm{m}$ apart along the boundary.

**Key idea.** Don't minimize the travel time $T(x)$ over the crossing point by calculus. Treat the minimal-time frontier as a *wavefront* and propagate it by Huygens' principle: every point of the current wavefront is a secondary source emitting a wavelet of radius (local speed)$\times dt$, and the next wavefront is the envelope of those wavelets. The ray (optimal path) is everywhere perpendicular to the wavefronts. Fermat's principle of least time guarantees the ray is the least-time path; Huygens' construction *produces* the ray without minimizing anything.

### Derivation of the refraction law from the wavefront

Let the ray in region $i$ make angle $\theta_i$ with the normal to the boundary, so the wavefront is inclined to the boundary at angle $\theta_i$.

Because the envelope touches each secondary wavelet along the radius, the wavefront advances **normal to itself at the local speed**: in time $dt$ it moves forward $v_i\,dt$.

Consider the point where the wavefront meets the boundary, and let $u$ be the speed at which that intersection slides *along* the boundary (the trace speed). In time $dt$ the boundary intersection moves $u\,dt$ along the boundary while the wavefront advances $v_1\,dt$ perpendicular to itself; since the wavefront is inclined to the boundary by $\theta_1$, the perpendicular advance equals the along-boundary slide times $\sin\theta_1$:
$$v_1\,dt=(u\,dt)\sin\theta_1\quad\Longrightarrow\quad u=\frac{v_1}{\sin\theta_1}.$$

The region-2 wavefront is built from sources on the *same* boundary, switched on in the same sequence and hence with the same along-boundary trace speed $u$. The identical right-triangle relation on the region-2 side gives
$$u=\frac{v_2}{\sin\theta_2}.$$

The intersection point is a single geometric point, so the two trace speeds must agree:
$$\frac{v_1}{\sin\theta_1}=\frac{v_2}{\sin\theta_2}
\qquad\Longleftrightarrow\qquad
\boxed{\dfrac{\sin\theta_1}{v_1}=\dfrac{\sin\theta_2}{v_2}}
\qquad\Longleftrightarrow\qquad
\frac{\sin\theta_1}{\sin\theta_2}=\frac{v_1}{v_2}.$$

This is Snell's law for the least-time path. The law is local: only the speeds and the two angles at the crossing point enter, while the positions of $A$ and $B$ determine which crossing point can satisfy it. That is why the same law appears as the stationarity condition of $T(x)$. (The same envelope-of-circles construction, applied to a source moving at $v$ through a single medium with wave speed $c$, gives the Mach-cone half-angle $\sin\beta=c/v$ — a sine equal to a speed ratio, by the same geometry.) Since $v_2<v_1$, the law forces $\theta_2<\theta_1$: the path bends *toward* the normal in the slower region.

### Applying it to the lifeguard

Place the foot of $A$ at the origin, the boundary along the $x$-axis, and the crossing point at $P=(x,0)$, with $A=(0,30)$ and $B=(100,-40)$. Then
$$\sin\theta_1=\frac{x}{\sqrt{x^2+30^2}},\qquad
\sin\theta_2=\frac{100-x}{\sqrt{(100-x)^2+40^2}},$$
and the refraction law $\sin\theta_1/\sin\theta_2=v_1/v_2=5/2$ becomes
$$\frac{x}{\sqrt{x^2+900}}=\frac{5}{2}\,\frac{100-x}{\sqrt{(100-x)^2+1600}}.$$
The left side increases in $x$ and the right side decreases, so the root is unique. Solving numerically,
$$x\approx 83.74\ \mathrm{m}.$$
Check: $\sin\theta_1\approx0.9414$, $\sin\theta_2\approx0.3766$, ratio $=2.500=v_1/v_2$; $\theta_1\approx70.3^\circ$, $\theta_2\approx22.1^\circ$, bent toward the normal in the water.

**Result.** The lifeguard should enter the water about $83.7\ \mathrm{m}$ down the shoreline from the foot of $A$, at $\theta_1\approx70.3^\circ$ to the normal, and swim out at $\theta_2\approx22.1^\circ$. The least total time is
$$T_{\min}=\frac{\sqrt{83.74^2+30^2}}{5}+\frac{\sqrt{16.26^2+40^2}}{2}\approx 17.79+21.59\approx \boxed{39.4\ \mathrm{s}}.$$
(The straight $A\!\to\!B$ dash crosses at $x\approx42.9\ \mathrm{m}$ and takes $\approx45.3\ \mathrm{s}$; the refracted path is faster because it trades fast-sand distance for less slow-water distance.)
