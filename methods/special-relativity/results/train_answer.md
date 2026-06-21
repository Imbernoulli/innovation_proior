Two enormously successful theories sit side by side and quietly contradict each other. Newtonian mechanics, through the Galilean principle, says the laws of motion are the same in every uniformly moving frame, that coordinates transform as $x' = x - vt$ with the time shared by all frames, $t' = t$, and that velocities simply add. Maxwell's electrodynamics governs the field and hands us a definite propagation speed $c = 1/\sqrt{\varepsilon_0\mu_0}$ — but it never says *with respect to which frame* that speed holds. The customary repair, an all-pervading luminiferous ether that is the rest frame of light, should make the measured speed of light depend on one's motion through it, the way the speed of sound depends on the wind. Yet every attempt to find that dependence — first-order optical interference and aberration experiments, and decisively the second-order Michelson–Morley interferometer — returns null. We need a single consistent kinematics in which Maxwell's electrodynamics holds in every inertial frame, the speed of light is the same in every inertial frame, and the unobservability of the ether is *explained* rather than patched, all without privileging any frame.

The existing options each fall short in the same telling way: they keep a privileged frame alive. Lorentz's theorem of corresponding states (1895) rewrites Maxwell's equations for a system moving through the ether and shows that to first order in $v/c$ they retain their form provided one feeds them not the true time but a doctored "local time" $t' = t - (v/c^2)\,x$ together with $x' = x - vt$; that explains the first-order nulls, but $t'$ is treated as a mere auxiliary calculational device while the "real" ether time $t$ survives underneath. To kill the second-order Michelson–Morley null, the FitzGerald–Lorentz contraction hypothesis bolts on a separate *dynamical* assumption that bodies physically shrink along their motion by $\sqrt{1 - v^2/c^2}$. Lorentz's 1904 transformation gets $\gamma = 1/\sqrt{1-v^2/c^2}$ into structurally central position but leaves an undetermined overall scale $l(v)$ open and still rests on the ether and the true-time/local-time split. Poincaré presses the relativity principle as a postulate, reads local time as what a moving observer gets by synchronizing clocks with light, and fixes $l(v)=1$ by a group requirement — but still keeps one foot in the ether and the "true time." The splinter common to all of them is the word *true*: as long as there is a true time underneath, there is a privileged frame, and the magnet-and-conductor asymmetry — one current, told as an electric field in one frame and a magnetic force in another — comes right back. That asymmetry is in the bookkeeping, not in nature, and the only thing the bookkeeping uses to distinguish the cases is a notion of absolute rest.

I propose the kinematics of special relativity, built honestly on two postulates and a single operational definition. The first postulate (P1), the principle of relativity, is that the laws of physics — mechanics *and* electrodynamics — take the same form in every inertial frame; no experiment picks out absolute rest. The second postulate (P2), the constancy of light, is that light propagates in vacuum at the one speed $c$ independent of the motion of the source, because I trust Maxwell's equations, which hand me $c$ as a constant of the field, not of any emitter. Put together with the ordinary Galilean rule these seem to collide: P1 and P2 say a light pulse has speed $c$ in both my frame and a frame gliding past at $v$, while velocity addition says I should measure $c - v$. The collision is real only because of a buried, unexamined third assumption hiding inside $t' = t$ — that simultaneity is absolute, that "now" is one shared fact for everybody. The whole resolution is to refuse that assumption and define simultaneity instead.

The definition has to be operational, because every time judgment is at bottom a judgment of coincidence, and a single clock dates only events at its own location; "simultaneous at a distance" is meaningless until a rule is laid down. The clean rule uses the one thing both postulates make universal — light. Put identical clocks at $A$ and $B$, send a flash from $A$ at $A$-time $t_A$, reflect it at $B$ at $B$-time $t_B$, and receive it back at $A$ at $A$-time $t'_A$; *define* the clocks to be synchronized when the light takes equally long each way,
$$t_B - t_A = t'_A - t_B,$$
which is the same as dating the remote reflection event at the midpoint of emission and return, and which builds in that the round-trip speed is $c$. This is a definition, not a discovery — there is nothing to discover until it is made — and it is symmetric and transitive, so it is consistent.

The decisive move is that simultaneity, so defined, inherits whatever the postulates say about light, and need not survive a change of frame. Take a rod of measured length $r_{AB}$ moving at $v$, with end-clocks that the rod-rider has synchronized by the light rule. Watching the same flash from the frame the rod moves through, light still goes at $c$ (P2), but the far end runs away from the outgoing flash and the near end runs into the returning one, so the two legs take
$$t_B - t_A = \frac{r_{AB}}{c - v}, \qquad t'_A - t_B = \frac{r_{AB}}{c + v},$$
which are unequal. The clocks the rider set equal are, to me, not synchronized: simultaneity is frame-relative. The assumption $t' = t$ is simply false, invisible in daily life only because $c$ is so large that the offset $\sim r\,v/c^2$ is negligible. With that prejudice removed the two postulates no longer conflict — they only ever conflicted *through* absolute time — and Lorentz's local time $t' = t - vx/c^2$ stops being a trick: that $-vx/c^2$ is precisely the position-dependent offset making the moving observer's light-synchronized clocks consistent, so it *is* his time, with no truer time beneath it.

With simultaneity pinned operationally, the transformation between frame $K$ with coordinates $(x,y,z,t)$ and frame $k$ moving at $v$ along the shared axis with coordinates $(\xi,\eta,\zeta,\tau)$ is forced. Homogeneity of space and time requires the map to be linear, so uniform motions and equal intervals are preserved. I fix $\tau$ by demanding it be the time the $k$-observer obtains from synchronizing his own clocks. Shooting a ray from $k$'s origin out to a co-moving point $x' \equiv x - vt$ and back, the midpoint condition $\tfrac12(\tau_0+\tau_2)=\tau_1$, with $K$-frame light-times $x'/(c-v)$ outbound and $x'/(c+v)$ back, expanded to first order in $x'$, gives the constraint
$$\frac{\partial\tau}{\partial x'} + \frac{v}{c^2 - v^2}\,\frac{\partial\tau}{\partial t} = 0,$$
and the transverse synchronizing rays give $\partial\tau/\partial y = \partial\tau/\partial z = 0$, so by linearity $\tau = a\,(t - \tfrac{v}{c^2-v^2}\,x')$ for some $a = a(v)$. The spatial part is then fixed by P2 in $k$: a ray sent along $+\xi$ satisfies $\xi = c\tau$ by $k$'s own measurement, and the same ray seen from $K$ has $x' = (c-v)t$; substituting yields $\xi$ in terms of $a$, and the transverse light-times give $\eta,\zeta$. Writing $\beta \equiv 1/\sqrt{1 - v^2/c^2}$ and collecting (using $c^2/(c^2-v^2)=\beta^2$, $c/\sqrt{c^2-v^2}=\beta$, $v/(c^2-v^2)=\beta^2 v/c^2$), everything folds into a single leftover scale $\varphi(v)$:
$$\tau = \varphi(v)\,\beta\!\left(t - \frac{vx}{c^2}\right), \quad \xi = \varphi(v)\,\beta\,(x - vt), \quad \eta = \varphi(v)\,y, \quad \zeta = \varphi(v)\,z.$$
That $\varphi(v)$ is exactly the scale Lorentz could not pin down, and the postulates should leave nothing free, so I kill it with two arguments. Reciprocity: composing the transformation at $+v$ then at $-v$ must be the identity, which forces $\varphi(v)\,\varphi(-v) = 1$. Isotropy: a rod moving *sideways* (perpendicular to its own length) cannot care about the sign of $v$, so $\varphi(v) = \varphi(-v)$. Together with $\varphi > 0$ this gives $\varphi(v) = 1$, and the transverse directions are therefore wholly unchanged. The transformation is rigid:
$$\tau = \beta\!\left(t - \frac{vx}{c^2}\right), \quad \xi = \beta(x - vt), \quad \eta = y, \quad \zeta = z, \qquad \beta = \frac{1}{\sqrt{1 - v^2/c^2}}.$$

The two postulates are then provably compatible. A spherical light pulse $x^2 + y^2 + z^2 = c^2t^2$ in $K$ stays a sphere of speed $c$ in $k$ because the transformation leaves the quadratic form invariant:
$$\xi^2 - c^2\tau^2 = \beta^2(1 - v^2/c^2)\,(x^2 - c^2t^2) = x^2 - c^2t^2,$$
the cross terms $\mp 2vxt$ cancelling cleanly, so $\xi^2+\eta^2+\zeta^2 - c^2\tau^2 = x^2+y^2+z^2 - c^2t^2 = 0$. I prefer reaching the invariance through the synchronization definition rather than postulating it, because that is where the physics lives: the invariance is a *consequence* of how time had to be redefined.

The laboratory consequences then read straight off the transformation with no new assumptions. Length contraction: a rest-frame sphere $\xi^2+\eta^2+\zeta^2=R^2$, taken at one $K$-instant $t=0$ (so $\xi=\beta x$, $\eta=y$, $\zeta=z$), becomes the ellipsoid $x^2/(1-v^2/c^2) + y^2 + z^2 = R^2$, squashed longitudinally by $\sqrt{1-v^2/c^2}$ and untouched sideways — Lorentz's contraction, but as a statement about simultaneity, not about ether forces squeezing matter, because "the length of the moving rod" *means* where its ends are at one instant of my frame, and that instant is no longer what the rider means. Time dilation: a clock at $k$'s origin ($x = vt$) reads
$$\tau = \beta\!\left(t - \frac{v^2 t}{c^2}\right) = t\sqrt{1 - v^2/c^2} \le t,$$
running slow, falling behind by $\approx \tfrac12 v^2/c^2$ per second, so a clock carried around a closed loop returns behind a stationary one by $\approx \tfrac12(v^2/c^2)t$. Velocity addition: a point moving at $w$ in $k$, composed with the frame velocity $v$ (collinear), moves in $K$ at
$$V = \frac{v + w}{1 + vw/c^2},$$
which is $v + w$ to first order — why ordinary life never noticed — but whose denominator keeps $c$ unsurpassable: two sub-$c$ velocities always compose below $c$, and $c$ composed with any $w$ returns exactly $c$, so light is a fixed point of the composition law and "chasing a light beam" is impossible. These transformations compose into one of the same form, so they form a group, the structural guarantee that "no privileged frame" is consistent. The ether is gone because there was never a true time to host it, and the magnet and conductor tell one story — relative motion, one current — because there was never a fact about which one was really moving.

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
