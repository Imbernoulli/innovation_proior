I will work out the speed of the spaceship by using the Lorentz invariance of the four-velocity scalar product, a strategy I will call the four-velocity invariant method for the 2023 Physics Cup spaceship problem. The ship undergoes four identical proper-acceleration burns in perpendicular directions, and the puzzle asks for the unique speed at which the magnitude of the Earth-frame velocity after the first burn equals the magnitude after the fourth burn. Rather than tracking ordinary velocity components through each turn, I reduce the whole motion to a single invariant time component of the Earth four-velocity as seen from the successive ship rest frames.

First I set up the kinematics. I use units in which the speed of light is one and write $\beta=v/c$. During each interval the astronauts feel the same constant proper acceleration of magnitude $g$ for the same proper time $\tau$, so each burn is the same Lorentz boost in the ship's instantaneous rest frame. I denote the Lorentz factor of that one-burn boost by $\gamma=1/\sqrt{1-\beta^2}$ and the corresponding rapidity by $\alpha=\operatorname{arctanh}\beta$. The first interval starts from rest relative to Earth, so the speed after the first burn is exactly the requested value $v$, and the associated Lorentz factor is $\gamma$.

The complication is that the boost axes rotate. After the first burn the engine is turned ninety degrees counterclockwise and fires along the $y$-direction, then it reverses along $x$, then reverses along $y$. If I try to compose ordinary three-velocities directly in the Earth frame, each new burn has both parallel and perpendicular components relative to the current motion, and the denominators in the relativistic velocity-addition formula become messy. Worse, after two non-collinear boosts the ship axes experience a Wigner rotation, so keeping track of directions in the Earth frame involves extra angular baggage that the final answer does not need. The question only asks for a speed, which is a scalar.

That scalar is most naturally captured by the Minkowski inner product of four-velocities. For two future-directed unit four-velocities $a$ and $b$ with metric signature $(+,-,-,-)$, the invariant product $a\cdot b$ equals the Lorentz factor $\gamma_{\rm rel}$ between the two frames. In the rest frame of $a$, the four-velocity of $b$ has time component $\gamma_{\rm rel}$, so the statement holds in every frame. Let $U_E$ be Earth's four-velocity and $U_k$ the ship's four-velocity after $k$ burns. Because consecutive ship rest frames differ by the same one-burn boost, I have $U_{k-1}\cdot U_k=\gamma$ for $k=1,2,3,4$. In particular, $U_E\cdot U_1=\gamma$ is the speed condition at $t=\tau$. The requirement that the speed at $t=4\tau$ is the same value $v$ translates into the identical scalar equation $U_E\cdot U_4=\gamma$. All directional information has disappeared.

To evaluate $U_E\cdot U_4$ I express Earth's four-velocity in each successive ship rest frame. Let $X_k$ denote the components of $U_E$ in the frame that is instantaneously at rest with the ship after $k$ burns, with the spatial axes relabeled after every burn so that the next engine firing always points along the local $x$-axis. The time component of $X_k$ is exactly $X_k^0=U_E\cdot U_k$, which is the quantity I need. At the start $X_0=(1,0,0)$.

One full step consists of a passive boost to the new rest frame followed by a passive rotation that reorients the axes. The boost along the local $x$-axis with relative speed $\beta$ is
\[
B=\begin{pmatrix}\gamma&-\gamma\beta&0\\ -\gamma\beta&\gamma&0\\ 0&0&1\end{pmatrix}.
\]
After the boost the engine points in the old $+y$ direction, and to make it the new $+x$ direction while preserving handedness I rotate axes so that $x_{\rm new}=y_{\rm old}$ and $y_{\rm new}=-x_{\rm old}$. That rotation is
\[
R=\begin{pmatrix}1&0&0\\ 0&0&1\\ 0&-1&0\end{pmatrix}.
\]
The combined passive transformation for one step is therefore
\[
\Lambda=RB=\begin{pmatrix}\gamma&-\gamma\beta&0\\ 0&0&1\\ \gamma\beta&-\gamma&0\end{pmatrix}.
\]
Because any final rotation after the fourth burn affects only spatial components, it cannot change the time component $X_4^0$, so I may simply apply $\Lambda$ four times to $X_0$.

Carrying out the matrix multiplications gives
\[
X_1=(\gamma,0,\gamma\beta),\qquad
X_2=(\gamma^2,\gamma\beta,\gamma^2\beta),
\]
\[
X_3=\bigl(\gamma^2(\gamma-\beta^2),\ \gamma^2\beta,\ \gamma^2\beta(\gamma-1)\bigr),
\]
and finally
\[
X_4=\begin{pmatrix}\gamma^3(\gamma-2\beta^2)\\ \gamma^2\beta(\gamma-1)\\ \gamma^3\beta(\gamma-\beta^2-1)\end{pmatrix}.
\]
The time component is the desired invariant:
\[
U_E\cdot U_4=X_4^0=\gamma^3(\gamma-2\beta^2).
\]
Using $\beta^2=(\gamma^2-1)/\gamma^2$, this becomes
\[
U_E\cdot U_4=\gamma^4-2\gamma^3+2\gamma.
\]
The condition that the speed at $t=4\tau$ equals the speed at $t=\tau$ is the equality of Lorentz factors, so
\[
\gamma^4-2\gamma^3+2\gamma=\gamma.
\]
Rearranging,
\[
\gamma^4-2\gamma^3+\gamma=\gamma(\gamma-1)(\gamma^2-\gamma-1)=0.
\]
The factor $\gamma=0$ is unphysical, and $\gamma=1$ corresponds to the trivial case where the ship never moves. The nontrivial motion must satisfy $\gamma^2-\gamma-1=0$, whose positive root is the golden ratio,
\[
\gamma=\frac{1+\sqrt5}{2}.
\]
Since $\gamma^2=\gamma+1$, the squared speed is
\[
\beta^2=1-\frac{1}{\gamma^2}=\frac{\gamma^2-1}{\gamma^2}=\frac{1}{\gamma}=\gamma-1=\frac{\sqrt5-1}{2}.
\]
Restoring $c$, the speed is
\[
v=c\sqrt{\frac{\sqrt5-1}{2}}\approx 0.78615\,c.
\]
Notably, the result is independent of the proper acceleration $g$ and the interval length $\tau$; those quantities only determine how hard the engines must work to reach the required one-burn Lorentz factor. The condition that the fourth-frame invariant returns to the same Lorentz factor as the first frame fixes the speed uniquely.

The following small Python script verifies the algebra numerically. It forms the one-step boost-rotation matrix from the golden-ratio Lorentz factor, applies it four times to the Earth four-velocity, and checks that the resulting time component equals $\gamma$ to within floating-point tolerance.

```python
import numpy as np

# The nontrivial Lorentz factor is the golden ratio.
gamma = 0.5 * (1.0 + np.sqrt(5.0))
beta = np.sqrt(1.0 - 1.0 / gamma**2)

# One-step passive boost B along the local x-axis, followed by axis relabeling R.
B = np.array([
    [gamma, -gamma * beta, 0.0],
    [-gamma * beta, gamma, 0.0],
    [0.0, 0.0, 1.0]
])
R = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, -1.0, 0.0]
])
Lambda = R @ B

# Earth's four-velocity in the initial ship rest frame.
X = np.array([1.0, 0.0, 0.0])
for _ in range(4):
    X = Lambda @ X

# The time component is the invariant Lorentz factor between Earth and the ship.
assert np.isclose(X[0], gamma), "Invariant did not return to gamma."

v_over_c = beta
print(f"v/c = {v_over_c:.6f}")
print(f"v   = {v_over_c:.6f} c")
print(f"gamma check: X_4^0 = {X[0]:.6f}, gamma = {gamma:.6f}")
```
