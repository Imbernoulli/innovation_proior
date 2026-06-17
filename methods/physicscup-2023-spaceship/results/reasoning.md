Let me first separate the size of each burn from the direction bookkeeping. I write $\beta=v/c$ and set $c=1$ until the final line. In the ship's instantaneous rest frame, one interval of constant proper acceleration $g$ for proper time $\tau$ adds the fixed rapidity $\alpha=g\tau/c$ along the engine axis. So each interval carries the ship from one rest frame to the next by the same boost, with $\beta=\tanh\alpha$ and $\gamma=\cosh\alpha=1/\sqrt{1-\beta^2}$. The first interval starts from rest with the Earth, so this one-interval speed is exactly the speed called $v$ at $t=\tau$.

The hard part is not the size of the boost. It is the fact that the boost axis turns. If I try to do everything with ordinary three-velocity addition, the second burn already splits into parallel and perpendicular pieces with different denominators. Then the third burn is worse, because after two non-collinear boosts the ship's axes have not merely boosted relative to the Earth; they have also picked up the Wigner rotation. So if I keep chasing Earth-frame velocity components, I am solving a direction problem even though the question only asks for a speed.

I want a scalar that ignores all of that orientation clutter. For two future-directed unit four-velocities $a$ and $b$, with metric $\operatorname{diag}(+,-,-,-)$, the inner product is the relative Lorentz factor. In the rest frame of $a$, I have $a=(1,0,0)$ and $b=(\gamma_{\rm rel},\gamma_{\rm rel}\mathbf{v}_{\rm rel})$, so $a\cdot b=\gamma_{\rm rel}$. Since the dot product is invariant, this is true in any frame. If I multiply by rest masses and write $p_a=m_a a$, $p_b=m_b b$, then $p_a\cdot p_b=m_a m_b\gamma_{\rm rel}$ while $c=1$; the four-momentum version is the same invariant with only mass factors attached.

Let $U_E$ be the Earth's four-velocity and let $U_k$ be the ship's four-velocity after $k$ intervals. Consecutive ship frames differ by one identical speed-$\beta$ boost, so
$$U_{k-1}\cdot U_k=\gamma\qquad(k=1,2,3,4).$$
The first equality $U_E\cdot U_1=\gamma$ is just the speed at $t=\tau$. The condition at the end is the same scalar statement,
$$U_E\cdot U_4=\gamma.$$
So I do not need the final direction of the ship at all. I only need the Lorentz factor between the Earth frame and the fourth ship frame.

I can make that scalar appear as a component if I keep expressing the Earth's four-velocity in the current ship frame. Call those components $X_k$. In the $k$th ship rest frame, $U_k=(1,0,0)$, so the time component of $X_k$ is exactly
$$X_k^0=U_E\cdot U_k.$$
That is the collapse I need: the entire chained motion can be reduced to the time component of one transformed four-vector. A passive spatial rotation of the ship axes may change the last two components of $X_k$, but it cannot change $X_k^0$ or any Minkowski product.

Now I need one repeatable frame change. At the start of an interval I choose the local $x$-axis along the current proper acceleration. Passing to the ship's new rest frame is the passive boost
$$B=\begin{pmatrix}\gamma&-\gamma\beta&0\\ -\gamma\beta&\gamma&0\\ 0&0&1\end{pmatrix}.$$
After that, I relabel the spatial axes so that the next acceleration direction is again the local $x$-axis. The next engine direction is the old $+y$ direction, so the relabeling must read $x_{\rm new}=y_{\rm old}$ and $y_{\rm new}=-x_{\rm old}$. That passive rotation is
$$R=\begin{pmatrix}1&0&0\\ 0&0&1\\ 0&-1&0\end{pmatrix}.$$
Thus one whole step, boost into the new rest frame and then rotate axes for the next burn, is
$$\Lambda=RB=\begin{pmatrix}\gamma&-\gamma\beta&0\\ 0&0&1\\ \gamma\beta&-\gamma&0\end{pmatrix}.$$
The final rotation after the fourth burn is harmless, because I only read the time component. Starting with the Earth four-velocity in the initial frame, $X_0=(1,0,0)$, I can apply the same $\Lambda$ four times.

Multiplying carefully,
$$X_1=\Lambda X_0=\begin{pmatrix}\gamma\\0\\\gamma\beta\end{pmatrix},\qquad
X_2=\Lambda X_1=\begin{pmatrix}\gamma^2\\\gamma\beta\\\gamma^2\beta\end{pmatrix}.$$
The first time component is $\gamma$, which checks the first burn. The second is $\gamma^2$, which is also what I expect from two equal orthogonal boosts at the level of the scalar product. Continuing,
$$X_3=\Lambda X_2=\begin{pmatrix}\gamma^2(\gamma-\beta^2)\\\gamma^2\beta\\\gamma^2\beta(\gamma-1)\end{pmatrix},$$
and
$$X_4=\Lambda X_3=\begin{pmatrix}\gamma^3(\gamma-2\beta^2)\\\gamma^2\beta(\gamma-1)\\\gamma^3\beta(\gamma-\beta^2-1)\end{pmatrix}.$$
Since $X_4^0=U_E\cdot U_4$, the final Lorentz factor relative to Earth is
$$U_E\cdot U_4=\gamma^3(\gamma-2\beta^2).$$

Now the directions are gone and only the scalar algebra remains. With $\beta^2=1-1/\gamma^2=(\gamma^2-1)/\gamma^2$,
$$\gamma^3(\gamma-2\beta^2)=\gamma^4-2\gamma^3\frac{\gamma^2-1}{\gamma^2}
=\gamma^4-2\gamma^3+2\gamma.$$
The required equality of speeds at $t=\tau$ and $t=4\tau$ is equality of Lorentz factors, so
$$\gamma^4-2\gamma^3+2\gamma=\gamma.$$
That gives
$$\gamma^4-2\gamma^3+\gamma=0
\quad\Longrightarrow\quad
\gamma(\gamma^3-2\gamma^2+1)=0.$$
The factor $\gamma=0$ lies outside Lorentz factors. The cubic has the root $\gamma=1$, which is the trivial zero-speed case, and
$$\gamma^3-2\gamma^2+1=(\gamma-1)(\gamma^2-\gamma-1).$$
The nonzero motion therefore forces
$$\gamma^2-\gamma-1=0.$$
The physical root is
$$\gamma=\frac{1+\sqrt5}{2}.$$

I turn that Lorentz factor back into the speed. Since $\gamma^2=\gamma+1$,
$$\beta^2=1-\frac1{\gamma^2}=\frac{\gamma^2-1}{\gamma^2}=\frac{\gamma}{\gamma^2}=\frac1\gamma=\gamma-1=\frac{\sqrt5-1}{2}.$$
Restoring $c$,
$$v=c\sqrt{\frac{\sqrt5-1}{2}}\approx 0.78615\,c.$$
The proper acceleration and the interval length only decide which one-burn speed is produced; the condition that the fourth-frame invariant $U_E\cdot U_4$ returns to the same Lorentz factor forces that speed to be the one above.
