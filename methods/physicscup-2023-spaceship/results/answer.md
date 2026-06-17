Set $\beta=v/c$ and use $c=1$ during the derivation. Each interval is the same proper-acceleration maneuver in the ship's instantaneous rest frame, so each interval is a Lorentz boost with the same relative speed $\beta$ and Lorentz factor
$$\gamma=\frac1{\sqrt{1-\beta^2}}.$$

For any two future-directed unit four-velocities $a,b$,
$$a\cdot b=\gamma_{\rm rel}$$
because in the rest frame of $a$, the vector $b$ has time component $\gamma_{\rm rel}$. Thus the speed condition at $t=4\tau$ is equivalently
$$U_E\cdot U_4=\gamma,$$
where $U_E$ is Earth's four-velocity and $U_4$ is the ship's final four-velocity. Equivalently, for four-momenta $P_a=m_a a$ and $P_b=m_b b$, the invariant is $P_a\cdot P_b=m_a m_b\gamma_{\rm rel}$ while $c=1$.

Compute this scalar by expressing Earth's four-velocity in the successive ship rest frames. Let $X_k$ be Earth's four-velocity components in the ship frame after $k$ intervals, with the spatial axes relabeled after each interval so the next acceleration is again along the local $x$-axis. Then $X_k^0=U_E\cdot U_k$. Starting from $X_0=(1,0,0)$, one step is
$$\Lambda=RB,$$
with
$$B=\begin{pmatrix}\gamma&-\gamma\beta&0\\ -\gamma\beta&\gamma&0\\ 0&0&1\end{pmatrix},\qquad
R=\begin{pmatrix}1&0&0\\ 0&0&1\\ 0&-1&0\end{pmatrix},$$
where $R$ implements $x_{\rm new}=y_{\rm old}$ and $y_{\rm new}=-x_{\rm old}$ after the counterclockwise turn. Hence
$$\Lambda=\begin{pmatrix}\gamma&-\gamma\beta&0\\ 0&0&1\\ \gamma\beta&-\gamma&0\end{pmatrix}.$$

Applying $\Lambda$ four times gives
$$X_4=\begin{pmatrix}
\gamma^3(\gamma-2\beta^2)\\
\gamma^2\beta(\gamma-1)\\
\gamma^3\beta(\gamma-\beta^2-1)
\end{pmatrix}.$$
Therefore
$$U_E\cdot U_4=X_4^0=\gamma^3(\gamma-2\beta^2).$$
Using $\beta^2=(\gamma^2-1)/\gamma^2$,
$$U_E\cdot U_4=\gamma^4-2\gamma^3+2\gamma.$$
The condition $U_E\cdot U_4=\gamma$ gives
$$\gamma^4-2\gamma^3+\gamma=0
\quad\Longrightarrow\quad
\gamma(\gamma-1)(\gamma^2-\gamma-1)=0.$$
Discarding $\gamma=0$ outside Lorentz factors and the trivial $\gamma=1$ rest case leaves
$$\gamma^2-\gamma-1=0,$$
so
$$\gamma=\frac{1+\sqrt5}{2}.$$

Hence
$$\beta^2=1-\frac1{\gamma^2}=\frac1\gamma=\gamma-1=\frac{\sqrt5-1}{2},$$
and restoring $c$,
$$\boxed{v=c\sqrt{\frac{\sqrt5-1}{2}}\approx 0.78615\,c}.$$
