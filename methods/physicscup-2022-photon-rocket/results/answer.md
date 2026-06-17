Set $c=1$ and use metric $(+,-,-,-)$. For one straight photon burn from rest mass $M$ to rest mass $\kappa M$, conservation gives a lightlike exhaust $P_\gamma=P_i-P_f$, hence
$$0=(P_i-P_f)^2=M^2-2ME+\kappa^2M^2.$$
Thus
$$E=\frac{M}{2}(1+\kappa^2).$$
The mass shell $E^2-p^2=(\kappa M)^2$ then gives
$$p^2=E^2-\kappa^2M^2=\frac{M^2}{4}(1-\kappa^2)^2,$$
so
$$p=\frac{M}{2}(1-\kappa^2).$$
Because collinear photon four-momenta add to another lightlike vector, this depends only on the mass ratio, not on the thrust profile.

Let the intermediate rest mass be $\eta m$ and the final rest mass be $fm$, where $f=1/4$. Evaluate the invariant $P_i\cdot P_f$ in two frames.

In the post-first-burn rest frame, the second burn starts from rest, so the onboard angle is the spatial angle of the final momentum:
$$E_f=\frac{m}{2}\left(\eta+\frac{f^2}{\eta}\right),\qquad |\vec p_f|=\frac{m}{2}\left(\eta-\frac{f^2}{\eta}\right).$$
The original rocket four-momentum in this frame is
$$E_i=\frac{m}{2}\left(\eta+\frac1\eta\right),\qquad \vec p_i=\left(-\frac{m}{2}\left(\frac1\eta-\eta\right),0,0\right).$$
Therefore
$$P_i\cdot P_f=\frac{m^2}{4}\left[\left(1+\eta^2+f^2+\frac{f^2}{\eta^2}\right)+\left(1-\eta^2+f^2-\frac{f^2}{\eta^2}\right)\cos\alpha\right].$$

In the original rest frame, $P_i=(m,0,0,0)$ and the final speed is $4/5$, so $\gamma_{\rm fin}=5/3$ and
$$P_i\cdot P_f=\gamma_{\rm fin}fm^2.$$
Equating the two expressions gives
$$\cos\alpha
=1+\frac{4\gamma_{\rm fin}f-2f^2-2}{1-\eta^2+f^2-f^2/\eta^2}.$$

For $f=1/4$ and $\gamma_{\rm fin}=5/3$, the numerator is $-11/24$. To minimize $\alpha$, maximize $\cos\alpha$, so maximize the positive denominator
$$1+f^2-\left(\eta^2+\frac{f^2}{\eta^2}\right).$$
It is positive on $f<\eta<1$ because it also equals
$$\left(1-\eta^2\right)\left(1-\frac{f^2}{\eta^2}\right).$$
Thus it is maximized by minimizing $\eta^2+f^2/\eta^2$, and AM-GM gives equality at $\eta=\sqrt f=1/2$, which lies in $f<\eta<1$. Then
$$\cos\alpha_{\min}=1+\frac{4\gamma_{\rm fin}f-2f^2-2}{(1-f)^2}.$$
With $f=1/4$ and $\gamma_{\rm fin}=5/3$,
$$\cos\alpha_{\min}
=1+\frac{-11/24}{(1-1/4)^2}
=1+\frac{-11/24}{9/16}
=\frac{5}{27}.$$

Therefore
$$\boxed{\alpha_{\min}=\arccos\left(\frac{5}{27}\right)\approx 79.3^\circ}.$$
