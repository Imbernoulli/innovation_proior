Let me set $c=1$ and draw the first burn along the $x$-axis. The rocket starts at rest with mass $m$. After the first burn it has some mass $m_1$ and moves along $x$, but the statement refuses to tell me the thrust profile or the amount of fuel burned. Then, in the rocket's instantaneous rest frame, the thrust direction is turned by $\alpha$, and the second burn ends with mass $m/4$ and speed $4/5$. The free middle state is not a nuisance detail; it is the thing I have to optimize over.

If I attack this directly, I have to name the intermediate speed $v_1$, boost into the post-first-burn rest frame, put the second-burn velocity at angle $\alpha$, boost back, and then relate both speed gains to photon exhaust and mass loss. The velocity-addition formulas immediately mix the parallel and transverse components. That is a bad sign, because the angle is measured in one frame while the final speed is specified in another. I need a quantity that is allowed to be evaluated in both frames without carrying the boost formulas through the whole calculation.

The invariant available for four-momenta is the Minkowski dot product, $P\cdot Q=E_P E_Q-\vec p_P\cdot\vec p_Q$. If I can make the angle appear in one evaluation of a dot product and make the final speed appear in another evaluation of the same dot product, the boost disappears into invariance. Before that, I need one clean fact about a photon rocket burn.

Take a body of rest mass $M$ initially at rest. It emits light in one fixed direction and recoils in the opposite direction, ending with rest mass $\kappa M$, where $0<\kappa<1$. In the initial rest frame, write the initial four-momentum as $P_i=(M,0,0,0)$ and the final body as $P_f=(E,p,0,0)$. The total exhaust from this straight phase is lightlike and, by four-momentum conservation, equals $P_\gamma=P_i-P_f$. Squaring that conservation equation in the Minkowski metric gives
$$0=(P_i-P_f)^2=P_i^2-2P_i\cdot P_f+P_f^2=M^2-2ME+\kappa^2M^2.$$
So $E=\frac{M}{2}(1+\kappa^2)$. The mass shell for the final body then gives
$$p^2=E^2-\kappa^2M^2=\frac{M^2}{4}(1-\kappa^2)^2,$$
so the recoil momentum magnitude is $p=\frac{M}{2}(1-\kappa^2)$. A time-varying thrust profile in one fixed direction changes nothing here: all emitted photons are proportional to the same lightlike vector, and their sum is still lightlike. So a straight burn is summarized only by its initial and final rest masses.

Now I can compress the unknown first burn into one parameter. Let the intermediate mass be $m_1=\eta m$, and let the final mass be $fm$ with $f=1/4$. For a genuine two-burn turn I should work with $f<\eta<1$; the endpoints would remove one of the two phases. I want the value of $\eta$ that permits the smallest turn.

I choose the dot product between the original rocket four-momentum $P_i$ and the final rocket four-momentum $P_f$. In the original rest frame, $P_i=(m,0,0,0)$, so the dot product is just $m$ times the final energy. The final rest mass is $fm$, and the final speed is $4/5$, so
$$\gamma_{\rm fin}=\frac{1}{\sqrt{1-(4/5)^2}}=\frac{5}{3},\qquad P_i\cdot P_f=\gamma_{\rm fin} f m^2.$$

The same dot product in the post-first-burn rest frame is where the onboard angle lives. In this frame the second burn starts from rest mass $\eta m$ and ends at rest mass $fm$. The single-burn formula, with $M=\eta m$ and $\kappa=f/\eta$, gives the final rocket energy and momentum magnitude in this frame:
$$E_f=\frac{m}{2}\left(\eta+\frac{f^2}{\eta}\right),\qquad |\vec p_f|=\frac{m}{2}\left(\eta-\frac{f^2}{\eta}\right).$$
Its spatial momentum points at angle $\alpha$ from the first-burn direction, so $\vec p_f=|\vec p_f|(\cos\alpha,\sin\alpha,0)$.

I still need the original rocket four-momentum in this same frame. In the original rest frame, the first burn sends mass $m$ to mass $\eta m$, so the intermediate rocket energy equals $\frac{m}{2}(1+\eta^2)$. Dividing by the intermediate rest mass $\eta m$ gives the relative Lorentz factor $\gamma=\frac12(\eta+1/\eta)$. Viewed from the intermediate rest frame, the original rocket moves backward along $x$, so
$$E_i=\frac{m}{2}\left(\eta+\frac{1}{\eta}\right),\qquad \vec p_i=\left(-\frac{m}{2}\left(\frac{1}{\eta}-\eta\right),0,0\right).$$

Now the dot product in the intermediate frame is
$$P_i\cdot P_f=E_iE_f-\vec p_i\cdot\vec p_f
=\frac{m^2}{4}\left[\left(\frac1\eta+\eta\right)\left(\eta+\frac{f^2}{\eta}\right)+\left(\frac1\eta-\eta\right)\left(\eta-\frac{f^2}{\eta}\right)\cos\alpha\right].$$
Expanding the two products, this is
$$P_i\cdot P_f=\frac{m^2}{4}\left[\left(1+\eta^2+f^2+\frac{f^2}{\eta^2}\right)+\left(1-\eta^2+f^2-\frac{f^2}{\eta^2}\right)\cos\alpha\right].$$

Equating the two evaluations and cancelling $m^2$ gives
$$4\gamma_{\rm fin}f=\left(1+\eta^2+f^2+\frac{f^2}{\eta^2}\right)+\left(1-\eta^2+f^2-\frac{f^2}{\eta^2}\right)\cos\alpha.$$
Solving for the cosine,
$$\cos\alpha=\frac{4\gamma_{\rm fin}f-\left(1+\eta^2+f^2+f^2/\eta^2\right)}{1-\eta^2+f^2-f^2/\eta^2}
=1+\frac{4\gamma_{\rm fin}f-2f^2-2}{1-\eta^2+f^2-f^2/\eta^2}.$$

For the given final data, the numerator in the added fraction is
$$4\cdot\frac53\cdot\frac14-2\cdot\frac1{16}-2=\frac53-\frac18-2=-\frac{11}{24}.$$
Inside the open interval $f<\eta<1$, the denominator is
$$1-\eta^2+f^2-\frac{f^2}{\eta^2}=1+f^2-\left(\eta^2+\frac{f^2}{\eta^2}\right),$$
and the same expression factors as
$$\left(1-\eta^2\right)\left(1-\frac{f^2}{\eta^2}\right),$$
so it is positive throughout that interval. Since the numerator is negative, maximizing $\cos\alpha$ means making this positive denominator as large as possible. That reduces the problem to minimizing $\eta^2+f^2/\eta^2$.

AM-GM gives
$$\eta^2+\frac{f^2}{\eta^2}\geq 2f,$$
with equality when $\eta^2=f$, so $\eta=\sqrt f$. For $f=1/4$, the optimum is $\eta=1/2$, an admissible intermediate mass. This means the two burns reduce the rest mass by equal ratios, $m\to m/2\to m/4$.

At $\eta=\sqrt f$, the denominator becomes
$$1-f+f^2-f=(1-f)^2.$$
So the optimized cosine, before putting in the numerical final state, is
$$\cos\alpha_{\min}=1+\frac{4\gamma_{\rm fin}f-2f^2-2}{(1-f)^2}.$$
Substituting $f=1/4$ and $\gamma_{\rm fin}=5/3$,
$$\cos\alpha_{\min}=1+\frac{-11/24}{(3/4)^2}
=1-\frac{11}{24}\cdot\frac{16}{9}
=1-\frac{22}{27}
=\frac{5}{27}.$$
So the smallest onboard turn is
$$\alpha_{\min}=\arccos\left(\frac{5}{27}\right)\approx 79.3^\circ.$$

Now I see why the hidden thrust history vanishes: a collinear photon exhaust has a lightlike total four-momentum. I also see why the frame mismatch vanishes: the same invariant $P_i\cdot P_f$ reads as an angle formula in the intermediate rest frame and as final energy in the original rest frame. The remaining freedom is only the mass split, and the equal-ratio split is forced by the AM-GM minimum.
