The fact that won't leave me alone is the SLAC deep-inelastic data. You strike a proton hard, at large momentum transfer, and the inclusive structure functions come out *scaling*: they depend only on the dimensionless ratio $x = Q^2/2M\nu$ and not separately on $Q^2$, exactly as if the charged constituents were free, point-like, spin-$\tfrac12$ particles carrying fractional baryon number. Quarks. And yet those same constituents are permanently confined — no one can knock one loose. So the strong force has to do two contradictory things at once: bind permanently at the hadronic scale, and effectively switch off when the constituents are probed at very short distance. The only honest tool for "what happens at short distance" is the renormalization group, which already taught us the coupling is not a number but a function of scale, with $\beta(g) = M\,dg/dM$ governing how it runs. Short distance means large $M$, so the whole question collapses to the sign of $\beta$ near $g=0$: if $\beta < 0$ the coupling shrinks toward the ultraviolet and the force turns off (and I can hope to explain scaling), while if $\beta > 0$ it grows the way it does in electrodynamics and I am dead.

The wall everyone slams into is that $\beta > 0$ is supposed to be a law of nature. Quantum electrodynamics is the paradigm and it *screens*: the vacuum is a polarizable medium of virtual $e^+e^-$ pairs, a bare charge polarizes it, the induced dipoles partially cancel the charge, and the effective coupling therefore grows toward short distance — $\beta > 0$, with Landau's eventual zero-charge catastrophe. Every $\beta$ anyone had computed — scalar $\phi^4$, Yukawa, Abelian gauge — came out positive, and the no-go arguments (a spectral-positivity argument, the Coleman–Gross result of 1973) prove that *no* renormalizable theory built from scalars, fermions with arbitrary Yukawa couplings, and Abelian gauge fields can have a coupling that vanishes in the ultraviolet. The S-matrix/bootstrap program gives soft high-momentum behavior, the opposite of the hard scaling seen; current algebra establishes only what scaling *would require* (free-field short-distance behavior), not which dynamics could deliver it. But there is exactly one renormalizable case the positivity machinery cannot close: **non-Abelian (Yang–Mills) gauge theory**, where the gauge bosons themselves carry the charge. The spectral reasoning that kills every other theory simply fails to go through for a charged, self-interacting gauge field. That silence is the whole opening, and the only way through it is to compute the one-loop $\beta$-function honestly.

I propose that the theory of the strong interaction is a **non-Abelian color gauge theory — specifically $SU(3)$ Yang–Mills coupled to quarks — and that it is asymptotically free**: its coupling vanishes in the ultraviolet because the charged gluons antiscreen the vacuum. Before the Feynman graphs, here is the physics that tells me why this theory can be different. In units $c=1$ the vacuum's dielectric constant and permeability obey $\varepsilon\mu = 1$, so electric screening ($\varepsilon > 1$) is *the same statement* as diamagnetism ($\mu < 1$). Decompose the vacuum's magnetic response to a color charge into an orbital/diamagnetic piece (the ordinary Lenz response, $-\tfrac13$ per unit charge-squared, the screening piece a spinless or fermionic source gives) and a *paramagnetic* spin piece. A charge-$q$, spin-$s$ particle contributes $\Delta\mu = [\,-\tfrac13 + (\gamma s)^2\,]\,q^2$ to the permeability, where $\gamma$ is its gyromagnetic ratio. The gluon is a charged spin-1 boson, and gauge invariance fixes its gyromagnetic ratio to $\gamma = 2$, so $\Delta\mu_{\text{gluon}} = (-\tfrac13 + 2^2)\,q^2 = +\tfrac{11}{3}\,q^2$ — paramagnetic, antiscreening: the $\tfrac{11}{3}$ is literally "$4$ (spin paramagnetism) $-\tfrac13$ (orbital diamagnetism)." In QED the photon is neutral, so this term never exists and you only ever see screening. The no-go theorems missed this case precisely because they had no charged spin-1 source.

Now the calculation, in a covariant gauge with Faddeev–Popov ghosts, since covariant quantization of a non-Abelian theory demands them. Here the gluon is charged, the simple QED Ward identity ($Z_1 = Z_2$) becomes the weaker Slavnov–Taylor identities with $Z_1 \neq Z_2$, and the coupling renormalization read off the quark–gluon vertex is
$$g_{\text{bare}} = g\,\frac{Z_1}{Z_2\,Z_3^{1/2}},$$
so writing each $Z = 1 + \delta$ and taking the log derivative, the one-loop $\beta$-function is built from the residues of the $1/\varepsilon$ poles,
$$\beta(g) = g\cdot\mathrm{Res}\big[\,2\delta_1 - 2\delta_2 - \delta_3\,\big]\cdot\frac{g^2}{16\pi^2}.$$
I need three counterterms. The **quark self-energy** $\delta_2$ is, stripped of color, exactly the QED electron self-energy times the quark Casimir $C(Q)$ (from $T^a T^a = C(Q)$ on the multiplet), giving $\mathrm{Res}[\delta_2] = -C(Q)$. The **quark–gluon vertex** $\delta_1$ has two diagrams: the QED-like one carries the color factor $T^b T^a T^b = (C(Q) - \tfrac12 C(G))\,T^a$ — the $-\tfrac12 C(G)$ being the first place the self-interaction sneaks in — and a genuinely new diagram with the three-gluon vertex inside, whose color reduces to a pure $C(G)$ piece and whose Lorentz algebra gives $+\tfrac32 C(G)$; together $(C(Q) - \tfrac12 C(G)) + \tfrac32 C(G) = C(Q) + C(G)$, so $\mathrm{Res}[\delta_1] = -(C(Q) + C(G))$. That $\delta_1 \neq \delta_2$ is the expected signature of the non-Abelian Ward identity, not an error.

The **gluon vacuum polarization** $\delta_3$ is where the magnetic intuition must be vindicated by signs, and it has five diagrams: the quark loop, the gluon loop (two three-gluon vertices), the four-gluon tadpole, the ghost loop, and the counterterm. The quark loop is the screening piece — color trace $\mathrm{tr}(T^aT^b) = R_{\text{net}}\,\delta^{ab}$ summed over flavors, times the QED $-\tfrac43$ — contributing $-\tfrac43 R_{\text{net}}$. The gluon loop alone is *not transverse*: its numerator splits into a "good" piece $\propto (k^2 g^{\mu\nu} - k^\mu k^\nu)$ and a "bad" $g^{\mu\nu}$ piece that would be a gauge-violating gluon mass. The four-gluon tadpole is purely "bad," and the ghost loop (anticommuting scalars, an overall fermionic minus sign, color $f^{acd}f^{bdc} = -C(G)\delta^{ab}$) carries its own good and bad pieces. The crucial check: the bad $g^{\mu\nu}$ pieces from the three gauge/ghost diagrams sum to a term $\propto (D/2 - 1)\,\Gamma(1 - D/2) + \Gamma(2 - D/2)$, which by the identity $y\,\Gamma(y) + \Gamma(y+1) = 0$ at $y = 1 - D/2$ is **exactly zero** — but only because the ghost was included with its correct minus sign. Transversality survives. The surviving good piece integrates over the Feynman parameter to $\int_0^1 [2 + 8x(1-x)]\,dx = \tfrac{10}{3}$, so the gauge+ghost self-energy counterterm is $+\tfrac53 C(G)$, and $\mathrm{Res}[\delta_3] = \tfrac53 C(G) - \tfrac43 R_{\text{net}}$.

Assembling through the Slavnov–Taylor combination,
$$\mathrm{Res}[\,2\delta_1 - 2\delta_2 - \delta_3\,] = 2\big(-C(Q)-C(G)\big) - 2\big(-C(Q)\big) - \Big(\tfrac53 C(G) - \tfrac43 R_{\text{net}}\Big) = -\tfrac{11}{3}\,C(G) + \tfrac43\,R_{\text{net}},$$
where the quark Casimir $C(Q)$ cancels completely — as it must, since the quark's own charge cannot enter the universal running of the coupling — and the $-2C(G)$ from the vertices combines with the $-\tfrac53 C(G)$ from the self-energy to give the antiscreening $-\tfrac{11}{3}C(G)$. Therefore
$$\beta(g) = \frac{g^3}{16\pi^2}\Big[\,-\tfrac{11}{3}\,C(G) + \tfrac43\,R_{\text{net}}\,\Big] + O(g^5),$$
with $C(G)$ the adjoint Casimir ($=N$ for $SU(N)$) and $R_{\text{net}} = \sum_r T(r)$ the sum of Dynkin indices of the fermion multiplets ($= n_f\cdot\tfrac12$ for $n_f$ Dirac fermions in the $SU(N)$ fundamental). For $SU(N)$ with $n_f$ fundamental flavors,
$$\beta(g) = \frac{g^3}{16\pi^2}\Big[\,-\tfrac{11}{3}\,N + \tfrac23\,n_f\,\Big],$$
and for color $SU(3)$,
$$\beta(g) = -\frac{g^3}{16\pi^2}\,\Big(11 - \tfrac23\,n_f\Big) \equiv -\frac{g^3}{16\pi^2}\,b_0,\qquad b_0 = 11 - \tfrac23\,n_f.$$
The gauge term is **negative** — for the first time in any renormalizable theory the leading coefficient has the sign of antiscreening — while the fermion term is the positive QED screening sign and opposes it. Asymptotic freedom holds iff $b_0 > 0$, i.e.
$$R_{\text{net}} < \tfrac{11}{4}\,C(G) \quad\Longleftrightarrow\quad n_f < \tfrac{11}{2}N \quad\Longleftrightarrow\quad (SU(3))\ n_f < 16.5,$$
so up to 16 quark flavors; the physical 6 give $b_0 = 11 - 4 = 7 > 0$, comfortably free. I do not trust a single sign without the four checks: gauge-parameter independence (the individual $\delta$'s move with the gauge, but $-\tfrac{11}{3}$ survives); the transversality cancellation above; the Abelian limit $C(G)\to 0$ leaving only $+\tfrac43 R_{\text{net}}$, recovering QED screening exactly; and recomputing the same $\beta$ from the ghost–gluon, three-gluon, and four-gluon vertices via the Slavnov–Taylor identities — four independent routes to the same number. The minus sign is real, and it matches the magnetic reading: the gluon's spin-1 paramagnetism ($+4$) overwhelms its orbital diamagnetism ($-\tfrac13$), giving $\tfrac{11}{3}$, while a spin-$\tfrac12$ quark gives $\Delta\mu = -(-\tfrac13 + 1)\,q^2 = -\tfrac23\,q^2$ of screening per flavor.

Cashing it in, $\beta < 0$ makes the origin an ultraviolet-stable fixed point. With $\beta = -b\,g^3$ ($b = b_0/16\pi^2 > 0$) and $t = \tfrac12\ln(s/M^2)$, the flow $d\bar g/dt = -b\,\bar g^3$ gives $d(1/\bar g^2)/dt = 2b$, so
$$\bar g^2(t) = \frac{g^2}{1 + 2b\,g^2\,t} \;\xrightarrow{t\to\infty}\; 0 \;\sim\; \frac{1}{2bt} \sim \frac{1}{\ln s}.$$
The coupling vanishes *logarithmically* in the ultraviolet, so matrix elements of currents approach their free-field values — that is Bjorken scaling, together with calculable logarithmic violations set by $b_0$. A theory can thus be strong at the hadronic scale (where $1 + 2bg^2t$ is order one) and make its constituents look free at short distance. Toward the infrared ($t < 0$) the denominator passes through zero and $\bar g^2$ blows up: perturbation theory loses control, consistent with confinement, which lies outside the one-loop result's validity. The honest statement is only about the ultraviolet, and there the coupling is provably weak.

Here is the calculation distilled to the symbolic assembly and the running coupling, the same arithmetic done by hand above.

```python
import numpy as np
from fractions import Fraction as F

def residues(C_G, C_Q, R_net):
    """One-loop counterterm residues (coeff of g^2/16pi^2 eps), covariant gauge + FP ghosts."""
    res_d2 = -C_Q                               # quark self-energy
    res_d1 = -(C_Q + C_G)                       # quark-gluon vertex
    res_d3 = F(5, 3)*C_G - F(4, 3)*R_net        # gluon vacuum polarization (5 diagrams)
    return res_d1, res_d2, res_d3

def one_loop_beta_coefficient(C_G, C_Q, R_net):
    """beta(g) = (g^3/16pi^2) * combo ;  combo = -(11/3)C_G + (4/3)R_net  (C_Q cancels)."""
    d1, d2, d3 = residues(C_G, C_Q, R_net)
    return 2*d1 - 2*d2 - d3

def su_n_data(N, n_f):
    C_G   = F(N)                                # adjoint Casimir
    C_Q   = F(N*N - 1, 2*N)                     # fundamental Casimir (cancels in beta)
    R_net = F(n_f, 2)                           # n_f Dirac fundamentals, T(fund)=1/2
    return C_G, C_Q, R_net

def run_coupling_sq(g0_sq, b0, t):
    """g^2(t) = g0^2 / (1 + 2 b g0^2 t),  b = b0/16pi^2,  t = (1/2) ln(s/M^2)."""
    b = b0 / (16 * np.pi**2)
    return g0_sq / (1.0 + 2.0 * b * g0_sq * t)

if __name__ == "__main__":
    N, n_f = 3, 6
    C_G, C_Q, R_net = su_n_data(N, n_f)
    combo = one_loop_beta_coefficient(C_G, C_Q, R_net)   # -7
    b0 = -combo                                          # 7  (beta = -(g^3/16pi^2) b0)
    print("combo = -(11/3)C_G + (4/3)R_net =", combo)
    print("b0 = 11 - 2 n_f/3 =", F(11) - F(2,3)*n_f, "->", b0)
    print("asymptotically free:", b0 > 0, " (need n_f < 16.5)")
    for t in [0.0, 5.0, 50.0, 500.0]:
        print(f"t={t:6.1f}  g^2(t)={run_coupling_sq(1.0, float(b0), t):.4f}")
```
