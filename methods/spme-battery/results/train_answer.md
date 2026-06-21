I want a lithium-ion cell model I can actually put to work — inside a battery-management system estimating state-of-charge in real time, inside a parameter-estimation loop, inside an optimizer that calls it thousands of times, inside a pack solver advancing hundreds of thermally coupled cells. The model physicists trust is the Doyle–Fuller–Newman porous-electrode / concentrated-solution description: volume-average the porous microstructure into a continuum, put a representative spherical active-material particle at every macroscopic point $x$ across the cell, let lithium diffuse radially inside each particle, let lithium ions diffuse and migrate through the electrolyte, conserve charge in both the solid and the electrolyte, and couple the two phases through a Butler–Volmer reaction at every particle surface. It is faithful, and it is structurally expensive. The two charge-conservation equations carry no time derivative — they are elliptic constraints — while the two diffusion equations are parabolic, so after discretization the whole thing is a stiff index-1 differential-algebraic system. A particle diffusion problem at every $x$ means, with 30 $x$-points per electrode and 15 $r$-points per particle, $2\times30\times15=900$ particle states alone, on the order of $1100$ states total to store and advance every step. Worse, the DAE form needs consistent initial conditions found by a Newton solve, and when the current steps — switching charge to discharge, exactly the regime a fast-charge controller lives in — the algebraic part can fail to converge. Too big and too brittle for the job.

The cheap escape the control community already uses is the single-particle model: represent each electrode by one particle, assume the reaction current is spread uniformly, assume the electrolyte is a constant — uniform concentration, flat potential. A handful of non-stiff states, robust, but with no electrolyte at all, and it is well documented to drift from the full model by tens of millivolts (exceeding $50\,\mathrm{mV}$) once the current passes roughly 1C–2C, because at high current the electrolyte develops real concentration gradients that shift the ionic conductivity, the overpotentials and the reaction distribution. The obvious repair, which many have made, is to bolt the electrolyte back on: add an electrolyte diffusion PDE and graft a pointwise concentration overpotential and a pointwise Ohmic loss onto the single-particle voltage. What bothers me about every such extended model is that the choices — which terms survive, whether to freeze the nonlinear $D_e(c_e)$, whether to keep the $c_e^{1/2}$ factor in the exchange-current density — are made by hand, sometimes six assumptions deep, and can only be checked *after* running the full model. You cannot tell from the cell parameters and the C-rate alone whether the model will be accurate. And there is a subtler defect: writing a *pointwise* voltage and then substituting an electrode-*averaged* particle concentration and reaction current silently asserts that the local reaction current equals its electrode average — that every particle behaves identically. That is precisely the single-particle assumption, true only to leading order, so the correction is built on an inconsistency and the models over- or under-correct in different regimes.

So I will not bolt anything on. I propose the SPMe — the Single Particle Model with electrolyte — obtained not by assembly but by *systematically* expanding the full Doyle–Fuller–Newman model, letting a distinguished-limit asymptotic expansion decide which terms survive and at what order, so that "which terms to keep" stops being a judgement call and the modelling error becomes something I can read off the parameters before running anything. The starting move is to nondimensionalize — scale $x$ by the cell thickness $L$, time by the discharge timescale $\tau_d = F c_{n,\max} L / I_{\text{typ}}$, the radial coordinate by $R_k$, solid concentration by $c_{s,\max}$, electrolyte concentration by a typical $c_{e,\text{typ}}$, potentials by the thermal voltage $RT/F$, currents by $I_{\text{typ}}$ — so that "small" finally means something. The dimensionless groups that fall out tell the whole story through their sizes: $C_k=\tau_k/\tau_d$ and $C_{r,k}=\tau_{r,k}/\tau_d$ are order one at the C-rates of interest; the conductivity ratios $\sigma_k\approx 476/C$ and $\hat\kappa_e\approx 5.2/C$ are *large*; and the ratio of the electrolyte-transport timescale to the discharge timescale, $C_e=\tau_e/\tau_d\approx 4.19\times10^{-3}\,(\text{C-rate})$, is *tiny*. That is my small parameter, and physically $C_e\to0$ says lithium ions cross the cell far faster than the cell discharges, so the electrolyte never strays far from equilibrium — exactly the regime where the single-particle model is meant to be good.

The delicate part is what to do with the large conductivities, because how they enter decides whether the Ohmic losses appear at leading order, at first order, or never. The tell is that their *products* with the small parameter are order one: $\sigma_k C_e\approx(476/C)(4.19\times10^{-3}C)\approx2$ and $\hat\kappa_e C_e\approx(5.2/C)(4.19\times10^{-3}C)\approx0.02$. The distinguished limit — the scaling that keeps the maximum physics alive at the same order — is therefore to hold $\sigma_k C_e$ and $\hat\kappa_e C_e$ fixed as $C_e\to0$, i.e. write $\sigma_k=\sigma_k'/C_e$ and $\hat\kappa_e=\hat\kappa_e'/C_e$ with the primed quantities $O(1)$. Letting the conductivities run to infinity independently of $C_e$ would drop the Ohmic losses entirely; this co-scaling makes the electrolyte concentration correction and the Ohmic drops appear *together* at first order, and that co-appearance matters.

Before expanding anything, I rewrite the terminal voltage, and this is the whole game. Rather than $V=\phi_{s,p}|_{x=1}-\phi_{s,n}|_{x=0}$, follow a current path: in at $x=0$, through the negative solid to a point $x_n$, across into the electrolyte through a reaction, across the electrolyte to a point $x_p$ in the positive electrode, back into the positive solid through another reaction, out at $x=1$. Summing the drops and using $\eta_k=\phi_s-\phi_e-U_k$ gives $V$ as a sum of named pieces evaluated at $(x_n,x_p)$ — a pointwise OCV $U_p(c_{s,p}|_{r=1})|_{x_p}-U_n(c_{s,n}|_{r=1})|_{x_n}$, the two reaction overpotentials, the electrolyte potential difference, and the solid Ohmic drops. Since $V$ is just the difference of the two endpoint solid potentials, *every* path gives the same $V$; therefore $V$ equals the average of the path expression over all paths. Averaging over $x_n\in[0,L_n]$ and $x_p\in[1-L_p,1]$ turns every piece into an electrode-*averaged* quantity. I want the averaged form precisely because, after reduction, I will only know the electrode-averaged current; the pointwise form would force me to assume the local current equals the average, which is false beyond leading order. The averaged problem — given the averaged current, find the averaged potential differences — is well posed with no assumption smuggled in. This is the step the ad-hoc models skip, and it is exactly where their first-order error enters. I also note one conservation relation for later: integrating electrolyte molar conservation over the whole cell with the no-flux ends and interface continuity telescopes to $\int c_{e,n}\,dx+\int c_{e,s}\,dx+\int c_{e,p}\,dx=1$ — the electrolyte is a closed reservoir, ions only slosh from one side to the other.

Now expand every variable as $f\sim f^0+C_e f^1+C_e^2 f^2+\dots$. At leading order the dimensionless molar conservation $C_e\varepsilon_k\gamma_e\,\partial_t c_e=-\gamma_e\,\partial_x N_e+C_e\,\partial_x i_e$ loses its time-derivative and migration terms (both carry an explicit $C_e$), leaving $\partial_x N_e^0=0$ with $N_e^0=-\varepsilon^b D_e(c_e^0)\partial_x c_e^0$; the no-flux ends and interface continuity force $N_e^0=0$ everywhere, so $\partial_x c_e^0=0$, and with the closed-reservoir integral $c_e^0=1$ — no electrolyte depletion at leading order. The MacInnes equation $i_e=\varepsilon^b\hat\kappa_e\kappa_e(c_e)\big(-\partial_x\phi_e+2(1-t^+)\partial_x\log c_e\big)$ with the *large* $\hat\kappa_e=\hat\kappa_e'/C_e$ forces the bracket to be $O(C_e)$, so $\partial_x\phi_e^0=0$; the same logic on solid Ohm's law $I-i_e=-\sigma_k\partial_x\phi_s$ with $\sigma_k=\sigma_k'/C_e$ gives $\partial_x\phi_s^0=0$. Flat potentials are now a *consequence* of high conductivity, not a postulate. With $c_e^0$, $\phi_e^0$, $\phi_s^0$ all $x$-independent, Butler–Volmer makes the reaction uniform; integrating $\partial_x i_e^0=j_k^0$ across each electrode with $i_e=0$ at the outer edge and $i_e=I$ at the inner edge gives $i_{e,n}^0=xI/L_n$, $j_n^0=I/L_n$ and $i_{e,p}^0=(1-x)I/L_p$, $j_p^0=-I/L_p$. Inverting Butler–Volmer, $\eta_n^0=2\sinh^{-1}(I/(j_{0,n}^0 L_n))$ and $\eta_p^0=-2\sinh^{-1}(I/(j_{0,p}^0 L_p))$. The averaged voltage at leading order is

$$V^0 = U_p(c_{s,p}^0) - U_n(c_{s,n}^0) - 2\sinh^{-1}\!\Big(\tfrac{I}{j_{0,p}^0 L_p}\Big) - 2\sinh^{-1}\!\Big(\tfrac{I}{j_{0,n}^0 L_n}\Big),$$

with $j_{0,k}^0=(\gamma_k/C_{r,k})(c_{s,k}^0)^{1/2}(1-c_{s,k}^0)^{1/2}$ and $c_{s,k}^0$ solving one spherical diffusion equation per electrode, $C_k\,\partial_t c_{s,k}^0=(1/r^2)\partial_r(r^2\partial_r c_{s,k}^0)$ with surface flux $\pm I/L_k$. That is the single-particle model exactly — confirmation that $C_e$ is the right small parameter, and a derivation of what assumptions the cheap model secretly rests on.

The corrections appear at $O(C_e)$. Molar conservation now reads $\partial_x N_e^1=(1/\gamma_e)\partial_x i_e^0$ (the $c_e^0=1$ constant kills the time term), and the known leading current integrates to a steady linear flux $N_{e,n}^1=Ix/(\gamma_e L_n)$, $N_{e,s}^1=I/\gamma_e$, $N_{e,p}^1=I(1-x)/(\gamma_e L_p)$ — ions injected uniformly on one side, carried flat across the separator, absorbed on the other. The flux law at this order, $N_{e,k}^1=-\varepsilon^b D_e(1)\partial_x c_{e,k}^1+(t^+/\gamma_e)i_{e,k}^0$, has $D_e$ frozen at $D_e(1)$, so the first-order electrolyte equation is *linear*; rearranging gives a right-hand side linear in $x$, hence $c_{e,k}^1$ quadratic in $x$ in each region, with the three integration constants fixed by concentration continuity at the interfaces and the $O(C_e)$ closed-reservoir condition. The averages that the voltage needs are $\bar c_{e,n}^1=\frac{(1-t^+)I}{6\gamma_e D_e(1)}\big(2(L_p^2/\varepsilon_p^b-L_n^2/\varepsilon_n^b)+2L_n/\varepsilon_n^b+(3L_s/\varepsilon_s^b)(L_p-L_n+1)\big)$ and $\bar c_{e,p}^1=\frac{(1-t^+)I}{6\gamma_e D_e(1)}\big(2(L_p^2/\varepsilon_p^b-L_n^2/\varepsilon_n^b)-2L_p/\varepsilon_p^b+(3L_s/\varepsilon_s^b)(L_p-L_n-1)\big)$; the $(1-t^+)$ factor is the fraction of the current the concentration gradient must carry, with $t^+$ riding along as migration. The electrolyte potential at this order separates cleanly: integrating the $O(C_e)$ MacInnes balance gives a *diffusion* (concentration) part $2(1-t^+)c_{e,k}^1$ and a geometry-dependent *migration Ohmic* part — the two physically distinct contributions the ad-hoc models muddle together, here disentangled automatically. The solid potential integrates to a quadratic whose electrode average is the clean $\bar{\Delta\Phi}_{\text{Solid}}=-(I/3)(L_p/\sigma_p+L_n/\sigma_n)$, the $1/3$ being the average of the quadratic profile.

The fact that justifies the averaged-voltage rewrite, and that makes the model consistent, lives in charge conservation at $O(C_e)$. The boundary conditions give $i_{e,k}^1=0$ at *both* ends of each electrode — the leading current already carried all of $I$ — so integrating $\partial_x i_e^1=j_k^1$ telescopes to $\int_{\text{electrode}} j_k^1\,dx=0$: the electrode-averaged first-order reaction current vanishes. That cascades. The averaged $O(C_e)$ particle problem then has zero surface flux and zero initial data, so $\bar c_{s,n}^1=\bar c_{s,p}^1=0$, and the first-order averaged OCV correction $\bar U_{\text{eq}}^1=U_p'(c_{s,p}^0)\bar c_{s,p}^1-U_n'(c_{s,n}^0)\bar c_{s,n}^1=0$. There is no first-order OCV correction. This is exactly why the averaged formulation is well posed where the pointwise one is not: I never needed the pointwise $j_k^1(x)$, only its electrode integral, which is zero — whereas the ad-hoc models implicitly assume $j_k^1(x)=0$ *pointwise*, which is false; only its average vanishes. That averaging buys consistency to $O(C_e^2)$.

So the first-order voltage correction is purely electrolyte and Ohmic, and purely algebraic. Linearizing Butler–Volmer and averaging — with $\int j_k^1=0$ and $\bar c_{s,k}^1=0$ killing the solid-potential and OCP-derivative pieces — leaves the overpotential correction carried only by the electrolyte through the exchange-current shift, $\bar\eta_k^1=-\bar c_{e,k}^1\tanh(\eta_k^0/2)$, which evaluates to the reaction contribution $\bar c_{e,p}^1 I/\sqrt{(j_{0,p}^0 L_p)^2+I^2}+\bar c_{e,n}^1 I/\sqrt{(j_{0,n}^0 L_n)^2+I^2}$ in the terminal voltage. The concentration overpotential is the diffusion part of the averaged electrolyte potential difference, $\bar\eta_c=2C_e(1-t^+)(\bar c_{e,p}^1-\bar c_{e,n}^1)$, and the migration parts assemble into $\bar{\Delta\Phi}_{\text{Elec}}=-\big(I/(\hat\kappa_e\kappa_e(1))\big)\big(L_n/(3\varepsilon_n^b)+L_s/\varepsilon_s^b+L_p/(3\varepsilon_p^b)\big)$ — the separator contributing its full thickness because the current is uniform there, the electrodes a third because it ramps linearly. There is a clean way to fold the reaction-overpotential correction back into single-particle form: define a dressed, electrode-averaged exchange-current density $\bar j_{0,k}=(1/L_k)\int(\gamma_k/C_{r,k})(c_{s,k}^0)^{1/2}(1-c_{s,k}^0)^{1/2}(1+C_e c_{e,k}^1)^{1/2}\,dx$; then $-2\sinh^{-1}(I/(\bar j_{0,k}L_k))$ expands to exactly the leading overpotential plus its $O(C_e)$ correction, so the electrolyte enters only through $\bar j_{0,k}$ and the structure of the single-particle voltage is preserved. The voltage accurate to $O(C_e^2)$ is then

$$V = \bar U_{\text{eq}} + \bar\eta_r + \bar\eta_c + \bar{\Delta\Phi}_{\text{Elec}} + \bar{\Delta\Phi}_{\text{Solid}},$$

with $\bar U_{\text{eq}}=U_p(c_{s,p}^0)-U_n(c_{s,n}^0)$, $\bar\eta_r=-2\sinh^{-1}(I/(\bar j_{0,p}L_p))-2\sinh^{-1}(I/(\bar j_{0,n}L_n))$, and the three correction terms above.

This steady, fully algebraic form — the SPMe(S), with the (S) for the quasi-steady electrolyte profile — is right whenever the current varies slowly against the electrolyte-diffusion timescale, but it is wrong in the window I most care about: right after a current step, when $c_{e,k}^1$ has not yet relaxed to its steady profile. The fix is to let $c_{e,k}^1$ be genuinely time-dependent by keeping the time-derivative term I dropped at leading order:

$$C_e\,\varepsilon_k\gamma_e\,\partial_t c_{e,k}^1 = -\gamma_e\,\partial_x N_{e,k}^1 + \begin{cases}+I/L_n & \text{(n)}\\ 0 & \text{(s)}\\ -I/L_p & \text{(p)}\end{cases},\qquad N_{e,k}^1=-\varepsilon_k^b D_e(1)\,\partial_x c_{e,k}^1 + \tfrac{t^+ I}{\gamma_e}\times\begin{cases}x/L_n\\ 1\\ (1-x)/L_p\end{cases}.$$

In steady state this collapses back to the algebraic $c_{e,k}^1$, so the composite is consistent with the SPMe(S); out of steady state it tracks the transient correctly across current steps. This — single-particle solid diffusion, plus a transient linear electrolyte PDE, plus the algebraic voltage — is the canonical SPMe. It is three independent linear parabolic PDE problems (negative particle, positive particle, electrolyte across $n|s|p$) that do not talk to each other within a timestep, since the electrolyte source is the *leading-order* applied current $\pm I/L_k$ rather than a coupled unknown — so the structure is naturally parallel, and being linear it discretizes to a well-conditioned ODE system rather than a stiff DAE. On a 30/20/30 ($x$) $\times$ 15 ($r$) finite-volume mesh that is roughly 110 states against the full model's $\sim$1120 — an order of magnitude less memory, larger time steps, no consistent-initialization step, and no current-step convergence failure. And because the model is *derived*, the error comes for free: it is accurate to $O(C_e^2)$ provided $C_e\ll1$, $RT\sigma_k/(FIL)\gg1$, $RT\kappa_e/(FIL)\gg1$ and the solid-diffusion and reaction timescales $\ll1/C_e$, with leading error of size $\max\big((I_{\text{typ}}L/(D_{e,\text{typ}}Fc_{n,\max}))^2,\ (I_{\text{typ}}L/D_{e,\text{typ}})^2(1/(FRT))|U_k''|\big)$. The second term carries the OCP curvature $|U_k''|$, which spikes near the end of discharge where the particles really do start behaving differently and $\bar c_{s,k}^1=0$ stops being a good story; chasing it would force resolving every particle separately, back to the full model's cost, so I leave it deliberately — the residual $O(C_e)$ particle error is the visible, parameter-readable price of keeping single-particle cheapness.

In code, the SPMe inherits the SPM's leading-order $x$-averaged particle and kinetic submodels — unchanged, because $\bar c_{s,k}^1=0$ means resolving more particles buys nothing at first order — and refills exactly three slots with the derived first-order physics: the solid potential becomes the composite first-order Ohmic loss $-(I/3)(L/\sigma)$, the electrolyte concentration becomes the transient Stefan–Maxwell mass balance, and the electrolyte potential becomes the composite submodel that builds $\phi_e^1$ and splits off $\bar\eta_c$ and $\bar{\Delta\Phi}_{\text{Elec}}$. The terminal voltage is assembled in the base model as the averaged sum I wrote at the start.

```python
import pybamm
from .spm import SPM


class SPMe(SPM):
    def __init__(self, options=None, name="Single Particle Model with electrolyte",
                 build=True):
        self.x_average = True
        super().__init__(options, name, build)

    def set_solid_submodel(self):
        for domain in ["negative", "positive"]:
            if self.options.electrode_types[domain] == "porous":
                solid_submodel = pybamm.electrode.ohm.Composite
            elif self.options.electrode_types[domain] == "planar":
                if self.options["surface form"] == "false":
                    solid_submodel = pybamm.electrode.ohm.LithiumMetalExplicit
                else:
                    solid_submodel = pybamm.electrode.ohm.LithiumMetalSurfaceForm
            self.submodels[f"{domain} electrode potential"] = solid_submodel(
                self.param, domain, self.options
            )

    def set_electrolyte_concentration_submodel(self):
        self.submodels["electrolyte diffusion"] = pybamm.electrolyte_diffusion.Full(
            self.param, self.options
        )

    def set_electrolyte_potential_submodel(self):
        surf_form = pybamm.electrolyte_conductivity.surface_potential_form

        if (
            self.options["surface form"] == "false"
            or self.options.electrode_types["negative"] == "planar"
        ):
            if self.options["electrolyte conductivity"] in ["default", "composite"]:
                self.submodels["electrolyte conductivity"] = (
                    pybamm.electrolyte_conductivity.Composite(
                        self.param, options=self.options
                    )
                )
            elif self.options["electrolyte conductivity"] == "integrated":
                self.submodels["electrolyte conductivity"] = (
                    pybamm.electrolyte_conductivity.Integrated(
                        self.param, options=self.options
                    )
                )

        if self.options["surface form"] == "false":
            surf_model = surf_form.Explicit
        elif self.options["surface form"] == "differential":
            surf_model = surf_form.CompositeDifferential
        elif self.options["surface form"] == "algebraic":
            surf_model = surf_form.CompositeAlgebraic

        for domain in ["negative", "positive"]:
            if self.options.electrode_types[domain] == "porous":
                self.submodels[f"{domain} surface potential difference [V]"] = (
                    surf_model(self.param, domain, self.options)
                )
```

The composite electrolyte-potential submodel builds the first-order electrolyte potential and reads off the two averaged voltage contributions, which are exactly $\bar\eta_c$ and $\bar{\Delta\Phi}_{\text{Elec}}$:

```python
# diffusion-potential part of phi_e, averaged into the concentration overpotential
macinnes_c_e_p = pybamm.x_average(
    self._higher_order_macinnes_function(c_e_p / c_e_av)
)
macinnes_c_e_n = pybamm.x_average(
    self._higher_order_macinnes_function(c_e_n / c_e_av)
)
eta_c_av = chi_av * RT_F_av * (macinnes_c_e_p - macinnes_c_e_n)

# migration Ohmic part of phi_e, averaged across the three regions
delta_phi_e_av = -i_boundary_cc * (
    L_n / (3 * kappa_n_av) + L_s / kappa_s_av + L_p / (3 * kappa_p_av)
)
```

The Stefan–Maxwell electrolyte mass-transport submodel (`Full`) implements $N_e=-\text{tor}\cdot D_e(c_e,T)\,\nabla c_e+t^+ i_e/F$ plus the framework's convection term, and $\partial(\varepsilon c_e)/\partial t=-\nabla\cdot N_e+\text{source}/F$. The particle submodel is the $x$-averaged Fickian diffusion inherited from the SPM. Usage:

```python
model = pybamm.lithium_ion.SPMe()
sim = pybamm.Simulation(model)
sim.solve([0, 3600])   # 1C constant-current discharge
```
