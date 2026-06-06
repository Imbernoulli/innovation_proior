# SPMe — the Single Particle Model with electrolyte

## Problem

The Doyle–Fuller–Newman (DFN / pseudo-2D) porous-electrode model is the faithful physics-based
model of a lithium-ion cell, but it is a stiff, strongly coupled micro/macro PDE system: a radial
solid-diffusion problem at every macroscopic point `x`, concentrated-solution electrolyte transport,
solid and electrolyte charge conservation, all coupled by Butler–Volmer. Discretized it becomes an
index-1 DAE with ~10³ states that is too slow and too brittle (current-step convergence failure) for
BMS, parameter estimation, optimization and pack simulation. The classic single-particle model (SPM)
is cheap but drops the electrolyte entirely and is only accurate below ~1–2C. The SPMe is the
reduced-order model obtained by *systematically* expanding the DFN model, so the kept/dropped terms
are justified and the error is bounded a-priori from the parameters.

## Key idea

Nondimensionalize the DFN model and identify the small parameter `C_e = τ_e/τ_d` (electrolyte-
transport timescale over discharge timescale, ≈ 4.19×10⁻³ × C-rate). Take the distinguished limit
`C_e → 0` with the large dimensionless conductivities held as `σ_k = σ_k'/C_e` and `κ̂_e = κ̂_e'/C_e`
(primes O(1)), so the Ohmic losses appear at first order. Two moves make the reduction work:

1. **Electrode-averaged voltage.** Write the terminal voltage along an arbitrary current path through
   the cell, then average over all paths (every path gives the same `V`). The voltage becomes a sum of
   electrode-averaged pieces — OCV, reaction overpotentials, electrolyte potential difference, solid
   Ohmic loss. This is well posed given only the electrode-averaged current; the pointwise form used by
   ad-hoc models implicitly (and falsely, beyond leading order) assumes the local reaction current
   equals its electrode average.

2. **Expand and solve order by order.** *Leading order* recovers the SPM exactly: electrolyte uniform
   (`c_e⁰ = 1`), potentials flat, reaction uniform (`j_k⁰ = ±I/L_k`), voltage = OCV difference minus two
   reaction overpotentials. *First order* gives the electrolyte correction. Crucially the `O(C_e)`
   electrolyte-current correction has `i_e¹ = 0` at both electrode ends, so `∫_electrode j_k¹ dx = 0`:
   the averaged reaction-current correction vanishes, hence `c̄_{s,k}¹ = 0` and `Ū_eq¹ = 0` — no
   first-order OCV correction. The first-order correction is therefore purely electrolyte + Ohmic and
   purely algebraic.

The first-order electrolyte concentration `c_{e,k}¹` solves a *linear* PDE (with `D_e` frozen at
`D_e(1)`). Keeping it transient (a composite valid on both the discharge and the electrolyte-diffusion
timescales) makes the model correct across current steps — this is the canonical **SPMe**; dropping the
time derivative gives the steady, fully-algebraic **SPMe(S)**.

## Final model (dimensionless canonical SPMe)

Two independent particle PDEs + one electrolyte PDE, all linear, plus an algebraic voltage.

Solid (one x-averaged particle per electrode):

    C_k ∂c_{s,k}⁰/∂t = (1/r²) ∂_r(r² ∂_r c_{s,k}⁰),
    ∂_r c_{s,k}⁰|_{r=0} = 0,   −(a_k γ_k/C_k) ∂_r c_{s,k}⁰|_{r=1} = +I/L_n (n), −I/L_p (p)

Electrolyte (transient, linear, across n|s|p):

    C_e ε_k γ_e ∂c_{e,k}¹/∂t = −γ_e ∂N_{e,k}¹/∂x + { +I/L_n (n), 0 (s), −I/L_p (p) }
    N_{e,k}¹ = −ε_k^b D_e(1) ∂_x c_{e,k}¹ + (t⁺ I/γ_e) × { x/L_n (n), 1 (s), (1−x)/L_p (p) }
    N_e¹ = 0 at x = 0, 1;  c_e¹, N_e¹ continuous at interfaces;  c_{e,k}¹(x,0) = 0

Terminal voltage (electrode-averaged, accurate to O(C_e²)):

    V = Ū_eq + η̄_r + η̄_c + ΔΦ̄_Elec + ΔΦ̄_Solid
    Ū_eq      = U_p(c_{s,p}⁰|_{r=1}) − U_n(c_{s,n}⁰|_{r=1})
    η̄_r       = −2 sinh⁻¹(I/(j̄_{0,p} L_p)) − 2 sinh⁻¹(I/(j̄_{0,n} L_n))
    η̄_c       = 2 C_e (1−t⁺) (c̄_{e,p}¹ − c̄_{e,n}¹)
    ΔΦ̄_Elec  = −(I/(κ̂_e κ_e(1))) (L_n/(3ε_n^b) + L_s/ε_s^b + L_p/(3ε_p^b))
    ΔΦ̄_Solid = −(I/3) (L_p/σ_p + L_n/σ_n)
    j̄_{0,k}  = (1/L_k) ∫ (γ_k/C_{r,k}) (c_{s,k}⁰)^½ (1−c_{s,k}⁰)^½ (1 + C_e c_{e,k}¹)^½ dx

The dressed exchange-current density `j̄_{0,k}` folds the first-order reaction-overpotential contribution to the terminal voltage,
`c̄_{e,k}¹ I/√((j_{0,k}⁰ L_k)² + I²)`, into the leading-order `sinh⁻¹` form.

## Dimensional summary

    ∂c_{s,k}*/∂t* = (1/r*²) ∂_{r*}(r*² D_{s,k}*(c_{s,k}*) ∂_{r*} c_{s,k}*)
    ε_k ∂c_{e,k}*/∂t* = −∂_{x*} N_{e,k}* + { I*/(F* L_n*) (n), 0 (s), −I*/(F* L_p*) (p) }
    N_{e,k}* = −ε_k^b D_e*(c_{e,typ}*) ∂_{x*} c_{e,k}* + (t⁺ I*/F*) × { x*/L_n* (n), 1 (s), (L*−x*)/L_p* (p) }
    V* = Ū_eq* + η̄_r* + η̄_c* + ΔΦ̄_Elec* + ΔΦ̄_Solid*
    η̄_c* = (2 R*T*/(F* c_{e,typ}*)) (1−t⁺)(c̄_{e,p}* − c̄_{e,n}*)
    ΔΦ̄_Elec* = −(I*/κ_e*(c_{e,typ}*)) (L_n*/(3ε_n^b) + L_s*/ε_s^b + L_p*/(3ε_p^b))
    ΔΦ̄_Solid* = −(I*/3)(L_p*/σ_p* + L_n*/σ_n*)

Validity conditions: `C_e ≪ 1`, `RT σ_k/(F I L) ≫ 1`, `RT κ_e/(F I L) ≫ 1`, solid-diffusion and
reaction timescales `≪ 1/C_e`. The leading model error is of size
`max((I_typ L/(D_e,typ F c_{n,max}))², (I_typ L/D_e,typ)² (1/(F R T)) |U_k''|)`; the second term
(OCP curvature `|U_k''|`) grows where the OCP is strongly nonlinear (deep discharge), which is the main
source of residual error at high C-rate.

## Cost

Three independent linear parabolic PDEs (naturally parallel) + an algebraic voltage. After a 30/20/30
(x) × 15 (r) finite-volume discretization: ≈ 110 states vs ≈ 1120 for the DFN model (~10% of the
memory), and a well-conditioned ODE system rather than a stiff DAE — larger time steps, no consistent-
initialization step, and no current-step convergence failure.

## Code (PyBaMM)

The SPMe inherits the SPM leading-order particle and kinetic submodels, then overrides the solid potential, electrolyte concentration, and electrolyte-potential pieces.

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

The composite electrolyte-potential submodel builds the first-order electrolyte potential and reads off
the two averaged voltage contributions, which are exactly `η̄_c` and `ΔΦ̄_Elec`:

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

The Stefan–Maxwell electrolyte mass-transport submodel (`Full`) implements `N_e = −tor·D_e(c_e,T)·grad c_e + t⁺ i_e/F` plus the framework's convection term, and `∂(ε c_e)/∂t = −div N_e + source/F`. The particle submodel is the x-averaged Fickian diffusion inherited from the SPM. Usage:

```python
model = pybamm.lithium_ion.SPMe()
sim = pybamm.Simulation(model)
sim.solve([0, 3600])   # 1C constant-current discharge
```
