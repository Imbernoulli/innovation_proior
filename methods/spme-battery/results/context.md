# Context: a fast, accurate physics-based lithium-ion cell model

## Research question

Mathematical models of lithium-ion cells are needed for cell design, for battery-management
systems (BMS) that estimate state-of-charge and state-of-health in real time, for parameter
estimation, for optimal-charging and optimization loops, and for pack-scale simulations where
many cells (and thermal coupling) must be advanced together. The standard physics-based model
captures the chemistry faithfully but is far too expensive for these jobs: it is a stiff,
strongly coupled system of partial differential equations that takes many seconds to integrate
a single discharge and that can fail to converge when the current is stepped. The cheap
alternative used in the control community is fast but only trusted at low currents, where it
agrees with the full model; at fast charge/discharge it visibly disagrees.

The goal is a reduced-order, physics-based cell model that (i) is dramatically cheaper than the
full model — few states, non-stiff, robust to current steps — yet (ii) stays accurate up to
higher C-rates by accounting for the electrolyte, and crucially (iii) is derived *systematically*,
so the terms kept and dropped are justified and the modelling error can be estimated from the
input parameters *before* running any comparison, rather than chosen by hand and validated only
after the fact.

## Background

**The cell and the physics.** A lithium-ion cell is a sandwich: negative electrode | porous
separator | positive electrode, flooded with a liquid electrolyte, between two current
collectors. Each electrode is a porous matrix of active-material particles (spheres of radius
`R_k`) in which lithium is stored. On discharge, lithium in a negative particle diffuses to the
particle surface, an electrochemical reaction strips an electron (which goes around the external
circuit) and releases a lithium ion into the electrolyte; the ion migrates/diffuses across to
the positive electrode, where the reverse reaction intercalates it into a positive particle. The
processes to model are therefore: solid diffusion inside particles, transport of lithium ions in
the electrolyte, charge transport in the solid and in the electrolyte, and the interfacial
reaction.

**Porous-electrode theory + concentrated-solution theory (Newman).** The standard physics-based
description, developed by Newman and collaborators (Doyle, Fuller, Newman, 1993–94;
*Electrochemical Systems*), volume-averages the porous microstructure into a continuum. Each
macroscopic point `x` across the cell carries a representative spherical particle with its own
radial diffusion problem (hence "pseudo-two-dimensional", P2D: one macroscopic `x` plus a
"pseudo" radial `r`). The electrolyte is described by concentrated-solution (Stefan–Maxwell)
theory, giving coupled transport of the lithium-ion concentration `c_e` and the electrolyte
potential `φ_e`. The full dimensional system is:

- Charge conservation in the electrolyte (MacInnes equation): the electrolyte current is driven
  by `−∂φ_e/∂x` plus a diffusional `2(1−t⁺)(RT/F)∂log(c_e)/∂x` term, with `∂i_e/∂x = a_k j_k` in
  the electrodes (reaction source) and `0` in the separator.
- Charge conservation in the solid (Ohm's law): `I − i_e = −σ_k ∂φ_s/∂x`.
- Molar conservation in the electrolyte: `ε_k ∂c_e/∂t = ∂N_e/∂x + (1/F)∂i_e/∂x`, with flux
  `N_e = ε_k^b D_e(c_e) ∂c_e/∂x + (t⁺/F) i_e` (diffusion + migration; `t⁺` transference number,
  `ε^b` Bruggeman porosity correction).
- Solid diffusion in each particle: `∂c_s/∂t = (1/r²)∂_r(r² D_s ∂_r c_s)`, with a surface flux
  set by the local reaction, `−D_s ∂_r c_s|_{r=R_k} = j_k/F`.
- Butler–Volmer kinetics: `j_k = j_{0,k} sinh(Fη_k/(2RT))`, exchange-current density
  `j_{0,k} = m_k c_s^½ (c_{s,max} − c_s)^½ c_e^½`, overpotential
  `η_k = φ_s − φ_e − U_k(c_s|_{r=R_k})`, with `U_k` the open-circuit potential (OCP).
- Terminal voltage `V = φ_{s,p}|_{x=L} − φ_{s,n}|_{x=0}`.

This is the reference for accuracy. Its cost is structural: after a finite-volume (or
finite-element / orthogonal-collocation) discretisation, the mixed parabolic (diffusion) and
elliptic (charge-conservation) PDEs become a stiff index-1 system of differential-algebraic
equations (DAEs). With, e.g., 30 points in each electrode, 20 in the separator and 15 in each
particle, it carries on the order of a thousand internal states — about 900 for the particle
concentrations alone, plus electrolyte concentration, electrolyte potential and electrode
potentials — that must be stored and advanced every step, and the DAE form is prone to
convergence failure on inconsistent initial conditions or large current steps (switching
charge↔discharge). It is solved with implicit DAE solvers (e.g. SUNDIALS/IDA).

**Asymptotic methods.** Perturbation/asymptotic analysis — nondimensionalise, identify a small
parameter `δ`, expand each unknown as `f ~ f0 + δ f1 + …`, and solve order by order (Hinch,
*Perturbation Methods*; Bender & Orszag) — is standard across applied mathematics but underused
in battery modelling. It has, however, been used to *derive* the porous-electrode model itself by
asymptotic homogenisation / multiple scales (Richardson, Denuault, Please 2012) and to reduce
lead-acid (Sulzer et al. 2019) and particle-free lithium-ion models (Moyles et al. 2018). The
relevant feature for a full cell is that several physical timescales are widely separated —
discharge, solid diffusion, electrolyte diffusion, reaction — and several Ohmic drops are small
relative to the thermal voltage `RT/F`; ratios of these are the natural small/large dimensionless
groups.

**Established empirical facts about the existing reduced model.** The simplest reduced model
(below) is well documented to be accurate only at low currents: it is commonly accepted as valid
below roughly 1C–2C, where electrolyte concentration gradients are negligible, and its terminal-
voltage error grows with C-rate, reaching tens of millivolts (exceeding ~50 mV) at high C-rate.
The cause is understood: it discards electrolyte concentration and potential gradients, which
become significant at high C-rate and shift the local ionic conductivity, overpotential and
reaction distribution. These are pre-existing observations about the cheap model.

## Baselines

**The full physics-based model (Doyle–Fuller–Newman / P2D / Newman).** The complete system above.
*Core idea:* resolve everything — a particle diffusion problem at every `x`, full concentrated-
solution electrolyte transport, Butler–Volmer coupling. *Math/algorithm:* the coupled PDE/DAE
system listed under Background. *Gap it leaves:* accurate but expensive and stiff; ~10³ states;
DAE convergence issues on current steps; too slow and heavy for BMS, control, optimization and
pack simulations. This is the accuracy reference a reduced model is measured against.

**The single-particle model (SPM).** *Core idea:* in the low-current limit every particle in an
electrode behaves identically, so each electrode is represented by ONE particle with a uniform
reaction current; the electrolyte is taken spatially uniform and the solid/electrolyte potentials
flat. *Math/algorithm:* solve one spherical diffusion equation per electrode,
`C_k ∂c_s/∂t = (1/r²)∂_r(r² ∂_r c_s)` with surface flux `∝ ±I/L_k`; the terminal voltage is the
difference of the two OCPs minus two reaction overpotentials `−2 sinh⁻¹(I/(j_{0,k} L_k))`. Drops
to a handful of states and a non-stiff problem. *Gap it leaves:* it has no electrolyte. The
uniform-reaction / constant-electrolyte assumption holds only at low current; above ~1–2C the
neglected electrolyte concentration and potential gradients drive the voltage error up to tens of
millivolts, so the SPM is unreliable at fast charge.

**Ad-hoc extended SPMs (SPM + electrolyte).** A family of models (Kemper & Kum 2013; Perez/Hu/
Moura 2016; Prada et al. 2012; Han et al. 2015; Rahimian et al. 2013; Tanim et al. 2015) that
bolt electrolyte terms onto the SPM to push accuracy to higher C-rate. *Core idea:* add an
electrolyte concentration equation and extra voltage terms (a concentration overpotential and an
electrolyte Ohmic loss) on top of the SPM. *Math/algorithm (typical):* solve an electrolyte
diffusion PDE, then add to the SPM voltage a *pointwise* concentration overpotential of the form
`2(1−t⁺) log((1+C_e c_{e,p}¹|_{x=1})/(1+C_e c_{e,n}¹|_{x=0}))` and a *pointwise* electrolyte
Ohmic loss; some keep a nonlinear electrolyte diffusivity `D_e(c_e)`, some take exchange-current
densities constant.
*Gaps they leave:* the terms kept/dropped are chosen by hand (one such model needs six assumptions
that can only be checked *after* comparing to the full model, e.g. assuming the current profile
takes a prescribed shape); the modelling error cannot be bounded a-priori from the parameters; and
they mix electrode-averaged and pointwise quantities in the voltage, which implicitly assumes the
local reaction current equals its electrode average — true only at the lowest order — so a
consistent error order cannot be guaranteed and over- or under-correction is observed.

## Evaluation settings

- **Reference for accuracy:** the full Doyle–Fuller–Newman model, same parameters, same numerical
  method (finite volume), same mesh, so that only modelling error is compared.
- **Cell/parameters:** a graphite negative electrode, LiPF₆ in EC:DMC electrolyte, lithium-cobalt-
  oxide positive electrode; parameter set adapted from Newman's DUALFOIL and Moura's fastDFN.
  Thicknesses `L_n=L_p=100 μm`, `L_s=25 μm`; particle radii `10 μm`; porosities `ε_n=ε_p=0.3`;
  transference `t⁺=0.4`; Bruggeman `b=1.5`; typical electrolyte concentration `10³ mol/m³`; solid
  conductivities `σ_n=100`, `σ_p=10 S/m`; `1C ≡ 24 A/m²`. The natural small group is the ratio of
  electrolyte-transport to discharge timescale, `C_e = τ_e/τ_d ≈ 4.19×10⁻³ · (C-rate)`.
- **Protocol:** constant-current discharge over a range of C-rates (e.g. 0.1C, 0.5C, 1C, 2C, 3C),
  from given initial stoichiometries (0.8 negative, 0.6 positive) to a voltage cutoff (3.2 V).
  Spatial discretisation by finite volume (e.g. 30/20/30 in electrode/separator/electrode, 15 in
  particles); time integration by an implicit DAE/ODE solver (SUNDIALS) through a common modelling
  framework (PyBaMM).
- **Metrics:** root-mean-square terminal-voltage error relative to the full model across the
  discharge; number of internal states (memory); computation time; and a comparison of internal-
  state profiles (particle stoichiometry, electrolyte concentration and potential) against the
  full model.

## Code framework

The primitives below already exist: a battery-modelling framework (expression tree of symbolic
variables, spatial discretisation onto a 1D `x`-mesh and a particle `r`-mesh, a DAE/ODE time
integrator), parameter sets, and the open-circuit-potential and transport-property functions
`U_k(c)`, `D_e(c)`, `κ_e(c)`. A model is assembled from interchangeable *submodels*, each
contributing variables, governing equations, and boundary/initial conditions for one physical
process. The cheap base model is built from the leading-order submodels; the slots a higher-
accuracy electrolyte model would fill are left empty.

```python
# --- existing primitives (framework) ---
import pybamm  # symbolic vars, grad/div, x_average, DAE/ODE solver, parameter functions

class BaseCellModel:
    """Assembles submodels into one system and reads off the terminal voltage."""
    def __init__(self, param, options):
        self.param, self.options, self.submodels = param, options, {}
        self.set_particle_submodel()                   # solid diffusion in 1 particle / electrode
        self.set_intercalation_kinetics_submodel()     # Butler-Volmer at particle surface
        self.set_solid_submodel()                      # solid-phase potential
        self.set_electrolyte_concentration_submodel()  # electrolyte mass transport
        self.set_electrolyte_potential_submodel()      # electrolyte charge + surface potentials

    def terminal_voltage(self, v):
        # V = OCV + reaction overpotential + concentration overpotential
        #     + electrolyte Ohmic loss + solid Ohmic loss
        return (v["ocv"] + v["eta_r"]
                + v["eta_c"] + v["delta_phi_e"] + v["delta_phi_s"])

# --- the cheap leading-order base model: one averaged particle, no electrolyte gradients ---
class SingleParticleBase(BaseCellModel):
    def set_particle_submodel(self):
        for dom in ["negative", "positive"]:
            # one r-resolved spherical diffusion problem per electrode (x-averaged particle)
            self.submodels[f"{dom} particle"] = pybamm.particle.FickianDiffusion(
                self.param, dom, self.options, x_average=True)

    def set_solid_submodel(self):
        # flat solid potential; delta_phi_s = 0
        for dom in ["negative", "positive"]:
            self.submodels[f"{dom} electrode potential"] = \
                pybamm.electrode.ohm.LeadingOrder(self.param, dom, self.options)

    def set_electrolyte_concentration_submodel(self):
        # electrolyte held spatially uniform: c_e = const
        self.submodels["electrolyte diffusion"] = \
            pybamm.electrolyte_diffusion.ConstantConcentration(self.param, self.options)

    def set_electrolyte_potential_submodel(self):
        # flat electrolyte potential; eta_c = 0, delta_phi_e = 0
        self.submodels["leading-order electrolyte conductivity"] = \
            pybamm.electrolyte_conductivity.LeadingOrder(self.param, options=self.options)
        surf_model = pybamm.electrolyte_conductivity.surface_potential_form.Explicit
        for dom in ["negative", "positive"]:
            self.submodels[f"{dom} surface potential difference"] = \
                surf_model(self.param, dom, options=self.options)


# A higher-accuracy electrolyte-aware model reuses the particle + kinetics of the base
# model and refills the transport and potential terms to be derived:
class ElectrolyteAwareModel(SingleParticleBase):
    def set_solid_submodel(self):
        # TODO: solid-phase potential correction
        pass

    def set_electrolyte_concentration_submodel(self):
        # TODO: electrolyte concentration transport across n|s|p
        pass

    def set_electrolyte_potential_submodel(self):
        # TODO: electrolyte potential, electrolyte current, and surface potentials
        pass
```
