# Adjoint-method photonic inverse design

## Problem

Design a passive nanophotonic device (a silicon-on-insulator waveguide splitter, crossing, MMI, or fiber coupler) by choosing the permittivity distribution ε(r) over a fixed design footprint so as to maximize a figure of merit F — typically the optical power coupled from the input mode into a target output mode. The forward model is Maxwell's equations, whose single evaluation is a full electromagnetic simulation, and the design space is one free variable per pixel (N from thousands to millions). Heuristic global search (genetic, particle swarm) only samples F and so needs a small parameterization; a finite-difference gradient costs N+1 solves per gradient. Both scale with N and are infeasible for a high-dimensional permittivity.

## Key idea

A first-order perturbation of the permittivity at a point is an induced dipole, and its effect on F propagates through the Maxwell Green's function. **Reciprocity** of that Green's function lets source and observation be swapped, so the sensitivity ∂F/∂ε at *every* pixel is obtained from just two simulations per iteration: one **forward** solve (driven from the input, gives the field e and F) and one **adjoint** solve (the same structure driven from the readout port). The gradient is then the pointwise product of the forward and adjoint fields. For a mode-transmission objective the adjoint source is simply the **target output mode launched backward** into the device.

Manufacturability is enforced inside the optimization, not after it: the design variable is a density ρ ∈ [0,1] (not ε directly), passed through a **length-scale filter** (kills single-pixel/checkerboard features) and a **tanh projection** (drives the design toward binary silicon/oxide), with the projection sharpness β annealed up over the run (β-continuation) so the gradient never vanishes. Because F is always evaluated on the (nearly) binary design, there is no post-hoc thresholding collapse.

## Algorithm

Maxwell in FDFD: A(ε) e = b, with A = μ₀⁻¹∇×∇× − ω²ε₀ε_r and b = −iωJ.

1. Forward solve A e = b; read F = |mode-overlap coefficient|² at the output monitor.
2. Adjoint field from Aᵀ e_aj = −(∂L/∂e)ᵀ. For a reciprocal medium Aᵀ = A, so the adjoint reuses the forward operator with a different source; that source is −∂L/∂e at the port, which for a mode objective is the target mode sent in reverse.
3. Gradient (per pixel), from ∂A/∂ε_r = −ω²ε₀:
   dF/dε_r = ∂L/∂ε − 2ω²ε₀ Re[e_aj · e]  (the conjugate-pair gives the factor 2 and the Re; treat e, e* as independent — Wirtinger calculus).
4. Map ρ → filtered ρ̃ → projected ρ̄ → ε(ρ̄) = ε_air + ρ̄(ε_Si − ε_air); chain dF/dε back through projection and filter to dF/dρ.
   Projection: ρ̄ = [tanh(βη) + tanh(β(ρ̃−η))] / [tanh(βη) + tanh(β(1−η))], η = 0.5.
5. Ascend on ρ (MMA / L-BFGS / steepest ascent) with box constraint ρ ∈ [0,1]; periodically raise β. Two simulations per iteration, independent of N.

## Code

```python
import numpy as np
import nlopt
import meep as mp
import meep.adjoint as mpa
from autograd import numpy as npa
from autograd import tensor_jacobian_product

SI   = mp.Medium(index=3.45)        # silicon  (rho = 1)
SIO2 = mp.Medium(index=1.44)        # oxide    (rho = 0)
WVL  = 1.55                          # telecom wavelength (um)
FREQ = 1.0 / WVL
RESOLUTION = 30

DESIGN_SIZE = mp.Vector3(2.0, 2.0, 0)          # central designable footprint
NX = int(DESIGN_SIZE.x * RESOLUTION) + 1
NY = int(DESIGN_SIZE.y * RESOLUTION) + 1
N  = NX * NY

WG_WIDTH = 0.5; PML = 1.0; PAD = 0.6
SX = 2*PML + 2*PAD + DESIGN_SIZE.x
SY = 2*PML + 2*PAD + DESIGN_SIZE.y
CELL = mp.Vector3(SX, SY, 0)

FILTER_RADIUS = 0.10                 # minimum feature size = length scale
ETA = 0.5                            # projection threshold

# density rho -> filtered (length scale) -> tanh-projected (binary) weights
def filter_and_project(rho, beta):
    rho = rho.reshape(NX, NY)
    rho_tilde = mpa.conic_filter(rho, FILTER_RADIUS,
                                 DESIGN_SIZE.x, DESIGN_SIZE.y, RESOLUTION)
    if beta == 0:
        return rho_tilde.flatten()
    rho_bar = mpa.tanh_projection(rho_tilde, beta, ETA)        # smoothed Heaviside
    return rho_bar.flatten()

# forward problem + adjoint wrapper: eps(rho) interpolated on a material grid
def make_opt():
    matgrid = mp.MaterialGrid(mp.Vector3(NX, NY, 0), SIO2, SI,
                              weights=np.ones((NX, NY)), do_averaging=False)
    design_region = mpa.DesignRegion(
        matgrid, volume=mp.Volume(center=mp.Vector3(), size=DESIGN_SIZE))

    geometry = [
        mp.Block(center=mp.Vector3(), size=mp.Vector3(mp.inf, WG_WIDTH, mp.inf),
                 material=SI),                                   # access waveguide
        mp.Block(center=design_region.center, size=design_region.size,
                 material=matgrid),                             # the design region
    ]
    src_x = -0.5*SX + PML + 0.3
    mon_x = +0.5*SX - PML - 0.3
    sources = [mp.EigenModeSource(                              # launch input mode
        src=mp.GaussianSource(FREQ, fwidth=0.1*FREQ),
        size=mp.Vector3(0, SY, 0), center=mp.Vector3(src_x, 0, 0),
        eig_band=1, eig_parity=mp.ODD_Z + mp.EVEN_Y)]

    sim = mp.Simulation(resolution=RESOLUTION, cell_size=CELL,
                        boundary_layers=[mp.PML(PML)], geometry=geometry,
                        sources=sources, default_material=SIO2)

    # F = power into the target output mode = |mode-overlap coefficient|^2
    out = mpa.EigenmodeCoefficient(
        sim, mp.Volume(center=mp.Vector3(mon_x, 0, 0), size=mp.Vector3(0, SY, 0)),
        mode=1, eig_parity=mp.ODD_Z + mp.EVEN_Y)

    def J(out):                       # objective L(e): transmission into mode
        return npa.abs(out) ** 2

    # one forward + one adjoint solve per call -> (F, dF/d(eps-weights));
    # the adjoint source is the output mode launched backward, and the
    # gradient is the pointwise forward * adjoint field product
    return mpa.OptimizationProblem(simulation=sim, objective_functions=[J],
                                   objective_arguments=[out],
                                   design_regions=[design_region],
                                   frequencies=[FREQ])

def main():
    opt = make_opt()
    rho = 0.5 * np.ones(N)                 # uniform gray slab: no topology bias
    for beta in [8, 16, 32, 64]:           # beta-continuation: anneal toward binary
        def objective(x, gradient):
            f0, dF_dw = opt([filter_and_project(x, beta)])   # 2 solves, any N
            f0 = float(f0[0]); dF_dw = dF_dw.reshape(-1)
            if gradient.size > 0:          # chain dF/deps back through proj+filter
                gradient[:] = tensor_jacobian_product(
                    filter_and_project, 0)(x, beta, dF_dw)
            return f0
        solver = nlopt.opt(nlopt.LD_MMA, N)        # bound-constrained ascent (MMA)
        solver.set_lower_bounds(0.0); solver.set_upper_bounds(1.0)
        solver.set_max_objective(objective)        # MAXIMIZE transmission
        solver.set_maxeval(25)
        rho = solver.optimize(rho)
    np.save("optimized_design.npy",
            filter_and_project(rho, 64).reshape(NX, NY))

if __name__ == "__main__":
    main()
```
