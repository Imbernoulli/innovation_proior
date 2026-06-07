I want to design a passive silicon-photonics component — say a Y-junction that splits a single waveguide into two — by choosing the silicon pattern inside a small footprint of chip. The input mode comes in, the design region reshapes it, and I read out how much power lands in the fundamental mode of the output waveguides. The straight waveguides are essentially lossless; all the trouble is in this little functional region, where evanescent fields leak outside the core and the abrupt silicon/oxide interfaces reflect, and that scattering loss eats the whole link budget. So the object I really want to choose is the permittivity ε(r) over the design region, and the thing I want to maximize is a scalar figure of merit F — the power coupled into the target output mode.

Let me be honest about the size of this. The forward model is Maxwell's equations. Time-harmonic, non-magnetic, frequency ω: ∇×∇×E − (ω²/c²) ε_r(r) E = iωμ₀ J. Discretize space — a Yee grid in finite-difference frequency domain — and this becomes a big sparse linear system A(ε) e = b, where A holds the curl-curl operator and the permittivity on its diagonal, e is the stacked field, b is proportional to the source current. One solve of A e = b gives me the field everywhere, and from the field on a monitor plane I compute F. That single solve is the expensive primitive — a full electromagnetic simulation. And the design space is the permittivity at every pixel of the region: N free variables, N anywhere from thousands to millions.

The way everyone designs these things now is heuristic global optimization — genetic algorithms, particle swarm. Pick a handful of shape parameters, simulate a whole population of random parameter sets, keep the good ones, breed, repeat. The silicon Y-splitter record was set this way with particle swarm by Zhang et al. (2013): a low-loss splitter after on the order of fifteen hundred full simulations. The thing to notice is *why* it costs that much. These methods never look inside the physics — they only ever evaluate F, they treat the simulator as an oracle that returns one number. With no gradient, the only way to explore is to sample, and to keep the sampling affordable you must keep the parameterization small, a few smooth knobs. That's fine for a clean geometry with a few parameters. But the whole promise of topology optimization is to let *every pixel* be free, and a sample-only method drowns the moment the geometry gets rich — the number of simulations explodes with the number of parameters.

So I want a gradient. If I had ∂F/∂ε at every pixel, I could just ascend: nudge the whole pattern uphill at once, every iteration, instead of randomly poking it. The question is how to get that gradient cheaply.

The obvious way is finite differences. Perturb pixel i by a little, re-run Maxwell, see how F changed: ∂F/∂ε_i ≈ (F(ε + δ e_i) − F(ε))/δ. But that's one extra full Maxwell solve *per pixel*. N pixels means N+1 solves for a single gradient, and then I need a gradient at every step of the optimization. With N in the thousands-to-millions and each solve a full EM problem, that's not slow, it's hopeless. Finite differencing is exactly as physics-blind as the heuristics — it pokes the oracle N times. I've gained nothing.

Let me stop poking and actually look at what a perturbation *does* to the field. Strip the problem to its bones: maximize FoM = |E(x₀)|², the field intensity at one measuring point x₀, and let me change ε in a tiny volume ΔV at one point x in the design region. To first order the change in the figure of merit is ΔFoM = Re[E_old(x₀) · ΔE(x₀)] — the new intensity minus the old, linearized. So everything reduces to: when I perturb the dielectric at x, how does the field *at x₀* move?

A small dielectric bump Δε_r over ΔV at x is, electromagnetically, an induced dipole sitting at x. Its polarization is p_ind = ε₀ Δε_r ΔV E(x) — the local field times the extra polarizability I just installed. And a dipole at x produces a field at x₀ through the Maxwell Green's function: ΔE(x₀) = G^EP(x₀, x) p_ind = ε₀ Δε_r ΔV G^EP(x₀, x) E_new(x), where G^EP is the electric-field-from-electric-dipole Green's function of the *current* structure. For a small enough perturbation E_new(x) ≈ E_old(x). (One subtlety I'll keep in the back of my mind: for a binary structure Δε_r is *not* small — it's the full silicon-to-air jump — but ΔV can be the small parameter instead, the perturbation lives on a thin shell at the boundary; the same Green's-function bookkeeping goes through if I'm careful about which components of E and D are continuous across the interface.)

Substitute back: ΔFoM/Δε_r = ε₀ ΔV Re[ E_old(x₀) · (G^EP(x₀, x) E_old(x)) ]. This is exact to first order and it's pretty, but it looks like it costs the same as before — to evaluate it at every x I'd need G^EP(x₀, x) for every x, i.e. the field at x₀ from a dipole at *each* design point, which is again one solve per pixel. The Green's function has its arguments in the wrong order: source ranges over all design points x, observation is fixed at x₀.

Now the move. The Maxwell Green's function is reciprocal: G^EP(x₀, x) = G^EP(x, x₀)^T. A dipole at x seen at x₀ equals a dipole at x₀ seen at x, with the tensor transposed. So I can swap the roles inside that expression:

ΔFoM/Δε_r = Re[ (ε₀ ΔV G^EP(x, x₀) E_old(x₀)) · E_old(x) ].

Stare at the parenthesized factor. ε₀ ΔV G^EP(x, x₀) E_old(x₀) is the field *at x* produced by a single dipole sitting at x₀ — the measuring point — driven with amplitude ε₀ ΔV E_old(x₀). Call that field E_adj(x). Then

ΔFoM/Δε_r = Re[ E_adj(x) · E_old(x) ].

That's it — that's the whole thing. The gradient at *every* design point x is just the pointwise product of two fields: the forward field E_old(x), which I already have from the simulation that computed F, and one new field E_adj(x). And E_adj is the solution of a single Maxwell problem: put one source at x₀ — the place I'm reading out F — and solve once. That one "adjoint" simulation hands me the field everywhere in the design region, and combining it pointwise with the forward field gives ∂F/∂ε at every pixel simultaneously. Two solves total — one forward, one adjoint — for the gradient with respect to arbitrarily many degrees of freedom. The reciprocity is what made the N collapse to 1: instead of asking "if I poke pixel i, what happens at x₀?" N separate times, I ask once "if I drive a source at x₀, what field does it leave across the whole design region?" and read all N sensitivities off that single field.

This is the adjoint method — it's been the workhorse of shape optimization in mechanical engineering and fluids for decades, and sensitivity analysis for quantum electronic devices used it too; I'm just realizing it lands cleanly on Maxwell because the Green's function is reciprocal.

Let me redo this discretely, on the actual linear system, both to trust it and to get the exact gradient I'll code. Maxwell discretized is A(ε) e = b with A = μ₀⁻¹ ∇×∇× − ω²ε₀ ε_r and b = −iωJ. My objective is a real scalar L that depends on the field solution e. I want dL/dε_r. Here's a wrinkle I have to respect: L is something like |e|² at the port, which depends on e and its conjugate e* — it's not holomorphic in e. So I can't just differentiate as if e were a single complex variable. I treat e and e* as independent variables (Wirtinger calculus), and write the total derivative with respect to a single design parameter ϕ:

dL/dϕ = ∂L/∂ϕ + (∂L/∂e)(de/dϕ) + (∂L/∂e*)(de*/dϕ).

The unknowns are de/dϕ and de*/dϕ — how the field responds to changing the design. To get them, differentiate the constraint f(e, e*, ϕ) = A(ε) e − b = 0. For the linear system A doesn't depend on e, so differentiating f gives ∂f/∂e = A and ∂f/∂e* = 0, and the constraint derivative reads A (de/dϕ) = −(∂A/∂ϕ) e, i.e. de/dϕ = −A⁻¹ (∂A/∂ϕ) e. Plug in:

dL/dϕ = ∂L/∂ϕ − (∂L/∂e) A⁻¹ (∂A/∂ϕ) e + (the conjugate term).

The expensive object hiding here is A⁻¹ — applying the inverse means a Maxwell solve, and naively I'd need (∂A/∂ϕ) e and then A⁻¹ of it for *each* ϕ. The whole trick is to not let A⁻¹ act per-parameter. Group it the other way: take (∂L/∂e) A⁻¹ and treat it as one row vector solved *once*. Define the adjoint field e_aj as the solution of

A^T e_aj = −(∂L/∂e)^T.

Then −(∂L/∂e) A⁻¹ = e_aj^T (because (A^T)⁻¹ (∂L/∂e)^T = −e_aj, transposing). So

dL/dϕ = ∂L/∂ϕ + 2 Re[ e_aj^T (∂A/∂ϕ) e ],

the factor of 2 and the Re coming from the e-term and its conjugate partner combining (using that L and ϕ are real). The adjoint field e_aj is solved exactly *once* — its defining equation A^T e_aj = −(∂L/∂e)^T doesn't mention ϕ at all — and then the gradient for every parameter is the cheap bilinear form e_aj^T (∂A/∂ϕ) e. That's the discrete echo of the continuous result: one extra solve, gradient for everything.

Two things to read off this. First, reciprocity again: for a reciprocal medium A is symmetric, A^T = A, so the adjoint problem A e_aj = −(∂L/∂e)^T uses the *same operator* as the forward problem — same solver, same factorization even — just a different source. (If the medium were non-reciprocal — magneto-optic, say — I'd genuinely need A^T, but for ordinary dielectrics this is free.) Second, the source −(∂L/∂e)^T sits exactly where L is measured, the output port. For the intensity objective that source is a dipole at the readout point; the adjoint simulation is the same structure driven from the place I care about.

Now make the gradient concrete for ϕ = ε_r. In A = μ₀⁻¹∇×∇× − ω²ε₀ ε_r, only the last term carries ε_r, and it's diagonal in the field components, so ∂A/∂ε_r = −ω²ε₀ (a diagonal block, one entry per pixel). Therefore

dL/dε_r = ∂L/∂ε − 2 ω² ε₀ Re[ e_aj^T e ],

with the product taken pixel by pixel (∂L/∂ε is usually zero — L lives at the port, not in the bulk). The gradient at pixel i is just −2ω²ε₀ Re[e_aj,i · e_i]: multiply the adjoint field by the forward field at that pixel, take the real part, scale. That matches the continuous Re[E_adj · E_old] exactly, sign, conjugate-handling and all. I'll keep the minus sign honest — I'm going to *maximize* F, so I ascend along +dL/dε.

I should also nail down what the adjoint source is for the real figure of merit, because it's not |E|² at a point — it's transmission into a guided mode. The FoM I actually want is the power coupled into the fundamental mode (E_m, H_m) of the output waveguide, read off the monitor surface S as a mode overlap:

FoM = (1/8) |∫ E×H_m* · dS + ∫ E_m*×H · dS|² / ∫ Re(E_m×H_m*) · dS,

i.e. the power transmission corrected for the overlap with the target mode — the forward fields (E, H) projected onto the mode profile, squared and mode-normalized. This is a smooth real function of the field on S, so the adjoint machinery applies unchanged; I just need ∂L/∂e for *this* L, which sets the adjoint source. Working it through (and bringing in the magnetic-dipole Green's function G^EM for the H-part of the overlap), the adjoint field is

E_adj(x) = A_amp ∫_S [ G^EP(x, x') (H_m(x')×n) − G^EM(x, x') (n×E_m(x'))/μ₀ ] dS',

with the complex amplitude A_amp = (1/4) ε₀ ΔV (∫E_old×H_m·dS + ∫E_m×H_old·dS) / ∫Re(E_m×H_m*)·dS fixing its phase from the forward fields. I don't need to assemble those Green's functions by hand — what the formula *says* is the punchline: the adjoint source is the target output mode (E_m, H_m) launched *backward* into the device from the output port. That's the physical reading of −(∂L/∂e): for a mode objective the "source at the measuring point" is the desired mode sent in reverse, with its amplitude and phase set by how the forward field already overlaps the mode. And FDTD is perfectly suited to both runs — they're just propagating-wave problems in a dielectric, one driven from the input, one driven backward from the output.

So I have a cheap gradient. Now the second half of the problem, which the gradient alone doesn't solve: manufacturability. If I let ε be a free real number at every pixel and just ascend, two things go wrong, and both are well documented for density-based topology optimization. First, an unregularized pixel grid grows single-pixel and checkerboard features — a solid/void checkerboard is a numerical artifact that low-order finite elements report as artificially stiff/efficient, so an optimizer left unconstrained will *exploit* it, and the pattern is unmanufacturable anyway. Second, and worse, the optimizer loves *gray* — intermediate permittivity between silicon and oxide — because a continuum of index lets it tune the local phase exactly, so it parks ε at fractional values everywhere. But there is no gray material; a foundry gives me silicon or oxide, two phases.

The tempting fix is to optimize the gray design and then threshold it at the midpoint to recover black-and-white. That fails badly, and I can see why without running it: the thresholded structure is a *different device*. The optimizer found its performance using the gray phase delays; binarizing changes the phase everywhere and the carefully arranged interference falls apart. In a reported metalens example the merit collapses from ≈18.2 to ≈4.7 on thresholding — same intent, wrecked physics. So I can't binarize after the fact; binary-ness has to be *in* the optimization.

Both pathologies point at the same conclusion: don't optimize ε directly. Optimize a *density* ρ ∈ [0,1] per pixel — the design variable — and define the physical permittivity as an interpolation, ε(ρ) = ε_air + ρ (ε_Si − ε_air), so ρ=0 is oxide, ρ=1 is silicon. Then process ρ through two transformations before it becomes ε.

The filter handles the length scale. Convolve ρ with a small smoothing kernel of radius r_f — a conic (linear-hat) or PDE filter — to get ρ̃. Any feature smaller than r_f gets blurred away, so single-pixel spikes and checkerboards simply can't exist in ρ̃: I've baked in a minimum feature size, which is also exactly the manufacturing constraint (a foundry's minimum width / radius of curvature). The filter is a fixed linear operator, trivially differentiable.

The projection handles binarization. Push ρ̃ through a smoothed Heaviside:

ρ̄ = [ tanh(βη) + tanh(β(ρ̃ − η)) ] / [ tanh(βη) + tanh(β(1 − η)) ],

with threshold η (use ½) and steepness β. At β=0 it's the identity (gray passes through); as β→∞ it becomes a hard step at η, mapping everything to 0 or 1. So large β *forces* the design binary while keeping a differentiable map. The chain is ρ → filter → ρ̃ → project → ρ̄ → ε(ρ̄) → Maxwell. The adjoint gives me dF/dε; I just backpropagate through ε(ρ̄), the projection, and the filter by the chain rule to get dF/dρ — all three are explicit differentiable functions, so this is mechanical.

One trap with the projection: if I crank β high from the start, ρ̄ is nearly a step function, so its derivative is ~zero everywhere except in a thin band around η. The gradient vanishes over most of the design and the optimizer freezes against a near-binary pattern it can't escape. So I anneal: start with small β (almost gray, smooth gradients, the optimizer arranges the rough topology freely), then ramp β up in stages, sharpening toward binary as the design firms up — β-continuation. Each time I raise β I re-optimize from where I was. By the last stage the design is essentially 0/1 and the figure of merit was computed *on that binary design*, so there's no post-hoc thresholding cliff to fall off — the collapse I worried about never happens because I never optimized gray and binarized; I optimized binary all along.

(An alternative representation, which is what the cleanest two-phase formulations use, is a level set: the silicon/oxide boundary is the zero contour of a function, and the shape derivative dF/dε on the boundary becomes a velocity that moves the interface outward where the gradient is positive, with a fixed-area step as the line-search rule. That keeps the material strictly binary at every iteration by construction. Density + filter + projection gets to the same place — strict-enough binary, a real length scale — while staying on a fixed grid that plugs straight into autodiff, so that's what I'll build.)

Assemble the loop. Initialize ρ to a uniform ½ — a featureless gray slab, no bias toward any topology. Each iteration: map ρ through filter and projection to get the permittivity weights; run the forward Maxwell solve to get the field and F; run the *one* adjoint solve with the source being the target output mode sent backward; form the per-pixel gradient −2ω²ε₀ Re[e_aj · e], i.e. the pointwise forward·adjoint product; chain it back through projection and filter to dF/dρ; take a gradient-ascent step on ρ with the box constraint ρ ∈ [0,1]; periodically raise β. Two simulations per iteration regardless of how many pixels, the whole pattern improving every step. A plain steepest-ascent or L-BFGS step works; I'll use MMA (the method of moving asymptotes), which is built for exactly this kind of large-scale bound-constrained design problem and handles the box and any extra constraints cleanly.

Here is the loop in code, using an FDTD solver for the forward and adjoint runs and autodiff to chain the density map.

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
    rho_bar = mpa.tanh_projection(rho_tilde, beta, ETA)   # smoothed Heaviside
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

    # OptimizationProblem runs ONE forward + ONE adjoint solve per call and
    # returns (F, dF/d(eps-weights)); the adjoint source is the output mode
    # launched backward, the gradient is the pointwise forward*adjoint product
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
        solver = nlopt.opt(nlopt.LD_MMA, N)        # bound-constrained ascent
        solver.set_lower_bounds(0.0); solver.set_upper_bounds(1.0)
        solver.set_max_objective(objective)        # MAXIMIZE transmission
        solver.set_maxeval(25)
        rho = solver.optimize(rho)
    np.save("optimized_design.npy",
            filter_and_project(rho, 64).reshape(NX, NY))

if __name__ == "__main__":
    main()
```

The causal chain: I want to choose ε over a footprint to maximize coupling into an output mode; the design space is one variable per pixel, and the only forward model is an expensive Maxwell solve, so heuristics that merely sample F and finite differences that re-solve per pixel both scale with N and die. Linearizing F under a pointwise dielectric perturbation turns the sensitivity into a Green's-function expression, and reciprocity of the Maxwell Green's function lets me swap source and observation so that *one* adjoint solve — the device driven from the readout port, which for a mode objective is the target mode sent backward — yields ∂F/∂ε at every pixel as the pointwise product of the forward and adjoint fields, −2ω²ε₀ Re[e_aj·e]. To make the result a manufacturable two-material device rather than a gray, checkerboarded one, I optimize a density ρ, push it through a length-scale filter and a tanh projection (annealing the projection sharpness β so the gradient never vanishes), and ascend on ρ with MMA — two simulations per iteration, the whole permittivity pattern improving at once.
