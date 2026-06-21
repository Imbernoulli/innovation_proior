I want to design a passive silicon-photonics component — a Y-junction splitter, a crossing, an MMI, a fiber coupler — by choosing the silicon pattern inside a small footprint of chip. The input mode enters, the design region reshapes it, and I read out how much power lands in the fundamental mode of the output waveguide. The straight access waveguides are essentially lossless; all the trouble lives in this little functional region, where evanescent fields leak outside the core and abrupt silicon/oxide interfaces reflect, and that scattering loss eats the whole link budget. So the object I really want to choose is the permittivity $\varepsilon(r)$ over the design region, and the quantity I want to maximize is a scalar figure of merit $F$ — the power coupled into the target output mode. The forward model is Maxwell's equations: time-harmonic, non-magnetic, at frequency $\omega$, $\nabla\times\nabla\times E - (\omega^2/c^2)\,\varepsilon_r(r)\,E = -i\omega\mu_0 J$, which on a discretized grid becomes a large sparse linear system $A(\varepsilon)\,e = b$ with $A = \mu_0^{-1}\nabla\times\nabla\times - \omega^2\varepsilon_0\varepsilon_r$ and $b = -i\omega J$. One solve of $A e = b$ gives the field everywhere, and from the field on a monitor plane I compute $F$. That single solve is the expensive primitive, and the design space is the permittivity at every pixel — $N$ free variables, anywhere from thousands to millions.

The way these devices are designed now is heuristic global optimization — genetic algorithms, particle swarm. Pick a handful of shape parameters, simulate a whole population of random parameter sets, keep the good ones, breed, repeat; the silicon Y-splitter record was set this way with particle swarm by Zhang et al. (2013), reaching low loss after on the order of fifteen hundred full simulations. The reason it costs that much is that these methods never look inside the physics: they only ever evaluate $F$, treating the simulator as an oracle that returns one number. With no gradient, the only way to explore is to sample, and to keep sampling affordable you must keep the parameterization small — a few smooth knobs. That works for a clean geometry, but the whole promise of topology optimization is to let *every pixel* be free, and a sample-only method drowns the moment the geometry gets rich. The obvious gradient alternative, finite differences, is no better: estimating $\partial F/\partial\varepsilon_i \approx (F(\varepsilon + \delta e_i) - F(\varepsilon))/\delta$ costs one extra full Maxwell solve per pixel, so $N+1$ solves for a single gradient at every step — exactly as physics-blind as the heuristics, and just as hopeless. What is needed is a gradient with respect to *all* $N$ variables that costs almost nothing extra.

I propose adjoint-method photonic inverse design. The key realization is that a first-order perturbation of the permittivity at a point is, electromagnetically, an induced dipole, and its effect on $F$ propagates through the Maxwell Green's function — and that Green's function is reciprocal, which collapses $N$ solves to one. Strip the figure of merit to $|E(x_0)|^2$ at a measuring point and perturb $\varepsilon$ in a tiny volume $\Delta V$ at a design point $x$. To first order $\Delta F = 2\,\mathrm{Re}[E_\text{old}(x_0)^* \cdot \Delta E(x_0)]$, where the factor of $2$, the conjugate and the real part come straight from $d|E|^2 = E^*\cdot dE + E\cdot dE^* = 2\,\mathrm{Re}[E^*\cdot dE]$. The dielectric bump installs a dipole $p_\text{ind} = \varepsilon_0\,\Delta\varepsilon_r\,\Delta V\,E(x)$, whose field at $x_0$ is $\Delta E(x_0) = G^{EP}(x_0,x)\,p_\text{ind}$, giving $\Delta F/\Delta\varepsilon_r = 2\varepsilon_0\Delta V\,\mathrm{Re}[E_\text{old}(x_0)^* \cdot G^{EP}(x_0,x)E_\text{old}(x)]$. This is exact to first order but the Green's function has its arguments in the wrong order — source ranges over all design points $x$, observation is pinned at $x_0$ — so evaluating it everywhere would again be one solve per pixel. The move is reciprocity: $G^{EP}(x_0,x) = G^{EP}(x,x_0)^T$, so I can swap source and observation,
$$\frac{\Delta F}{\Delta\varepsilon_r} = 2\,\mathrm{Re}\!\left[\big(\varepsilon_0\Delta V\,G^{EP}(x,x_0)E_\text{old}(x_0)^*\big)\cdot E_\text{old}(x)\right] = 2\,\mathrm{Re}[E_\text{adj}(x)\cdot E_\text{old}(x)],$$
where $E_\text{adj}(x)$ is the field at $x$ produced by a *single* dipole sitting at the readout point $x_0$. The gradient at every design point is the pointwise product of two fields — the forward field $E_\text{old}$, which I already have from the solve that computed $F$, and one new adjoint field $E_\text{adj}$ from a single Maxwell problem driven from $x_0$. Two solves total for the gradient with respect to arbitrarily many degrees of freedom. Reciprocity is what makes $N$ collapse to $1$: instead of asking "if I poke pixel $i$, what happens at $x_0$?" $N$ separate times, I ask once "if I drive a source at $x_0$, what field does it leave across the whole design region?" and read all $N$ sensitivities off that one field.

The discrete derivation pins down the exact gradient to code, and it has to respect that the objective $L$ — something like $|e|^2$ at the port — depends on both $e$ and $e^*$ and so is not holomorphic. Treating $e$ and $e^*$ as independent (Wirtinger calculus), $dL/d\phi = \partial L/\partial\phi + (\partial L/\partial e)(de/d\phi) + (\partial L/\partial e^*)(de^*/d\phi)$. Differentiating the constraint $A(\varepsilon)e - b = 0$ gives $de/d\phi = -A^{-1}(\partial A/\partial\phi)e$, and the whole trick is to not let $A^{-1}$ — a Maxwell solve — act per-parameter. Group it the other way and define the adjoint field as the solution of
$$A^T e_\text{aj} = -(\partial L/\partial e)^T,$$
so that $-(\partial L/\partial e)A^{-1} = e_\text{aj}^T$ and
$$\frac{dL}{d\phi} = \frac{\partial L}{\partial\phi} + 2\,\mathrm{Re}\big[e_\text{aj}^T (\partial A/\partial\phi)\,e\big],$$
the factor of $2$ and the $\mathrm{Re}$ coming from the $e$-term and its conjugate partner combining since $L$ and $\phi$ are real. The adjoint field is solved exactly once — its defining equation never mentions $\phi$ — and then the gradient for every parameter is the cheap bilinear form $e_\text{aj}^T(\partial A/\partial\phi)e$. Two facts fall out. For a reciprocal medium $A$ is symmetric, $A^T = A$, so the adjoint problem reuses the *same operator* as the forward solve — same solver, same factorization — only with a different source; and that source $-(\partial L/\partial e)^T$ sits exactly at the output port where $L$ is measured. Specializing to $\phi = \varepsilon_r$: only the $-\omega^2\varepsilon_0\varepsilon_r$ term of $A$ carries $\varepsilon_r$ and it is diagonal, so $\partial A/\partial\varepsilon_r = -\omega^2\varepsilon_0$ and
$$\frac{dL}{d\varepsilon_r} = \frac{\partial L}{\partial\varepsilon} - 2\,\omega^2\varepsilon_0\,\mathrm{Re}[e_\text{aj}\cdot e]$$
pixel by pixel ($\partial L/\partial\varepsilon$ is usually zero, since $L$ lives at the port, not in the bulk). This matches the continuous $2\,\mathrm{Re}[E_\text{adj}\cdot E_\text{old}]$ exactly. For the real figure of merit — transmission into a guided mode, $F = \tfrac18|\int E\times H_m^*\cdot dS + \int E_m^*\times H\cdot dS|^2 / \int\mathrm{Re}(E_m\times H_m^*)\cdot dS$ — working the adjoint source through shows the physical punchline: $-(\partial L/\partial e)$ for a mode objective is simply the **target output mode $(E_m,H_m)$ launched backward** into the device from the output port, with amplitude and phase fixed by how the forward field already overlaps the mode. FDTD handles both runs naturally: one driven from the input, one driven backward from the output.

A cheap gradient is only half the problem; the other half is manufacturability, which the gradient alone does not solve. If I let $\varepsilon$ be a free real number per pixel and just ascend, two documented pathologies appear. First, an unregularized grid grows single-pixel and checkerboard features — a solid/void checkerboard is a numerical artifact that low-order finite elements report as artificially efficient, so the optimizer exploits it, and it is unmanufacturable anyway. Second, the optimizer parks $\varepsilon$ at *gray* intermediate values, because a continuum of index lets it tune local phase exactly — but a foundry gives only silicon or oxide. Thresholding a converged gray design at the midpoint to recover black-and-white fails badly, because the thresholded structure is a *different device*: binarizing changes the phase everywhere and the arranged interference falls apart (in a reported metalens example the merit collapsed from $\approx 18.2$ to $\approx 4.7$). So binary-ness must be *in* the optimization. The fix is to not optimize $\varepsilon$ directly but a density $\rho \in [0,1]$ per pixel, processed through two differentiable transformations before it becomes permittivity. A length-scale filter — convolution with a small conic kernel of radius $r_f$ — blurs $\rho$ into $\tilde\rho$, so any feature smaller than $r_f$ cannot exist; this kills checkerboards and bakes in the minimum feature size, which is exactly the manufacturing constraint. A tanh projection then drives the result toward binary through a smoothed Heaviside,
$$\bar\rho = \frac{\tanh(\beta\eta) + \tanh\!\big(\beta(\tilde\rho - \eta)\big)}{\tanh(\beta\eta) + \tanh\!\big(\beta(1-\eta)\big)},$$
with threshold $\eta = \tfrac12$ and steepness $\beta$: at $\beta = 0$ it is the identity (gray passes through), and as $\beta\to\infty$ it becomes a hard step that maps everything to $0$ or $1$. The physical permittivity is the interpolation $\varepsilon(\bar\rho) = \varepsilon_\text{air} + \bar\rho(\varepsilon_\text{Si} - \varepsilon_\text{air})$, and since the chain $\rho \to \text{filter} \to \tilde\rho \to \text{project} \to \bar\rho \to \varepsilon$ is three explicit differentiable maps, I backpropagate the adjoint's $dF/d\varepsilon$ through them to $dF/d\rho$ mechanically. The one trap is that cranking $\beta$ high from the start makes $\bar\rho$ a near-step whose derivative vanishes everywhere except a thin band around $\eta$, freezing the optimizer. So I anneal: start small (almost gray, smooth gradients, the rough topology arranged freely) and ramp $\beta$ up in stages, re-optimizing from where I was — $\beta$-continuation. Because $F$ is always evaluated on the increasingly binary design, by the last stage the structure is essentially $0/1$ and there is no post-hoc thresholding cliff to fall off; the collapse never happens because I optimized binary all along.

The loop assembles directly. Initialize $\rho$ to a uniform $\tfrac12$ — a featureless gray slab with no topology bias. Each iteration: map $\rho$ through filter and projection to get permittivity weights; run one forward Maxwell solve for the field and $F$; run the one adjoint solve with the target output mode sent backward as the source; form the per-pixel gradient $-2\omega^2\varepsilon_0\,\mathrm{Re}[e_\text{aj}\cdot e]$ as the pointwise forward–adjoint product; chain it back through projection and filter to $dF/d\rho$; take a bound-constrained ascent step on $\rho \in [0,1]$; and periodically raise $\beta$. Two simulations per iteration regardless of pixel count, with the whole pattern improving every step. I use MMA (the method of moving asymptotes), built for exactly this large-scale, box-constrained design problem. An alternative two-phase representation is a level set, where the silicon/oxide boundary is the zero contour of a function moved by the shape-derivative velocity, keeping the material strictly binary by construction; density-plus-filter-plus-projection reaches the same place while staying on a fixed grid that plugs straight into autodiff, so that is what I build.

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
