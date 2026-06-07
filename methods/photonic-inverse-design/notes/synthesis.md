# Synthesis — adjoint-based photonic inverse design

## Sources (all read this run)
1. PRIMARY (full): Lalau-Keraly, Bhargava, Miller, Yablonovitch, "Adjoint shape optimization applied to electromagnetic design," Opt. Express 21(18):21693 (2013). Author-hosted PDF: optoelectronics.eecs.berkeley.edu/ey2013oe2118.pdf. Read all 9 pages incl. all equations.
2. BACKGROUND-rigorous (full main text pp.1-4): Hughes, Minkov, Williamson, Fan, "Adjoint method and inverse design for nonlinear nanophotonic devices," arXiv:1811.01255. Gives the discrete FDFD adjoint with A(ε)e=b, Lagrangian/implicit-diff derivation, dL/dε = ∂L/∂ε − 2ω²ε₀ Re(e_ajᵀ e), adjoint source −(∂L/∂e)ᵀ, reciprocity Aᵀ=A, density ρ, filtering+projection, L-BFGS.
3. BACKGROUND-topopt (read pp.1-2, 6-11 key sections): Christiansen & Sigmund, "Inverse design in photonics by topology optimization: tutorial," arXiv:2008.11816. Gives density-based TopOpt, conic/PDE filter (radius r_f), tanh Heaviside projection eq (18), grayscale penalization, threshold-at-0.5 failure, robust (eroded/dilated) formulation, min-lengthscale.
4. THIRD-PARTY explainer + CANONICAL CODE: Meep adjoint-solver docs (NanoComp/meep Adjoint_Solver.md). MaterialGrid, DesignRegion, OptimizationProblem, EigenmodeCoefficient (S-params), conic_filter+tanh_projection mapping, β-continuation, subpixel smoothing, MMA/epigraph, two-simulation gradient.

## The problem (research question)
Design a nanophotonic device (Si waveguide splitter/crossing/coupler) = choose ε(r) over the design region to maximize a figure of merit F (transmission/coupling into a target output mode). Straight Si waveguides are low-loss but functional components (Y-splitters, crossings, MMIs) leak via evanescent fields + interface reflections → scattering loss. Sub-wavelength Si photonics + CMOS integration makes this matter. Want systematic topology optimization over a pixelated/level-set ε, not hand-tuned shapes.

## Why brute force / heuristics fail (context, pre-method facts)
- Maxwell solve is expensive; each FoM eval = one full EM simulation.
- Heuristics (genetic, particle swarm) rely on a limited parameterization + random testing of MANY parameter sets. Only feasible for simple geometries (few params). The Y-splitter prior record [Zhang 2013] used PSO, 1500 simulations, reached −0.13 dB.
- Finite-difference gradient: to get ∂F/∂ε_i for N pixels needs N+1 simulations. N ~ 10³–10⁶ → hopeless.

## The derivation (re-derive in reasoning.md)
Continuous version (primary paper, simplest):
- FoM = |E(x₀)|² (eq1). Small ε perturbation Δε_r in volume ΔV at x: ΔFoM = Re[E^old(x₀)·ΔE(x₀)] (eq2).
- ΔE(x₀) = G^EP(x₀,x) p^ind = ε₀ Δε_r ΔV G^EP(x₀,x) E^new(x), with E^new≈E^old for small Δε (eq3). G^EP = electric-field-from-electric-dipole Green's function.
- So ΔFoM/Δε_r = ε₀ΔV Re[E^old(x₀)·(G^EP(x₀,x) E^old(x))] (eq4).
- Reciprocity of Green's function: G^EP(x₀,x) = G^EP(x,x₀)ᵀ → ΔFoM/Δε_r = Re[(ε₀ΔV G^EP(x,x₀) E^old(x₀))·E^old(x)] ≡ Re[E^adj(x)·E^old(x)] (eq5).
- E^adj(x) = ε₀ΔV G^EP(x,x₀) E^old(x₀) (eq6) = field induced at x by a dipole AT x₀ with amplitude ε₀ΔV E^old(x₀). ONE adjoint simulation gives dF/dε at EVERY x.

Discrete/rigorous version (Hughes), the Maxwell-as-constraint Lagrangian:
- Maxwell FDFD: A(ε)e = b, A = μ₀⁻¹∇×∇× − ω²ε₀ε_r, b = −iωJ. Objective L(e,e*,ε).
- treat e,e* independent; dL/dϕ = ∂L/∂ϕ + ∂L/∂e de/dϕ + c.c. Differentiate constraint A e − b=0: A de/dϕ = −(∂A/∂ϕ)e → de/dϕ = −A⁻¹(∂A/∂ϕ)e.
- dL/dϕ = ∂L/∂ϕ − ∂L/∂e A⁻¹(∂A/∂ϕ)e + c.c. Define adjoint e_aj: Aᵀ e_aj = −(∂L/∂e)ᵀ (eq7,11). Reciprocity Aᵀ=A → same operator, source at the measuring point.
- dL/dϕ = ∂L/∂ϕ + 2Re[e_ajᵀ (∂A/∂ϕ)e] (eq8/12). For ϕ=ε_r: ∂A/∂ε_r = −ω²ε₀ (diagonal) → dL/dε_r = ∂L/∂ε − 2ω²ε₀ Re[e_ajᵀ e] (eq13). Matches eq5.
- KEY: e_aj solved ONCE (one adjoint sim), gradient for ALL pixels = cheap elementwise product of forward and adjoint fields.

Mode-overlap FoM (primary eq7): FoM = (1/8)|∫E×H_m*·dS + ∫E_m*×H·dS|² / ∫Re(E_m×H_m*)·dS — power transmission corrected for mode overlap. Adjoint source = the target mode launched BACKWARD into the device from the output port (eq8, amplitude A eq9). This is the physical reading of −(∂L/∂e): the adjoint "source" is the output mode sent in reverse.

## Manufacturability machinery (context + reasoning)
- Raw pixel ε gives single-pixel/checkerboard features & grayscale ε∈(0,1) (intermediate index unphysical / unmanufacturable). Threshold-at-0.5 of a grayscale design destroys FoM (tutorial: 18.2→4.7).
- Density ρ∈[0,1], ε(ρ)=ε_air+ρ(ε_Si−ε_air) (linear) — design variable is ρ not ε.
- Filter: conic/PDE filter of radius r_f → ρ̃ enforces a length scale, kills single-pixel features.
- Projection: smoothed Heaviside ρ̄ = [tanh(βη)+tanh(β(ρ̃−η))]/[tanh(βη)+tanh(β(1−η))], β↑ pushes toward 0/1, η=threshold. β-continuation (anneal β up) avoids getting stuck.
- Chain ρ → filter → project → ε(ρ̄) → Maxwell. Backprop gradient through filter+projection (chain rule) — adjoint gives dF/dε, then chain through projection & filter to dF/dρ.
- Level-set (primary): boundary as zero level set, gradient = velocity field that moves the boundary (out where dF/dε>0). Two-phase binary throughout. Fixed-area step (2D)/fixed-volume (3D) = step-size rule.

## Design-decision → why
- treat e and e* as independent → needed because L depends on |field|², non-holomorphic; Wirtinger calculus.
- adjoint vs forward/finite-diff → N+1 sims vs 2 sims; the whole point.
- adjoint source at output port = −∂L/∂e → for mode FoM it's the target mode reversed (reciprocity makes "sense backwards" exact).
- reciprocity Aᵀ=A → adjoint uses SAME solver/operator, just different source; for non-reciprocal need Aᵀ.
- density+filter+projection → manufacturability: length scale + binarization; grayscale penalized physically (imaginary-ε "pamping") or via projection.
- β-continuation → projection at high β is near-discontinuous, gradients vanish except near η; ramp β so optimizer keeps moving.
- subpixel smoothing in MaterialGrid → at high β gradients blow up at η±1/β; smoothing regularizes.
- level set vs density → primary uses level set for strict binary; tutorial/Meep use density. Both valid; density+projection is the modern canonical (Meep).
- 2D→3D (effective index n=2.8) → cheap 2D opt first, refine in 3D; effective-index mimics 3D in-plane propagation.
- optimizer: steepest descent (primary), L-BFGS (Hughes), MMA/CCSA (Meep) — gradient-based, MMA handles constraints/bounds.

## Code grounding (Meep canonical)
mp.MaterialGrid(grid, SiO2, Si, weights, beta, do_averaging), mpa.DesignRegion, mpa.EigenmodeCoefficient (S-params at ports), mpa.OptimizationProblem(sim, objective_functions, objective_arguments, design_regions, frequencies). mapping = conic_filter then tanh_projection. opt([v]) returns (f0, dJ/dρ) using two sims. Loop: update ρ via optimizer (MMA/grad-ascent), ramp β.

## Excluded (proposed-method results)
Y-splitter −0.07 dB, 51 iters/102 sims, beats −0.13 dB/1500 sims PSO record, Hughes switch transmission 98.2%/3.1%, tutorial FoM 17.84. NONE of these go in reasoning.md as outcomes of the method.
