# Synthesis — diffsim-am

## Pain point
AM build quality (microstructure, residual stress, geometric accuracy, melt-pool depth) is governed by the transient thermal field, which is in turn controlled by process parameters: laser power, scan speed, beam radius, toolpath, plus material properties. We want to *design* those parameters to hit a target thermal/melt-pool behavior. The decision space is huge: time-series laser power over ~20,000 explicit FE time steps is effectively thousands of free variables.

Derivative-free / FD optimization of an expensive transient FE simulation is infeasible: a forward sim of the hourglass test case costs minutes (Taichi 5 min/iter; PyTorch 40 min; JAX 10 h). FD needs one extra forward sim *per parameter* to get a gradient → for a time-series of thousands of steps that is thousands of sims per gradient. Even for the 5 static params in case I, FD is 5+ sims/grad. The cost scales with the number of parameters → wall for high-dim temporal design.

## The move
Make the simulator itself differentiable. Implement the explicit time-stepping thermal FE solver inside an autodiff framework so reverse-mode AD gives ∂loss/∂(all parameters) in ONE backward pass, regardless of parameter count. This is the *discrete adjoint* of the solver, obtained automatically (no hand-derived continuous adjoint). Reverse mode is cheap precisely because loss is R^P→R (many params, one scalar): cost ≈ a small constant × one forward sim, independent of P. (Contrast forward-mode/FD: cost ∝ P.)

## Why reverse mode (the R^n→R argument)
- Forward mode: one sweep per *input* → O(P)·forward cost for P params.
- Reverse mode: one sweep per *output* → O(1)·forward cost for a scalar loss.
- Loss is one scalar; params are P ~ thousands. So reverse mode wins by factor ~P. This is exactly backprop.

## Memory / BPTT cost
- The explicit scheme is recurrent: T^{n+1} = T^n + Δt M^{-1}[...]. Backprop-through-time must hold every intermediate state (every T^n at all ~20k steps × n nodes) to evaluate local gradients on the backward pass → O(n_steps) memory.
- Taichi/DiffTaichi: lightweight tape records only kernel launches + scalar params (intermediates already live in global tensors). Memory still O(n_steps) for the temperature history.
- Checkpointing trades memory for recompute: store states only at segment boundaries (S steps apart), recompute within a segment on the backward pass → O(S + n/S), minimized at S=√n → O(√n) memory, time still O(n). This is the standard BPTT memory knob.

## Why Taichi (library choice)
FEM needs: flexible/random indexing (scatter-assembly into global K, M, rhs), large-scale atomic adds (element→node assembly), dynamic active-element set (element birth). DL-oriented libs (TF lacks flexible assembly indexing entirely; PyTorch 127 GB mem; JAX 10 h/sim) choke because they optimize dense high-level ops, not many low-level scatter/atomic ops. Taichi's megakernel + imperative parallel-for + flexible indexing + native atomics → 1 GB, 5 min/iter (2× less mem, 8× faster than next best). So Taichi.

## Differentiability hazards (load-bearing design choices)
1. **Geometric/material-deposition discontinuities** (element birth, surface birth) are discrete jumps → cannot be smoothly differentiated. FIX/hypothesis: freeze the geometric & boundary-related discontinuities (precompute element/surface birth times, gate with `if birth<=t*dt`), so the only things the gradient flows through are the *continuous* material & process params. The toolpath/geometry are held fixed; power/properties are optimized.
2. **Step functions kill gradients.** Computing melt-pool depth as "deepest node above T_melt" is differentiable a.e. but its gradient is 0 everywhere (piecewise constant) → no signal. FIX: build a *continuous* melt-pool depth: interpolate temperature directly below the laser at several height levels (9-nearest-neighbor / 9th-degree interp per level), then a pairwise-linear solve for the depth where T crosses melting → smooth, nonzero gradient.
3. **Recursive tanh saturation** = vanishing/exploding gradients. Relevant to the NN parameterization of laser power; mitigated by Adam (momentum + per-coord LR scaling) and the observation that gradients stay healthy over 20k steps.
4. **`min(T, max_temp)` clamp** stops gradient above max_temp — a deliberate stability cap.

## Parameterizations
- Case I (param inference): static scalars (cp, cond, h_conv, Qin, r_beam, ambient) as leaf differentiable fields; MSE on top-layer nodal temps (partial observation); Adam. Demonstrates AD through assembly/lumping/distribution/temporal-map ops. Note coupling/non-identifiability: under-cp vs over-power give same response.
- Case II (thermal history): time-series laser power = small FC net (input = normalized time, 2 hidden layers of 50, tanh, → 0..1000 W). Net produces P(t) for ~20k steps; MSE between current and target thermal history; optimize net weights via Adam. NN as universal function approximator of the temporal control; integrates data-driven model with physics sim.
- Case III (melt pool): same NN→P(t), but loss = MSE of continuous melt-pool depth vs target depth. Needs the differentiable melt-pool calc (hazard 2).

## FEM formulation (eqs)
- PDE: ρ c_p ∂T/∂t − ∇·(k∇T) − s = 0.
- BCs: Dirichlet T=T1 on Γ1; Neumann flux q=−q_s on Γ2; convection q=−h(T−Tamb) on Γ3; radiation q=−εσ(T^4−Tamb^4) on Γ4.
- Explicit update: {T^{n+1}} = {T^n} + Δt [M]^{-1} [ {R_G} − {R_F} − {R_C} − {R_R} − [K]{T^n} ]. M = (lumped) capacitance, K = conduction, R_F external flux (laser), R_C convection, R_R radiation, R_G internal gen.
- Lumped mass M (row-sum lumping) → diagonal → explicit update is just elementwise divide by m_vec (no linear solve) → cheap & trivially differentiable.
- Laser: Goldak-style surface Gaussian q = 3 Q P(t) on(t) /(π r²) · exp(−3 r²_dist / r²). (code uses 3.14 for π, 3.0 the Goldak concentration factor.)
- Radiation σ ≈ 5.6704e-14 (consistent unit system, mm/s/K), h_rad emissivity factor.
- Latent heat: cp += latent between solidus(1533.15) and liquidus(1609.15), latent=272/(liq−sol).
- Hex8 elements, 8 Gauss points, shape fns N, derivative B, Jacobian per ip.

## Code structure (canonical, Taichi notebook)
- Global `ti.field(..., needs_grad=True)` for every state/param/NN weight.
- Kernels: nn1/nn2/nn3 (FC net), clac_cp/update_matprop, clac_mass/update_mvec, calc_stiffness/update_stiffness, update_fluxes_m (laser), update_fluxes_cr (conv+rad), time_integrate, compute_loss.
- `simulate()`: loop t=1..steps: clear → nn → matprop → mvec → stiffness → fluxes → integrate; then loss.
- `with ti.Tape(loss): simulate()` → reverse-mode AD fills `.grad`.
- `update_weights_adam` / Adam on static params; lr=1e-2, beta1=.9, beta2=.999, eps=1e-7; iters 60/300/200.

## Sources (three-source bottom line)
1. PRIMARY: arXiv 2107.10919 "Additive manufacturing process design with differentiable simulations" (full text) + repo github.com/mojtabamozaffar/differentiable-simulation-am (notebook full).
2. BACKGROUND: AM thermal FEM lineage — Goldak moving Gaussian heat source, Rosenthal, element birth/explicit integration; Mozaffar et al. 2019 "Acceleration strategies for explicit FE analysis of metal powder-based AM using GPUs" (ref [12]); reverse-mode AD cost — Baydin et al. 2018 AD survey (arXiv 1502.05767).
3. THIRD-PARTY EXPLAINER / AD framework: DiffTaichi, Hu et al. 2019/ICLR2020 (arXiv 1910.00935) — two-scale AD, lightweight tape, global-data-access rules, checkpointing O(√n).
