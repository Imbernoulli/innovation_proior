# FNO synthesis notes (Phase 1.5)

## Pain point / research question
Many science/engineering tasks require solving a *family* of PDEs — the same equation for many different coefficient functions / initial conditions a(x). Classical solvers (FEM/FDM/pseudospectral) solve **one instance at a time**, must discretize space, and pay a resolution tradeoff: coarse=fast/inaccurate, fine=accurate/slow. For inverse design / Bayesian inversion you need thousands of forward solves → infeasible.

The object we actually want is the **solution operator** G†: a(x) ↦ u(x), a map between infinite-dimensional function spaces (Banach spaces A, U on domain D⊂R^d). Learn it once from data {a_j, u_j}, then each new instance is a forward pass.

## Load-bearing ancestors and exactly where each falls short

1. **Classical numerical solvers (FEM/FDM/pseudospectral).** Discretize D into a mesh, solve a (sparse/dense) linear or nonlinear system per instance. Mesh-bound; resolution tradeoff; one instance per solve. Pseudospectral NS solver here: 2.2s/instance even on GPU.

2. **Finite-dimensional neural surrogates (CNN image-to-image: Guo 2016, Zabaras/Zhu 2018 FCN, U-Net, ResNet, TF-Net).** Parameterize G as a CNN between R^n → R^n on a *fixed grid*. Mesh-dependent BY DEFINITION: the learned filters/weights are tied to the training resolution and geometry. Error grows when you change resolution (Fig: CNN error rises with resolution). Cannot query at off-grid points; no transfer between meshes.

3. **Neural-FEM / PINNs (E & Yu 2018 Deep Ritz; Raissi 2019 PINN).** Parameterize the *solution function* u itself as a neural net, minimize PDE residual. Mesh-free, accurate, BUT models ONE instance — new a(x) ⇒ retrain from scratch (same per-instance cost as classical). Also requires knowing the PDE.

4. **Operator-learning forerunners.**
   - **PCANN / model reduction (Bhattacharya–Kovachki–Stuart 2020):** PCA-encode a and u into finite latent spaces, learn a NN map between latents. Operator-ish but the basis is data-fixed/linear.
   - **DeepONet (Lu–Jin–Karniadakis 2019):** branch net (samples of a) × trunk net (query point x), output ⟨branch, trunk⟩. Mesh-free in output but input sensor locations fixed; low-rank kernel structure.
   - **Random features in Banach space (Nelsen–Stuart 2020).**

5. **Graph Neural Operator / Nonlocal Neural Operator line (Li et al. 2020 "Neural Operator: Graph Kernel Network", 2003.03485; Multipole GNO 2006.09535).** THE direct ancestor and the framework FNO inherits. Defines the operator as an **iterative architecture** with a kernel integral operator:
   - lift a ↦ v_0 = P(a) to channel dim d_v,
   - iterate v_{t+1}(x) = σ( W v_t(x) + (K(a)v_t)(x) ),
   - project u = Q(v_T).
   - The kernel integral operator: (K(a)v_t)(x) = ∫_D κ_φ(x,y,a(x),a(y)) v_t(y) dy, with κ_φ a NN R^{2(d+d_a)}→R^{d_v×d_v}.
   - Computed by **message passing on a graph** (Nyström approximation of the integral by sampling nodes). This is mesh-free and resolution-transferable — solved problem 2 — but the integral is **O(N²)** in the number of evaluation/quadrature points: every output point integrates against every input point. MGNO uses multipole/multi-scale decomposition to cut it but is still expensive and **does not converge on turbulent Navier–Stokes**.
   - GNO/MGNO Burgers error ~0.055/0.024; FCN/PCANN baselines. The cost of evaluating the integral operator is THE bottleneck — that is the sentence "neural operators have not yielded efficient numerical algorithms ... due to the cost of evaluating integral operators."

6. **Green's functions (classical).** For a *linear* PDE L u = f with constant coefficients, the solution operator is exactly an integral operator with the Green's function as kernel: u(x) = ∫ G(x,y) f(y) dy, and translation invariance ⇒ G(x,y)=G(x−y), so u = G * f, a **convolution**. This is the conceptual justification for (a) using an integral kernel operator at all (it's literally the solution operator of linear PDEs) and (b) restricting the kernel to depend only on x−y.

7. **Convolution theorem + spectral methods.** Spectral PDE methods use the FFT because differentiation = multiplication in Fourier space. Convolution theorem: F(f*g) = F(f)·F(g) (pointwise). So a convolution — O(N²) in physical space — becomes a **pointwise multiply** in Fourier space, computed via FFT in O(N log N). Fan et al. 2019 (BCR-Net, multiscale), Kashinath 2020 — spectral methods meeting NNs.

## The derivation chain (discovery order)
1. Want G†: a↦u, mesh-free, resolution-invariant. ⇒ must define the architecture in *function space*, not on a fixed grid.
2. Natural function-space analog of a linear layer = a kernel **integral operator** (Green's function is exactly this for linear PDEs). Stack with pointwise nonlinearity σ and a pointwise linear W to get nonlinear operators: v_{t+1}=σ(W v_t + K v_t). (Inherited GNO framework.)
3. The integral (K v)(x)=∫κ(x,y,...)v(y)dy is O(N²) and that's why GNO is slow / won't scale to turbulence. **Wall.**
4. Restrict κ to a convolution: drop the a-dependence and set κ(x,y)=κ(x−y). Justified by Green's function of constant-coeff linear PDE. Now (K v)(x)=∫_D κ(x−y)v(y)dy = (κ * v)(x).
5. Convolution theorem: κ*v = F⁻¹(F(κ)·F(v)). Let R = F(κ) be the thing we learn directly in Fourier space. ⇒ (K v)(x) = F⁻¹(R · F(v))(x). **The integral is now a pointwise (per-mode) multiply.**
6. κ periodic ⇒ Fourier *series*, discrete modes k∈Z^d. Truncate to lowest k_max modes (parameterization + regularizer; justified because the data spectra decay — and even when they don't, the σ between layers regenerates high modes). R is then a complex tensor (k_max × d_v × d_v); per mode it's a d_v×d_v channel-mixing matrix. κ real ⇒ conjugate symmetry R(−k)=R*(k).
7. Discrete/FFT: v_t∈R^{n×d_v}, F(v_t)∈C^{n×d_v}; truncate to C^{k_max×d_v}; (R·F v_t)_{k,l}=Σ_j R_{k,l,j}(F v_t)_{k,j}. With uniform grid use FFT (O(N log N)); modes truncated ⇒ even general DFT only O(n k_max). Quasi-linear.
8. Resolution invariance: parameters (R) live in Fourier space, independent of n. Evaluating in physical space = projecting onto e^{2πi⟨x,k⟩}, defined for any x ⇒ zero-shot super-resolution. CNN error grows with resolution; FNO error flat.
9. Lifting P, projection Q (shallow MLPs, pointwise in x) give the channel space d_v that makes per-mode mixing expressive.
10. W (pointwise linear, the "bias" branch / 1×1 conv) keeps the non-periodic / non-convolutional part — the FFT branch is periodic, W tracks boundary info; together they handle non-periodic BCs. σ on the spatial domain (not Fourier) recovers high frequencies that the truncation dropped.

## Design-decision → why table
- **Operator between function spaces (not R^n→R^n):** so one set of params works at any discretization; resolution-invariant. Alternative (CNN): mesh-bound, error grows with resolution.
- **Kernel integral operator as the "linear layer":** it IS the solution operator for linear PDEs (Green's fn); function-space analog of matrix multiply. Alternative (pointwise only): can't capture nonlocal/global dependence.
- **σ + W around the linear K:** compose linear global operators with local nonlinearities → nonlinear operators (mirrors standard NN linear+ReLU). Without σ the whole stack collapses to one linear operator.
- **Restrict kernel to convolution κ(x−y):** Green's-function/translation-invariance justification; AND it's what unlocks the FFT speedup. Alternative (full κ(x,y,a(x),a(y))): the GNO integral, O(N²), too slow, won't converge on NS.
- **Parameterize R=F(κ) directly in Fourier space:** skip ever forming κ in physical space; learn the multiplier. Alternative (learn κ then FFT it): wasteful.
- **Truncate to k_max modes:** finite parameterization + acts as regularizer/low-pass; justified by spectral decay of solutions; cheap. high modes regenerated by σ and Q. Alternative (keep all n modes): n grows with resolution → params resolution-dependent, defeats the point.
- **Z_kmax = box/"corners" (|k_j|≤k_max,j) not ℓ1-ball:** allows a trivial parallel slice-and-matmul implementation. (ℓ1 ball is "canonical low modes" but awkward to index.)
- **rfft (real FFT):** input real ⇒ conjugate symmetry ⇒ only need last-dim modes 0..n/2; halves storage/compute. In 2D this is why SpectralConv2d keeps TWO corners ([:m1,:m2] and [-m1:,:m2]) — the first (non-rfft) axis is full so both low-positive and high (= negative) frequencies must be kept; the rfft axis keeps only [:m2]. 3D ⇒ 4 corners (weights1..4). Conjugate symmetry recovers the rest.
- **Per-mode complex weight R∈C^{k_max×d_v×d_v}:** each retained mode gets its own d_v×d_v channel-mixing matrix (independent linear transform per frequency). einsum "bix,iox->box": batch×in_channel×mode times in×out×mode → batch×out×mode.
- **scale = 1/(in*out) init:** keep representation norm roughly preserved (modes not counted because the transform acts per-mode independently).
- **Lifting P / projection Q (1-layer Linear up to width, then Linear→128→1):** raise to channel dim d_v=32/64 so the per-mode d_v×d_v mixing has capacity; project back at the end. Q (nonlinear) also helps recover high modes.
- **W as 1×1 conv (Conv1d/2d/3d kernel 1):** pointwise-in-space linear; the "W v_t(x)" branch. Keeps the residual/local + non-periodic boundary info the periodic FFT branch can't represent.
- **GELU/ReLU activation on spatial domain (paper says ReLU+BN; canonical code uses GELU):** nonlinearity must be in physical space — a pointwise nonlinearity in Fourier space = spatial convolution, which destroys its meaning and can't regenerate high modes.
- **4 Fourier layers:** depth enough to compose; deeper Fourier stacks stop helping (noted in discussion).
- **k_max,j=12 (2d) / 16 (1d), d_v=32/64:** empirically sufficient; spectral analysis shows 12 modes + learned dependence beats truncating the true solution at 20 modes (≤1% vs ~2% error).
- **FNO-2D + RNN-in-time vs FNO-3D space-time conv:** 2D+RNN propagates to arbitrary T in Δt steps; 3D convolves space-time directly (fixed [0,T] window) — more expressive, easier to train.
- **Concatenate grid coords (x) / (x,y) / (x,y,t) to input:** gives the pointwise networks positional information.
- **Loss = relative L2 (LpLoss), Adam, lr 1e-3 halved/100ep, weight_decay 1e-4:** standard; relative L2 is the natural function-space norm error.

## Equations to derive inline in reasoning.md
- Operator-learning objective min_θ E_a C(G(a,θ), G†(a)).
- v_{t+1}(x)=σ(W v_t(x) + (K(a;φ)v_t)(x)).
- (K(a;φ)v_t)(x)=∫_D κ_φ(x,y,a(x),a(y)) v_t(y) dy.
- κ(x,y)=κ(x−y) ⇒ convolution; Green's-function motivation u=∫G(x−y)f(y)dy.
- Convolution theorem ⇒ (K v_t)(x)=F⁻¹(F(κ)·F(v_t))(x)=F⁻¹(R·F(v_t))(x).
- Fourier transform pair (continuous), Fourier series for periodic κ, truncation Z_kmax.
- (R·(F v_t))_{k,l}=Σ_{j=1}^{d_v} R_{k,l,j}(F v_t)_{k,j}.
- DFT/FFT pair on uniform grid; Z_kmax "corners" with s_j−k_j≤k_max,j.
- complexity: inner O(k_max), FFT O(n log n), truncated DFT O(n k_max).
- conjugate symmetry R(−k)=R*(k); rfft halving.
```
```
## Code structure to mirror (canonical Zongyi Li fourier_1d/2d/3d.py)
- SpectralConvNd: weights = scale*rand(in,out,*modes, cfloat); forward: x_ft=rfft; out_ft zeros; fill corner(s) via einsum compl_mul; irfft. 2D=2 corners, 3D=4 corners.
- FNONd: fc0 lift; 4× (convN + wN[=Conv1d/2d/3d kernel1]) with gelu between (last no gelu); optional F.pad for non-periodic; fc1→128→fc2→1; get_grid concat.
- training: relative LpLoss, Adam lr1e-3 wd1e-4, StepLR gamma0.5 step100/50, 500 epochs.
```
