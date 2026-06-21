## Research question

I have a pretrained **diffusion bridge model** for image-to-image translation: a stochastic process pinned to start at the clean target `x_0` and arrive almost surely at an informative source endpoint `x_T` — a sketch, a degraded photo, a masked image. To generate I run the reverse process from the source `x_T` at `t = t_max` down to a target sample at `t = 0`. The network is fixed; I am not allowed to retrain. The single thing being designed is the **per-step transition rule of the sampler** — how the bridge state at the current time `s` becomes the state at the next, smaller time `t`, given the network's data prediction and the analytic schedule coefficients. Everything else (the trained model, the schedule, the dataset handling, the evaluation scripts) is frozen.

Every denoiser call is one forward pass of a large U-Net, so wall-clock is essentially proportional to the **number of function evaluations (NFE)**. The harness wraps the denoiser with a counter and **rejects any run that exceeds NFE = 5 denoiser calls per sample** — the `(len(ts)+1)`-th call raises `RuntimeError: NFE_BUDGET_EXCEEDED`. The research question is: *what transition rule produces the best conditional image quality (lowest FID) with at most five network evaluations?* How those five calls are allocated and how stochasticity is scheduled across the trajectory is the design space.

## Prior art / Background / Baselines

These are the methods and facts currently available.

- **Score-based diffusion + EDM.** A forward SDE `dx = f x dt + g dw` turns data into Gaussian noise; generation reverses the probability-flow ODE `dx = [f x − ½ g² ∇log p_t] dt` with a learned score, and dedicated high-order solvers (Heun on the ODE, with EDM's ρ=7 power-law time grid and optional stochastic "churn") reach good quality in ~10 NFE.

- **Doob's h-transform bridges.** Pinning a diffusion to land at a fixed endpoint `x_T = y` almost surely by adding the drift `g² ∇log p(x_T | x_t)` gives a bridge whose doubly-conditioned forward kernel is an analytic Gaussian `q(x_t | x_0, x_T) = N(a_t x_T + b_t x_0, c_t² I)`, with coefficients fixed by the underlying VP schedule: `a_t = (α_t/α_T)(SNR_T/SNR_t)`, `b_t = α_t(1 − SNR_T/SNR_t)`, `c_t² = σ_t²(1 − SNR_T/SNR_t)`, `SNR_t = α_t²/σ_t²`.

- **Marginals-only training objective.** The training loss depends on the model only through the per-time marginals `q(x_t | x_0, x_T)`, not through the joint over the whole trajectory.

## Fixed substrate / Code framework

A pretrained VP-schedule bridge model and its evaluation harness are frozen. The model exposes a **data predictor** `denoiser(x, t) → x̂_0` — the EDM-preconditioned network's estimate of the clean target from a noisy bridge state — and it is the NFE-counted resource. The `diffusion` object exposes the schedule through `diffusion.noise_schedule`:

- `get_abc(t) → (a_t, b_t, c_t)` — the bridge coefficients (`a_t` multiplies `x_T`, `b_t` multiplies `x_0`, `c_t` is the noise scale).
- `get_alpha_rho(t) → (α_t, α_t/α_T, ρ_t, ρ̄_t)` — raw schedule values, where `ρ_t = σ_t/α_t` (= 1/√SNR_t, the schedule's own "rho", **not** an injected stochasticity) and `ρ̄_t² = ρ_T² − ρ_t²`.
- `get_f_g2(t) → (f_t, g²_t)` — the base SDE drift `f_t = (log α_t)'` and `g²_t`, from which the analytic time-derivatives of `a_t, b_t, c_t` can be formed exactly.
- `diffusion.bridge_sample(x0, xT, t, noise) → a_t xT + b_t x0 + c_t noise`, `diffusion.t_max`, `diffusion.t_min`.

The outer `sample.py` loop supplies the source `x_T` (= the input `x`), a monotonically decreasing interior time schedule `ts` (first element just below `t_max`), an optional `mask` (for restoration / inpainting), and a seeded noise generator `BatchedSeedGenerator(seed)`; it wraps `denoiser` with the hard NFE counter. The helper `append_dims(v, ndim)` broadcasts a per-sample scalar across image dimensions.

## Editable interface

Exactly one region is editable: the body of `sample_dbim` in `ddbm/karras_diffusion.py`. The signature and the six-value return tuple are part of the contract and must not change:

- **Contract.** `sample_dbim(denoiser, diffusion, x, ts, eta=1.0, mask=None, seed=None, **kwargs)` must return exactly `(x, path, nfe, pred_x0, ts, first_noise)` — the final image, the list of intermediate states, the function-eval count, the list of predicted `x̂_0`, the (possibly modified) time schedule, and the initial noise. The harness uses the returned `nfe` to locate the generated files, and rejects the run if the real number of `denoiser` calls exceeds `len(ts)`.
- `eta` is the stochasticity scale; `mask` must be preserved for inpainting; external hyperparameters (parsed elsewhere from environment variables) must not be touched.

Every method on the ladder is a fill of this one function body. The starting point is the scaffold default — **not implemented**:

```python
@torch.no_grad()
def sample_dbim(
    denoiser,
    diffusion,
    x,
    ts,
    eta=1.0,
    mask=None,
    seed=None,
    **kwargs,
):
    # =================================================================================
    # 🚨 CRITICAL CONSTRAINTS - DO NOT IGNORE! 🚨
    # 1. Function Signature: You must NOT modify the function name, arguments, or return structure.
    # 2. NFE Match (FATAL I/O ERROR): The framework uses the final returned `nfe` to locate
    #    generated files (e.g., expecting `samples_..._nfe5.npz`). You MUST return
    #    `nfe = len(ts) - 1` regardless of the internal call count.
    # =================================================================================

    # TODO: Implement your novel sampling kernel here.
    # Ensure the return structure is: return x, path, nfe, pred_x0, ts, first_noise

    raise NotImplementedError("Custom sampler not implemented yet.")
```

## Evaluation settings

Three image-to-image / restoration workloads, each with the same pretrained bridge per task and the hard **NFE = 5** budget, seed 42:

- **Edges→Handbags** (`run_e2h.sh`, 64×64 translation).
- **ImageNet** center-inpainting (`run_Imagenet.sh`, 256×256, a masked region to complete — `mask` semantics are live here).
- **DIODE-Outdoor** (`run_DIODE.sh`, 256×256 translation).

**Metric: Fréchet Inception Distance (FID), lower is better**, reported per workload (`best_fid_edges2handbags`, `best_fid_Imagenet`, `best_fid_DIODE` / `fid_DIODE`). The parser also verifies the actual per-sample denoiser-call count and rejects any run over budget. A separate DDBM reference is run at 50 NFE (≈10× the compute and ≈8× the wall-clock) purely as an upper-bound reference point; that budget is **not** available to the agent's sampler.
