**Problem.** DDIM at 20 steps holds quality (hopper 1.0045, halfcheetah 0.4754) but still pays a 14% NFE tax (penalty 0.861). The clean win is 10 steps (penalty 1.0, a 1.16× multiplier) — but DDIM is first-order: it freezes the prediction at each step's left endpoint, so at 10 steps each step is twice as wide and the frozen-prediction error grows. The rung must hold DDIM's returns at 10 steps by modeling how the prediction changes across the step.

**Key idea.** DPM-Solver++ 2M. Deterministic sampling is solving the probability-flow ODE; that ODE is semi-linear, so variation of constants solves the linear part exactly and the change of variable to the half-log-SNR λ leaves a single exponentially-weighted integral of the network to approximate. DDIM is its first-order member; climbing to **second order** in this expansion holds quality at wider steps. Two regime-specific choices define the variant:
- **`++` (data-prediction).** Expand the *clean-action* face x_θ=(x_t−σ_t ε_θ)/α_t, not ε_θ. The carried x_0 estimate stays inside the action box [−1,1] (in-distribution), and the data face is the stable one under the sharp conditioning a Q-favored policy induces, where high-order *noise*-prediction solvers degrade.
- **`2M` (second-order multistep).** Get the extra λ-derivative by finite-differencing the current and *previous* step's predictions — reusing a network call already paid for — so it is one NFE per step (10 steps from a 10-NFE budget), versus a singlestep order-2 method's 2 NFE per step (only 5 steps).

**Per-step update (data face, multistep).**
`x_{t_i} = (σ_{t_i}/σ_{t_{i-1}})·x_{t_{i-1}} − α_{t_i}(e^{−h_i}−1)·x_θ(x_{t_{i-1}}) − (α_{t_i}/2)(e^{−h_i}−1)(1/r_i)·(x_θ(x_{t_{i-1}})−x_θ(x_{t_{i-2}}))`, with h_i the λ-step and r_i=h_{i-1}/h_i; first step falls back to first-order.

**Why 2M not 3M.** Third order needs a third λ-derivative and its marginal gain is eaten by reduced stability in the few-step, sharply-conditioned regime. Second order models the leading prediction change across each wider step — exactly DDIM-at-10's failure — at one NFE per step.

**Hyperparameters.** `solver: ode_dpmsolver++_2M`; `sampling_steps: 10`. NFE penalty 1.0 (the floor); a 1.16× multiplier over DDIM before quality.

**What to watch.** Expect to clear DDIM on both axes: hopper ~1.00–1.01, walker2d toward ~0.88, halfcheetah toward ~0.50, at half the steps. Coming in below DDIM's 20-step returns would mean even second order cannot hold quality at 10 for this policy — but this data-face / multistep / 2nd-order combination is the strongest the two-line edit surface offers at the NFE floor.

```yaml
# EDITABLE region of CleanDiffuser/configs/custom/mujoco/mujoco.yaml — step 3: DPM-Solver++ 2M, 10 steps
diffusion_steps: 100
solver: ode_dpmsolver++_2M   # line 15 — deterministic ODE, data-prediction (++), 2nd-order multistep (2M)
predict_noise: True
sampling_steps: 10           # line 17 — at the NFE floor: penalty = 1.0
ema_rate: 0.9999
```
