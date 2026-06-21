The reference told me what I needed, in numbers that are almost embarrassing for the floor. DDBM at 50 NFE landed FID 11.139 on Edges→Handbags, 10.556 on ImageNet inpainting, and 15.811 on DIODE — real images, so the framework and the trained model are sound — but it spent 2149 seconds on DIODE against the $\approx$270 the budgeted samplers get, and 11 FID on a $64\times64$ translation is mediocre. The diagnosis is sharp and it is not "the model is weak": I am paying for genericity. DDBM treats the reverse bridge dynamics as one opaque vector field and takes many small Heun-plus-churn steps because that is all a black-box solver can safely do, paying discretization error on every part of the drift including the linear part. Fifty calls is the price of that ignorance. Five calls now have to get *closer* to the truth than fifty did — which means exploiting the one thing the black-box solver ignored: the bridge has a known analytic structure, and the trained network only cares about its marginals.

That last clause is the whole lever. The denoising-bridge-score-matching loss depends on the model *only through the per-time marginals* $q(x_t \mid x_0, x_T)$, never through the full joint over the trajectory — the network never saw a trajectory, only $(x_0, x_T, t)$ triples with $x_t$ drawn from the marginal kernel. So any inference process that *agrees with these same marginals* is one the network is already optimal for, and a different joint can take far bigger steps. This is exactly the move that made DDIM fast for ordinary diffusion. I cannot copy that formula, though, because the diffusion construction is welded to a single Gaussian endpoint — mean $\sqrt{\alpha_t}\,x_0$, no second endpoint — while my bridge mean carries the extra $a_t x_T$ term and noise scale $c_t$, not $1 - \alpha_t$.

So I propose **DBIM**, the bridge analogue of DDIM, redone from the bridge kernel and forced to preserve the bridge marginals. Over timesteps $0 = t_0 < t_1 < \dots < t_N = t_{\max}$, I posit a family of reverse conditionals indexed by a per-step injected standard deviation $\rho_n$, each $x_{t_n}$ conditioned on the next-later state $x_{t_{n+1}}$ and on $x_0$, Gaussian with variance $\rho_n^2$ and mean
$$a_{t_n} x_T + b_{t_n} x_0 + \sqrt{c_{t_n}^2 - \rho_n^2}\;\frac{x_{t_{n+1}} - a_{t_{n+1}} x_T - b_{t_{n+1}} x_0}{c_{t_{n+1}}}.$$
The last factor is precisely the standardized Gaussian $\hat\varepsilon$ that generated $x_{t_{n+1}}$: I am recycling a fraction of the later step's realized noise as a deterministic direction, scaled by $\sqrt{c_{t_n}^2 - \rho_n^2}$, and adding fresh noise of variance $\rho_n^2$. The total noise at $t_n$ is $(c_{t_n}^2 - \rho_n^2) + \rho_n^2 = c_{t_n}^2$, the marginal variance the network expects.

What makes the network reusable as-is is that the marginals are *provably* preserved for every admissible $\rho$. By backward induction from $n = N-1$: the base case forces the boundary $\rho_{N-1} = c_{t_{N-1}}$, because there $t_{n+1} = t_{\max}$, the recycled coefficient $\sqrt{c^2 - \rho^2} = 0$ kills the borrowed-noise term, and the conditional collapses to exactly the bridge kernel. The inductive step is the standard linear-Gaussian marginalization: substituting the mean of $x_{t_k}$ into the recycled-direction term sends its argument to $(a_{t_k} x_T + b_{t_k} x_0) - a_{t_k} x_T - b_{t_k} x_0 = 0$, so the deterministic direction averages to zero, the mean collapses to $a_{t_{k-1}} x_T + b_{t_{k-1}} x_0$, and the variance sums to $c_{t_{k-1}}^2$ after the $c_{t_k}^2$ cancels. The ELBO confirms the same conclusion: it reduces to a weighted sum of data-prediction errors, which converts to score matching with a per-time reweighting that does not move the minimizer. No retraining; I just choose $\rho$.

That leaves one dial. The step replaces the unknown $x_0$ with $\hat x_0 = \mathrm{denoiser}(x_{t_{n+1}}, t_{n+1})$, and I parameterize the injected noise by a scalar $\eta \in [0,1]$ via $\rho_n = \eta\,\sigma_{t_n}\sqrt{1 - \mathrm{SNR}_{t_{n+1}}/\mathrm{SNR}_{t_n}}$. At $\eta = 1$ the induced forward process becomes Markovian — the $x_T$ term cancels and the update is a DDPM-like ancestral sampler; at $\eta = 0$ there is no fresh noise, the update is a deterministic implicit map, the bridge analogue of DDIM that takes clean sharp jumps in few steps. I expose it because the ends genuinely trade off: the deterministic map is sharp and invertible, ideal when source and target are tightly correlated, while injected noise acts like a Langevin correction that washes out accumulated discretization error and helps on diverse tasks. In the harness the schedule's `rho` is $\sigma_t/\alpha_t$, not my injected $\rho_n$, so the injected std reads $\eta\,(\alpha_t \rho_t)\sqrt{1 - \rho_t^2/\rho_s^2}$ with $\alpha_t \rho_t = \sigma_t$.

Two boundary subtleties remain, and both fall out of the same fact. The deterministic first step, where $t_{n+1} = t_{\max}$, divides by $c_{t_{\max}} = 0$ — the bridge is pinned exactly at $x_T$ with no spread — and this is the same one-to-many stochasticity that forced DDBM to inject noise: $p(x_t \mid x_T)$ is genuinely not a Dirac. The fix is already in hand, since the proof's forced boundary $\rho_{N-1} = c_{t_{N-1}}$ is exactly the Markovian boundary at step one: it zeros the recycled coefficient and annihilates the singular $c_{t_{\max}}$ denominator, leaving a single injection of fresh Gaussian noise of scale $c_{t_{N-1}}$ — the **booting noise** — which seeds the first interior state via $x = a x_T + b \hat x_0 + c\cdot\text{noise}$ and accounts for the spread of $x_0$ given $x_T$. Where DDBM spent many churn injections, I spend exactly one, and I return it as the sixth tuple value, the latent that controls diversity. The mirror concern is the end: fresh injected noise on the final step would land straight on the output with nothing left to denoise it, so I drop the fresh-noise term there and keep only the deterministic part.

The full sampler thus predicts at $t_{\max}$ and seeds with the booting noise; then each step evaluates the predictor once at the current larger time $s$, forms the recycled-direction coefficient $\sqrt{c_t^2 - \rho_n^2}/c_s$ and the two endpoint coefficients, and writes the next state as a closed-form linear combination of $\hat x_0$, $x_T$, the current state, and fresh noise (dropped on the final step). One denoiser call per step — five calls buy four interior transitions plus the booting prediction, a genuine 5-NFE sampler. For inpainting I keep the observed pixels fixed each call by re-blending $\hat x_0$ with $x_T$ under the mask. Unlike the reference rung, this editable `sample_dbim` honors the caller's `eta` (default 1.0) and `ts` directly.

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
    x_T = x
    path = []
    pred_x0 = []

    ones = x.new_ones([x.shape[0]])
    indices = range(len(ts) - 1)
    indices = tqdm(indices, disable=(dist.get_rank() != 0))

    nfe = 0
    x0_hat = denoiser(x, diffusion.t_max * ones)
    generator = BatchedSeedGenerator(seed)
    noise = generator.randn_like(x0_hat)
    first_noise = noise
    if mask is not None:
        x0_hat = x0_hat * mask + x_T * (1 - mask)
    x = diffusion.bridge_sample(x0_hat, x_T, ts[0] * ones, noise)
    path.append(x.detach().cpu())
    pred_x0.append(x0_hat.detach().cpu())
    nfe += 1

    for _, i in enumerate(indices):
        s = ts[i]
        t = ts[i + 1]

        x0_hat = denoiser(x, s * ones)
        if mask is not None:
            x0_hat = x0_hat * mask + x_T * (1 - mask)

        a_s, b_s, c_s = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(s * ones)]
        a_t, b_t, c_t = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(t * ones)]

        _, _, rho_s, _ = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_alpha_rho(s * ones)]
        alpha_t, _, rho_t, _ = [
            append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_alpha_rho(t * ones)
        ]

        omega_st = eta * (alpha_t * rho_t) * (1 - rho_t**2 / rho_s**2).sqrt()
        tmp_var = (c_t**2 - omega_st**2).sqrt() / c_s
        coeff_xs = tmp_var
        coeff_x0_hat = b_t - tmp_var * b_s
        coeff_xT = a_t - tmp_var * a_s

        noise = generator.randn_like(x0_hat)

        x = coeff_x0_hat * x0_hat + coeff_xT * x_T + coeff_xs * x + (1 if i != len(ts) - 2 else 0) * omega_st * noise

        path.append(x.detach().cpu())
        pred_x0.append(x0_hat.detach().cpu())
        nfe += 1

    return x, path, nfe, pred_x0, ts, first_noise
```
